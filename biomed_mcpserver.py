import os
import json
import requests
import re
import time
import traceback
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from urllib.parse import quote_plus, urlparse
import xml.etree.ElementTree as ET
from datetime import datetime
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('mcp-server')

app = Flask(__name__)

# PubMed API configuration
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_SEARCH_URL = f"{PUBMED_BASE_URL}/esearch.fcgi"
PUBMED_FETCH_URL = f"{PUBMED_BASE_URL}/efetch.fcgi"
PUBMED_SUMMARY_URL = f"{PUBMED_BASE_URL}/esummary.fcgi"

# You should replace this with your own NCBI API key if you have one
# Get one here: https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/
NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")  

# If you have an API key, you get higher rate limits
REQUEST_DELAY = 0.34 if NCBI_API_KEY else 1.0  # seconds between requests (NCBI guidelines)

# User agent for web requests
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# DuckDuckGo search URL
DUCKDUCKGO_SEARCH_URL = "https://duckduckgo.com/html"


class PubMedSearchTool:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.params = {"api_key": api_key} if api_key else {}
        
    def search(self, query, max_results=10, sort="relevance"):
        """
        Search PubMed for articles matching the query
        
        Parameters:
        query (str): The search query
        max_results (int): Maximum number of results to return
        sort (str): Sort order - "relevance", "pub_date", or "first_author"
        
        Returns:
        dict: Search results with PMIDs, titles, and metadata
        """
        logger.info(f"Searching PubMed for: {query}")
        
        # Convert sort parameter to PubMed format
        sort_param = None
        if sort == "pub_date":
            sort_param = "pub+date"
        elif sort == "first_author":
            sort_param = "first+author"
            
        # Build search parameters
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max_results,
            **self.params
        }
        
        if sort_param:
            search_params["sort"] = sort_param
        
        # Execute search to get PMIDs
        try:
            response = requests.get(PUBMED_SEARCH_URL, params=search_params)
            response.raise_for_status()
            search_data = response.json()
            
            if 'esearchresult' not in search_data or 'idlist' not in search_data['esearchresult']:
                logger.warning(f"Unexpected response format from PubMed search: {search_data}")
                return {"status": "error", "message": "Invalid response from PubMed"}
                
            pmids = search_data['esearchresult']['idlist']
            
            if not pmids:
                return {"status": "success", "count": 0, "results": []}
                
            # Get detailed information for each article
            articles = self.fetch_article_details(pmids)
            
            return {
                "status": "success",
                "count": len(articles),
                "query": query,
                "results": articles
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"PubMed search request failed: {str(e)}")
            return {"status": "error", "message": f"PubMed search request failed: {str(e)}"}
        except json.JSONDecodeError:
            logger.error(f"Failed to parse PubMed response as JSON")
            return {"status": "error", "message": "Failed to parse PubMed response"}
        except Exception as e:
            logger.error(f"Unexpected error in PubMed search: {str(e)}")
            return {"status": "error", "message": f"Unexpected error: {str(e)}"}
    
    def fetch_article_details(self, pmids):
        """
        Fetch detailed information for articles by their PMIDs
        
        Parameters:
        pmids (list): List of PubMed IDs
        
        Returns:
        list: Detailed article information
        """
        if not pmids:
            return []
            
        pmids_str = ",".join(pmids)
        
        # Use esummary to get article details
        summary_params = {
            "db": "pubmed",
            "id": pmids_str,
            "retmode": "json",
            **self.params
        }
        
        try:
            response = requests.get(PUBMED_SUMMARY_URL, params=summary_params)
            response.raise_for_status()
            
            summary_data = response.json()
            if 'result' not in summary_data:
                logger.warning(f"Unexpected summary response format from PubMed")
                return []
                
            # Extract the useful fields from each article
            articles = []
            for pmid in pmids:
                if pmid not in summary_data['result']:
                    continue
                    
                article_data = summary_data['result'][pmid]
                
                # Extract and format authors
                authors = []
                if 'authors' in article_data:
                    for author in article_data['authors']:
                        if 'name' in author:
                            authors.append(author['name'])
                
                # Extract publication date
                pub_date = None
                if 'pubdate' in article_data:
                    pub_date = article_data['pubdate']
                
                # Build article object
                article = {
                    "pmid": pmid,
                    "title": article_data.get('title', 'No title available'),
                    "authors": authors,
                    "journal": article_data.get('fulljournalname', article_data.get('source', 'Unknown journal')),
                    "pub_date": pub_date,
                    "abstract": self.fetch_abstract(pmid),
                    "doi": article_data.get('elocationid', '').replace('doi: ', '') if 'elocationid' in article_data else None,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                }
                
                articles.append(article)
                
            return articles
            
        except requests.exceptions.RequestException as e:
            logger.error(f"PubMed summary request failed: {str(e)}")
            return []
        except json.JSONDecodeError:
            logger.error(f"Failed to parse PubMed summary response as JSON")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in PubMed summary fetch: {str(e)}")
            return []
    
    def fetch_abstract(self, pmid):
        """
        Fetch the abstract for a specific article
        
        Parameters:
        pmid (str): PubMed ID of the article
        
        Returns:
        str: Abstract text or None if not available
        """
        abstract_params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml",
            **self.params
        }
        
        try:
            response = requests.get(PUBMED_FETCH_URL, params=abstract_params)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            abstract_elements = root.findall('.//AbstractText')
            
            if abstract_elements:
                # Combine all abstract sections
                abstract_parts = []
                for elem in abstract_elements:
                    label = elem.get('Label')
                    text = elem.text or ""
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
                
                return " ".join(abstract_parts)
            else:
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"PubMed abstract request failed for PMID {pmid}: {str(e)}")
            return None
        except ET.ParseError:
            logger.error(f"Failed to parse PubMed XML response for PMID {pmid}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching abstract for PMID {pmid}: {str(e)}")
            return None


class QueryOptimizationTool:
    """
    Tool to provide prompts for LLMs to craft effective PubMed search queries
    """
    
    # Default prompt template for PubMed query generation
    PUBMED_QUERY_PROMPT_TEMPLATE = """
You are an expert medical librarian specializing in crafting precise and effective PubMed search queries.

## PubMed Search Syntax Rules:
- A PubMed query consists of search terms joined with the logical operators AND, OR, and NOT (must be CAPITALIZED).
- Multi-word terms must be enclosed in double quotes: "heart attack".
- Group terms with parentheses: (heart attack OR "myocardial infarction") AND aspirin.
- Use these common field tags to refine searches:
  * [mesh] - For Medical Subject Headings (controlled vocabulary terms)
  * [tiab] - Searches title and abstract fields
  * [au] - Author search
  * [affl] - Affiliation search
  * [dp] - Date of publication in YYYY/MM/DD format
  * [pt] - Publication type (e.g., review, clinical trial)
  * [majr] - MeSH Major Topic (focuses on key concepts)
  * [subh] - MeSH Subheading
  * [tw] - Text word (searches multiple text fields)
  * [la] - Language

## Advanced PubMed Search Techniques:
- Use MeSH terms to capture all related concepts: "myocardial infarction"[mesh] is more comprehensive than just the text search.
- For comprehensive searches of a concept, combine MeSH terms with text terms: hypertension[mesh] OR hypertension[tiab]
- For recent content not yet indexed with MeSH, use the [tiab] tag.
- Date ranges use format: ("2020"[dp] : "2023"[dp])
- Use "filters" for specific article types: "clinical trial"[pt]
- Use the "explosion" feature of MeSH by default (searches narrower terms automatically)
- More specific searches use multiple concepts joined with AND
- More sensitive (comprehensive) searches use OR to combine synonyms

## Task:
Based on these rules, construct a PubMed query for the following question:

<question>{question}</question>

Create a search strategy that:
1. Includes all key concepts from the question
2. Uses appropriate MeSH terms where possible
3. Includes synonyms for important concepts (combined with OR)
4. Uses field tags appropriately to focus the search
5. Balances specificity and sensitivity based on the question's needs

Return ONLY the final PubMed query string, ready to be copied and pasted into PubMed's search box.
"""
    
    @staticmethod
    def get_pubmed_query_crafting_prompt(human_question: str) -> str:
        """
        Generate a prompt for an LLM to craft an optimized PubMed search query
        
        Parameters:
        human_question (str): Plain language question about medical research
        
        Returns:
        str: Prompt for LLM to generate a PubMed query
        """
        return QueryOptimizationTool.PUBMED_QUERY_PROMPT_TEMPLATE.format(question=human_question)


class WebSearchTool:
    """
    Tool to search the web for information relevant to a question
    """
    
    @staticmethod
    def websearch(question: str, max_results: int = 20) -> List[Dict[str, str]]:
        """
        Perform a web search using DuckDuckGo and return relevant URLs
        
        Parameters:
        question (str): Plain language question to search for
        max_results (int): Maximum number of results to return
        
        Returns:
        list: List of dictionaries containing title and URL for each result
        """
        logger.info(f"Performing web search for: {question}")
        
        try:
            # Format the search query
            query = quote_plus(question)
            
            # Set up headers to mimic a browser
            headers = {
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://duckduckgo.com/",
                "DNT": "1"  # Do Not Track request header
            }
            
            # Request parameters
            params = {
                "q": query,
                "kl": "us-en"  # Region and language: US English
            }
            
            # Make the search request
            response = requests.get(
                DUCKDUCKGO_SEARCH_URL, 
                headers=headers, 
                params=params
            )
            response.raise_for_status()
            
            # Parse the HTML response
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract search results
            results = []
            result_elements = soup.select('.result')
            
            for element in result_elements:
                if len(results) >= max_results:
                    break
                    
                # Extract title and URL
                title_element = element.select_one('.result__a')
                if not title_element:
                    continue
                    
                title = title_element.get_text(strip=True)
                url = title_element.get('href')
                
                # DuckDuckGo links are redirects, extract the actual URL
                if url and url.startswith('/'):
                    url_params = urlparse(url).query.split('&')
                    for param in url_params:
                        if param.startswith('uddg='):
                            url = param[5:]
                            break
                
                if url:
                    url = requests.utils.unquote(url)
                    
                    # Skip certain non-content URLs
                    if any(domain in url for domain in ['youtube.com', 'reddit.com', 'twitter.com', 'facebook.com']):
                        continue
                        
                    # Add to results
                    results.append({
                        "title": title,
                        "url": url
                    })
            
            logger.info(f"Web search found {len(results)} results")
            return results
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Web search request failed: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in web search: {str(e)}\n{traceback.format_exc()}")
            return []


class WebContentTool:
    """
    Tool to retrieve and process web content
    """
    
    @staticmethod
    def webget(url: str, max_length: int = 2000) -> str:
        """
        Retrieve and process content from a web URL
        
        Parameters:
        url (str): URL to retrieve content from
        max_length (int): Maximum length of content to return
        
        Returns:
        str: Processed web content optimized for LLM consumption
        """
        logger.info(f"Retrieving web content from: {url}")
        
        try:
            # Set up headers to mimic a browser
            headers = {
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml",
                "Accept-Language": "en-US,en;q=0.9"
            }
            
            # Make the request
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()
            
            # Extract title
            title = soup.title.string if soup.title else "No title"
            
            # Try to extract the main content
            main_content = WebContentTool._extract_main_content(soup)
            
            # Format the content
            content = f"Title: {title}\nURL: {url}\n\nContent:\n{main_content}"
            
            # Truncate to max_length if needed
            if len(content) > max_length:
                content = content[:max_length] + "... [content truncated]"
            
            return content
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Web content request failed for {url}: {str(e)}")
            return f"Error retrieving content from {url}: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error retrieving web content from {url}: {str(e)}\n{traceback.format_exc()}")
            return f"Error processing content from {url}: {str(e)}"
    
    @staticmethod
    def _extract_main_content(soup: BeautifulSoup) -> str:
        """
        Extract the main content from a webpage
        
        Parameters:
        soup (BeautifulSoup): Parsed HTML content
        
        Returns:
        str: Extracted main content
        """
        # Try to find the main content container
        main_candidates = [
            soup.find('main'),
            soup.find('article'),
            soup.find(id=re.compile(r'content|main|article', re.I)),
            soup.find(class_=re.compile(r'content|main|article', re.I))
        ]
        
        content_element = next((e for e in main_candidates if e is not None), soup.body)
        
        if not content_element:
            # Fallback to body if no specific content container found
            content_element = soup.body
        
        if not content_element:
            return "Could not extract content from webpage."
        
        # Extract paragraphs
        paragraphs = []
        
        # Get headings
        headings = content_element.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        for heading in headings:
            text = heading.get_text(strip=True)
            if text:
                heading_level = int(heading.name[1])
                heading_prefix = '#' * heading_level
                paragraphs.append(f"{heading_prefix} {text}")
        
        # Get paragraphs
        for p in content_element.find_all('p'):
            text = p.get_text(strip=True)
            if text and len(text) > 20:  # Skip very short paragraphs
                paragraphs.append(text)
        
        # Get list items
        for ul in content_element.find_all('ul'):
            for li in ul.find_all('li'):
                text = li.get_text(strip=True)
                if text:
                    paragraphs.append(f"• {text}")
        
        for ol in content_element.find_all('ol'):
            for i, li in enumerate(ol.find_all('li')):
                text = li.get_text(strip=True)
                if text:
                    paragraphs.append(f"{i+1}. {text}")
        
        # Join the content with newlines
        content = "\n\n".join(paragraphs)
        
        # Handle empty content
        if not content.strip():
            # Fallback to all text
            content = content_element.get_text(separator='\n\n', strip=True)
        
        return content


# Initialize tools
pubmed_tool = PubMedSearchTool(api_key=NCBI_API_KEY)
query_optimizer = QueryOptimizationTool()
web_search_tool = WebSearchTool()
web_content_tool = WebContentTool()

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify the server is running
    """
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "MCP Server"
    })

@app.route('/search', methods=['GET'])
def search_pubmed():
    """
    Endpoint to search PubMed based on query parameters
    
    Query parameters:
    - q: Search query (required)
    - max_results: Maximum number of results to return (default: 10)
    - sort: Sort order - "relevance", "pub_date", or "first_author" (default: "relevance")
    """
    query = request.args.get('q')
    if not query:
        return jsonify({"status": "error", "message": "Query parameter 'q' is required"}), 400
        
    try:
        max_results = int(request.args.get('max_results', 10))
        if max_results < 1:
            max_results = 10
        elif max_results > 100:
            max_results = 100  # Limit to reasonable number
    except ValueError:
        max_results = 10
        
    sort = request.args.get('sort', 'relevance')
    if sort not in ['relevance', 'pub_date', 'first_author']:
        sort = 'relevance'
        
    # Execute search
    result = pubmed_tool.search(query, max_results=max_results, sort=sort)
    
    return jsonify(result)

@app.route('/article/<pmid>', methods=['GET'])
def get_article(pmid):
    """
    Endpoint to get detailed information about a specific article by PMID
    """
    try:
        # Fetch the single article
        articles = pubmed_tool.fetch_article_details([pmid])
        
        if not articles:
            return jsonify({"status": "error", "message": f"Article with PMID {pmid} not found"}), 404
            
        return jsonify({"status": "success", "article": articles[0]})
        
    except Exception as e:
        logger.error(f"Error retrieving article {pmid}: {str(e)}")
        return jsonify({"status": "error", "message": f"Error retrieving article: {str(e)}"}), 500

@app.route('/get_pubmed_query_crafting_prompt', methods=['POST'])
def get_pubmed_query_crafting_prompt():
    """
    Endpoint to get a prompt for an LLM to craft an optimized PubMed query
    
    Request body (JSON):
    {
        "question": "Plain language question about medical research"
    }
    
    Returns:
    JSON with a prompt for LLM to generate a PubMed query
    """
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                "status": "error", 
                "message": "Request must include 'question' field"
            }), 400
            
        question = data['question']
        prompt = query_optimizer.get_pubmed_query_crafting_prompt(question)
        
        return jsonify({
            "status": "success",
            "original_question": question,
            "prompt": prompt
        })
        
    except Exception as e:
        logger.error(f"Error creating PubMed query prompt: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "status": "error", 
            "message": f"Error creating PubMed query prompt: {str(e)}"
        }), 500

@app.route('/websearch', methods=['GET'])
def web_search():
    """
    Endpoint to search the web for information
    
    Query parameters:
    - q: Search query (required)
    - max_results: Maximum number of results to return (default: 20)
    
    Returns:
    JSON with list of search results (titles and URLs)
    """
    question = request.args.get('q')
    if not question:
        return jsonify({
            "status": "error", 
            "message": "Query parameter 'q' is required"
        }), 400
        
    try:
        max_results = int(request.args.get('max_results', 20))
        if max_results < 1:
            max_results = 20
        elif max_results > 50:
            max_results = 50  # Limit to reasonable number
    except ValueError:
        max_results = 20
        
    # Execute search
    results = web_search_tool.websearch(question, max_results=max_results)
    
    return jsonify({
        "status": "success",
        "query": question,
        "count": len(results),
        "results": results
    })

@app.route('/webget', methods=['GET'])
def web_get():
    """
    Endpoint to retrieve and process content from a URL
    
    Query parameters:
    - url: URL to retrieve content from (required)
    - max_length: Maximum length of content to return (default: 2000)
    
    Returns:
    JSON with processed web content
    """
    url = request.args.get('url')
    if not url:
        return jsonify({
            "status": "error", 
            "message": "Query parameter 'url' is required"
        }), 400
        
    try:
        max_length = int(request.args.get('max_length', 2000))
        if max_length < 100:
            max_length = 100
        elif max_length > 10000:
            max_length = 10000  # Limit to reasonable length
    except ValueError:
        max_length = 2000
        
    # Retrieve and process content
    content = web_content_tool.webget(url, max_length=max_length)
    
    return jsonify({
        "status": "success",
        "url": url,
        "content": content
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5152))
    
    # Print startup message
    print(f"Starting MCP Server on port {port}")
    print(f"NCBI API Key: {'Configured' if NCBI_API_KEY else 'Not configured - using slower rate limits'}")
    print("Available endpoints:")
    print("  - GET /health: Service health check")
    print("  - GET /search?q=QUERY&max_results=10&sort=relevance: Search PubMed")
    print("  - GET /article/PMID: Get detailed article information")
    print("  - POST /get_pubmed_query_crafting_prompt: Get a prompt for crafting PubMed queries")
    print("  - GET /websearch?q=QUERY&max_results=20: Search the web")
    print("  - GET /webget?url=URL&max_length=2000: Retrieve content from URL")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=False)