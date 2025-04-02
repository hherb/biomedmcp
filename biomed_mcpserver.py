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
from typing import List, Dict, Any, Optional, Union, Literal
from flask_jsonrpc import JSONRPC

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('mcp-server')

# Configure Flask app
app = Flask(__name__)

# Initialize JSON-RPC
jsonrpc = JSONRPC(app, '/mcp/v1/jsonrpc')

# Define API paths - standard for MCP protocol
MCP_PATH = "/mcp/v1"

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

# MCP Protocol constants
MCP_PROTOCOL_VERSION = "0.1"
MCP_CONTENT_BLOCK_DELIMITER = "```"


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


# MCP Protocol Handler classes
class MCPMessage:
    """Base class for MCP messages"""
    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement to_dict method")


class MCPRequest(MCPMessage):
    """Represents an MCP request from the model to the server"""
    def __init__(
        self,
        request_id: str,
        tool_name: str,
        parameters: Dict[str, Any],
        input_value: Optional[str] = None
    ):
        self.request_id = request_id
        self.tool_name = tool_name
        self.parameters = parameters
        self.input_value = input_value

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "request_id": self.request_id,
            "tool_name": self.tool_name,
            "parameters": self.parameters
        }
        if self.input_value is not None:
            result["input_value"] = self.input_value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPRequest':
        return cls(
            request_id=data.get("request_id", ""),
            tool_name=data.get("tool_name", ""),
            parameters=data.get("parameters", {}),
            input_value=data.get("input_value")
        )


class MCPResponse(MCPMessage):
    """Represents an MCP response from the server to the model"""
    def __init__(
        self,
        request_id: str,
        status: Literal["success", "error"],
        response: Any,
        error: Optional[str] = None
    ):
        self.request_id = request_id
        self.status = status
        self.response = response
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "request_id": self.request_id,
            "status": self.status,
            "response": self.response
        }
        if self.error is not None:
            result["error"] = self.error
        return result


class MCPToolDefinition:
    """Describes an MCP tool definition"""
    def __init__(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        authentication: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.authentication = authentication

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema
        }
        if self.authentication is not None:
            result["authentication"] = self.authentication
        return result


class MCPManifest:
    """Describes the MCP server manifest"""
    def __init__(
        self,
        protocol_version: str,
        tools: List[MCPToolDefinition]
    ):
        self.protocol_version = protocol_version
        self.tools = tools

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "tools": [tool.to_dict() for tool in self.tools]
        }


# Initialize tools
pubmed_tool = PubMedSearchTool(api_key=NCBI_API_KEY)
query_optimizer = QueryOptimizationTool()
web_search_tool = WebSearchTool()
web_content_tool = WebContentTool()


# Define MCP Tool schemas
pubmed_search_schema = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "PubMed search query"
        },
        "max_results": {
            "type": "integer",
            "default": 10,
            "minimum": 1,
            "maximum": 100,
            "description": "Maximum number of results to return"
        },
        "sort": {
            "type": "string",
            "enum": ["relevance", "pub_date", "first_author"],
            "default": "relevance",
            "description": "Sort order for results"
        }
    },
    "required": ["query"]
}

article_retrieval_schema = {
    "type": "object",
    "properties": {
        "pmid": {
            "type": "string",
            "description": "PubMed ID (PMID) of the article to retrieve"
        }
    },
    "required": ["pmid"]
}

query_crafting_schema = {
    "type": "object",
    "properties": {
        "question": {
            "type": "string",
            "description": "Medical question to create a PubMed query for"
        }
    },
    "required": ["question"]
}

web_search_schema = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Web search query"
        },
        "max_results": {
            "type": "integer",
            "default": 20,
            "minimum": 1,
            "maximum": 50,
            "description": "Maximum number of search results to return"
        }
    },
    "required": ["query"]
}

web_content_schema = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "format": "uri",
            "description": "URL to retrieve content from"
        },
        "max_length": {
            "type": "integer",
            "default": 2000,
            "minimum": 100,
            "maximum": 10000,
            "description": "Maximum length of content to return"
        }
    },
    "required": ["url"]
}


# Define MCP tool definitions
MCP_TOOLS = [
    MCPToolDefinition(
        name="pubmed_search",
        description="Search PubMed for articles matching a query",
        input_schema=pubmed_search_schema
    ),
    MCPToolDefinition(
        name="get_article",
        description="Get detailed information about a specific PubMed article by PMID",
        input_schema=article_retrieval_schema
    ),
    MCPToolDefinition(
        name="get_pubmed_query_crafting_prompt",
        description="Get a prompt for an LLM to craft an optimized PubMed query from a question",
        input_schema=query_crafting_schema
    ),
    MCPToolDefinition(
        name="web_search",
        description="Search the web for information on a topic",
        input_schema=web_search_schema
    ),
    MCPToolDefinition(
        name="web_content",
        description="Retrieve and process content from a web URL",
        input_schema=web_content_schema
    )
]

# Create MCP manifest
MCP_MANIFEST = MCPManifest(
    protocol_version=MCP_PROTOCOL_VERSION,
    tools=MCP_TOOLS
)


def execute_tool(request: MCPRequest) -> MCPResponse:
    """
    Execute the specified tool and return the response
    
    Parameters:
    request (MCPRequest): The MCP request containing the tool to execute
    
    Returns:
    MCPResponse: The MCP response containing the result
    """
    tool_name = request.tool_name
    parameters = request.parameters
    request_id = request.request_id
    
    try:
        if tool_name == "pubmed_search":
            query = parameters.get("query")
            max_results = parameters.get("max_results", 10)
            sort = parameters.get("sort", "relevance")
            
            if not query:
                return MCPResponse(
                    request_id=request_id,
                    status="error",
                    response=None,
                    error="Missing required parameter: query"
                )
                
            result = pubmed_tool.search(query, max_results=max_results, sort=sort)
            return MCPResponse(request_id=request_id, status="success", response=result)
            
        elif tool_name == "get_article":
            pmid = parameters.get("pmid")
            
            if not pmid:
                return MCPResponse(
                    request_id=request_id,
                    status="error",
                    response=None,
                    error="Missing required parameter: pmid"
                )
                
            articles = pubmed_tool.fetch_article_details([pmid])
            
            if not articles:
                return MCPResponse(
                    request_id=request_id,
                    status="error",
                    response=None,
                    error=f"Article with PMID {pmid} not found"
                )
                
            return MCPResponse(
                request_id=request_id,
                status="success",
                response={"article": articles[0]}
            )
            
        elif tool_name == "get_pubmed_query_crafting_prompt":
            question = parameters.get("question")
            
            if not question:
                return MCPResponse(
                    request_id=request_id,
                    status="error",
                    response=None,
                    error="Missing required parameter: question"
                )
                
            prompt = query_optimizer.get_pubmed_query_crafting_prompt(question)
            
            return MCPResponse(
                request_id=request_id,
                status="success",
                response={
                    "original_question": question,
                    "prompt": prompt
                }
            )
            
        elif tool_name == "web_search":
            query = parameters.get("query")
            max_results = parameters.get("max_results", 20)
            
            if not query:
                return MCPResponse(
                    request_id=request_id,
                    status="error",
                    response=None,
                    error="Missing required parameter: query"
                )
                
            results = web_search_tool.websearch(query, max_results=max_results)
            
            return MCPResponse(
                request_id=request_id,
                status="success",
                response={
                    "query": query,
                    "count": len(results),
                    "results": results
                }
            )
            
        elif tool_name == "web_content":
            url = parameters.get("url")
            max_length = parameters.get("max_length", 2000)
            
            if not url:
                return MCPResponse(
                    request_id=request_id,
                    status="error",
                    response=None,
                    error="Missing required parameter: url"
                )
                
            content = web_content_tool.webget(url, max_length=max_length)
            
            return MCPResponse(
                request_id=request_id,
                status="success",
                response={
                    "url": url,
                    "content": content
                }
            )
            
        else:
            return MCPResponse(
                request_id=request_id,
                status="error",
                response=None,
                error=f"Unknown tool: {tool_name}"
            )
            
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {str(e)}\n{traceback.format_exc()}")
        return MCPResponse(
            request_id=request_id,
            status="error",
            response=None,
            error=f"Error executing tool: {str(e)}"
        )


# --- MCP API Endpoints ---

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

# Update MCP endpoints with proper API version prefix
@app.route(f'{MCP_PATH}/tools', methods=['GET'])
def get_tools():
    """
    Get the MCP manifest describing available tools
    """
    return jsonify(MCP_MANIFEST.to_dict())

@app.route(f'{MCP_PATH}/execute', methods=['POST'])
def execute():
    """
    Execute a tool request according to the MCP protocol
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "error": "Invalid JSON request"}), 400
            
        mcp_request = MCPRequest.from_dict(data)
        mcp_response = execute_tool(mcp_request)
        
        return jsonify(mcp_response.to_dict())
        
    except Exception as e:
        logger.error(f"Error processing MCP request: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "status": "error",
            "error": f"Error processing request: {str(e)}"
        }), 500

# Add a root endpoint to provide API information
@app.route('/', methods=['GET'])
def api_info():
    """
    Root endpoint that lists available API endpoints
    """
    return jsonify({
        "service": "Biomedical MCP Server",
        "version": "1.0.0",
        "mcp_version": MCP_PROTOCOL_VERSION,
        "endpoints": {
            "health": "/health",
            "mcp_tools": f"{MCP_PATH}/tools",
            "mcp_execute": f"{MCP_PATH}/execute",
            "legacy_search": "/search",
            "legacy_article": "/article/<pmid>",
            "legacy_query_prompt": "/get_pubmed_query_crafting_prompt",
            "legacy_websearch": "/websearch",
            "legacy_webget": "/webget"
        }
    })


# --- Legacy API Endpoints (Maintained for backward compatibility) ---

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


@jsonrpc.method('MCP.execute')
def jsonrpc_execute(request_id: str, tool_name: str, parameters: dict, input_value: Optional[str] = None):
    """
    JSON-RPC method to execute a tool request according to the MCP protocol
    """
    try:
        mcp_request = MCPRequest(
            request_id=request_id,
            tool_name=tool_name,
            parameters=parameters,
            input_value=input_value
        )
        mcp_response = execute_tool(mcp_request)
        return mcp_response.to_dict()
    except Exception as e:
        logger.error(f"Error processing JSON-RPC MCP request: {str(e)}\n{traceback.format_exc()}")
        return {
            "status": "error",
            "error": f"Error processing request: {str(e)}"
        }


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5152))
    
    # Print startup message
    print(f"Starting MCP Server on port {port}")
    print(f"NCBI API Key: {'Configured' if NCBI_API_KEY else 'Not configured - using slower rate limits'}")
    print("Available endpoints:")
    print("  - GET /mcp/v1/tools: Get the MCP tool manifest")
    print("  - POST /mcp/v1/execute: Execute a tool according to the MCP protocol")
    print("  - GET /health: Service health check")
    print("  - GET /search?q=QUERY&max_results=10&sort=relevance: Search PubMed (legacy)")
    print("  - GET /article/PMID: Get detailed article information (legacy)")
    print("  - POST /get_pubmed_query_crafting_prompt: Get a prompt for crafting PubMed queries (legacy)")
    print("  - GET /websearch?q=QUERY&max_results=20: Search the web (legacy)")
    print("  - GET /webget?url=URL&max_length=2000: Retrieve content from URL (legacy)")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=False)