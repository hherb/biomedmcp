"""
PubMedTools.py
This module provides basic pubmed search tools for searching PubMed and retrieving article details.
"""
import os
import json
import requests
import logging
import xml.etree.ElementTree as ET
from typing import List, Dict, Any

# Configure logging
logger = logging.getLogger('mcp-server')

# PubMed API configuration
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_SEARCH_URL = f"{PUBMED_BASE_URL}/esearch.fcgi"
PUBMED_FETCH_URL = f"{PUBMED_BASE_URL}/efetch.fcgi"
PUBMED_SUMMARY_URL = f"{PUBMED_BASE_URL}/esummary.fcgi"

class PubMedSearchTool:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.params = {"api_key": api_key} if api_key else {}
        
    def search(self, query, max_results=10, sort="relevance"):
        """
        Search PubMed for articles matching the query
        
        Parameters:
        query (str): The search query. Consider using a tool to craft the query to ensure it is optimized for PubMed.
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
You translate medical questions to PubMed queries.

ANALYZE:
-PICO: Patient/Population, Intervention, Comparison, Outcome
-Extract: conditions, treatments, timeframes, demographics, study types

SYNTAX:
-MeSH: "term"[mesh], "term"[majr], "term"[mesh:noexp]
-Fields: [ti]=title, [tiab]=title/abstract, [tw]=text word, [au]=author, [pt]=publication type, [dp]=date, [la]=language, [subh]=subheading
-Boolean: AND, OR, NOT (capitalized)
-Group with (parentheses)
-Phrases use "quotes"
-Truncation: word*
-Dates: ("2018"[dp]:"3000"[dp])

FILTERS:
-Types: "review"[pt], "clinical trial"[pt], "randomized controlled trial"[pt]
-Other: "humans"[mesh], "english"[la], "free full text[sb]"

STRATEGY:
1.Core concepts→MeSH+[tiab]
2.Group synonyms with OR in (parentheses)
3.Connect concept groups with AND
4.Add filters last

OUTPUT:
1.PubMed query string (complete, copy-paste ready)
2.Brief component explanation
3.Alternative terms if needed

EXAMPLE:
Question: "SGLT2 inhibitors for heart failure in diabetics?"
Query: ("sodium glucose transporter 2 inhibitors"[mesh] OR sglt2 inhibitor*[tiab]) AND ("heart failure"[mesh] OR "heart failure"[tiab]) AND ("diabetes mellitus"[mesh] OR diabetes[tiab]) AND ("treatment outcome"[mesh] OR efficacy[tiab])

## Task:
Now create an optimal PubMed query for this question:

<question>{question}</question>

## OUTPUT:
Return ONLY the final PubMed query string, ready to be copied and pasted into PubMed's search box.
Do not include any explanations or additional text.
## Example:
User: What are the latest advances in using AI for cancer diagnosis?
Assitant: ("artificial intelligence"[tiab] OR "AI"[tiab] OR "machine learning"[tiab]) AND ("cancer diagnosis"[tiab] OR "oncology"[mesh]) AND ("2020/01/01"[dp] : "2023/12/31"[dp])
"""
    
    @staticmethod
    def get_pubmed_query_crafting_prompt(human_question: str) -> str:
        """
        Generate a prompt for an LLM to craft an optimized PubMed search query. 
        This prompt can the n be used to run a pubmed query.
        The prompt includes PubMed search syntax rules, advanced techniques, and an example.
        
        Parameters:
        human_question (str): Plain language question about medical research
        
        Returns:
        str: Prompt for LLM to generate a PubMed query that can be used to run a query on the pubmed server
        """
        return QueryOptimizationTool.PUBMED_QUERY_PROMPT_TEMPLATE.format(question=human_question)