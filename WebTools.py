"""
WebTools.py
This module provides tools for web searching and content retrieval.
It includes a web search tool using DuckDuckGo and a content retrieval tool
using BeautifulSoup.
"""

import requests
import re
import traceback
import logging
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urlparse
from typing import List, Dict, Any

# Configure logging
logger = logging.getLogger('mcp-server')

# User agent for web requests
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# DuckDuckGo search URL
DUCKDUCKGO_SEARCH_URL = "https://duckduckgo.com/html"

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