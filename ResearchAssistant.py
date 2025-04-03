#!/usr/bin/env python3
"""
Research Assistant using Ollama LLM with Model Context Protocol (MCP)

This application uses a local LLM model via Ollama to answer research questions.
It autonomously decides whether to:
1. Answer directly from the model's knowledge (if highly confident)
2. Perform a web search (for general information)
3. Search PubMed (for medical/scientific information)

The application adheres to Anthropic's Model Context Protocol (MCP) for communication with tools.
It implements retry mechanisms and provides detailed tracking of tool usage.
"""

import re
import json
import logging
import argparse
import requests
import textwrap
from typing import Dict, List, Any, Optional, Tuple, Union, Literal, Callable
import ollama
import time

from MCPClient import MCPClient, DEFAULT_MCP_URL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('research-assistant')

# Default model configuration
DEFAULT_MODEL_NAME = "phi4:latest"


class PubMedTool:
    """Tool for searching PubMed via the MCP server"""

    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client

    def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Search PubMed for articles matching the query

        Args:
            query: The search query
            max_results: Maximum number of results to return

        Returns:
            Dictionary containing search results or error message
        """
        result = self.mcp_client.execute_tool(
            tool_name="pubmed_search",
            parameters={
                "query": query,
                "max_results": max_results,
                "sort": "relevance"
            }
        )
        return result.get("response", {})

    def get_article(self, pmid: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific article

        Args:
            pmid: PubMed ID of the article

        Returns:
            Dictionary containing article details or error message
        """
        result = self.mcp_client.execute_tool(
            tool_name="get_article",
            parameters={"pmid": pmid}
        )

        return result.get("response", {})

    def get_query_crafting_prompt(self, question: str) -> str:
        """
        Get a prompt for crafting an effective PubMed query

        Args:
            question: The question to create a query for

        Returns:
            A prompt string for an LLM to craft a PubMed query
        """
        result = self.mcp_client.execute_tool(
            tool_name="get_pubmed_query_crafting_prompt",
            parameters={"question": question}
        )

        response_data = result.get("response", {})
        return response_data.get("prompt", "")

    def format_results(self, results: Dict[str, Any]) -> str:
        """
        Format search results into a readable text format

        Args:
            results: Dictionary containing search results

        Returns:
            Formatted string with search results
        """
        if results.get("status") != "success":
            return f"Error: {results.get('message', 'Unknown error')}"

        count = results.get("count", 0)
        if count == 0:
            return "No articles found."

        formatted = f"Found {count} articles:\n\n"

        for idx, article in enumerate(results.get("results", []), 1):
            title = article.get("title", "No title")
            pmid = article.get("pmid", "Unknown")
            authors = article.get("authors", [])
            journal = article.get("journal", "Unknown journal")
            year = "Unknown"

            if article.get("pub_date"):
                # Extract year from pub_date
                year_match = re.search(r'\d{4}', article.get("pub_date", ""))
                if year_match:
                    year = year_match.group(0)

            # Format authors
            author_text = "No authors listed"
            if authors:
                if len(authors) <= 3:
                    author_text = ", ".join(authors)
                else:
                    author_text = f"{', '.join(authors[:3])} et al."

            formatted += f"{idx}. {title}\n"
            formatted += f"   Authors: {author_text}\n"
            formatted += f"   Journal: {journal} ({year})\n"
            formatted += f"   PMID: {pmid}\n\n"

        return formatted

    def format_article(self, article_data: Dict[str, Any]) -> str:
        """
        Format a single article's details into a readable text format

        Args:
            article_data: Dictionary containing article details

        Returns:
            Formatted string with article details
        """
        article = article_data.get("article", {})
        if not article:
            return "No article data available."

        pmid = article.get("pmid", "Unknown")
        title = article.get("title", "No title")
        authors = article.get("authors", [])
        journal = article.get("journal", "Unknown journal")
        pub_date = article.get("pub_date", "Unknown date")
        abstract = article.get("abstract")

        # Format authors
        author_text = "No authors listed"
        if authors:
            if len(authors) <= 5:
                author_text = ", ".join(authors)
            else:
                author_text = f"{', '.join(authors[:5])} et al."

        formatted = f"Title: {title}\n"
        formatted += f"Authors: {author_text}\n"
        formatted += f"Journal: {journal}\n"
        formatted += f"Publication Date: {pub_date}\n"
        formatted += f"PMID: {pmid}\n"

        if abstract:
            formatted += f"\nAbstract:\n{textwrap.fill(abstract, width=80)}\n"
        else:
            formatted += "\nNo abstract available for this article.\n"

        return formatted


class WebTool:
    """Tool for web search and content retrieval via the MCP server"""

    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client

    def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Search the web for information

        Args:
            query: The search query
            max_results: Maximum number of results to return

        Returns:
            Dictionary containing search results
        """
        result = self.mcp_client.execute_tool(
            tool_name="web_search",
            parameters={
                "query": query,
                "max_results": max_results
            }
        )

        return result.get("response", {})

    def get_content(self, url: str, max_length: int = 2000) -> Dict[str, Any]:
        """
        Retrieve content from a URL

        Args:
            url: URL to retrieve content from
            max_length: Maximum length of content to return

        Returns:
            Dictionary containing web content
        """
        result = self.mcp_client.execute_tool(
            tool_name="web_content",
            parameters={
                "url": url,
                "max_length": max_length
            }
        )

        return result.get("response", {})
    
    def format_search_results(self, results: Dict[str, Any]) -> str:
        """
        Format web search results into a readable text format

        Args:
            results: Dictionary containing web search results

        Returns:
            Formatted string with search results
        """
        if "results" not in results:
            return "No web search results found."
        
        count = len(results.get("results", []))
        if count == 0:
            return "No web results found."
        
        formatted = f"Found {count} web results:\n\n"
        
        for idx, result in enumerate(results.get("results", []), 1):
            title = result.get("title", "No title")
            url = result.get("url", "No URL")
            snippet = result.get("snippet", "No description available")
            
            formatted += f"{idx}. {title}\n"
            formatted += f"   URL: {url}\n"
            formatted += f"   Description: {snippet}\n\n"
            
        return formatted
    
    def format_content(self, content_data: Dict[str, Any]) -> str:
        """
        Format web content into a readable text format

        Args:
            content_data: Dictionary containing web content

        Returns:
            Formatted string with web content
        """
        url = content_data.get("url", "Unknown URL")
        content = content_data.get("content", "No content available")
        
        formatted = f"Content from {url}:\n\n"
        formatted += f"{content}\n"
        
        return formatted


class ResearchAssistant:
    """Research assistant using Ollama models with MCP tool capabilities"""

    def __init__(
        self, 
        model_name: str = DEFAULT_MODEL_NAME, 
        mcp_url: str = DEFAULT_MCP_URL,
        default_step_callback: Optional[Callable] = None
    ):
        self.model_name = model_name

        # Initialize MCP client and tools
        self.mcp_client = MCPClient(base_url=mcp_url)
        self.pubmed_tool = PubMedTool(mcp_client=self.mcp_client)
        self.web_tool = WebTool(mcp_client=self.mcp_client)
        
        # Set default step callback if none provided
        self.default_step_callback = default_step_callback or self._default_step_callback

        # Check if Ollama is available
        try:
            models = ollama.list()['models']
            # Check if model is available
            model_names = [model.get('model') for model in models]
            if model_name not in model_names:
                logger.warning(f"Model {model_name} not found. Please pull it with: ollama pull {model_name}")
        except Exception as e:
            logger.error(f"Error connecting to Ollama: {str(e)}")
            logger.error("Make sure Ollama is installed and running.")
            raise RuntimeError("Failed to connect to Ollama server")

    def _default_step_callback(self, why_use_tool: str, tool_used: str, tool_use_results: Dict[str, Any]) -> None:
        """
        Default callback for tool usage steps

        Args:
            why_use_tool: Reason for using the tool
            tool_used: Name of the tool used
            tool_use_results: Results from using the tool
        """
        logger.info(f"Tool usage: {tool_used}")
        logger.info(f"Reason: {why_use_tool}")
        logger.debug(f"Results: {tool_use_results}")
        
        print(f"\n[Tool used: {tool_used}]")
        print(f"[Reason: {why_use_tool}]")
    
    def generate_decision_prompt(self, question: str) -> str:
        """
        Generate a prompt to decide how to answer the question

        Args:
            question: User's question

        Returns:
            Prompt string for the model
        """
        return f"""You are a research assistant powered by an LLM that can use tools to find information.

For the following question, you need to decide the best approach to answer it:
{question}

Choose ONE of these approaches:
1. DIRECT_ANSWER - If you're very confident you have accurate and up-to-date knowledge to answer this question directly without needing to look anything up.
2. WEB_SEARCH - If this is a general knowledge question that would benefit from current information from the web.
3. PUBMED_SEARCH - If this is a medical/scientific question that requires technical or scientific information from peer-reviewed publications.

For PUBMED_SEARCH, also determine if you need help crafting an effective PubMed query:
- PUBMED_WITH_CRAFTING_HELP - If you need assistance crafting an effective PubMed query
- PUBMED_DIRECT_QUERY - If you can create a good PubMed query directly

Answer ONLY with one of these exact options:
- "DIRECT_ANSWER" 
- "WEB_SEARCH"
- "PUBMED_WITH_CRAFTING_HELP"
- "PUBMED_DIRECT_QUERY"
"""

    def decide_approach(self, question: str) -> str:
        """
        Determine the best approach to answer the question

        Args:
            question: User's question

        Returns:
            Decision string (DIRECT_ANSWER, WEB_SEARCH, PUBMED_WITH_CRAFTING_HELP, PUBMED_DIRECT_QUERY)
        """
        prompt = self.generate_decision_prompt(question)

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )

            decision = response['message']['content'].strip()
            logger.info(f"Approach decision: {decision}")

            # Normalize and validate the decision
            decision = decision.upper()
            valid_decisions = [
                "DIRECT_ANSWER", 
                "WEB_SEARCH", 
                "PUBMED_WITH_CRAFTING_HELP", 
                "PUBMED_DIRECT_QUERY"
            ]
            
            if decision not in valid_decisions:
                logger.warning(f"Invalid decision: {decision}, defaulting to WEB_SEARCH")
                return "WEB_SEARCH"
            
            return decision

        except Exception as e:
            logger.error(f"Error querying Ollama for decision: {str(e)}")
            # Default to web search if there's an error
            return "WEB_SEARCH"

    def craft_pubmed_query(self, question: str) -> str:
        """
        Craft an effective PubMed query using the query crafting prompt

        Args:
            question: User's question

        Returns:
            Optimized PubMed query string
        """
        # Get the query crafting prompt from MCP server
        prompt = self.pubmed_tool.get_query_crafting_prompt(question)

        if not prompt:
            logger.warning("Failed to get query crafting prompt, using question as query")
            return question

        try:
            # Ask the LLM to craft the query
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )

            crafted_query = response['message']['content'].strip()
            logger.info(f"Crafted PubMed query: {crafted_query}")

            return crafted_query

        except Exception as e:
            logger.error(f"Error crafting PubMed query: {str(e)}")
            return question  # Fallback to using the question directly
    
    def modify_pubmed_query(self, original_query: str, search_results: Dict[str, Any]) -> str:
        """
        Modify a PubMed query to potentially get better results

        Args:
            original_query: Original PubMed query
            search_results: Results from the previous search

        Returns:
            Modified PubMed query string
        """
        count = search_results.get("count", 0)
        
        prompt = f"""You are a PubMed search expert. You need to modify this query to get better results.

Original query: {original_query}

Current results: {count} articles found.

{"If too many results were found, make the query more specific." if count > 20 else ""}
{"If too few or no results were found, make the query more general or use alternative terms." if count < 3 else ""}

Please provide a modified PubMed query that might yield better results.
Use proper PubMed syntax including operators (AND, OR, NOT) and field tags ([mesh], [tiab], etc.) where appropriate.

Modified query:"""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            
            modified_query = response['message']['content'].strip()
            logger.info(f"Modified PubMed query: {modified_query}")
            
            return modified_query
            
        except Exception as e:
            logger.error(f"Error modifying PubMed query: {str(e)}")
            # Add a small modification as fallback
            if count > 20:
                return original_query + " AND review[pt]"
            elif count < 3:
                return re.sub(r'\[mesh\]', '[tiab]', original_query)
            else:
                return original_query
    
    def modify_web_search_query(self, original_query: str, search_results: Dict[str, Any]) -> str:
        """
        Modify a web search query to potentially get better results

        Args:
            original_query: Original web search query
            search_results: Results from the previous search

        Returns:
            Modified web search query string
        """
        results_count = len(search_results.get("results", []))
        
        prompt = f"""You are a web search expert. You need to modify this query to get more relevant results.

Original query: {original_query}

Current results: {results_count} results found.

{"If the results don't seem relevant or are too few, broaden the query or use alternative terms." if results_count < 3 else ""}
{"If too many irrelevant results were found, make the query more specific." if results_count > 10 else ""}

Please provide a modified web search query that might yield better results.
Be concise and focused on the key concepts.

Modified query:"""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            
            modified_query = response['message']['content'].strip()
            logger.info(f"Modified web search query: {modified_query}")
            
            return modified_query
            
        except Exception as e:
            logger.error(f"Error modifying web search query: {str(e)}")
            # Basic fallback modification
            if results_count < 3:
                # Remove quotes if they exist to broaden
                return re.sub(r'"([^"]+)"', r'\1', original_query)
            else:
                # Make more specific by adding "guide" or "explained"
                return original_query + " guide explained"
    
    def are_search_results_satisfactory(self, results: Dict[str, Any], is_pubmed: bool = False) -> bool:
        """
        Determine if search results are satisfactory

        Args:
            results: Search results to evaluate
            is_pubmed: Whether these are PubMed results (vs web)

        Returns:
            Boolean indicating if results are satisfactory
        """
        if is_pubmed:
            # Check if we have any results for PubMed
            count = results.get("count", 0)
            return count > 0
        else:
            # Check for web results
            web_results = results.get("results", [])
            return len(web_results) > 0
    
    def are_results_relevant(self, question: str, search_results: Any, is_pubmed: bool) -> bool:
        """
        Determine if search results are relevant to the question

        Args:
            question: Original user question
            search_results: Results from search (formatted as string)
            is_pubmed: Whether these are PubMed results (vs web)

        Returns:
            Boolean indicating if results are relevant
        """
        result_type = "PubMed" if is_pubmed else "web search"
        
        prompt = f"""You're a research analyst determining if search results are relevant to a question.

Question: {question}

{result_type} results:
{search_results}

Are these results relevant to answering the question? Answer with ONLY 'YES' or 'NO'.
"""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            
            answer = response['message']['content'].strip().upper()
            if "YES" in answer:
                return True
            else:
                return False
            
        except Exception as e:
            logger.error(f"Error evaluating result relevance: {str(e)}")
            # Assume results are relevant if we can't evaluate
            return True

    def answer_with_pubmed(
        self, 
        question: str, 
        max_retries: int = 3, 
        step_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Answer a question using PubMed search with retries

        Args:
            question: User's research question
            max_retries: Maximum number of query attempts
            step_callback: Callback function for tool usage steps

        Returns:
            Dictionary with answer and tool usage data
        """
        callback = step_callback or self.default_step_callback
        tools_called = []
        context_used = []
        
        # Determine if we need query crafting help
        decision = self.decide_approach(question)
        needs_crafting_help = decision == "PUBMED_WITH_CRAFTING_HELP"
        
        # First attempt at query construction
        if needs_crafting_help:
            # Get prompt and craft query
            callback(
                "Need assistance crafting an effective PubMed query for medical/scientific literature",
                "get_pubmed_query_crafting_prompt",
                {"question": question}
            )
            
            tools_called.append({
                "tool": "get_pubmed_query_crafting_prompt",
                "parameters": {"question": question},
                "reason": "Get assistance for PubMed query formulation"
            })
            
            pubmed_query = self.craft_pubmed_query(question)
        else:
            # Create direct query
            pubmed_query = question
            callback(
                "Creating direct PubMed query based on question",
                "direct_query_construction",
                {"query": pubmed_query}
            )
            
            tools_called.append({
                "tool": "direct_query_construction",
                "parameters": {"question": question},
                "reason": "Create direct PubMed query"
            })
        
        # Search PubMed with retries
        current_query = pubmed_query
        search_results = None
        formatted_results = None
        attempts = 0
        success = False
        
        while attempts < max_retries and not success:
            callback(
                f"Searching PubMed medical literature (attempt {attempts+1}/{max_retries})",
                "pubmed_search",
                {"query": current_query}
            )
            
            tools_called.append({
                "tool": "pubmed_search",
                "parameters": {"query": current_query},
                "reason": f"Search PubMed (attempt {attempts+1})"
            })
            
            # Execute search
            search_results = self.pubmed_tool.search(current_query, max_results=10)
            formatted_results = self.pubmed_tool.format_results(search_results)
            
            # Check if we have satisfactory results
            if self.are_search_results_satisfactory(search_results, is_pubmed=True):
                # Check if results are relevant
                if self.are_results_relevant(question, formatted_results, is_pubmed=True):
                    success = True
                    break
            
            # If we're here, results weren't satisfactory or relevant
            if attempts < max_retries - 1:  # Only modify if we have retries left
                current_query = self.modify_pubmed_query(current_query, search_results)
                time.sleep(1)  # Small delay between retries
            
            attempts += 1
        
        # Get detailed information about top article if available
        article_text = ""
        if search_results and search_results.get("count", 0) > 0 and len(search_results.get("results", [])) > 0:
            top_article = search_results.get("results", [])[0]
            pmid = top_article.get("pmid")
            
            if pmid:
                callback(
                    "Retrieving detailed information about the top research article",
                    "get_article",
                    {"pmid": pmid}
                )
                
                tools_called.append({
                    "tool": "get_article",
                    "parameters": {"pmid": pmid},
                    "reason": "Get detailed article information"
                })
                
                article_details = self.pubmed_tool.get_article(pmid)
                article_text = self.pubmed_tool.format_article(article_details)
                
                # Add to context
                context_used.append({
                    "type": "article_details",
                    "content": article_text
                })
        
        # Add search results to context
        if formatted_results:
            context_used.append({
                "type": "pubmed_results",
                "content": formatted_results
            })
        
        # Combine search results with detailed view of top article
        if article_text:
            pubmed_data = f"{formatted_results}\nDetailed view of top result:\n\n{article_text}"
        else:
            pubmed_data = formatted_results or "No relevant medical literature found."
        
        # Generate answer using PubMed data
        answer = self.generate_answer_with_context(question, pubmed_data)
        
        return {
            "final_answer": answer,
            "context_used_for_final_answer": context_used,
            "tools_called": tools_called
        }

    def answer_with_web_search(
        self, 
        question: str, 
        max_retries: int = 3, 
        step_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Answer a question using web search with retries

        Args:
            question: User's research question
            max_retries: Maximum number of query attempts
            step_callback: Callback function for tool usage steps

        Returns:
            Dictionary with answer and tool usage data
        """
        callback = step_callback or self.default_step_callback
        tools_called = []
        context_used = []
        
        # Initial web search query
        web_query = question
        search_results = None
        formatted_results = None
        attempts = 0
        success = False
        
        while attempts < max_retries and not success:
            callback(
                f"Searching the web for information (attempt {attempts+1}/{max_retries})",
                "web_search",
                {"query": web_query}
            )
            
            tools_called.append({
                "tool": "web_search",
                "parameters": {"query": web_query},
                "reason": f"Search the web (attempt {attempts+1})"
            })
            
            # Execute search
            search_results = self.web_tool.search(web_query, max_results=10)
            formatted_results = self.web_tool.format_search_results(search_results)
            
            # Check if we have satisfactory results
            if self.are_search_results_satisfactory(search_results):
                # Check if results are relevant
                if self.are_results_relevant(question, formatted_results, is_pubmed=False):
                    success = True
                    break
            
            # If we're here, results weren't satisfactory or relevant
            if attempts < max_retries - 1:  # Only modify if we have retries left
                web_query = self.modify_web_search_query(web_query, search_results)
                time.sleep(1)  # Small delay between retries
            
            attempts += 1
        
        # Get content from top result if available
        content_text = ""
        if search_results and len(search_results.get("results", [])) > 0:
            top_result = search_results.get("results", [])[0]
            url = top_result.get("url")
            
            if url:
                callback(
                    "Retrieving detailed content from the top web result",
                    "web_content",
                    {"url": url}
                )
                
                tools_called.append({
                    "tool": "web_content",
                    "parameters": {"url": url},
                    "reason": "Get detailed web content"
                })
                
                content_data = self.web_tool.get_content(url)
                content_text = self.web_tool.format_content(content_data)
                
                # Add to context
                context_used.append({
                    "type": "web_content",
                    "content": content_text
                })
        
        # Add search results to context
        if formatted_results:
            context_used.append({
                "type": "web_search_results",
                "content": formatted_results
            })
        
        # Combine search results with detailed content
        if content_text:
            web_data = f"{formatted_results}\nDetailed content from top result:\n\n{content_text}"
        else:
            web_data = formatted_results or "No relevant web information found."
        
        # Generate answer using web data
        answer = self.generate_answer_with_context(question, web_data)
        
        return {
            "final_answer": answer,
            "context_used_for_final_answer": context_used,
            "tools_called": tools_called
        }

    def answer_directly(
        self, 
        question: str,
        step_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Answer a question directly using the model's knowledge

        Args:
            question: User's research question
            step_callback: Callback function for tool usage steps

        Returns:
            Dictionary with answer and tool usage data
        """
        callback = step_callback or self.default_step_callback
        
        callback(
            "Answering directly from model knowledge (high confidence)",
            "direct_answer",
            {"question": question}
        )
        
        prompt = f"""You are a research assistant powered by a language model with expertise in various fields.

Question: {question}

Please provide a comprehensive and accurate answer based on your knowledge.
Your answer should:
1. Directly address the question
2. Be accurate and evidence-based
3. Explain concepts clearly
4. Acknowledge limitations in your knowledge when relevant
5. Maintain a professional tone

Answer:"""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )

            answer = response['message']['content'].strip()
            
            return {
                "final_answer": answer,
                "context_used_for_final_answer": [],
                "tools_called": [{
                    "tool": "direct_answer",
                    "parameters": {"question": question},
                    "reason": "Question answerable from model knowledge"
                }]
            }

        except Exception as e:
            logger.error(f"Error querying model directly: {str(e)}")
            return {
                "final_answer": f"I apologize, but I encountered an error while generating your answer: {str(e)}",
                "context_used_for_final_answer": [],
                "tools_called": [{
                    "tool": "direct_answer",
                    "parameters": {"question": question},
                    "reason": "Attempted direct answer but failed"
                }]
            }

    def generate_answer_with_context(self, question: str, context: str) -> str:
        """
        Generate answer using retrieved context

        Args:
            question: User's research question
            context: Context information to use for answering

        Returns:
            Answer string
        """
        prompt = f"""You are a research assistant powered by a language model with expertise in various fields.

Question: {question}

I've gathered the following relevant information:

{context}

Please provide a comprehensive answer to the question based on this information.
Your answer should:
1. Directly address the question
2. Synthesize the relevant information from the provided context
3. Cite specific sources when referencing their findings
4. Acknowledge limitations or gaps in the available information when present
5. Maintain a professional, evidence-based tone

Answer:"""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )

            return response['message']['content'].strip()

        except Exception as e:
            logger.error(f"Error generating answer with context: {str(e)}")
            return f"I apologize, but I encountered an error while generating your answer: {str(e)}"

    def answer(
        self, 
        question: str, 
        force_approach: Optional[str] = None, 
        max_retries: int = 3,
        step_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Answer a research question, determining the best approach

        Args:
            question: User's research question
            force_approach: Force a specific approach (DIRECT_ANSWER, WEB_SEARCH, PUBMED_SEARCH)
            max_retries: Maximum number of search retries
            step_callback: Callback function for tool usage steps

        Returns:
            Dictionary with answer and metadata in JSON format
        """
        # Use provided step callback or default
        callback = step_callback or self.default_step_callback
        
        # Determine approach
        if force_approach:
            approach = force_approach.upper()
        else:
            approach = self.decide_approach(question)
        
        # Answer based on selected approach
        if approach == "DIRECT_ANSWER":
            result = self.answer_directly(question, step_callback=callback)
        elif approach in ["PUBMED_WITH_CRAFTING_HELP", "PUBMED_DIRECT_QUERY"]:
            result = self.answer_with_pubmed(question, max_retries, step_callback=callback)
        else:  # WEB_SEARCH or fallback
            result = self.answer_with_web_search(question, max_retries, step_callback=callback)
        
        # Add question and approach to result
        result["question"] = question
        result["approach_used"] = approach
        
        return result


def check_mcp_server(mcp_url: str) -> bool:
    """Check if the MCP server is running"""
    try:
        response = requests.get(f"{mcp_url}/health")
        if response.status_code == 200:
            return True
        return False
    except requests.exceptions.RequestException:
        return False


def main():
    """Main entry point for the CLI application"""
    parser = argparse.ArgumentParser(description="Research Assistant with Tool Integration using MCP")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Ollama model name to use")
    parser.add_argument("--mcp-url", default=DEFAULT_MCP_URL, help="URL of the MCP server")
    parser.add_argument("--force-direct", action="store_true", help="Force using direct model answers")
    parser.add_argument("--force-web", action="store_true", help="Force using web search")
    parser.add_argument("--force-pubmed", action="store_true", help="Force using PubMed search")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of search retries")
    parser.add_argument("--output-json", action="store_true", help="Output raw JSON response")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        # Make requests library more verbose to see HTTP traffic
        requests_logger = logging.getLogger('urllib3')
        requests_logger.setLevel(logging.DEBUG)
        requests_logger.propagate = True
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    logger.info(f"Connecting to MCP server at {args.mcp_url}")

    # Check if MCP server is running
    if not check_mcp_server(args.mcp_url):
        logger.error(f"MCP server at {args.mcp_url} is not running or unreachable")
        print(f"Error: MCP server at {args.mcp_url} is not running or unreachable.")
        print("Please ensure the server is running before starting the assistant.")
        return 1
    else:
        logger.info("MCP server is running")

    # Check conflicting arguments
    force_count = sum([args.force_direct, args.force_web, args.force_pubmed])
    if force_count > 1:
        parser.error("Cannot specify more than one force option")

    force_approach = None
    if args.force_direct:
        force_approach = "DIRECT_ANSWER"
    elif args.force_web:
        force_approach = "WEB_SEARCH"
    elif args.force_pubmed:
        force_approach = "PUBMED_WITH_CRAFTING_HELP"

    try:
        # Initialize assistant
        logger.info(f"Initializing assistant with model: {args.model}")
        assistant = ResearchAssistant(model_name=args.model, mcp_url=args.mcp_url)

        print(f"\n🔍 Research Assistant (MCP Protocol version)")
        print("Type 'exit', 'quit', or Ctrl+C to exit\n")

        while True:
            try:
                question = input("\nEnter your research question: ")

                if question.lower() in ["exit", "quit"]:
                    break

                print("\nProcessing your question...")

                result = assistant.answer(
                    question, 
                    force_approach=force_approach,
                    max_retries=args.max_retries
                )

                # Output format based on user preference
                if args.output_json:
                    print("\n" + json.dumps(result, indent=2))
                else:
                    # Print a user-friendly version
                    approach_used = result.get("approach_used", "")
                    if "DIRECT" in approach_used:
                        print("\n[Answering based on model knowledge]\n")
                    elif "WEB" in approach_used:
                        print("\n[Answering based on web search results]\n")
                    elif "PUBMED" in approach_used:
                        print("\n[Answering based on PubMed medical literature]\n")
                    
                    print(result.get("final_answer", "No answer generated"))

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error processing question: {str(e)}")
                print(f"\nI encountered an error: {str(e)}")

    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        print(f"Failed to initialize: {str(e)}")

    print("\nThank you for using the Research Assistant!")
    return 0


if __name__ == "__main__":
    exit(main())