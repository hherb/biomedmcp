#!/usr/bin/env python3
"""
Phi-4 Medical Assistant with PubMed Tool Integration

This application uses the Phi-4 model via Ollama to answer medical questions.
It can autonomously decide when to use a PubMed search tool to provide evidence-based responses.
"""

import os
import re
import json
import logging
import argparse
import requests
import textwrap
from typing import Dict, List, Any, Optional, Tuple

import ollama

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mcp-pubmed')

# MCP PubMed Server configuration
DEFAULT_MCP_URL = "http://localhost:5152"
DEFAULT_MODEL_NAME = "phi4:latest"

class PubMedTool:
    """Tool for searching PubMed via the MCP server"""
    
    def __init__(self, mcp_url: str = DEFAULT_MCP_URL):
        self.mcp_url = mcp_url
        self.base_url = f"{mcp_url}"
        
    def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Search PubMed for articles matching the query
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary containing search results or error message
        """
        try:
            url = f"{self.base_url}/search"
            params = {
                "q": query,
                "max_results": max_results,
                "sort": "relevance"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to MCP server: {str(e)}")
            return {"status": "error", "message": f"Error connecting to MCP server: {str(e)}"}
            
    def get_article(self, pmid: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific article
        
        Args:
            pmid: PubMed ID of the article
            
        Returns:
            Dictionary containing article details or error message
        """
        try:
            url = f"{self.base_url}/article/{pmid}"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching article {pmid}: {str(e)}")
            return {"status": "error", "message": f"Error fetching article: {str(e)}"}
    
    def get_query_crafting_prompt(self, question: str) -> str:
        """
        Get a prompt for crafting an effective PubMed query
        
        Args:
            question: The medical question to create a query for
            
        Returns:
            A prompt string for an LLM to craft a PubMed query
        """
        try:
            url = f"{self.base_url}/get_pubmed_query_crafting_prompt"
            data = {"question": question}
            
            response = requests.post(url, json=data, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            if result.get("status") == "success":
                return result.get("prompt", "")
            else:
                logger.error(f"Error getting query prompt: {result.get('message')}")
                return ""
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to MCP server: {str(e)}")
            return ""

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
        
    def format_article(self, result: Dict[str, Any]) -> str:
        """
        Format a single article's details into a readable text format
        
        Args:
            result: Dictionary containing article details
            
        Returns:
            Formatted string with article details
        """
        if result.get("status") != "success":
            return f"Error: {result.get('message', 'Unknown error')}"
            
        article = result.get("article", {})
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

class Assistant:
    f"""Medical assistant using Ollama models ({DEFAULT_MODEL_NAME}) with tool capabilities"""
    
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, mcp_url: str = DEFAULT_MCP_URL):
        self.model_name = model_name
        self.pubmed_tool = PubMedTool(mcp_url=mcp_url)
        
        # Check if Ollama is available
        try:
            models = ollama.list()['models']
            #logger.info(f"Connected to Ollama. Available models: {[m.get('name') for m in self.models.get('models', [])]}")
            # Check if model is available
            model_names = [model.get('model') for model in models]
            if model_name not in model_names:
                logger.warning(f"Model {model_name} not found. Please pull it with: ollama pull {model_name}")
        except Exception as e:
            logger.error(f"Error connecting to Ollama: {str(e)}")
            logger.error("Make sure Ollama is installed and running.")
            raise RuntimeError("Failed to connect to Ollama server")
            
    def generate_tool_prompt(self, question: str) -> str:
        """
        Generate a prompt to ask if the PubMed tool should be used
        
        Args:
            question: User's medical question
            
        Returns:
            Prompt string for the model
        """
        return f"""You are a medical assistant powered by the Phi-4 language model that can use a PubMed search tool when needed.

For the following medical question, decide if you need to search for recent medical literature using the PubMed tool.

Question: {question}

Before answering, determine if this question:
1. Requires current medical research or evidence
2. Would benefit from citing specific studies or journals
3. Involves recent medical advances, treatments, or guidelines
4. Asks about the state of research on a specific condition or treatment

If ANY of these are true, respond with: "NEEDS_PUBMED_SEARCH: <yes_or_no>"
Where <yes_or_no> is replaced with:
- "YES_CRAFT_QUERY" if you need a PubMed search AND need help crafting an effective PubMed query
- "YES_DIRECT_QUERY" if you need a PubMed search AND can create a direct query yourself

Otherwise, if you can answer without PubMed, respond with: "NO_PUBMED_SEARCH_NEEDED"

Your response (ONLY choose one of these exact formats):"""
        
    def generate_answer_prompt(self, question: str, pubmed_results: str) -> str:
        """
        Generate a prompt to answer the question using PubMed results
        
        Args:
            question: User's medical question
            pubmed_results: Formatted PubMed search results
            
        Returns:
            Prompt string for the model
        """
        return f"""You are a medical assistant powered by the Phi-4 language model with expertise in emergency medicine and medical informatics.

Question: {question}

I've searched the medical literature and found the following relevant information:

{pubmed_results}

Please provide a comprehensive answer to the question based on this information. 
Your answer should:
1. Directly address the question
2. Synthesize the relevant information from the literature
3. Cite the specific studies when referencing their findings (use PMID numbers)
4. Acknowledge limitations or gaps in the evidence when present
5. Maintain a professional, evidence-based tone

Answer:"""

    def should_use_pubmed(self, question: str) -> Tuple[bool, bool, Optional[str]]:
        """
        Determine if the question requires PubMed search and how to formulate the query
        
        Args:
            question: User's medical question
            
        Returns:
            Tuple of (should_use_pubmed, needs_crafting_help, search_terms)
        """
        prompt = self.generate_tool_prompt(question)
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            
            decision_text = response['message']['content'].strip()
            logger.info(f"Tool decision: {decision_text}")
            
            if "YES_CRAFT_QUERY" in decision_text:
                # Needs PubMed search and query crafting help
                return True, True, None
            elif "YES_DIRECT_QUERY" in decision_text:
                # Extract search terms from the model's response or use default
                search_terms = question  # Default to using the question
                
                # Try to extract specific search terms if provided after the decision
                if ":" in decision_text:
                    search_terms = decision_text.split(":", 1)[1].strip()
                    if search_terms == "YES_DIRECT_QUERY":
                        # No specific terms provided, use the question
                        search_terms = question
                
                return True, False, search_terms
            else:
                # No PubMed search needed
                return False, False, None
                
        except Exception as e:
            logger.error(f"Error querying Ollama: {str(e)}")
            # Default to not using PubMed if there's an error
            return False, False, None
    
    def craft_pubmed_query(self, question: str) -> str:
        """
        Craft an effective PubMed query using the query crafting prompt
        
        Args:
            question: User's medical question
            
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
            
    def answer_with_pubmed(self, question: str, search_terms: Optional[str] = None, needs_crafting: bool = False) -> str:
        """
        Answer a question using PubMed search results
        
        Args:
            question: User's medical question
            search_terms: Terms to search for in PubMed (or None to craft them)
            needs_crafting: Whether query crafting help is needed
            
        Returns:
            Answer string
        """
        # Determine search terms
        if needs_crafting or search_terms is None:
            logger.info("Crafting optimized PubMed query...")
            search_terms = self.craft_pubmed_query(question)
        
        logger.info(f"Searching PubMed for: {search_terms}")
        
        # Search PubMed
        search_results = self.pubmed_tool.search(search_terms, max_results=5)
        formatted_results = self.pubmed_tool.format_results(search_results)
        
        # Get article details for the top result if available
        if search_results.get("status") == "success" and search_results.get("count", 0) > 0:
            top_article = search_results.get("results", [])[0]
            pmid = top_article.get("pmid")
            
            if pmid:
                article_details = self.pubmed_tool.get_article(pmid)
                article_text = self.pubmed_tool.format_article(article_details)
                
                # Combine search results with detailed view of top article
                pubmed_data = f"{formatted_results}\nDetailed view of top result:\n\n{article_text}"
            else:
                pubmed_data = formatted_results
        else:
            pubmed_data = formatted_results
            
        # Generate answer using PubMed data
        prompt = self.generate_answer_prompt(question, pubmed_data)
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response['message']['content'].strip()
            
        except Exception as e:
            logger.error(f"Error querying Ollama: {str(e)}")
            return f"I apologize, but I encountered an error while generating your answer: {str(e)}"
            
    def answer_directly(self, question: str) -> str:
        """
        Answer a question without using PubMed
        
        Args:
            question: User's medical question
            
        Returns:
            Answer string
        """
        prompt = f"""You are a medical assistant powered by the Phi-4 language model with expertise in emergency medicine and medical informatics.

Question: {question}

Please provide a comprehensive and accurate answer based on your medical knowledge.
Your answer should:
1. Directly address the question
2. Be accurate and evidence-based
3. Explain medical concepts clearly
4. Maintain a professional tone

Answer:"""
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response['message']['content'].strip()
            
        except Exception as e:
            logger.error(f"Error querying Ollama: {str(e)}")
            return f"I apologize, but I encountered an error while generating your answer: {str(e)}"
            
    def answer(self, question: str, force_tool: Optional[bool] = None) -> Dict[str, Any]:
        """
        Answer a medical question, using PubMed if necessary
        
        Args:
            question: User's medical question
            force_tool: If True, force using PubMed. If False, force not using it. If None, decide automatically.
            
        Returns:
            Dictionary with answer and metadata
        """
        result = {
            "question": question,
            "used_pubmed": False,
            "used_query_crafting": False,
            "search_terms": None,
            "answer": ""
        }
        
        # Determine if we should use PubMed
        if force_tool is None:
            use_pubmed, needs_crafting, search_terms = self.should_use_pubmed(question)
        else:
            use_pubmed = force_tool
            needs_crafting = True if force_tool else False
            search_terms = None if force_tool else None
            
        # Answer with or without PubMed
        if use_pubmed:
            result["used_pubmed"] = True
            result["used_query_crafting"] = needs_crafting
            
            if needs_crafting:
                # Get a crafted query
                crafted_query = self.craft_pubmed_query(question)
                result["search_terms"] = crafted_query
                result["answer"] = self.answer_with_pubmed(question, crafted_query, False)
            else:
                # Use direct search terms
                result["search_terms"] = search_terms
                result["answer"] = self.answer_with_pubmed(question, search_terms, False)
        else:
            result["answer"] = self.answer_directly(question)
            
        return result

def main():
    # Check if the MCP server is running - no  point continuing without it
    try:
        response = requests.get(f"{DEFAULT_MCP_URL}/health")
        if response.status_code != 200:
            raise RuntimeError("MCP server is not running or unreachable.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to MCP server: {str(e)}")
        raise RuntimeError("MCP server is not running or unreachable.")

    """Main entry point for the CLI application"""
    parser = argparse.ArgumentParser(description="Phi-4 Medical Assistant with PubMed Tool Integration")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Ollama model name to use")
    parser.add_argument("--mcp-url", default=DEFAULT_MCP_URL, help="URL of the MCP PubMed server")
    parser.add_argument("--force-pubmed", action="store_true", help="Force using PubMed search")
    parser.add_argument("--no-pubmed", action="store_true", help="Force not using PubMed search")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Check conflicting arguments
    if args.force_pubmed and args.no_pubmed:
        parser.error("Cannot specify both --force-pubmed and --no-pubmed")
        
    force_tool = None
    if args.force_pubmed:
        force_tool = True
    elif args.no_pubmed:
        force_tool = False
    
    try:
        # Initialize assistant
        assistant = Assistant(model_name=args.model, mcp_url=args.mcp_url)
        
        print("\n🩺 Biomedical Research Assistant with PubMed Integration 🔍")
        print("Type 'exit', 'quit', or Ctrl+C to exit\n")
        
        while True:
            try:
                question = input("\nEnter your medical question: ")
                
                if question.lower() in ["exit", "quit"]:
                    break
                    
                print("\nProcessing your question...")
                
                result = assistant.answer(question, force_tool=force_tool)
                
                if result["used_pubmed"]:
                    if result["used_query_crafting"]:
                        print(f"\n[Using crafted PubMed query: '{result['search_terms']}']\n")
                    else:
                        print(f"\n[Using PubMed search: '{result['search_terms']}']\n")
                else:
                    print("\n[Answering based on model knowledge]\n")
                    
                print(result["answer"])
                
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
        
    print("\nThank you for using the Phi-4 Medical Assistant!")

if __name__ == "__main__":
    main()