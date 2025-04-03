#!/usr/bin/env python3
"""
Autonomous Research Agent using Local LLM with Model Context Protocol (MCP)

This application uses a local LLM model via Ollama to answer research questions.
The Agent dynamically discovers available tools from the MCP server upon initialization.
When prompted with a question, the agent will:
1. Evaluate if it has sufficient knowledge to answer accurately
2. If not, determine which tools could help gather relevant information
3. Prioritize query optimization tools before executing search queries when available
4. Continue gathering context until it has sufficient information for a well-referenced answer
5. Provide a comprehensive answer with proper citations including verifiable URLs when available
6. Log all tool interactions with their parameters and results

The application adheres to the Model Context Protocol (MCP) for communication with tools.
It implements retry mechanisms and provides detailed tracking of tool usage.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Generator, Tuple, Union, Literal
import logging
import datetime
import json
import time
import re
import urllib.parse
from enum import Enum

import ollama

from MCPClient import MCPClient, DEFAULT_MCP_URL

# Configure logging
logger = logging.getLogger('autonomous-agent')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class ToolUseFeedback:
    """
    Feedback on the usage of a tool including parameters and results
    """
    timestamp: str
    tool_id: str
    tool_name: str
    tool_args: Dict[str, Any]
    tool_response: Any
    success: bool
    error_message: Optional[str] = None
    retry_count: int = 0


@dataclass
class FinalAnswer:
    """
    Final answer provided by the agent with references and tool usage tracking
    """
    timestamp: str
    answer: str
    references: List[str]
    tool_uses: List[ToolUseFeedback]
    confidence: float
    feedback: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0


class ContextStatus(str, Enum):
    """
    Enum for context sufficiency status
    """
    SUFFICIENT = "SUFFICIENT"
    NEED_MORE_CONTEXT = "NEED_MORE_CONTEXT"
    CANNOT_ANSWER = "CANNOT_ANSWER"


class ResearchAgent:
    """
    Autonomous Research Agent using an LLM with Model Context Protocol (MCP)
    
    This agent discovers available tools from an MCP server and uses them to answer
    research questions with proper citations and references.
    """

    def __init__(self, mcp_url: str = DEFAULT_MCP_URL, model_name: str = "phi4:latest"):
        """
        Initialize the Research Agent
        
        Parameters:
        mcp_url (str): URL for the MCP server
        model_name (str): Name of the Ollama model to use
        """
        self.mcp_client = MCPClient(mcp_url)
        self.model_name = model_name
        # Initialize tools from the tool_map property
        self.tools = []
        for tool_name, tool_data in self.mcp_client.tool_map.items():
            self.tools.append(tool_data)
        self.tool_uses: List[ToolUseFeedback] = []
        self.final_answer: Optional[FinalAnswer] = None
        logger.info(f"Initialized Research Agent with {len(self.tools)} tools")
        
        # Log discovered tools
        for tool in self.tools:
            logger.debug(f"Discovered tool: {tool.get('name')} - {tool.get('description', 'No description')}")

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        return datetime.datetime.now().isoformat()
        
    def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any], max_retries: int) -> ToolUseFeedback:
        """
        Execute a tool with retry logic and log the results
        
        Parameters:
        tool_name (str): Name of the tool
        tool_args (Dict[str, Any]): Arguments for the tool
        max_retries (int): Maximum number of retries
        
        Returns:
        ToolUseFeedback: Feedback on tool usage including parameters and results
        """
        retry_count = 0
        while retry_count <= max_retries:
            try:
                logger.info(f"Executing tool {tool_name} with args: {json.dumps(tool_args)}")
                response = self.mcp_client.execute_tool(tool_name, tool_args)
                
                feedback = ToolUseFeedback(
                    timestamp=self._get_timestamp(),
                    tool_id=tool_name,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_response=response,
                    success=True,
                    retry_count=retry_count
                )
                self.tool_uses.append(feedback)
                
                # Log the successful tool execution results
                logger.info(f"Tool {tool_name} executed successfully")
                logger.debug(f"Tool {tool_name} response: {json.dumps(response)[:500]}...")
                
                return feedback
                
            except Exception as e:
                retry_count += 1
                error_msg = f"Error executing tool {tool_name}: {str(e)}"
                logger.error(f"{error_msg} (Retry {retry_count}/{max_retries})")
                
                if retry_count > max_retries:
                    feedback = ToolUseFeedback(
                        timestamp=self._get_timestamp(),
                        tool_id=tool_name,
                        tool_name=tool_name,
                        tool_args=tool_args,
                        tool_response=None,
                        success=False,
                        error_message=error_msg,
                        retry_count=retry_count
                    )
                    self.tool_uses.append(feedback)
                    return feedback
                
                # Wait before retrying with exponential backoff
                time.sleep(2 * retry_count)

    def _generate_prompt(self, question: str, collected_context: List[Dict[str, Any]]) -> str:
        """
        Generate a prompt for the LLM based on the question and collected context
        
        Parameters:
        question (str): The research question
        collected_context (List[Dict[str, Any]]): Context collected from tool usage
        
        Returns:
        str: Prompt for the LLM
        """
        prompt = f"""You are a helpful research assistant who provides accurate, factual information.
Please answer the following question based on the context provided:

QUESTION: {question}

CONTEXT:
"""
        
        for idx, context in enumerate(collected_context):
            prompt += f"\n--- Source {idx+1}: {context.get('source', 'Unknown')} ---\n"
            content = context.get('content', '')
            # If content is a dict, convert it to a formatted string
            if isinstance(content, dict):
                prompt += json.dumps(content, indent=2)
            else:
                prompt += str(content)
            prompt += "\n"
            
        prompt += """
Based on the context provided, please:
1. Determine if there is enough context to answer the question confidently.
2. If there is enough context, provide a comprehensive answer with references to the sources used.
3. If there is NOT enough context, respond with EXACTLY the phrase "NEED_MORE_CONTEXT" followed by a brief explanation of what additional information would be helpful and which tools might provide that information.
4. If you believe the question cannot be answered even with more context, respond with EXACTLY the phrase "CANNOT_ANSWER" followed by an explanation why.
"""
        return prompt

    def _evaluate_context_sufficiency(self, question: str, collected_context: List[Dict[str, Any]]) -> Tuple[ContextStatus, str]:
        """
        Evaluate if the collected context is sufficient to answer the question
        
        Parameters:
        question (str): The research question
        collected_context (List[Dict[str, Any]]): Context collected from tool usage
        
        Returns:
        Tuple[ContextStatus, str]: Status of the context and explanation
        """
        if not collected_context:
            return ContextStatus.NEED_MORE_CONTEXT, "No context has been collected yet"
        
        prompt = self._generate_prompt(question, collected_context)
        
        try:
            logger.info("Evaluating context sufficiency")
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            
            answer = response['message']['content']
            
            if answer.strip().startswith("NEED_MORE_CONTEXT"):
                explanation = answer.replace("NEED_MORE_CONTEXT", "", 1).strip()
                logger.info(f"Context evaluation: Need more context - {explanation}")
                return ContextStatus.NEED_MORE_CONTEXT, explanation
            elif answer.strip().startswith("CANNOT_ANSWER"):
                explanation = answer.replace("CANNOT_ANSWER", "", 1).strip()
                logger.info(f"Context evaluation: Cannot answer - {explanation}")
                return ContextStatus.CANNOT_ANSWER, explanation
            else:
                logger.info("Context evaluation: Sufficient context available")
                return ContextStatus.SUFFICIENT, answer
        
        except Exception as e:
            logger.error(f"Error evaluating context sufficiency: {str(e)}")
            return ContextStatus.NEED_MORE_CONTEXT, f"Error evaluating context: {str(e)}"

    def _identify_query_optimization_tools(self) -> List[str]:
        """
        Identify tools that can help optimize search queries
        
        Returns:
        List[str]: List of tool names that appear to be query optimization tools
        """
        query_optimization_tools = []
        optimization_keywords = ["query", "craft", "optimize", "refine", "suggestion", "prompt"]
        
        for tool in self.tools:
            tool_name = tool.get("name", "").lower()
            tool_desc = tool.get("description", "").lower()
            
            # Check if the tool name or description suggests query optimization capability
            if any(keyword in tool_name for keyword in optimization_keywords):
                query_optimization_tools.append(tool.get("name"))
            elif any(keyword in tool_desc for keyword in optimization_keywords) and "search" not in tool_name:
                query_optimization_tools.append(tool.get("name"))
                
        return query_optimization_tools

    def _identify_search_tools(self) -> List[str]:
        """
        Identify tools that can perform searches or retrieve information
        
        Returns:
        List[str]: List of tool names that appear to be search tools
        """
        search_tools = []
        search_keywords = ["search", "lookup", "find", "retrieve", "query", "get"]
        
        for tool in self.tools:
            tool_name = tool.get("name", "").lower()
            tool_desc = tool.get("description", "").lower()
            
            # Check if the tool name or description suggests search capability
            if any(keyword in tool_name for keyword in search_keywords):
                search_tools.append(tool.get("name"))
            elif any(keyword in tool_desc for keyword in search_keywords) and "craft" not in tool_name:
                search_tools.append(tool.get("name"))
                
        return search_tools

    def _select_next_tools(self, question: str, collected_context: List[Dict[str, Any]], explanation: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Autonomously select the next tools to use based on the question and current context
        
        Parameters:
        question (str): The research question
        collected_context (List[Dict[str, Any]]): Context collected so far
        explanation (str): Explanation of why more context is needed
        
        Returns:
        List[Tuple[str, Dict[str, Any]]]: List of tools to use next with their arguments
        """
        # Create a description of available tools and their capabilities
        available_tools = []
        for tool in self.tools:
            tool_info = {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": {}
            }
            
            input_schema = tool.get("input_schema", {})
            if "properties" in input_schema:
                for param_name, param_info in input_schema["properties"].items():
                    required = "required" in input_schema and param_name in input_schema["required"]
                    tool_info["parameters"][param_name] = {
                        "description": param_info.get("description", ""),
                        "required": required
                    }
            
            available_tools.append(tool_info)
            
        # Create a summary of context already collected
        used_tools = []
        for ctx in collected_context:
            source = ctx.get("source", "Unknown")
            used_tools.append(source)
        
        # Create a prompt to recommend the best next tools to use
        system_prompt = """You are an AI research assistant that helps select the most appropriate tools to gather information to answer a research question.

For the given question, context already collected, and explanation of why more context is needed, your task is to identify which tools should be used next.

First query optimization tools should be used to craft better search queries, and then search tools should be used with the optimized queries.

Return your response as a valid JSON array containing objects with 'tool_name' and 'tool_args' fields.

Example:
[
    {
        "tool_name": "query_optimizer",
        "tool_args": {
            "question": "What are the latest treatments for heart failure?"
        }
    }
]
"""

        user_prompt = f"""QUESTION: {question}

AVAILABLE TOOLS:
{json.dumps(available_tools, indent=2)}

CONTEXT ALREADY COLLECTED FROM:
{", ".join(used_tools) if used_tools else "No context collected yet"}

EXPLANATION OF WHY MORE CONTEXT IS NEEDED:
{explanation}

Please identify the most appropriate next tool(s) to use. If no context has been collected yet and a query crafting tool is available, use that first to optimize the search query.

Return ONLY a JSON array with no additional text."""

        try:
            # Ask the model to select tools
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            content = response['message']['content']
            logger.debug(f"Tool selection response: {content}")
            
            # Extract JSON from the response using more robust methods
            try:
                # First try: Look for JSON blocks
                json_match = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', content)
                if json_match:
                    json_str = json_match.group(1)
                    selected_tools = json.loads(json_str)
                else:
                    # Second try: Strip any text before [ and after ]
                    json_start = content.find('[')
                    json_end = content.rfind(']') + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_str = content[json_start:json_end]
                        try:
                            selected_tools = json.loads(json_str)
                        except json.JSONDecodeError:
                            # Try to clean up the JSON string
                            json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas
                            selected_tools = json.loads(json_str)
                    else:
                        raise ValueError("No JSON array found in response")
                
                # Convert to the expected format
                result = []
                for tool in selected_tools:
                    if "tool_name" in tool and "tool_args" in tool:
                        result.append((tool["tool_name"], tool["tool_args"]))
                
                if result:
                    logger.info(f"Selected {len(result)} tools for next inquiry: {[t[0] for t in result]}")
                    return result
                else:
                    raise ValueError("No valid tools found in JSON")
                    
            except (json.JSONDecodeError, ValueError) as json_error:
                logger.error(f"JSON parsing error: {json_error}. Content: {content}")
                raise
            
        except Exception as e:
            logger.error(f"Error selecting next tools: {str(e)}")
            
            # Fall back to heuristic tool selection if the LLM-based selection fails
            # This is a generic approach that doesn't assume specific tool names
            
            # If no context has been collected yet, prioritize query optimization tools first
            if not collected_context:
                query_tools = self._identify_query_optimization_tools()
                search_tools = self._identify_search_tools()
                
                if query_tools:
                    # Use the first query optimization tool
                    logger.info(f"No context yet - selecting query optimization tool: {query_tools[0]}")
                    return [(query_tools[0], {"question": question})]
                elif search_tools:
                    # If no query tools, fall back to a search tool
                    logger.info(f"No context yet - falling back to search tool: {search_tools[0]}")
                    return [(search_tools[0], {"query": question, "max_results": 5})]
            
            # For subsequent queries after some context is collected
            else:
                # Try to use search tools if we have query optimization results
                search_tools = self._identify_search_tools()
                query_tools_used = [t for t in used_tools if t in self._identify_query_optimization_tools()]
                
                if query_tools_used and search_tools:
                    # We've used a query tool before, now use a search tool
                    logger.info(f"Using search tool after query optimization: {search_tools[0]}")
                    return [(search_tools[0], {"query": question, "max_results": 5})]
                
                # Try to use any unused tools
                available_tool_names = [tool["name"] for tool in self.tools]
                unused_tools = [t for t in available_tool_names if t not in used_tools]
                
                if unused_tools:
                    logger.info(f"Trying unused tool: {unused_tools[0]}")
                    # Try to guess reasonable parameters based on the tool name
                    if "search" in unused_tools[0].lower() or "query" in unused_tools[0].lower():
                        return [(unused_tools[0], {"query": question, "max_results": 5})]
                    else:
                        return [(unused_tools[0], {"question": question})]
            
            # If all else fails, return an empty list
            logger.warning("Could not determine next tools to use")
            return []

    def _extract_url_from_text(self, text: str) -> Optional[str]:
        """
        Extract a URL from text if present
        
        Parameters:
        text (str): Text to search for URLs
        
        Returns:
        Optional[str]: URL if found, None otherwise
        """
        url_pattern = re.compile(r'https?://[^\s<>"\']+|www\.[^\s<>"\']+')
        match = url_pattern.search(text)
        if match:
            return match.group()
        return None

    def _format_reference_with_url(self, ref: str) -> str:
        """
        Format a reference string with a URL if possible
        
        Parameters:
        ref (str): Reference string
        
        Returns:
        str: Formatted reference with URL if available
        """
        # Check if reference already has a URL
        if "http://" in ref or "https://" in ref:
            return ref
            
        # Extract identifiers that might be used to construct URLs
        identifiers = {
            "doi": re.search(r'doi:?\s*([^\s]+)', ref, re.IGNORECASE),
            "pmid": re.search(r'pmid:?\s*([^\s]+)', ref, re.IGNORECASE),
            "isbn": re.search(r'isbn:?\s*([^\s]+)', ref, re.IGNORECASE),
        }
        
        # Construct URLs based on identifiers
        if identifiers["doi"] and identifiers["doi"].group(1):
            doi = identifiers["doi"].group(1).strip()
            return f"{ref} [https://doi.org/{doi}]"
        elif identifiers["pmid"] and identifiers["pmid"].group(1):
            pmid = identifiers["pmid"].group(1).strip()
            return f"{ref} [https://pubmed.ncbi.nlm.nih.gov/{pmid}/]"
        elif identifiers["isbn"] and identifiers["isbn"].group(1):
            isbn = identifiers["isbn"].group(1).strip()
            return f"{ref} [https://www.worldcat.org/search?q=bn:{isbn}]"
            
        return ref

    def _generate_references(self, collected_context: List[Dict[str, Any]]) -> List[str]:
        """
        Generate references from collected context with URLs when possible
        
        Parameters:
        collected_context (List[Dict[str, Any]]): Context collected from tool usage
        
        Returns:
        List[str]: References with URLs when available
        """
        references = []
        
        for idx, context in enumerate(collected_context):
            source = context.get("source", "Unknown")
            tool_args = context.get("tool_args", {})
            content = context.get("content", {})
            
            # For sources that might contain article metadata
            if isinstance(content, dict) and any(key in content for key in ["title", "authors", "article", "articles"]):
                # Handle sources with article lists
                articles = content.get("articles", [])
                if articles and isinstance(articles, list):
                    for i, article in enumerate(articles[:5]):  # Limit to first 5 articles
                        if not isinstance(article, dict):
                            continue
                            
                        ref_parts = []
                        
                        # Build reference with available metadata
                        if "title" in article:
                            ref_parts.append(f'"{article["title"]}"')
                            
                        if "authors" in article and article["authors"]:
                            authors = article["authors"]
                            if isinstance(authors, list):
                                if len(authors) == 1:
                                    ref_parts.append(f"by {authors[0]}")
                                elif len(authors) == 2:
                                    ref_parts.append(f"by {authors[0]} and {authors[1]}")
                                else:
                                    ref_parts.append(f"by {authors[0]} et al.")
                        
                        pub_info = []
                        if "journal" in article:
                            pub_info.append(article["journal"])
                        if "year" in article:
                            pub_info.append(str(article["year"]))
                        if pub_info:
                            ref_parts.append(f"({', '.join(pub_info)})")
                        
                        # Add source identifiers that could be used for URLs
                        identifiers = []
                        for id_type in ["doi", "pmid", "pmc", "url"]:
                            if id_type in article and article[id_type]:
                                identifiers.append(f"{id_type.upper()}: {article[id_type]}")
                                
                        if identifiers:
                            ref_parts.append(", ".join(identifiers))
                        
                        # Combine all parts into a reference string
                        ref_str = f"Source {idx+1}.{i+1}: {' '.join(ref_parts)}"
                        
                        # Try to add a URL if not already present
                        ref_str = self._format_reference_with_url(ref_str)
                        references.append(ref_str)
                
                # Handle single article source
                elif "article" in content and isinstance(content["article"], dict):
                    article = content["article"]
                    ref_parts = []
                    
                    # Build reference with available metadata
                    if "title" in article:
                        ref_parts.append(f'"{article["title"]}"')
                        
                    if "authors" in article and article["authors"]:
                        authors = article["authors"]
                        if isinstance(authors, list):
                            if len(authors) == 1:
                                ref_parts.append(f"by {authors[0]}")
                            elif len(authors) == 2:
                                ref_parts.append(f"by {authors[0]} and {authors[1]}")
                            else:
                                ref_parts.append(f"by {authors[0]} et al.")
                    
                    pub_info = []
                    if "journal" in article:
                        pub_info.append(article["journal"])
                    if "year" in article:
                        pub_info.append(str(article["year"]))
                    if pub_info:
                        ref_parts.append(f"({', '.join(pub_info)})")
                    
                    # Add source identifiers that could be used for URLs
                    identifiers = []
                    for id_type in ["doi", "pmid", "pmc", "url"]:
                        if id_type in article and article[id_type]:
                            identifiers.append(f"{id_type.upper()}: {article[id_type]}")
                            
                    if identifiers:
                        ref_parts.append(", ".join(identifiers))
                    
                    # Combine all parts into a reference string
                    ref_str = f"Source {idx+1}: {' '.join(ref_parts)}"
                    
                    # Try to add a URL if not already present
                    ref_str = self._format_reference_with_url(ref_str)
                    references.append(ref_str)
            
            # For web content or other sources
            elif isinstance(content, dict) and "url" in content:
                title = content.get("title", "Web Page")
                url = content["url"]
                ref_str = f"Source {idx+1}: {title} [{url}]"
                references.append(ref_str)
                
            # For text content that might contain URLs
            elif isinstance(content, str) or (isinstance(content, dict) and "text" in content):
                text_content = content if isinstance(content, str) else content.get("text", "")
                url = self._extract_url_from_text(text_content)
                
                if url:
                    ref_str = f"Source {idx+1}: Content from {url}"
                else:
                    ref_str = f"Source {idx+1}: {source} {json.dumps(tool_args)}"
                    
                references.append(ref_str)
                
            # Generic fallback for other sources
            else:
                ref_str = f"Source {idx+1}: {source} query parameters: {json.dumps(tool_args)}"
                references.append(ref_str)
        
        return references

    def answer_question(self, question: str, max_retries: int = 3) -> Generator[Union[ToolUseFeedback, FinalAnswer], None, None]:
        """
        Answer a research question using the LLM and available tools.
        It returns a generator that yields each step (tool uses, final answer).
        
        Parameters:
        question (str): The research question to answer
        max_retries (int): Maximum number of retries for tool usage
        
        Yields:
        Union[ToolUseFeedback, FinalAnswer]: Tool usage feedback or final answer
        """
        logger.info(f"Answering question: {question}")
        
        # Reset state for new question
        self.tool_uses = []
        self.final_answer = None
        
        try:
            collected_context = []
            max_context_gathering_iterations = 5  # Prevent infinite loops
            current_iteration = 0
            performed_web_search = False
            skip_web_search = False  # Flag to track if web search should be skipped
            
            while current_iteration < max_context_gathering_iterations:
                current_iteration += 1
                
                # Evaluate if we have enough context
                context_status, explanation = self._evaluate_context_sufficiency(question, collected_context)
                
                # Check if "no web search" or similar is mentioned in the question
                if re.search(r'no\s+web\s+search|without\s+web\s+search|skip\s+web\s+search', question.lower()):
                    skip_web_search = True
                    logger.info("Web search explicitly skipped based on question")
                
                if context_status == ContextStatus.SUFFICIENT:
                    # We have enough context, but still perform web search if not done yet unless explicitly skipped
                    if not performed_web_search and not skip_web_search:
                        logger.info("Context is sufficient, but performing web search for recency and verification")
                        web_search_tool = self._find_web_search_tool()
                        
                        if web_search_tool:
                            logger.info(f"Performing web search using tool: {web_search_tool}")
                            
                            # Try to create a more focused search query based on the collected context
                            search_query = self._create_focused_web_query(question, collected_context)
                            tool_args = {"query": search_query, "max_results": 3}
                            
                            try:
                                tool_feedback = self._execute_tool(web_search_tool, tool_args, max_retries)
                                performed_web_search = True
                                yield tool_feedback
                                
                                if tool_feedback.success:
                                    context_entry = {
                                        "source": web_search_tool,
                                        "content": tool_feedback.tool_response.get("response", {}),
                                        "tool_args": tool_args
                                    }
                                    collected_context.append(context_entry)
                                    
                                    # Re-evaluate context sufficiency with web results
                                    context_status, explanation = self._evaluate_context_sufficiency(question, collected_context)
                            except Exception as e:
                                logger.error(f"Error performing web search: {str(e)}")
                        else:
                            logger.info("No web search tool found, proceeding with current context")
                            performed_web_search = True  # Mark as performed to avoid further attempts
                    
                    # Now we can generate the final answer
                    logger.info("Context is sufficient. Generating final answer.")
                    break
                    
                elif context_status == ContextStatus.CANNOT_ANSWER:
                    # We can't answer the question even with more context
                    logger.info("Cannot answer the question even with more context.")
                    final_answer = FinalAnswer(
                        timestamp=self._get_timestamp(),
                        answer=f"I'm unable to answer this question: {explanation}",
                        references=[],
                        tool_uses=self.tool_uses,
                        confidence=0.0,
                        error_message=explanation
                    )
                    self.final_answer = final_answer
                    yield final_answer
                    return
                    
                else:  # NEED_MORE_CONTEXT
                    # We need more context, select and execute more tools
                    logger.info(f"Need more context. Reason: {explanation}")
                    logger.info(f"Context gathering iteration {current_iteration}/{max_context_gathering_iterations}")
                    
                    # If we're on the last iteration and haven't done web search yet, prioritize it
                    if current_iteration == max_context_gathering_iterations - 1 and not performed_web_search and not skip_web_search:
                        web_search_tool = self._find_web_search_tool()
                        if web_search_tool:
                            logger.info(f"Last iteration, prioritizing web search using tool: {web_search_tool}")
                            search_query = self._create_focused_web_query(question, collected_context)
                            tool_args = {"query": search_query, "max_results": 3}
                            
                            try:
                                tool_feedback = self._execute_tool(web_search_tool, tool_args, max_retries)
                                performed_web_search = True
                                yield tool_feedback
                                
                                if tool_feedback.success:
                                    context_entry = {
                                        "source": web_search_tool,
                                        "content": tool_feedback.tool_response.get("response", {}),
                                        "tool_args": tool_args
                                    }
                                    collected_context.append(context_entry)
                                continue
                            except Exception as e:
                                logger.error(f"Error performing web search: {str(e)}")
                    
                    next_tools = self._select_next_tools(question, collected_context, explanation)
                    
                    # Track if this iteration includes web search
                    if not performed_web_search and not skip_web_search:
                        for tool_name, _ in next_tools:
                            if self._is_web_search_tool(tool_name):
                                performed_web_search = True
                                break
                    
                    if not next_tools:
                        logger.warning("No additional tools selected, but more context is needed.")
                        
                        # If we haven't performed web search yet and we're out of tools, try it
                        if not performed_web_search and not skip_web_search:
                            web_search_tool = self._find_web_search_tool()
                            if web_search_tool:
                                logger.info(f"No tools selected but web search not performed. Using web search tool: {web_search_tool}")
                                search_query = self._create_focused_web_query(question, collected_context)
                                tool_args = {"query": search_query, "max_results": 3}
                                
                                try:
                                    tool_feedback = self._execute_tool(web_search_tool, tool_args, max_retries)
                                    performed_web_search = True
                                    yield tool_feedback
                                    
                                    if tool_feedback.success:
                                        context_entry = {
                                            "source": web_search_tool,
                                            "content": tool_feedback.tool_response.get("response", {}),
                                            "tool_args": tool_args
                                        }
                                        collected_context.append(context_entry)
                                        continue
                                except Exception as e:
                                    logger.error(f"Error performing web search: {str(e)}")
                        
                        break  # Break the loop and use whatever context we have
                    
                    # Execute each selected tool
                    for tool_name, tool_args in next_tools:
                        try:
                            tool_feedback = self._execute_tool(tool_name, tool_args, max_retries)
                            yield tool_feedback
                            
                            if tool_feedback.success:
                                # For search tools, enrich the context with source information
                                context_entry = {
                                    "source": tool_name,
                                    "content": tool_feedback.tool_response.get("response", {}),
                                    "tool_args": tool_args
                                }
                                
                                collected_context.append(context_entry)
                                
                                # Update web search tracking
                                if self._is_web_search_tool(tool_name):
                                    performed_web_search = True
                        except Exception as tool_error:
                            logger.error(f"Error with tool {tool_name}: {str(tool_error)}")
            
            # Final check for web search before generating answer
            if not performed_web_search and not skip_web_search:
                web_search_tool = self._find_web_search_tool()
                if web_search_tool:
                    logger.info(f"Performing final web search before generating answer using tool: {web_search_tool}")
                    search_query = self._create_focused_web_query(question, collected_context)
                    tool_args = {"query": search_query, "max_results": 3}
                    
                    try:
                        tool_feedback = self._execute_tool(web_search_tool, tool_args, max_retries)
                        performed_web_search = True
                        yield tool_feedback
                        
                        if tool_feedback.success:
                            context_entry = {
                                "source": web_search_tool,
                                "content": tool_feedback.tool_response.get("response", {}),
                                "tool_args": tool_args
                            }
                            collected_context.append(context_entry)
                    except Exception as e:
                        logger.error(f"Error performing final web search: {str(e)}")
            
            # Generate final answer using collected context
            try:
                # Don't use the explanation from context evaluation as the answer
                # Instead, always generate a targeted, focused answer to the actual question
                current_date = datetime.datetime.now().strftime("%B %d, %Y")
                
                # Generate a prompt specifically for the final answer
                prompt = f"""You are a helpful research assistant delivering accurate, factual information with proper citations.

QUESTION: {question}

CONTEXT:
"""
                for idx, context in enumerate(collected_context):
                    prompt += f"\n--- Source {idx+1}: {context.get('source', 'Unknown')} ---\n"
                    content = context.get('content', '')
                    # If content is a dict, convert it to a formatted string
                    if isinstance(content, dict):
                        prompt += json.dumps(content, indent=2)
                    else:
                        prompt += str(content)
                    prompt += "\n"
                
                prompt += f"""
Based on the context above, answer the original question: "{question}"

Your answer should:
1. Be comprehensive and directly address the question
2. Integrate information from multiple sources when available
3. Include specific evidence and findings from the context
4. Use numbered citations in square brackets [1], [2], etc. that correspond to the source numbers
5. Focus solely on the information from the provided sources
6. Be written in a clear, professional style
7. Provide a conclusion that summarizes the key findings
8. If web search results were included, compare them with other sources and note any discrepancies or updates

Today's date is {current_date}.

Do NOT include any additional headers like "FINAL ANSWER" or evaluation statements about the context sufficiency.
"""
                
                logger.info("Generating final answer")
                response = ollama.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                answer_text = response['message']['content']
                
                # Generate references from the collected context
                references = self._generate_references(collected_context)
                
                # Create and store final answer
                final_answer = FinalAnswer(
                    timestamp=self._get_timestamp(),
                    answer=answer_text,
                    references=references,
                    tool_uses=self.tool_uses,
                    confidence=0.8 if context_status == ContextStatus.SUFFICIENT else 0.6,
                    retry_count=0
                )
                
                self.final_answer = final_answer
                yield final_answer
                
            except Exception as e:
                error_msg = f"Error generating final answer: {str(e)}"
                logger.error(error_msg)
                
                final_answer = FinalAnswer(
                    timestamp=self._get_timestamp(),
                    answer="Sorry, I couldn't generate an answer due to an error.",
                    references=[],
                    tool_uses=self.tool_uses,
                    confidence=0.0,
                    error_message=error_msg,
                    retry_count=0
                )
                
                self.final_answer = final_answer
                yield final_answer
                
        except Exception as e:
            error_msg = f"Error during research process: {str(e)}"
            logger.error(error_msg)
            
            final_answer = FinalAnswer(
                timestamp=self._get_timestamp(),
                answer=f"Sorry, I encountered an error while researching your question: {str(e)}",
                references=[],
                tool_uses=self.tool_uses,
                confidence=0.0,
                error_message=error_msg,
                retry_count=0
            )
            
            self.final_answer = final_answer
            yield final_answer

    def get_tool_usage_summary(self) -> Dict[str, Any]:
        """
        Get a summary of tool usage with parameters and results
        
        Returns:
        Dict[str, Any]: Summary of tool usage including success/failure stats
        """
        tool_counts = {}
        for tool_use in self.tool_uses:
            tool_name = tool_use.tool_name
            if tool_name in tool_counts:
                tool_counts[tool_name] += 1
            else:
                tool_counts[tool_name] = 1
        
        # Include detailed tool usage logs
        tool_logs = []
        for tool_use in self.tool_uses:
            tool_log = {
                "timestamp": tool_use.timestamp,
                "tool_name": tool_use.tool_name,
                "parameters": tool_use.tool_args,
                "success": tool_use.success,
                "retry_count": tool_use.retry_count
            }
            
            if not tool_use.success and tool_use.error_message:
                tool_log["error"] = tool_use.error_message
                
            tool_logs.append(tool_log)
                
        return {
            "total_tool_uses": len(self.tool_uses),
            "tool_counts": tool_counts,
            "successful_calls": sum(1 for t in self.tool_uses if t.success),
            "failed_calls": sum(1 for t in self.tool_uses if not t.success),
            "tool_logs": tool_logs
        }

    def reset(self) -> None:
        """Reset the agent state"""
        self.tool_uses = []
        self.final_answer = None

    def _is_web_search_tool(self, tool_name: str) -> bool:
        """
        Determine if a tool is a web search tool based on name and description
        
        Parameters:
        tool_name (str): Name of the tool to check
        
        Returns:
        bool: True if the tool appears to be a web search tool
        """
        # First check if we can find the tool in our tools list
        tool_data = None
        for tool in self.tools:
            if tool.get("name", "") == tool_name:
                tool_data = tool
                break
                
        if not tool_data:
            return False
        
        # Check name for web search keywords
        web_keywords = ["web", "internet", "online", "google", "bing", "search"]
        
        tool_name_lower = tool_name.lower()
        if any(keyword in tool_name_lower for keyword in web_keywords):
            return True
            
        # Check description for web search mentions
        description = tool_data.get("description", "").lower()
        if "web" in description and "search" in description:
            return True
            
        return False
        
    def _find_web_search_tool(self) -> Optional[str]:
        """
        Find a suitable web search tool from available tools
        
        Returns:
        Optional[str]: Name of web search tool if found, None otherwise
        """
        for tool in self.tools:
            tool_name = tool.get("name", "")
            if self._is_web_search_tool(tool_name):
                return tool_name
                
        # If no specific web search tool found, look for generic search tools
        for tool in self.tools:
            tool_name = tool.get("name", "")
            description = tool.get("description", "").lower()
            
            # Check for any general search tool as fallback
            if ("search" in tool_name.lower() and 
                "pubmed" not in tool_name.lower() and 
                "query" not in tool_name.lower()):
                return tool_name
                
        return None
        
    def _create_focused_web_query(self, question: str, collected_context: List[Dict[str, Any]]) -> str:
        """
        Create a focused web search query based on the question and collected context
        
        Parameters:
        question (str): The original research question
        collected_context (List[Dict[str, Any]]): Context collected so far
        
        Returns:
        str: A focused search query for web search
        """
        if not collected_context:
            # If no context, just use the original question with some search optimizations
            # Remove question words and add quotes for key phrases
            search_query = question
            search_query = re.sub(r'^(what|who|where|when|why|how|is|are|was|were|do|does|did)\s+', '', search_query, flags=re.IGNORECASE)
            search_query = re.sub(r'\?$', '', search_query)  # Remove trailing question mark
            
            # Add current year for recency
            current_year = datetime.datetime.now().year
            if not re.search(r'\b\d{4}\b', search_query):  # Only add year if not already present
                search_query += f" {current_year}"
                
            return search_query
        
        # If we have context, try to use it to create a more targeted query
        try:
            # Create a prompt for the LLM to generate a focused search query
            prompt = f"""Based on the following information, create a focused web search query to find the most relevant and up-to-date information.

Original Question: {question}

Context already collected:
"""
            for idx, context in enumerate(collected_context):
                source = context.get("source", "Unknown")
                tool_args = context.get("tool_args", {})
                
                # Add a brief summary of this context
                prompt += f"\n- Source {idx+1}: {source}"
                if isinstance(tool_args, dict) and "query" in tool_args:
                    prompt += f" (query: {tool_args['query']})"
            
            prompt += """

Create a web search query that:
1. Focuses on aspects not covered by the existing context
2. Includes specific keywords and terms relevant to the topic
3. Is optimized for web search (no unnecessary words like "what", "how", etc.)
4. Includes the current year for recency
5. Uses quotes for specific phrases
6. Is no longer than 10-15 words

Return only the search query text with no explanations or other text.
"""
            
            # Use the LLM to generate a focused query
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            
            focused_query = response['message']['content'].strip()
            
            # Clean up the query: remove quotes, ensure no more than 15 words
            focused_query = focused_query.strip('"\'')
            focused_query = re.sub(r'\s+', ' ', focused_query)  # Normalize whitespace
            
            words = focused_query.split()
            if len(words) > 15:
                focused_query = ' '.join(words[:15])
                
            # Add current year if not present
            current_year = datetime.datetime.now().year
            if not re.search(r'\b\d{4}\b', focused_query):
                focused_query += f" {current_year}"
                
            logger.info(f"Created focused web query: {focused_query}")
            return focused_query
            
        except Exception as e:
            logger.error(f"Error creating focused web query: {str(e)}")
            # Fall back to original question with year
            current_year = datetime.datetime.now().year
            return f"{question.rstrip('?')} {current_year}"


if __name__ == "__main__":
    import argparse
    
    # Configure argument parser
    parser = argparse.ArgumentParser(description="Autonomous Research Agent")
    parser.add_argument("--model", type=str, default="phi4:latest", 
                        help="Ollama model to use (default: phi4:latest)")
    parser.add_argument("--mcp-url", type=str, default=DEFAULT_MCP_URL,
                        help=f"MCP server URL (default: {DEFAULT_MCP_URL})")
    parser.add_argument("--verbose", action="store_true", 
                        help="Enable verbose logging")
    parser.add_argument("--max-iterations", type=int, default=5,
                        help="Maximum number of context gathering iterations (default: 5)")
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger('autonomous-agent').setLevel(logging.DEBUG)
    
    # Initialize the agent
    agent = ResearchAgent(mcp_url=args.mcp_url, model_name=args.model)
    print(f"Research Agent initialized with model: {args.model}")
    print(f"Connected to MCP server at: {args.mcp_url}")
    print(f"Available tools: {len(agent.tools)}")
    
    # Simple interactive loop
    try:
        while True:
            question = input("\nEnter your research question (or 'exit' to quit): ")
            if question.lower() in ('exit', 'quit', 'q'):
                break
                
            print("\nResearching your question...\n")
            
            # Process each step from the generator
            for step in agent.answer_question(question):
                if isinstance(step, ToolUseFeedback):
                    print(f"Using tool: {step.tool_name}")
                    if not step.success:
                        print(f"  ❌ Error: {step.error_message}")
                    else:
                        print(f"  ✅ Success")
                        
                elif isinstance(step, FinalAnswer):
                    print("\n" + "="*80)
                    print("FINAL ANSWER:")
                    print("="*80)
                    print(step.answer)
                    print("="*80)
                    
                    # Show summary
                    summary = agent.get_tool_usage_summary()
                    print(f"\nResearch completed using {summary['total_tool_uses']} tool calls "
                          f"({summary['successful_calls']} successful, {summary['failed_calls']} failed)")
                    
                    if step.references:
                        print("\nReferences:")
                        for ref in step.references:
                            print(f"- {ref}")
            
    except KeyboardInterrupt:
        print("\nExiting Research Agent...")
    except Exception as e:
        print(f"\nError: {str(e)}")


