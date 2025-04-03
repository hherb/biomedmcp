#!/usr/bin/env python3
"""
Research Assistant using Ollama LLM with Model Context Protocol (MCP)

This application uses a local LLM model via Ollama to answer research questions.
The Agent initially enquires the MCP server for tools available for use.
When prompted for an answer by the user, the agent will consider whether any of the available
tools are helpful to answer the question.
If the Agent is not highly confident it can answer from prior knowledge, it will then select 
one or more tools in order to provide sufficient context for a good answer.
If the Agent is not satisfied that the context suffices for a good answer, it may call tools 
repeatedly, including with different parameters, until it is satisfied with the context.

The application adheres to Anthropic's Model Context Protocol (MCP) for communication with tools.
It implements retry mechanisms and provides detailed tracking of tool usage.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Generator, Tuple, Union, Literal
import logging
import datetime
import json
import time
import re
from enum import Enum

import ollama

from MCPClient import MCPClient, DEFAULT_MCP_URL

# Configure logging
logger = logging.getLogger('research-agent')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class ToolUseFeedback:
    """
    Feedback on the usage of a tool
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
    Final answer provided by the agent
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
    Enum for context status
    """
    SUFFICIENT = "SUFFICIENT"
    NEED_MORE_CONTEXT = "NEED_MORE_CONTEXT"
    CANNOT_ANSWER = "CANNOT_ANSWER"

class ResearchAgent:
    """
    Research Assistant Agent using an LLM with Model Context Protocol (MCP)
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

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        return datetime.datetime.now().isoformat()
        
    def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any], max_retries: int) -> ToolUseFeedback:
        """
        Execute a tool with retry logic
        
        Parameters:
        tool_name (str): Name of the tool
        tool_args (Dict[str, Any]): Arguments for the tool
        max_retries (int): Maximum number of retries
        
        Returns:
        ToolUseFeedback: Feedback on tool usage
        """
        retry_count = 0
        while retry_count <= max_retries:
            try:
                logger.info(f"Executing tool {tool_name} with args: {tool_args}")
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
                
                # Wait before retrying
                time.sleep(2 * retry_count)  # Exponential backoff

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
                return ContextStatus.NEED_MORE_CONTEXT, explanation
            elif answer.strip().startswith("CANNOT_ANSWER"):
                explanation = answer.replace("CANNOT_ANSWER", "", 1).strip()
                return ContextStatus.CANNOT_ANSWER, explanation
            else:
                return ContextStatus.SUFFICIENT, answer
        
        except Exception as e:
            logger.error(f"Error evaluating context sufficiency: {str(e)}")
            return ContextStatus.NEED_MORE_CONTEXT, f"Error evaluating context: {str(e)}"

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
        "tool_name": "get_pubmed_query_crafting_prompt",
        "tool_args": {
            "question": "What is the cutoff value for ONSD to determine raised ICP?"
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
            
            # Use special discovery logic for the first tool selection
            if not collected_context:
                # If no context has been collected yet, prioritize query optimization
                available_tool_names = [tool["name"] for tool in self.tools]
                
                # First try to use query optimization if available
                if "get_pubmed_query_crafting_prompt" in available_tool_names:
                    logger.info("No context yet - selecting query crafting tool first")
                    return [("get_pubmed_query_crafting_prompt", {"question": question})]
                # Then try other search tools
                elif "pubmed_search" in available_tool_names:
                    logger.info("No context yet - falling back to pubmed_search")
                    return [("pubmed_search", {"query": question, "max_results": 5})]
                elif "web_search" in available_tool_names:
                    logger.info("No context yet - falling back to web_search")
                    return [("web_search", {"query": question, "max_results": 5})]
            
            # For subsequent queries after some context is collected
            else:
                # If we've used the query crafting prompt, look for references to optimized queries
                if "get_pubmed_query_crafting_prompt" in used_tools:
                    # Try to extract optimized query from the context
                    for ctx in collected_context:
                        if ctx.get("source") == "get_pubmed_query_crafting_prompt":
                            content = ctx.get("content", {})
                            if isinstance(content, dict) and "prompt" in content:
                                # Use pubmed_search with the optimized prompt content
                                logger.info("Using pubmed_search with previously optimized query guidance")
                                return [("pubmed_search", {"query": question, "max_results": 10})]
                
                # Look for pmids to get article details
                for ctx in collected_context:
                    if ctx.get("source") == "pubmed_search":
                        content = ctx.get("content", {})
                        if isinstance(content, dict) and "articles" in content:
                            articles = content.get("articles", [])
                            if articles and "get_article" in [tool["name"] for tool in self.tools]:
                                # Get details for the first article
                                if "pmid" in articles[0]:
                                    logger.info(f"Getting details for article {articles[0]['pmid']}")
                                    return [("get_article", {"pmid": articles[0]["pmid"]})]
            
            # If all else fails, return an empty list
            return []

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
            
            while current_iteration < max_context_gathering_iterations:
                current_iteration += 1
                
                # Evaluate if we have enough context
                context_status, explanation = self._evaluate_context_sufficiency(question, collected_context)
                
                if context_status == ContextStatus.SUFFICIENT:
                    # We have enough context, generate the final answer
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
                    
                    next_tools = self._select_next_tools(question, collected_context, explanation)
                    
                    if not next_tools:
                        logger.warning("No additional tools selected, but more context is needed.")
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
                                
                                # For PubMed search, extract document information
                                if tool_name == "pubmed_search" and isinstance(context_entry["content"], dict):
                                    articles = context_entry["content"].get("articles", [])
                                    if articles and len(articles) > 0:
                                        sources = []
                                        for article in articles[:5]:  # Limit to first 5 articles for references
                                            if isinstance(article, dict):
                                                source_info = {}
                                                if "title" in article:
                                                    source_info["title"] = article["title"]
                                                if "authors" in article:
                                                    source_info["authors"] = article["authors"]
                                                if "journal" in article:
                                                    source_info["journal"] = article["journal"]
                                                if "year" in article:
                                                    source_info["year"] = article["year"]
                                                if "pmid" in article:
                                                    source_info["pmid"] = article["pmid"]
                                                sources.append(source_info)
                                        context_entry["sources"] = sources
                                
                                collected_context.append(context_entry)
                        except Exception as tool_error:
                            logger.error(f"Error with tool {tool_name}: {str(tool_error)}")
            
            # Generate final answer using collected context
            try:
                # Use the explanation if it's from the SUFFICIENT evaluation
                if context_status == ContextStatus.SUFFICIENT:
                    answer_text = explanation
                else:
                    # Generate a new prompt for the final answer
                    prompt = f"""You are a helpful research assistant who provides accurate, factual information.
Please answer the following question based on the context provided:

QUESTION: {question}

CONTEXT:
"""
                    for idx, context in enumerate(collected_context):
                        prompt += f"\n--- Source {idx+1}: {context.get('source', 'Unknown')} ---\n"
                        content = context.get('content', '')
                        if isinstance(content, dict):
                            prompt += json.dumps(content, indent=2)
                        else:
                            prompt += str(content)
                        prompt += "\n"
                    
                    prompt += """
Please provide a comprehensive answer with references to the sources used. If the context doesn't contain enough information, acknowledge the limitations of your response.

In your answer, when referencing specific articles or sources, include the article title, authors, and year where available.
"""
                    
                    logger.info("Generating final answer")
                    response = ollama.chat(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    
                    answer_text = response['message']['content']
                
                # Extract detailed references from the collected context
                references = []
                for idx, context in enumerate(collected_context):
                    source = context.get("source", "Unknown")
                    tool_args = context.get("tool_args", {})
                    
                    if source == "pubmed_search":
                        # Add detailed source information for pubmed results
                        sources = context.get("sources", [])
                        if sources:
                            for i, source_info in enumerate(sources):
                                if isinstance(source_info, dict):
                                    ref_str = f"PubMed article {i+1}: "
                                    if "title" in source_info:
                                        ref_str += f"\"{source_info['title']}\" "
                                    if "authors" in source_info and source_info["authors"]:
                                        # Format authors nicely
                                        authors = source_info["authors"]
                                        if isinstance(authors, list) and len(authors) > 0:
                                            if len(authors) == 1:
                                                ref_str += f"by {authors[0]} "
                                            elif len(authors) == 2:
                                                ref_str += f"by {authors[0]} and {authors[1]} "
                                            else:
                                                ref_str += f"by {authors[0]} et al. "
                                    if "journal" in source_info:
                                        ref_str += f"({source_info['journal']} "
                                        if "year" in source_info:
                                            ref_str += f"{source_info['year']}"
                                        ref_str += ") "
                                    if "pmid" in source_info:
                                        ref_str += f"PMID: {source_info['pmid']}"
                                    references.append(ref_str)
                        else:
                            # Fall back to basic reference if no detailed sources
                            references.append(f"{source}: {json.dumps(tool_args)}")
                    elif source == "get_article":
                        # For individual article retrieval, extract article details
                        if isinstance(context["content"], dict) and "article" in context["content"]:
                            article = context["content"]["article"]
                            if isinstance(article, dict):
                                ref_str = "PubMed article: "
                                if "title" in article:
                                    ref_str += f"\"{article['title']}\" "
                                if "authors" in article and article["authors"]:
                                    authors = article["authors"]
                                    if isinstance(authors, list) and len(authors) > 0:
                                        if len(authors) == 1:
                                            ref_str += f"by {authors[0]} "
                                        elif len(authors) == 2:
                                            ref_str += f"by {authors[0]} and {authors[1]} "
                                        else:
                                            ref_str += f"by {authors[0]} et al. "
                                if "journal" in article:
                                    ref_str += f"({article['journal']} "
                                    if "year" in article:
                                        ref_str += f"{article['year']}"
                                    ref_str += ") "
                                if "pmid" in article:
                                    ref_str += f"PMID: {article['pmid']}"
                                references.append(ref_str)
                            else:
                                references.append(f"{source}: {json.dumps(tool_args)}")
                        else:
                            references.append(f"{source}: {json.dumps(tool_args)}")
                    else:
                        # Standard reference format for other tools
                        references.append(f"{source}: {json.dumps(tool_args)}")
                
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
        Get a summary of tool usage
        
        Returns:
        Dict[str, Any]: Summary of tool usage
        """
        tool_counts = {}
        for tool_use in self.tool_uses:
            tool_name = tool_use.tool_name
            if tool_name in tool_counts:
                tool_counts[tool_name] += 1
            else:
                tool_counts[tool_name] = 1
                
        return {
            "total_tool_uses": len(self.tool_uses),
            "tool_counts": tool_counts,
            "successful_calls": sum(1 for t in self.tool_uses if t.success),
            "failed_calls": sum(1 for t in self.tool_uses if not t.success)
        }

    def reset(self) -> None:
        """Reset the agent state"""
        self.tool_uses = []
        self.final_answer = None


if __name__ == "__main__":
    import argparse
    
    # Configure argument parser
    parser = argparse.ArgumentParser(description="Research Agent Demo")
    parser.add_argument("--model", type=str, default="phi4:latest", 
                        help="Ollama model to use (default: phi4:latest)")
    parser.add_argument("--mcp-url", type=str, default=DEFAULT_MCP_URL,
                        help=f"MCP server URL (default: {DEFAULT_MCP_URL})")
    parser.add_argument("--verbose", action="store_true", 
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger('research-agent').setLevel(logging.DEBUG)
    
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


