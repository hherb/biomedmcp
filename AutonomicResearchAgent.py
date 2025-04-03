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
        # Create a system prompt that asks the model to select appropriate tools
        system_prompt = """You are a research assistant that selects appropriate tools to gather information.
Based on the question, the context already collected, and the explanation of why more context is needed,
select the most appropriate tool(s) to use next.

You have access to the following tools:
{tool_descriptions}

Provide your selection as a JSON array of objects with 'tool_name' and 'tool_args' fields.
Example:
[
    {
        "tool_name": "pubmed_search",
        "tool_args": {
            "query": "diabetes treatment advances 2024",
            "max_results": 5
        }
    }
]
"""

        # Create a description of available tools
        tool_descriptions = ""
        for tool in self.tools:
            tool_name = tool.get("name", "")
            description = tool.get("description", "")
            
            tool_descriptions += f"- {tool_name}: {description}\n"
            
            # Add parameter information
            input_schema = tool.get("input_schema", {})
            if "properties" in input_schema:
                tool_descriptions += "  Parameters:\n"
                for param_name, param_info in input_schema["properties"].items():
                    param_desc = param_info.get("description", "")
                    required = "required" in input_schema and param_name in input_schema["required"]
                    tool_descriptions += f"  - {param_name}: {param_desc} {'(required)' if required else '(optional)'}\n"
        
        # Create a summary of context already collected
        context_summary = ""
        used_tools = []
        for ctx in collected_context:
            source = ctx.get("source", "Unknown")
            used_tools.append(source)
            tool_args = ctx.get("tool_args", {})
            context_summary += f"- Used {source} with args: {json.dumps(tool_args)}\n"
        
        # Create the user prompt
        user_prompt = f"""QUESTION: {question}

CONTEXT ALREADY COLLECTED FROM:
{context_summary if context_summary else "No context collected yet"}

EXPLANATION OF WHY MORE CONTEXT IS NEEDED:
{explanation}

Based on this information, select the next tool(s) to use to gather more context. 
Consider using different tools or different parameters than what has been tried already.
Return your selection as JSON only."""

        try:
            # Ask the model to select tools
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt.format(tool_descriptions=tool_descriptions)},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            content = response['message']['content']
            
            # Extract JSON from the response
            json_match = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', content)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON array directly
                json_match = re.search(r'(\[\s*\{.*\}\s*\])', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Last resort: try to find the first [ and last ]
                    json_start = content.find('[')
                    json_end = content.rfind(']') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = content[json_start:json_end]
                    else:
                        logger.warning("Failed to extract JSON from tool selection response")
                        return []
            
            # Parse the JSON into a list of tool selections
            selected_tools = json.loads(json_str)
            logger.info(f"Selected {len(selected_tools)} tools for next inquiry")
            
            # Convert to the expected format
            result = []
            for tool in selected_tools:
                if "tool_name" in tool and "tool_args" in tool:
                    result.append((tool["tool_name"], tool["tool_args"]))
            
            return result
            
        except Exception as e:
            logger.error(f"Error selecting next tools: {str(e)}")
            
            # Fallback to a simple heuristic approach if the model-based selection fails
            available_tool_names = [tool["name"] for tool in self.tools]
            result = []
            
            # If no search has been done yet, try a search tool
            if "pubmed_search" in available_tool_names and "pubmed_search" not in used_tools:
                result.append(("pubmed_search", {"query": question, "max_results": 5}))
            elif "web_search" in available_tool_names and "web_search" not in used_tools:
                result.append(("web_search", {"query": question, "max_results": 5}))
            
            return result

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
                                collected_context.append({
                                    "source": tool_name,
                                    "content": tool_feedback.tool_response.get("response", {}),
                                    "tool_args": tool_args
                                })
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
                    
                    prompt += "\nPlease provide a comprehensive answer with references to the sources used. If the context doesn't contain enough information, acknowledge the limitations of your response."
                    
                    logger.info("Generating final answer")
                    response = ollama.chat(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    
                    answer_text = response['message']['content']
                
                # Extract references from the answer
                references = []
                for idx, context in enumerate(collected_context):
                    source = context.get("source", "Unknown")
                    tool_args = context.get("tool_args", {})
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


