"""
proof-of-concept framework free minimalistic(hopefully) MCP compliant  server 
for basic PubMed and web search
Supports both network mode (via Flask) and local mode (via stdin/stdout)
"""

import os
import json
import requests
import re
import time
import traceback
import uuid
import sys
import select
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from urllib.parse import quote_plus, urlparse
import xml.etree.ElementTree as ET
from datetime import datetime
import logging
from typing import List, Dict, Any, Optional, Union, Literal
from flask_jsonrpc import JSONRPC
from PubMedTools import PubMedSearchTool, QueryOptimizationTool
from WebTools import WebSearchTool, WebContentTool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('mcp-server')

# Configure Flask app
app = Flask(__name__)

# Initialize JSON-RPC but use a different endpoint to avoid conflicts
jsonrpc = JSONRPC(app, '/mcp/v1/jsonrpc_internal')

# Define API paths - standard for MCP protocol
MCP_PATH = "/mcp/v1"

# PubMed API configuration
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_SEARCH_URL = f"{PUBMED_BASE_URL}/esearch.fcgi"
PUBMED_FETCH_URL = f"{PUBMED_BASE_URL}/efetch.fcgi"
PUBMED_SUMMARY_URL = f"{PUBMED_BASE_URL}/esummary.fcgi"
NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")  

# If you have an API key, you get higher rate limits
REQUEST_DELAY = 0.34 if NCBI_API_KEY else 1.0  # seconds between requests (NCBI guidelines)

# MCP Protocol constants
MCP_PROTOCOL_VERSION = "2024-11-05"  # Updated to match the official MCP protocol version
MCP_CONTENT_BLOCK_DELIMITER = "```"
JSONRPC_VERSION = "2.0"

# Standard JSON-RPC error codes as defined in MCP spec
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


# MCP Protocol Handler classes
class MCPMessage:
    """Base class for MCP messages"""
    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement to_dict method")


# Define capability interfaces
class ServerCapabilities:
    """Server capabilities according to MCP spec"""
    def __init__(self):
        self.tools = {"listChanged": True}  # We support tool list changes notification
        self.experimental = {}  # No experimental capabilities

    def to_dict(self) -> Dict[str, Any]:
        capabilities = {}
        if hasattr(self, "tools"):
            capabilities["tools"] = self.tools
        if hasattr(self, "experimental") and self.experimental:
            capabilities["experimental"] = self.experimental
        if hasattr(self, "resources") and hasattr(self.resources, "subscribe"):
            capabilities["resources"] = self.resources
        if hasattr(self, "prompts") and hasattr(self.prompts, "listChanged"):
            capabilities["prompts"] = self.prompts
        if hasattr(self, "logging"):
            capabilities["logging"] = self.logging
        return capabilities


class ClientCapabilities:
    """Client capabilities according to MCP spec"""
    def __init__(self, capabilities_dict: Dict[str, Any] = None):
        self.experimental = {}
        if capabilities_dict:
            # Extract capabilities from client
            self.experimental = capabilities_dict.get("experimental", {})
            if "sampling" in capabilities_dict:
                self.sampling = capabilities_dict.get("sampling", {})
            if "roots" in capabilities_dict:
                self.roots = capabilities_dict.get("roots", {})


class Implementation:
    """Implementation info according to MCP spec"""
    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version
        }


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


class Tool:
    """Tool definition according to MCP spec"""
    def __init__(
        self,
        name: str,
        description: str = None,
        input_schema: Dict[str, Any] = None,
    ):
        self.name = name
        self.description = description
        self.input_schema = input_schema or {"type": "object", "properties": {}}

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "input_schema": self.input_schema
        }
        if self.description:
            result["description"] = self.description
        return result


class MCPManifest:
    """Describes the MCP server manifest"""
    def __init__(
        self,
        protocol_version: str,
        tools: List[Tool]
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
    "title": "PubMed Search Parameters",
    "description": "Parameters for searching PubMed medical literature database",
    "version": "1.0.0",
    "properties": {
        "query": {
            "type": "string",
            "description": "PubMed search query using standard PubMed syntax",
            "examples": ["\"heart attack\"[mesh] AND aspirin[tiab]"],
            "minLength": 2
        },
        "max_results": {
            "type": "integer",
            "default": 10,
            "minimum": 1,
            "maximum": 100,
            "description": "Maximum number of results to return (limited to 100 by PubMed API)"
        },
        "sort": {
            "type": "string",
            "enum": ["relevance", "pub_date", "first_author"],
            "default": "relevance",
            "description": "Sort order for results: by relevance, publication date, or first author"
        }
    },
    "additionalProperties": False,
    "required": ["query"]
}

article_retrieval_schema = {
    "type": "object",
    "title": "Article Retrieval Parameters",
    "description": "Parameters for retrieving a specific article from PubMed",
    "version": "1.0.0",
    "properties": {
        "pmid": {
            "type": "string",
            "description": "PubMed ID (PMID) of the article to retrieve",
            "pattern": "^[0-9]+$",
            "examples": ["32493627"]
        }
    },
    "additionalProperties": False,
    "required": ["pmid"]
}

query_crafting_schema = {
    "type": "object",
    "title": "Query Crafting Parameters",
    "description": "Parameters for generating a PubMed query crafting prompt",
    "version": "1.0.0",
    "properties": {
        "question": {
            "type": "string",
            "description": "Medical question to create a PubMed query for",
            "minLength": 5,
            "examples": ["What are the latest treatments for type 2 diabetes?"]
        }
    },
    "additionalProperties": False,
    "required": ["question"]
}

web_search_schema = {
    "type": "object",
    "title": "Web Search Parameters",
    "description": "Parameters for searching the web for information",
    "version": "1.0.0",
    "properties": {
        "query": {
            "type": "string",
            "description": "Web search query",
            "minLength": 2,
            "examples": ["latest research on artificial pancreas"]
        },
        "max_results": {
            "type": "integer",
            "default": 20,
            "minimum": 1,
            "maximum": 50,
            "description": "Maximum number of search results to return"
        }
    },
    "additionalProperties": False,
    "required": ["query"]
}

web_content_schema = {
    "type": "object",
    "title": "Web Content Parameters",
    "description": "Parameters for retrieving and processing content from a web URL",
    "version": "1.0.0",
    "properties": {
        "url": {
            "type": "string",
            "format": "uri",
            "description": "URL to retrieve content from",
            "examples": ["https://www.nih.gov/news-events/nih-research-matters/artificial-pancreas-improves-type-1-diabetes-management"]
        },
        "max_length": {
            "type": "integer",
            "default": 2000,
            "minimum": 100,
            "maximum": 10000,
            "description": "Maximum length of content to return in characters"
        }
    },
    "additionalProperties": False,
    "required": ["url"]
}


# Define MCP tool definitions
MCP_TOOLS = [
    Tool(
        name="pubmed_search",
        description="Search PubMed for articles matching a query",
        input_schema=pubmed_search_schema
    ),
    Tool(
        name="get_article",
        description="Get detailed information about a specific PubMed article by PMID",
        input_schema=article_retrieval_schema
    ),
    Tool(
        name="get_pubmed_query_crafting_prompt",
        description="Get a prompt for an LLM to craft an optimized PubMed query from a question",
        input_schema=query_crafting_schema
    ),
    Tool(
        name="web_search",
        description="Search the web for information on a topic",
        input_schema=web_search_schema
    ),
    Tool(
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

# Server implementation info
SERVER_INFO = Implementation(
    name="BioMed-MCP-Server",
    version="0.1.0"
)

# Server capabilities
SERVER_CAPABILITIES = ServerCapabilities()


def handle_initialize_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle the initialize request from the client according to the MCP spec
    
    Parameters:
    request_data (Dict[str, Any]): The JSON-RPC request data
    
    Returns:
    Dict[str, Any]: JSON-RPC response
    """
    request_id = request_data.get("id")
    params = request_data.get("params", {})
    
    try:
        # Extract client capabilities and info
        client_protocol_version = params.get("protocolVersion")
        client_capabilities_dict = params.get("capabilities", {})
        client_info = params.get("clientInfo", {})
        
        logger.info(f"Client initialized connection: {client_info.get('name')}, version: {client_info.get('version')}")
        logger.info(f"Client protocol version: {client_protocol_version}")
        
        # Store client capabilities for future use
        client_capabilities = ClientCapabilities(client_capabilities_dict)
        
        # Create server response with our protocol version, capabilities and info
        result = {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": SERVER_CAPABILITIES.to_dict(),
            "serverInfo": SERVER_INFO.to_dict(),
            "instructions": "This is a biomedical research MCP server that provides tools for PubMed search, article retrieval, query optimization, and web search."
        }
        
        return {
            "jsonrpc": JSONRPC_VERSION,
            "result": result,
            "id": request_id
        }
        
    except Exception as e:
        logger.error(f"Error handling initialize request: {str(e)}\n{traceback.format_exc()}")
        return {
            "jsonrpc": JSONRPC_VERSION,
            "error": {"code": INTERNAL_ERROR, "message": f"Internal error: {str(e)}"},
            "id": request_id
        }


def handle_ping_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle a ping request from the client or server
    
    Parameters:
    request_data (Dict[str, Any]): The JSON-RPC request data
    
    Returns:
    Dict[str, Any]: JSON-RPC response with empty result
    """
    request_id = request_data.get("id")
    
    return {
        "jsonrpc": JSONRPC_VERSION,
        "result": {},  # Empty result as per spec
        "id": request_id
    }


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
            
            # Execute the search
            search_results = pubmed_tool.search(query, max_results=max_results, sort=sort)
            return MCPResponse(
                request_id=request_id,
                status="success",
                response=search_results
            )
        
        elif tool_name == "get_article":
            pmid = parameters.get("pmid")
            
            if not pmid:
                return MCPResponse(
                    request_id=request_id,
                    status="error",
                    response=None,
                    error="Missing required parameter: pmid"
                )
            
            # Retrieve the article
            article_details = pubmed_tool.fetch_article_details([pmid])
            if not article_details or len(article_details) == 0:
                return MCPResponse(
                    request_id=request_id,
                    status="error",
                    response=None,
                    error=f"Article with PMID {pmid} not found"
                )
                
            return MCPResponse(
                request_id=request_id,
                status="success",
                response={"article": article_details[0]}
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
            
            # Generate the query crafting prompt
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
            
            # Execute the web search
            web_results = web_search_tool.websearch(query, max_results=max_results)
            return MCPResponse(
                request_id=request_id,
                status="success",
                response={
                    "query": query,
                    "count": len(web_results),
                    "results": web_results
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
            
            # Retrieve the web content
            web_content = web_content_tool.webget(url, max_length=max_length)
            return MCPResponse(
                request_id=request_id,
                status="success",
                response={
                    "url": url,
                    "content": web_content
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


def handle_local_jsonrpc():
    """
    Handle JSON-RPC communication over stdin/stdout for local connections.
    This function implements an event loop that reads from stdin and writes to stdout.
    """
    logger.info("Starting MCP server in local mode (stdin/stdout)")
    logger.info("JSON-RPC requests will be read from stdin and responses written to stdout")
    
    try:
        # Disable Flask debug logs for cleaner stdout
        werkzeug_log = logging.getLogger('werkzeug')
        werkzeug_log.setLevel(logging.ERROR)
        
        # Configure stderr logger to avoid interfering with stdout JSON messages
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.handlers = [stderr_handler]
        
        logger.info("Local mode initialized. Waiting for JSON-RPC requests on stdin...")
        
        # Main processing loop
        while True:
            # Check if we have data on stdin without blocking
            if select.select([sys.stdin], [], [], 0.1)[0]:
                try:
                    # Read a line from stdin
                    line = sys.stdin.readline()
                    if not line:
                        # EOF - stdin closed
                        logger.info("Stdin closed. Exiting...")
                        break
                    
                    # Parse the JSON-RPC request
                    request_data = json.loads(line)
                    logger.debug(f"Received JSON-RPC request: {request_data}")
                    
                    # Process the request and generate response
                    if "method" not in request_data:
                        response = {
                            "jsonrpc": JSONRPC_VERSION,
                            "error": {"code": INVALID_REQUEST, "message": "Invalid request: missing method"},
                            "id": request_data.get("id")
                        }
                    else:
                        method = request_data.get("method")
                        
                        if method == "initialize":
                            response = handle_initialize_request(request_data)
                        elif method == "ping":
                            response = handle_ping_request(request_data)
                        elif method == "tools/list":
                            tools = [tool.to_dict() for tool in MCP_TOOLS]
                            response = {
                                "jsonrpc": JSONRPC_VERSION,
                                "result": {"tools": tools},
                                "id": request_data.get("id")
                            }
                        elif method == "tools/call":
                            params = request_data.get("params", {})
                            tool_name = params.get("name")
                            arguments = params.get("arguments", {})
                            
                            if not tool_name:
                                response = {
                                    "jsonrpc": JSONRPC_VERSION,
                                    "error": {"code": INVALID_PARAMS, "message": "Missing required parameter: name"},
                                    "id": request_data.get("id")
                                }
                            else:
                                # Find the tool
                                tool = next((t for t in MCP_TOOLS if t.name == tool_name), None)
                                if not tool:
                                    response = {
                                        "jsonrpc": JSONRPC_VERSION,
                                        "error": {"code": METHOD_NOT_FOUND, "message": f"Tool not found: {tool_name}"},
                                        "id": request_data.get("id")
                                    }
                                else:
                                    # Create MCPRequest and execute tool
                                    mcp_request = MCPRequest(
                                        request_id=str(uuid.uuid4()),
                                        tool_name=tool_name,
                                        parameters=arguments
                                    )
                                    
                                    mcp_response = execute_tool(mcp_request)
                                    
                                    if mcp_response.status == "success":
                                        # Convert to proper CallToolResult
                                        tool_result = {
                                            "content": [
                                                {
                                                    "type": "text",
                                                    "text": json.dumps(mcp_response.response, indent=2)
                                                }
                                            ]
                                        }
                                        
                                        response = {
                                            "jsonrpc": JSONRPC_VERSION,
                                            "result": tool_result,
                                            "id": request_data.get("id")
                                        }
                                    else:
                                        # Tool execution failed
                                        error_message = mcp_response.error or "Unknown error"
                                        tool_result = {
                                            "content": [
                                                {
                                                    "type": "text",
                                                    "text": f"Error: {error_message}"
                                                }
                                            ],
                                            "isError": True
                                        }
                                        
                                        response = {
                                            "jsonrpc": JSONRPC_VERSION,
                                            "result": tool_result,
                                            "id": request_data.get("id")
                                        }
                        elif method == "MCP.getTools":
                            response = {
                                "jsonrpc": JSONRPC_VERSION,
                                "result": MCP_MANIFEST.to_dict(),
                                "id": request_data.get("id")
                            }
                        elif method == "MCP.execute":
                            params = request_data.get("params", {})
                            
                            if isinstance(params, dict):
                                tool_request_id = params.get("request_id", str(uuid.uuid4()))
                                tool_name = params.get("tool_name")
                                tool_parameters = params.get("parameters", {})
                                input_value = params.get("input_value")
                                
                                if not tool_name:
                                    response = {
                                        "jsonrpc": JSONRPC_VERSION,
                                        "error": {"code": INVALID_PARAMS, "message": "Missing required parameter: tool_name"},
                                        "id": request_data.get("id")
                                    }
                                else:
                                    # Create MCPRequest from the parameters
                                    mcp_request = MCPRequest(
                                        request_id=tool_request_id,
                                        tool_name=tool_name,
                                        parameters=tool_parameters,
                                        input_value=input_value
                                    )
                                    
                                    # Execute the tool
                                    mcp_response = execute_tool(mcp_request)
                                    
                                    response = {
                                        "jsonrpc": JSONRPC_VERSION,
                                        "result": mcp_response.to_dict(),
                                        "id": request_data.get("id")
                                    }
                            else:
                                response = {
                                    "jsonrpc": JSONRPC_VERSION,
                                    "error": {"code": INVALID_PARAMS, "message": "Invalid params format"},
                                    "id": request_data.get("id")
                                }
                        else:
                            response = {
                                "jsonrpc": JSONRPC_VERSION,
                                "error": {"code": METHOD_NOT_FOUND, "message": f"Method not found: {method}"},
                                "id": request_data.get("id")
                            }
                    
                    # Write the response to stdout
                    sys.stdout.write(json.dumps(response) + "\n")
                    sys.stdout.flush()
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON received: {e}")
                    error_response = {
                        "jsonrpc": JSONRPC_VERSION,
                        "error": {"code": PARSE_ERROR, "message": "Parse error: Invalid JSON"},
                        "id": None
                    }
                    sys.stdout.write(json.dumps(error_response) + "\n")
                    sys.stdout.flush()
                    
                except Exception as e:
                    logger.error(f"Error processing request: {str(e)}\n{traceback.format_exc()}")
                    error_response = {
                        "jsonrpc": JSONRPC_VERSION,
                        "error": {"code": INTERNAL_ERROR, "message": f"Internal error: {str(e)}"},
                        "id": None if not request_data or not isinstance(request_data, dict) else request_data.get("id")
                    }
                    sys.stdout.write(json.dumps(error_response) + "\n")
                    sys.stdout.flush()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Exiting...")
    except Exception as e:
        logger.error(f"Unhandled exception in local mode: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1)


@app.route('/jsonrpc', methods=['POST'])
def handle_jsonrpc():
    """Handle JSON-RPC requests as per MCP specifications."""
    try:
        request_data = request.get_json()
        if not request_data:
            logger.error("Invalid JSON request received")
            return jsonify({
                "jsonrpc": JSONRPC_VERSION,
                "error": {"code": PARSE_ERROR, "message": "Parse error: Invalid JSON was received"},
                "id": None
            }), 400

        method = request_data.get("method")
        request_id = request_data.get("id")
        params = request_data.get("params", {})

        logger.debug(f"Incoming JSON-RPC request: method={method}, id={request_id}")
        
        # Handle standard MCP protocol methods
        if method == "initialize":
            # Handle initialization request
            return jsonify(handle_initialize_request(request_data))
            
        elif method == "ping":
            # Handle ping request
            return jsonify(handle_ping_request(request_data))
            
        elif method == "tools/list":
            # Return the list of tools
            tools = [tool.to_dict() for tool in MCP_TOOLS]
            return jsonify({
                "jsonrpc": JSONRPC_VERSION,
                "result": {"tools": tools},
                "id": request_id
            })
            
        elif method == "tools/call":
            # Call a tool
            try:
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                
                if not tool_name:
                    return jsonify({
                        "jsonrpc": JSONRPC_VERSION,
                        "error": {"code": INVALID_PARAMS, "message": "Missing required parameter: name"},
                        "id": request_id
                    })
                
                # Find the tool
                tool = next((t for t in MCP_TOOLS if t.name == tool_name), None)
                if not tool:
                    return jsonify({
                        "jsonrpc": JSONRPC_VERSION,
                        "error": {"code": METHOD_NOT_FOUND, "message": f"Tool not found: {tool_name}"},
                        "id": request_id
                    })
                
                # Create MCPRequest and execute tool
                mcp_request = MCPRequest(
                    request_id=str(uuid.uuid4()),
                    tool_name=tool_name,
                    parameters=arguments
                )
                
                mcp_response = execute_tool(mcp_request)
                
                # Format response according to MCP spec
                if mcp_response.status == "success":
                    # Convert to proper CallToolResult
                    tool_result = {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(mcp_response.response, indent=2)
                            }
                        ]
                    }
                    
                    return jsonify({
                        "jsonrpc": JSONRPC_VERSION,
                        "result": tool_result,
                        "id": request_id
                    })
                else:
                    # Tool execution failed, but we still return a proper CallToolResult
                    # with isError=true as per spec
                    error_message = mcp_response.error or "Unknown error"
                    tool_result = {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Error: {error_message}"
                            }
                        ],
                        "isError": True
                    }
                    
                    return jsonify({
                        "jsonrpc": JSONRPC_VERSION,
                        "result": tool_result,
                        "id": request_id
                    })
                    
            except Exception as e:
                logger.error(f"Error calling tool: {str(e)}\n{traceback.format_exc()}")
                return jsonify({
                    "jsonrpc": JSONRPC_VERSION,
                    "error": {"code": INTERNAL_ERROR, "message": f"Internal error: {str(e)}"},
                    "id": request_id
                })
        
        # Legacy methods for backward compatibility
        elif method == "MCP.getTools":
            # Return the full MCP manifest with complete tool definitions
            tools_response = {
                "jsonrpc": JSONRPC_VERSION,
                "result": MCP_MANIFEST.to_dict(),
                "id": request_id
            }
            logger.debug(f"Generated response for MCP.getTools: {tools_response}")
            return app.response_class(
                response=json.dumps(tools_response),
                status=200,
                mimetype='application/json'
            )
            
        elif method == "MCP.execute":
            logger.debug(f"Processing MCP.execute with params: {params}")
            
            try:
                # For MCP.execute, we expect params to be a dict with request_id, tool_name, parameters
                if isinstance(params, dict):
                    # Extract parameters from the request
                    tool_request_id = params.get("request_id", str(uuid.uuid4()))
                    tool_name = params.get("tool_name")
                    tool_parameters = params.get("parameters", {})
                    input_value = params.get("input_value")
                else:
                    logger.error(f"Invalid params format: {params}")
                    return jsonify({
                        "jsonrpc": JSONRPC_VERSION,
                        "error": {"code": INVALID_PARAMS, "message": "Invalid params format"},
                        "id": request_id
                    }), 400
                
                logger.debug(f"Extracted tool request: id={tool_request_id}, tool={tool_name}")
                
                if not tool_name:
                    logger.error("Missing tool_name in MCP.execute request")
                    return jsonify({
                        "jsonrpc": JSONRPC_VERSION,
                        "error": {"code": INVALID_PARAMS, "message": "Missing required parameter: tool_name"},
                        "id": request_id
                    }), 400
                
                # Create MCPRequest from the parameters
                mcp_request = MCPRequest(
                    request_id=tool_request_id,
                    tool_name=tool_name,
                    parameters=tool_parameters,
                    input_value=input_value
                )
                
                # Execute the tool
                logger.debug(f"Executing tool {tool_name}")
                mcp_response = execute_tool(mcp_request)
                
                # Return the result as proper JSON-RPC response
                return jsonify({
                    "jsonrpc": JSONRPC_VERSION,
                    "result": mcp_response.to_dict(),
                    "id": request_id
                })
                
            except Exception as e:
                logger.error(f"Error executing MCP.execute: {str(e)}\n{traceback.format_exc()}")
                return jsonify({
                    "jsonrpc": JSONRPC_VERSION,
                    "error": {"code": INTERNAL_ERROR, "message": f"Internal error: {str(e)}"},
                    "id": request_id
                }), 500

        # Handle unsupported methods
        logger.warning(f"Unsupported method called: {method}")
        return jsonify({
            "jsonrpc": JSONRPC_VERSION,
            "error": {"code": METHOD_NOT_FOUND, "message": f"Method not found: {method}"},
            "id": request_id
        }), 404

    except Exception as e:
        # Handle internal server errors
        logger.error(f"JSON-RPC error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "jsonrpc": JSONRPC_VERSION,
            "error": {"code": INTERNAL_ERROR, "message": "Internal error", "data": str(e)},
            "id": None if not request_data or not isinstance(request_data, dict) else request_data.get("id")
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for the server"""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "version": MCP_PROTOCOL_VERSION,
        "tools_available": len(MCP_TOOLS)
    })


# Server running mode - set via command-line arguments
SERVER_MODE = "network"  # Default to network mode, can be "local" for stdin/stdout

# Import argparse for command-line argument handling
import argparse

# Entry point for running the server
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='BioMed MCP Server')
    parser.add_argument('--mode', choices=['local', 'network'], default='network',
                      help='Communication mode: "local" for stdin/stdout or "network" for Flask HTTP server')
    parser.add_argument('--port', type=int, default=5152,
                      help='Port number for network mode (default: 5152)')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode')
    args = parser.parse_args()
    
    # Set server mode from arguments
    SERVER_MODE = args.mode
    port = args.port
    debug_mode = args.debug or os.environ.get("DEBUG", "False").lower() == "true"
    
    # Log the server configuration
    logger.info(f"Starting BioMed MCP Server in {SERVER_MODE} mode")
    if SERVER_MODE == "network":
        logger.info(f"Server will listen on port {port}")
    else:
        logger.info("Server will communicate via stdin/stdout")
    
    logger.info(f"Debug mode: {debug_mode}")
    if NCBI_API_KEY:
        logger.info("NCBI API key configured (higher rate limits available)")
    else:
        logger.warning("No NCBI API key configured. Rate limits will be stricter.")
    
    # Launch in the appropriate mode
    if SERVER_MODE == "local":
        handle_local_jsonrpc()
    else:
        # Start the Flask server
        app.run(host="0.0.0.0", port=port, debug=debug_mode)