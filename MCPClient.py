"""
framework free implementation of a basic MCP client
using the requests library for HTTP communication
"""
import logging
import json
import uuid
import requests
import time
import sseclient
from typing import Dict, List, Any, Optional, Tuple, Union, Literal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mcp-client')

DEFAULT_MCP_URL = "http://localhost:5152"

class MCPRequest:
    """Represents an MCP request to the server"""
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


class MCPClient:
    """MCP Client for communicating with an MCP server using JSON-RPC"""

    def __init__(self, base_url: str = DEFAULT_MCP_URL):
        self.base_url = base_url
        self.manifest = None
        self.tool_map = {}
        self.server_capabilities = {}
        self.protocol_version = None
        self.server_info = None
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        self.initialized = False
        self.initialize()  # Perform initialization immediately

    def initialize(self) -> bool:
        """
        Initialize the connection to the MCP server
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            # Create client info
            client_info = {
                "name": "BioMed-MCP-Agent",
                "version": "0.1.0"
            }
            
            # Create client capabilities
            client_capabilities = {}  # We don't advertise any special capabilities
            
            # Create initialize request
            initialize_params = {
                "protocolVersion": "2024-11-05",  # Use latest official MCP version
                "clientInfo": client_info,
                "capabilities": client_capabilities
            }
            
            logger.info("Initializing connection to MCP server...")
            response = self._send_request_with_retry("initialize", **initialize_params)
            
            if "result" not in response:
                logger.error("Invalid initialize response from server")
                return False
                
            # Extract server information
            result = response["result"]
            self.protocol_version = result.get("protocolVersion")
            self.server_capabilities = result.get("capabilities", {})
            self.server_info = result.get("serverInfo", {})
            instructions = result.get("instructions", "")
            
            logger.info(f"Connected to MCP server: {self.server_info.get('name', 'Unknown')} {self.server_info.get('version', '')}")
            logger.info(f"Protocol version: {self.protocol_version}")
            if instructions:
                logger.debug(f"Server instructions: {instructions}")
                
            # Send initialized notification (no response expected)
            self._send_notification("notifications/initialized")
            
            # Fetch tools list
            self._fetch_tools()
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP connection: {str(e)}")
            return False

    def _fetch_tools(self) -> None:
        """Fetch and store the MCP tools list"""
        try:
            logger.debug("Fetching tools list from the server")
            response = self._send_request_with_retry("tools/list")
            
            # Validate the response structure
            if not isinstance(response, dict) or "result" not in response:
                logger.error("Invalid response format received from server")
                self.tool_map = {}
                return

            tools = response["result"].get("tools", [])
            logger.debug(f"Received {len(tools)} tools from server")
            
            # Build a tool map for easy access
            self.tool_map = {}
            for tool in tools:
                if isinstance(tool, dict) and "name" in tool:
                    self.tool_map[tool.get("name")] = tool

            logger.info(f"Fetched {len(self.tool_map)} tools from MCP server")
            logger.debug(f"Available tools: {list(self.tool_map.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to fetch MCP tools list: {str(e)}")
            self.tool_map = {}

    def _send_notification(self, method: str, **params) -> None:
        """
        Send a JSON-RPC notification (no response expected)
        
        Args:
            method: The JSON-RPC method to call
            **params: Parameters for the method
        """
        try:
            # Create a proper JSON-RPC notification object
            notification_data = {
                "jsonrpc": "2.0",
                "method": method,
                "params": params if params else {}
            }
            
            # Send the notification directly
            logger.debug(f"Sending JSON-RPC notification: {method}")
            
            headers = {"Content-Type": "application/json"}
            requests.post(
                f"{self.base_url}/jsonrpc", 
                headers=headers,
                json=notification_data,
                timeout=5
            )
            
        except Exception as e:
            logger.warning(f"Error sending notification {method}: {str(e)}")
            # We don't raise an exception for notifications

    def _send_request_with_retry(self, method: str, **params) -> Dict[str, Any]:
        """
        Send a JSON-RPC request with retry logic
        
        Args:
            method: The JSON-RPC method to call
            params: Parameters for the method
            
        Returns:
            The JSON-RPC response
        """
        attempts = 0
        last_error = None
        
        while attempts < self.max_retries:
            try:
                # Create a proper JSON-RPC request object
                request_data = {
                    "jsonrpc": "2.0",
                    "method": method,
                    "params": params if params else {},
                    "id": str(uuid.uuid4())
                }
                
                # Send the request directly without using jsonrpcclient
                logger.debug(f"Sending direct JSON-RPC request: {method}")
                
                headers = {"Content-Type": "application/json"}
                response = requests.post(
                    f"{self.base_url}/jsonrpc", 
                    headers=headers,
                    json=request_data,
                    timeout=30
                )
                response.raise_for_status()
                
                # Parse the response
                response_data = response.json()
                
                # Check for errors in the JSON-RPC response
                if "error" in response_data:
                    error = response_data["error"]
                    logger.error(f"JSON-RPC error: {error.get('message', 'Unknown error')}")
                    raise RuntimeError(f"JSON-RPC error: {error.get('message', 'Unknown error')}")
                
                # Return the result
                return response_data
                
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error (attempt {attempts+1}/{self.max_retries}): {str(e)}")
                last_error = e
            except requests.exceptions.Timeout as e:
                logger.warning(f"Request timeout (attempt {attempts+1}/{self.max_retries}): {str(e)}")
                last_error = e
            except Exception as e:
                logger.error(f"Error in request (attempt {attempts+1}/{self.max_retries}): {str(e)}")
                last_error = e
            
            # Exponential backoff
            time.sleep(self.retry_delay * (2 ** attempts))
            attempts += 1
        
        logger.error(f"Failed after {self.max_retries} attempts. Last error: {str(last_error)}")
        raise last_error or RuntimeError("Failed to send request after multiple attempts")

    def _stream_request(self, method: str, **params) -> Any:
        """
        Send a JSON-RPC request and handle streaming responses via SSE.

        Args:
            method: The JSON-RPC method to call
            params: Parameters for the method

        Returns:
            Generator yielding streamed data
        """
        try:
            # Create a proper JSON-RPC request object
            request_data = {
                "jsonrpc": "2.0",
                "method": method,
                "params": params if params else {},
                "id": str(uuid.uuid4())
            }

            headers = {"Content-Type": "application/json"}
            response = requests.post(
                f"{self.base_url}/jsonrpc",
                headers=headers,
                json=request_data,
                stream=True,
                timeout=30
            )

            # Use sseclient to handle the streaming response
            client = sseclient.SSEClient(response)
            for event in client.events():
                yield event.data

        except Exception as e:
            logger.error(f"Error in streaming request: {str(e)}")
            raise RuntimeError(f"Streaming request failed: {str(e)}")

    def ping(self) -> bool:
        """
        Send a ping to check if the server is alive
        
        Returns:
            bool: True if server responded, False otherwise
        """
        try:
            response = self._send_request_with_retry("ping")
            return True
        except Exception as e:
            logger.error(f"Ping failed: {str(e)}")
            return False

    def execute_tool(self, tool_name: str, parameters: Dict[str, Any], input_value: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a tool via the MCP protocol using JSON-RPC
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            input_value: Optional input value for the tool
            
        Returns:
            The tool execution result
        """
        if not self.initialized:
            logger.warning("Attempting to execute tool before initialization")
            if not self.initialize():
                return {
                    "status": "error",
                    "error": "Failed to initialize connection to MCP server"
                }

        # Use tools/call instead of MCP.execute to be spec-compliant
        try:
            # Verify the tool exists
            if tool_name not in self.tool_map:
                logger.warning(f"Unknown tool: {tool_name}")
                # Try to refresh tools list
                self._fetch_tools()
                if tool_name not in self.tool_map:
                    return {
                        "status": "error",
                        "error": f"Unknown tool: {tool_name}"
                    }
            
            # Call the tool
            response = self._send_request_with_retry("tools/call", 
                name=tool_name, 
                arguments=parameters
            )
            
            # Parse the result
            result = response.get("result", {})
            
            # Check for tool error
            if result.get("isError", False):
                content = result.get("content", [])
                error_text = ""
                for item in content:
                    if item.get("type") == "text":
                        error_text += item.get("text", "")
                
                return {
                    "status": "error",
                    "error": error_text or "Tool execution failed"
                }
                
            # Extract text content from the result
            content = result.get("content", [])
            text_content = ""
            
            for item in content:
                if item.get("type") == "text":
                    text_content += item.get("text", "")
            
            # Try to parse the text content as JSON
            try:
                parsed_content = json.loads(text_content)
                return {
                    "status": "success",
                    "response": parsed_content
                }
            except json.JSONDecodeError:
                # Return as raw text if not valid JSON
                return {
                    "status": "success",
                    "response": {"text": text_content}
                }
                
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            return {
                "status": "error",
                "error": f"Error executing tool: {str(e)}"
            }

    def execute_tool_stream(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """
        Execute a tool via the MCP protocol using JSON-RPC with streaming support.

        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters

        Returns:
            Generator yielding streamed data
        """
        if not self.initialized:
            logger.warning("Attempting to execute tool before initialization")
            if not self.initialize():
                raise RuntimeError("Failed to initialize connection to MCP server")

        # Verify the tool exists
        if tool_name not in self.tool_map:
            logger.warning(f"Unknown tool: {tool_name}")
            self._fetch_tools()
            if tool_name not in self.tool_map:
                raise ValueError(f"Unknown tool: {tool_name}")

        # Stream the tool execution
        return self._stream_request("tools/call", name=tool_name, arguments=parameters)