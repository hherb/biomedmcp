"""
Framework-free implementation of a basic MCP client for local mode
using subprocess and stdin/stdout communication instead of HTTP
"""
import logging
import json
import uuid
import time
import subprocess
import threading
import queue
import os
import sys
from typing import Dict, List, Any, Optional, Tuple, Union, Literal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mcp-local-client')

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


class MCPLocalClient:
    """
    MCP Client for communicating with an MCP server in local mode
    using subprocess and stdin/stdout communication instead of HTTP
    """

    def __init__(self, server_path: str = None, python_executable: str = "python"):
        """
        Initialize the MCP Local Client
        
        Args:
            server_path: Path to the MCP server script. If None, will use default path
            python_executable: Python executable to use for launching the server
        """
        self.server_path = server_path or os.path.join(os.path.dirname(__file__), "biomed_mcpserver.py")
        self.python_executable = python_executable
        self.server_process = None
        self.manifest = None
        self.tool_map = {}
        self.server_capabilities = {}
        self.protocol_version = None
        self.server_info = None
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        self.initialized = False
        self.response_queue = queue.Queue()
        self.response_map = {}  # Map request IDs to responses
        self.reader_thread = None
        
        # Start the server
        self._start_server()
        
        # Initialize the connection
        self.initialize()
        
    def _start_server(self) -> None:
        """Start the MCP server as a subprocess communicating via stdin/stdout"""
        try:
            logger.info(f"Starting MCP server from {self.server_path} in local mode")
            # Start the server process
            self.server_process = subprocess.Popen(
                [self.python_executable, self.server_path, "--mode", "local"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line-buffered
            )
            
            # Start a thread to read server responses
            self.reader_thread = threading.Thread(
                target=self._read_server_output,
                daemon=True
            )
            self.reader_thread.start()
            
            # Start a thread to read server stderr
            stderr_thread = threading.Thread(
                target=self._read_server_stderr,
                daemon=True
            )
            stderr_thread.start()
            
            logger.info("MCP server started in local mode")
        except Exception as e:
            logger.error(f"Failed to start MCP server: {str(e)}")
            raise RuntimeError(f"Failed to start MCP server: {str(e)}")
            
    def _read_server_output(self) -> None:
        """Read and process server responses from stdout"""
        while self.server_process and self.server_process.poll() is None:
            try:
                # Read a line from the server's stdout
                line = self.server_process.stdout.readline()
                if not line:
                    # End of file - server closed stdout
                    logger.warning("Server stdout closed")
                    break
                    
                # Parse the JSON response
                try:
                    response = json.loads(line)
                    logger.debug(f"Received response: {response}")
                    
                    # Get the request ID
                    request_id = response.get("id")
                    if request_id:
                        # Store the response in the map
                        self.response_map[request_id] = response
                        # Signal that a response is available
                        self.response_queue.put(request_id)
                    else:
                        # Handle notifications (no ID)
                        logger.debug(f"Received notification: {response}")
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON response from server: {line}")
            except Exception as e:
                logger.error(f"Error reading server output: {str(e)}")
                
        logger.info("Server output reader thread exiting")
    
    def _read_server_stderr(self) -> None:
        """Read and log server stderr output"""
        while self.server_process and self.server_process.poll() is None:
            try:
                # Read a line from the server's stderr
                line = self.server_process.stderr.readline()
                if not line:
                    # End of file - server closed stderr
                    break
                    
                # Log the server's stderr output
                logger.debug(f"Server stderr: {line.strip()}")
            except Exception as e:
                logger.error(f"Error reading server stderr: {str(e)}")
                
        logger.info("Server stderr reader thread exiting")
            
    def __del__(self):
        """Clean up resources when the client is destroyed"""
        self.close()
        
    def close(self) -> None:
        """Close the connection to the MCP server and terminate the subprocess"""
        try:
            if self.server_process:
                logger.info("Terminating MCP server process")
                # Close stdin to signal the server to exit
                if self.server_process.stdin:
                    self.server_process.stdin.close()
                    
                # Wait for a short time for the process to exit
                for _ in range(10):
                    if self.server_process.poll() is not None:
                        break
                    time.sleep(0.1)
                    
                # If still running, terminate
                if self.server_process.poll() is None:
                    self.server_process.terminate()
                    self.server_process.wait(timeout=2)
                    
                # Force kill if still running
                if self.server_process.poll() is None:
                    self.server_process.kill()
                    
                self.server_process = None
        except Exception as e:
            logger.error(f"Error closing MCP server: {str(e)}")

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
            
            logger.info("Initializing connection to MCP local server...")
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
            
            # Send the notification directly via stdin
            logger.debug(f"Sending JSON-RPC notification: {method}")
            
            # Write the notification to the server's stdin
            json_str = json.dumps(notification_data) + "\n"
            self.server_process.stdin.write(json_str)
            self.server_process.stdin.flush()
            
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
                # Generate a unique request ID
                request_id = str(uuid.uuid4())
                
                # Create a proper JSON-RPC request object
                request_data = {
                    "jsonrpc": "2.0",
                    "method": method,
                    "params": params if params else {},
                    "id": request_id
                }
                
                logger.debug(f"Sending JSON-RPC request: {method} (id: {request_id})")
                
                # Write the request to the server's stdin
                json_str = json.dumps(request_data) + "\n"
                self.server_process.stdin.write(json_str)
                self.server_process.stdin.flush()
                
                # Wait for the response
                try:
                    # Wait for the response to be available
                    wait_time = 30  # seconds
                    start_time = time.time()
                    
                    while time.time() - start_time < wait_time:
                        # Check if the response is already in the map
                        if request_id in self.response_map:
                            response = self.response_map.pop(request_id)
                            return response
                            
                        # Wait for a notification from the reader thread
                        try:
                            received_id = self.response_queue.get(timeout=1.0)
                            if received_id == request_id:
                                response = self.response_map.pop(request_id)
                                return response
                        except queue.Empty:
                            # No response yet, continue waiting
                            pass
                            
                        # Check if the server process is still alive
                        if self.server_process.poll() is not None:
                            logger.error(f"Server process exited with code {self.server_process.returncode}")
                            raise RuntimeError(f"Server process exited with code {self.server_process.returncode}")
                    
                    # If we get here, the wait timed out
                    raise TimeoutError(f"Timeout waiting for response to {method} request")
                    
                except Exception as e:
                    logger.error(f"Error waiting for response: {str(e)}")
                    raise
                
            except TimeoutError as e:
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