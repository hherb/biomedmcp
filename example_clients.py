#!/usr/bin/env python3
"""
Example usage of both MCP client types:
1. MCPClient - Network-based client (HTTP)
2. MCPLocalClient - Local client using stdin/stdout

This example shows how to use both client types to connect to the same MCP server
and execute the same tools.
"""

import sys
import os
import time
import logging
import json
from typing import Dict, Any, Optional

# Import both client types
from MCPClient import MCPClient
from MCPLocalClient import MCPLocalClient


def run_example_queries(client, client_type: str):
    """Run some example tool executions using the provided client"""
    print(f"\n=== Running examples with {client_type} ===")
    
    # Example 1: Run a PubMed search
    print("\n--- Example 1: PubMed Search ---")
    result = client.execute_tool(
        "pubmed_search",
        {
            "query": """'artificial intelligence' AND medicine""",
            "max_results": 3
        }
    )
    
    if result["status"] == "success":
        articles = result["response"].get("articles", [])
        print(f"Found {len(articles)} articles:")
        for article in articles:
            print(f"- {article.get('title', 'No title')} (PMID: {article.get('pmid', 'N/A')})")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    # Example 2: Get query crafting prompt
    print("\n--- Example 2: Query Crafting Prompt ---")
    result = client.execute_tool(
        "get_pubmed_query_crafting_prompt",
        {
            "question": "What are the latest advances in using AI for cancer diagnosis?"
        }
    )
    
    if result["status"] == "success":
        prompt = result["response"].get("prompt", "")
        print(f"Generated prompt (truncated): {prompt[:100]}...")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    # Example 3: Web search
    print("\n--- Example 3: Web Search ---")
    result = client.execute_tool(
        "web_search",
        {
            "query": "latest medical AI research",
            "max_results": 3
        }
    )
    
    if result["status"] == "success":
        results = result["response"].get("results", [])
        print(f"Found {len(results)} web search results:")
        for idx, res in enumerate(results, 1):
            print(f"{idx}. {res.get('title', 'No title')} - {res.get('url', 'No URL')}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")


def main():
    """Run examples with both client types"""
    # Configure basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Example 1: Use network client
    try:
        print("\n=== Testing Network Client (MCPClient) ===")
        print("Starting MCP server in network mode...")
        
        # Start the server in a separate process (optional if it's already running)
        import subprocess
        server_process = subprocess.Popen(
            [sys.executable, "biomed_mcpserver.py", "--mode", "network"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Give the server a moment to start up
        time.sleep(2)
        
        # Create network client
        network_client = MCPClient()
        
        # Run the examples
        run_example_queries(network_client, "Network Client")
        
        # Cleanup
        print("\nClosing network client")
        if server_process:
            server_process.terminate()
            server_process.wait(timeout=5)
            
    except Exception as e:
        print(f"Error with network client: {str(e)}")
    
    # Example 2: Use local client
    try:
        print("\n=== Testing Local Client (MCPLocalClient) ===")
        
        # Create local client (automatically starts the server)
        local_client = MCPLocalClient()
        
        # Run the examples
        run_example_queries(local_client, "Local Client")
        
        # Cleanup
        print("\nClosing local client")
        local_client.close()
        
    except Exception as e:
        print(f"Error with local client: {str(e)}")


if __name__ == "__main__":
    main()