#!/usr/bin/env python3
"""
WebSocket connection test with CORS headers
"""

import asyncio
import websockets
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_websocket_connection():
    """Test WebSocket connection to the server."""
    
    # Test WebSocket connection
    uri = "ws://localhost:8001/ws"
    
    try:
        # Add origin header to simulate frontend request
        headers = {
            "Origin": "https://test-app.vercel.app"
        }
        
        logger.info(f"Connecting to WebSocket at {uri}")
        
        async with websockets.connect(uri, additional_headers=headers) as websocket:
            logger.info("WebSocket connected successfully!")
            
            # Wait for welcome message
            welcome_message = await websocket.recv()
            logger.info(f"Received welcome: {welcome_message}")
            
            # Send test message
            test_message = {
                "type": "test",
                "message": "Hello from test client!",
                "timestamp": "2025-08-26T15:22:00Z"
            }
            
            await websocket.send(json.dumps(test_message))
            logger.info(f"Sent test message: {test_message}")
            
            # Wait for echo response
            response = await websocket.recv()
            logger.info(f"Received response: {response}")
            
            # Send plain text message
            await websocket.send("Plain text test message")
            plain_response = await websocket.recv()
            logger.info(f"Received plain text response: {plain_response}")
            
        logger.info("WebSocket test completed successfully!")
        
    except Exception as e:
        logger.error(f"WebSocket connection failed: {e}")
        return False
    
    return True

async def main():
    """Main test function."""
    logger.info("Starting WebSocket CORS test...")
    
    success = await test_websocket_connection()
    
    if success:
        logger.info("✅ WebSocket connection test PASSED")
    else:
        logger.error("❌ WebSocket connection test FAILED")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())