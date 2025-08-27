#!/usr/bin/env python3
"""
Simple WebSocket test to verify the server configuration
"""
import asyncio
import json
import websockets
import sys

async def test_websocket():
    """Test WebSocket connection and basic functionality"""
    try:
        # Connect to WebSocket
        print("Connecting to WebSocket at ws://localhost:8001/ws")
        async with websockets.connect("ws://localhost:8001/ws") as websocket:
            print("✓ WebSocket connected successfully")
            
            # Wait for welcome message
            welcome_message = await websocket.recv()
            welcome_data = json.loads(welcome_message)
            print(f"✓ Received welcome message: {welcome_data}")
            
            # Test sending a JSON message
            test_message = {
                "type": "test",
                "message": "Hello from WebSocket test",
                "timestamp": "2025-08-26T14:43:00Z"
            }
            await websocket.send(json.dumps(test_message))
            print(f"✓ Sent test message: {test_message}")
            
            # Wait for response
            response = await websocket.recv()
            response_data = json.loads(response)
            print(f"✓ Received response: {response_data}")
            
            # Test sending plain text
            await websocket.send("Plain text message")
            print("✓ Sent plain text message")
            
            # Wait for response
            response = await websocket.recv()
            response_data = json.loads(response)
            print(f"✓ Received response: {response_data}")
            
            print("\n🎉 All WebSocket tests passed!")
            
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_websocket())