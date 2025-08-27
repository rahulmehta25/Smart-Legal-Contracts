#!/usr/bin/env python3
"""
WebSocket Integration Test Script

This script tests the WebSocket integration between frontend and backend.
"""

import asyncio
import websockets
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSocketTester:
    def __init__(self, url="ws://localhost:8000/ws"):
        self.url = url
        self.websocket = None
        
    async def connect(self):
        """Connect to the WebSocket server"""
        try:
            logger.info(f"Connecting to {self.url}")
            self.websocket = await websockets.connect(self.url)
            logger.info("WebSocket connected successfully")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    async def send_message(self, message):
        """Send a message to the WebSocket server"""
        if not self.websocket:
            logger.error("Not connected to WebSocket")
            return None
            
        try:
            if isinstance(message, dict):
                message = json.dumps(message)
            
            logger.info(f"Sending message: {message}")
            await self.websocket.send(message)
            
            # Wait for response
            response = await self.websocket.recv()
            logger.info(f"Received response: {response}")
            return json.loads(response)
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return None
    
    async def test_basic_connectivity(self):
        """Test basic WebSocket connectivity"""
        logger.info("Testing basic connectivity...")
        
        if not await self.connect():
            return False
        
        # Wait for welcome message
        try:
            welcome = await self.websocket.recv()
            logger.info(f"Welcome message: {welcome}")
            welcome_data = json.loads(welcome)
            
            if welcome_data.get("type") == "connected":
                logger.info("✅ Basic connectivity test passed")
                return True
            else:
                logger.error("❌ Unexpected welcome message format")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to receive welcome message: {e}")
            return False
    
    async def test_message_echo(self):
        """Test message echo functionality"""
        logger.info("Testing message echo...")
        
        test_message = {
            "type": "test",
            "data": "Hello WebSocket!",
            "timestamp": datetime.now().isoformat()
        }
        
        response = await self.send_message(test_message)
        if response and response.get("type") == "echo":
            logger.info("✅ Message echo test passed")
            return True
        else:
            logger.error("❌ Message echo test failed")
            return False
    
    async def test_text_message(self):
        """Test plain text message handling"""
        logger.info("Testing plain text message...")
        
        response = await self.send_message("Plain text test message")
        if response and response.get("type") == "text_received":
            logger.info("✅ Plain text message test passed")
            return True
        else:
            logger.error("❌ Plain text message test failed")
            return False
    
    async def close(self):
        """Close the WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            logger.info("WebSocket connection closed")
    
    async def run_all_tests(self):
        """Run all WebSocket tests"""
        logger.info("🚀 Starting WebSocket Integration Tests")
        logger.info("=" * 50)
        
        test_results = {}
        
        # Test 1: Basic Connectivity
        test_results["connectivity"] = await self.test_basic_connectivity()
        
        # Test 2: Message Echo
        if test_results["connectivity"]:
            test_results["echo"] = await self.test_message_echo()
            
            # Test 3: Plain Text Messages
            test_results["text_message"] = await self.test_text_message()
        else:
            test_results["echo"] = False
            test_results["text_message"] = False
        
        # Close connection
        await self.close()
        
        # Print results
        logger.info("=" * 50)
        logger.info("📊 Test Results Summary:")
        logger.info(f"Basic Connectivity: {'✅ PASS' if test_results['connectivity'] else '❌ FAIL'}")
        logger.info(f"Message Echo: {'✅ PASS' if test_results['echo'] else '❌ FAIL'}")
        logger.info(f"Text Messages: {'✅ PASS' if test_results['text_message'] else '❌ FAIL'}")
        
        all_passed = all(test_results.values())
        logger.info("=" * 50)
        logger.info(f"🎯 Overall Result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
        
        return all_passed


async def main():
    """Main test function"""
    tester = WebSocketTester()
    
    try:
        success = await tester.run_all_tests()
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Test failed with unexpected error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)