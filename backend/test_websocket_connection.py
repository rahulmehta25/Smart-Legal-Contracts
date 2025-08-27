#!/usr/bin/env python3
"""
WebSocket Connection Test

Specific test for WebSocket connectivity without authentication.
"""

import asyncio
import websockets
import json
from datetime import datetime

WS_URL = "ws://localhost:8000/ws"

async def test_websocket_connection():
    """Test WebSocket connection."""
    print("🔌 Testing WebSocket Connection...")
    print(f"URL: {WS_URL}")
    
    results = []
    
    # Test 1: Connection without token (should fail)
    print("\n1. Testing connection without authentication token...")
    try:
        async with websockets.connect(WS_URL, timeout=5) as websocket:
            print("✅ Connected successfully (unexpected)")
            results.append({"test": "no_auth", "status": "success", "note": "Connected without auth"})
            
            # Try sending a message
            test_message = {"type": "ping", "data": "test"}
            await websocket.send(json.dumps(test_message))
            
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=2)
                print(f"   📨 Received: {response}")
                results.append({"test": "message_no_auth", "status": "success", "response": response})
            except asyncio.TimeoutError:
                print("   ⏱️ No response received (timeout)")
                results.append({"test": "message_no_auth", "status": "timeout"})
                
    except websockets.exceptions.ConnectionClosed as e:
        print(f"❌ Connection closed: {e}")
        results.append({"test": "no_auth", "status": "closed", "error": str(e)})
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        results.append({"test": "no_auth", "status": "failed", "error": str(e)})
    
    # Test 2: Connection with token parameter (invalid token)
    print("\n2. Testing connection with token parameter...")
    ws_url_with_token = f"{WS_URL}?token=invalid_token"
    try:
        async with websockets.connect(ws_url_with_token, timeout=5) as websocket:
            print("✅ Connected with token parameter")
            results.append({"test": "with_token", "status": "success", "note": "Connected with invalid token"})
            
            # Try sending a message
            test_message = {"type": "ping", "data": "test_with_token"}
            await websocket.send(json.dumps(test_message))
            
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=2)
                print(f"   📨 Received: {response}")
                results.append({"test": "message_with_token", "status": "success", "response": response})
            except asyncio.TimeoutError:
                print("   ⏱️ No response received (timeout)")
                results.append({"test": "message_with_token", "status": "timeout"})
                
    except websockets.exceptions.ConnectionClosed as e:
        print(f"❌ Connection closed: {e}")
        results.append({"test": "with_token", "status": "closed", "error": str(e)})
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        results.append({"test": "with_token", "status": "failed", "error": str(e)})
    
    # Test 3: Connection with headers
    print("\n3. Testing connection with authorization header...")
    try:
        headers = {"Authorization": "Bearer invalid_token"}
        async with websockets.connect(WS_URL, extra_headers=headers, timeout=5) as websocket:
            print("✅ Connected with authorization header")
            results.append({"test": "with_header", "status": "success", "note": "Connected with auth header"})
            
            # Try sending a message
            test_message = {"type": "ping", "data": "test_with_header"}
            await websocket.send(json.dumps(test_message))
            
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=2)
                print(f"   📨 Received: {response}")
                results.append({"test": "message_with_header", "status": "success", "response": response})
            except asyncio.TimeoutError:
                print("   ⏱️ No response received (timeout)")
                results.append({"test": "message_with_header", "status": "timeout"})
                
    except websockets.exceptions.ConnectionClosed as e:
        print(f"❌ Connection closed: {e}")
        results.append({"test": "with_header", "status": "closed", "error": str(e)})
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        results.append({"test": "with_header", "status": "failed", "error": str(e)})
    
    # Summary
    print("\n" + "="*50)
    print("🔌 WEBSOCKET TEST RESULTS")
    print("="*50)
    
    success_count = len([r for r in results if r["status"] == "success"])
    total_count = len(results)
    
    print(f"✅ Successful connections: {len([r for r in results if r['test'] in ['no_auth', 'with_token', 'with_header'] and r['status'] == 'success'])}")
    print(f"❌ Failed connections: {len([r for r in results if r['test'] in ['no_auth', 'with_token', 'with_header'] and r['status'] == 'failed'])}")
    print(f"🔒 Closed connections: {len([r for r in results if r['test'] in ['no_auth', 'with_token', 'with_header'] and r['status'] == 'closed'])}")
    
    print(f"\nDetailed Results:")
    for result in results:
        status_emoji = "✅" if result["status"] == "success" else "❌" if result["status"] == "failed" else "🔒" if result["status"] == "closed" else "⏱️"
        print(f"{status_emoji} {result['test']}: {result['status']}")
        if result.get('note'):
            print(f"   Note: {result['note']}")
        if result.get('error'):
            print(f"   Error: {result['error']}")
        if result.get('response'):
            print(f"   Response: {result['response'][:100]}...")
    
    print("\n🔧 WebSocket Endpoint Analysis:")
    if any(r["status"] == "success" for r in results if r["test"] in ["no_auth", "with_token", "with_header"]):
        print("✅ WebSocket endpoint is accessible")
        print("✅ WebSocket server is running")
        print("⚠️  Authentication may be optional or not implemented")
    else:
        print("❌ WebSocket endpoint not accessible")
        print("🔧 May need proper authentication setup")
    
    return results

async def main():
    await test_websocket_connection()

if __name__ == "__main__":
    asyncio.run(main())