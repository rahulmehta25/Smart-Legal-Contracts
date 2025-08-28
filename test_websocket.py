#!/usr/bin/env python3
"""
Test script for WebSocket real-time functionality
"""

import asyncio
import json
import websockets
from datetime import datetime
import uuid


async def test_websocket_connection():
    """Test basic WebSocket connection"""
    uri = "ws://localhost:8000/ws"
    token = "demo_token"  # Use a demo token for testing
    
    try:
        # Connect with token in query parameter
        async with websockets.connect(f"{uri}?token={token}") as websocket:
            print(f"✅ Connected to WebSocket server at {uri}")
            
            # Send a test event
            test_event = {
                "event_type": "heartbeat",
                "event_id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "data": {"ping": True}
            }
            
            await websocket.send(json.dumps(test_event))
            print("📤 Sent heartbeat event")
            
            # Wait for response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                response_data = json.loads(response)
                print(f"📨 Received response: {response_data}")
                
                if response_data.get("event_type") == "heartbeat":
                    print("✅ Heartbeat response received successfully")
                else:
                    print(f"ℹ️ Received event: {response_data.get('event_type')}")
                    
            except asyncio.TimeoutError:
                print("⚠️ No response received within timeout")
            
            # Test analysis progress simulation
            analysis_id = f"test_analysis_{uuid.uuid4().hex[:8]}"
            
            analysis_start_event = {
                "event_type": "analysis_start",
                "event_id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "analysis_id": analysis_id,
                    "document_name": "test_document.txt"
                }
            }
            
            await websocket.send(json.dumps(analysis_start_event))
            print(f"📤 Started test analysis: {analysis_id}")
            
            # Simulate progress updates
            for progress in [25, 50, 75, 100]:
                await asyncio.sleep(1)  # Wait 1 second between updates
                
                progress_event = {
                    "event_type": "analysis_progress",
                    "event_id": str(uuid.uuid4()),
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {
                        "analysis_id": analysis_id,
                        "progress": progress,
                        "stage": f"step_{progress}",
                        "details": {"current_step": f"Processing step {progress}%"}
                    }
                }
                
                await websocket.send(json.dumps(progress_event))
                print(f"📤 Sent progress update: {progress}%")
            
            # Send completion event
            completion_event = {
                "event_type": "analysis_complete",
                "event_id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "analysis_id": analysis_id,
                    "result": {
                        "clauses_found": 3,
                        "confidence": 85.5
                    },
                    "completed_at": datetime.utcnow().isoformat(),
                    "duration": 4,
                    "success": True
                }
            }
            
            await websocket.send(json.dumps(completion_event))
            print(f"📤 Analysis completed: {analysis_id}")
            
            # Test notification
            notification_event = {
                "event_type": "notification_new",
                "event_id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "title": "Test Notification",
                    "message": "This is a test notification from the WebSocket test script",
                    "type": "info",
                    "created_at": datetime.utcnow().isoformat()
                }
            }
            
            await websocket.send(json.dumps(notification_event))
            print("📤 Sent test notification")
            
            print("✅ All test events sent successfully")
            print("🔄 Listening for incoming events for 10 seconds...")
            
            # Listen for incoming events
            end_time = asyncio.get_event_loop().time() + 10
            while asyncio.get_event_loop().time() < end_time:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    response_data = json.loads(response)
                    event_type = response_data.get("event_type", "unknown")
                    print(f"📨 Received {event_type} event: {response_data}")
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    print(f"❌ Error receiving message: {e}")
                    break
            
            print("✅ WebSocket test completed successfully")
            
    except ConnectionRefusedError:
        print("❌ Connection refused. Make sure the WebSocket server is running on localhost:8000")
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")


async def test_socketio_connection():
    """Test Socket.IO connection"""
    import socketio
    
    print("\n🔌 Testing Socket.IO connection...")
    
    sio = socketio.AsyncClient()
    
    @sio.event
    async def connect():
        print("✅ Connected to Socket.IO server")
        
        # Join a test room
        await sio.emit('join_room', {
            'room_id': 'test_room_123',
            'room_type': 'document',
            'auto_create': True
        })
        
        # Send a test websocket event
        test_event = {
            "event_type": "heartbeat",
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "data": {"ping": True, "message": "Socket.IO test"}
        }
        
        await sio.emit('websocket_event', test_event)
        print("📤 Sent test event via Socket.IO")
    
    @sio.event
    async def disconnect():
        print("🔌 Disconnected from Socket.IO server")
    
    @sio.event
    async def websocket_event(data):
        print(f"📨 Received Socket.IO event: {data}")
    
    @sio.event
    async def user_joined(data):
        print(f"👤 User joined: {data}")
    
    @sio.event
    async def user_left(data):
        print(f"👤 User left: {data}")
    
    try:
        await sio.connect(
            'http://localhost:8000',
            socketio_path='/socket.io/',
            auth={'token': 'demo_token'}
        )
        
        # Wait for events
        await asyncio.sleep(5)
        
        await sio.disconnect()
        print("✅ Socket.IO test completed")
        
    except Exception as e:
        print(f"❌ Socket.IO test failed: {e}")


async def main():
    """Run all WebSocket tests"""
    print("🚀 Starting WebSocket functionality tests...")
    print("=" * 50)
    
    print("\n1️⃣ Testing basic WebSocket connection...")
    await test_websocket_connection()
    
    print("\n2️⃣ Testing Socket.IO connection...")
    try:
        await test_socketio_connection()
    except ImportError:
        print("⚠️ Socket.IO client not available, skipping Socket.IO test")
        print("   Install with: pip install python-socketio")
    
    print("\n" + "=" * 50)
    print("🎉 WebSocket tests completed!")
    print("\nNext steps:")
    print("1. Start the backend server: python -m app.main")
    print("2. Start the frontend: npm run dev")
    print("3. Open the application and test real-time features")


if __name__ == "__main__":
    asyncio.run(main())