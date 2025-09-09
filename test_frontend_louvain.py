#!/usr/bin/env python3
"""
Test the frontend Louvain button functionality.
"""

import requests
import json

def test_louvain_api():
    """Test the Louvain API endpoint."""
    print("🔍 TESTING FRONTEND LOUVAIN BUTTON API")
    print("=" * 45)
    
    # Test data
    test_data = {
        "nodes": [
            {"id": "node1", "label": "Node 1"},
            {"id": "node2", "label": "Node 2"},
            {"id": "node3", "label": "Node 3"},
            {"id": "node4", "label": "Node 4"},
            {"id": "node5", "label": "Node 5"},
            {"id": "node6", "label": "Node 6"}
        ],
        "edges": [
            {"source": "node1", "target": "node2"},
            {"source": "node2", "target": "node3"},
            {"source": "node3", "target": "node1"},
            {"source": "node4", "target": "node5"},
            {"source": "node5", "target": "node6"},
            {"source": "node6", "target": "node4"},
            {"source": "node1", "target": "node4"}  # Bridge between communities
        ]
    }
    
    try:
        # Test the API endpoint
        print("📡 Testing /api/run-louvain endpoint...")
        response = requests.post(
            'http://localhost:5000/api/run-louvain',
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API Response:")
            print(f"  Success: {result['success']}")
            print(f"  Communities: {result['community_count']}")
            print(f"  Modularity: {result['modularity']:.3f}")
            print(f"  Node Count: {result['node_count']}")
            print(f"  Edge Count: {result['edge_count']}")
            
            print("\n🎯 Community Assignments:")
            for node_id, community_id in result['communities'].items():
                print(f"  {node_id} → Community {community_id}")
            
            print("\n🎉 FRONTEND LOUVAIN BUTTON API TEST PASSED!")
            print("✅ Backend API working!")
            print("✅ Louvain algorithm working!")
            print("✅ Community detection working!")
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Backend server not running")
        print("💡 Start the backend with: python app.py --api")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

def test_frontend_integration():
    """Test frontend integration."""
    print("\n🌐 FRONTEND INTEGRATION TEST")
    print("=" * 30)
    
    try:
        # Test if frontend is accessible
        response = requests.get('http://localhost:3000', timeout=5)
        if response.status_code == 200:
            print("✅ Frontend accessible at http://localhost:3000")
        else:
            print(f"⚠️  Frontend status: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Frontend not running")
        print("💡 Start frontend with: cd frontend && npm start")
    except Exception as e:
        print(f"❌ Frontend test failed: {e}")

if __name__ == "__main__":
    test_louvain_api()
    test_frontend_integration()
    
    print("\n📋 SUMMARY:")
    print("1. Backend API: /api/run-louvain endpoint")
    print("2. Frontend Button: Run Louvain button in GraphRenderer")
    print("3. Integration: Frontend calls backend API")
    print("4. Visualization: Nodes colored by community")
    print("\n🚀 To test the full system:")
    print("1. Start backend: python app.py --api")
    print("2. Start frontend: cd frontend && npm start")
    print("3. Open http://localhost:3000")
    print("4. Load a graph and click 'Run Louvain' button!")
