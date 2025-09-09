#!/usr/bin/env python3
"""
Test the Louvain route directly.
"""

from src.api import app
import json

def test_louvain_route():
    """Test the Louvain route directly."""
    print("🔍 TESTING LOUVAIN ROUTE DIRECTLY")
    print("=" * 40)
    
    # Test data
    test_data = {
        "nodes": [
            {"id": "node1", "label": "Node 1"},
            {"id": "node2", "label": "Node 2"},
            {"id": "node3", "label": "Node 3"},
            {"id": "node4", "label": "Node 4"}
        ],
        "edges": [
            {"source": "node1", "target": "node2"},
            {"source": "node2", "target": "node3"},
            {"source": "node3", "target": "node1"},
            {"source": "node4", "target": "node1"}
        ]
    }
    
    with app.test_client() as client:
        # Test POST request
        response = client.post('/api/run-louvain', 
                             json=test_data,
                             content_type='application/json')
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.get_json()}")
        
        if response.status_code == 200:
            result = response.get_json()
            print("✅ Route working!")
            print(f"Communities: {result['community_count']}")
            print(f"Modularity: {result['modularity']:.3f}")
        else:
            print("❌ Route failed!")

if __name__ == "__main__":
    test_louvain_route()
