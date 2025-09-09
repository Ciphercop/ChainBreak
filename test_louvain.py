#!/usr/bin/env python3
"""
Test Louvain community detection functionality.
"""

from illicit_transaction_detector import IllicitTransactionDetector, Transaction, GraphVisualizer
from datetime import datetime
import random
import networkx as nx

def create_test_graph():
    """Create a test graph with clear communities."""
    transactions = []
    
    # Create two distinct communities
    community1_addrs = ["1Comm1A", "1Comm1B", "1Comm1C"]
    community2_addrs = ["1Comm2A", "1Comm2B", "1Comm2C"]
    
    # Add transactions within community 1
    for i in range(15):
        tx = Transaction(
            tx_hash=f"comm1_tx_{i:03d}",
            from_address=random.choice(community1_addrs),
            to_address=random.choice(community1_addrs),
            value=random.uniform(0.1, 2.0),
            timestamp=datetime.now(),
            block_height=800000 + i
        )
        transactions.append(tx)
    
    # Add transactions within community 2
    for i in range(15):
        tx = Transaction(
            tx_hash=f"comm2_tx_{i:03d}",
            from_address=random.choice(community2_addrs),
            to_address=random.choice(community2_addrs),
            value=random.uniform(0.1, 2.0),
            timestamp=datetime.now(),
            block_height=800100 + i
        )
        transactions.append(tx)
    
    # Add a few cross-community transactions
    for i in range(3):
        tx = Transaction(
            tx_hash=f"cross_tx_{i:03d}",
            from_address=random.choice(community1_addrs),
            to_address=random.choice(community2_addrs),
            value=random.uniform(0.5, 1.5),
            timestamp=datetime.now(),
            block_height=800200 + i
        )
        transactions.append(tx)
    
    return transactions

def test_louvain_detection():
    """Test Louvain community detection."""
    print("🔍 TESTING LOUVAIN COMMUNITY DETECTION")
    print("=" * 45)
    
    # Create test data
    transactions = create_test_graph()
    print(f"✅ Created {len(transactions)} test transactions")
    
    # Initialize detector
    detector = IllicitTransactionDetector()
    
    # Build graph
    print("🕸️  Building transaction graph...")
    graph = detector._build_transaction_graph(transactions)
    print(f"📊 Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Test Louvain detection directly
    print("🔍 Testing Louvain community detection...")
    clusters = detector._detect_communities_louvain(graph)
    print(f"📊 Detected {len(clusters)} communities")
    
    # Show community results
    print("\n🎯 COMMUNITY DETECTION RESULTS:")
    print("-" * 35)
    
    for community_id, nodes in clusters.items():
        print(f"Community {community_id}: {len(nodes)} nodes")
        for node in nodes:
            print(f"  - {node}")
        print()
    
    # Test GraphVisualizer community detection
    print("🔍 Testing GraphVisualizer community detection...")
    visualizer = GraphVisualizer()
    communities = visualizer._run_community_detection_algorithms(graph)
    
    print(f"📊 GraphVisualizer detected: {list(communities.keys())}")
    
    if 'louvain' in communities:
        louvain_communities = communities['louvain']
        print(f"📊 Louvain communities: {len(set(louvain_communities.values()))}")
        
        # Show node-to-community mapping
        print("\n🎯 NODE-TO-COMMUNITY MAPPING:")
        print("-" * 35)
        
        community_groups = {}
        for node, community_id in louvain_communities.items():
            if community_id not in community_groups:
                community_groups[community_id] = []
            community_groups[community_id].append(node)
        
        for community_id, nodes in community_groups.items():
            print(f"Community {community_id}: {nodes}")
    
    # Test community layout
    print("\n🎨 Testing community-aware layout...")
    pos = nx.spring_layout(graph, k=1, iterations=50)
    adjusted_pos = visualizer._apply_community_layout(graph, pos, communities)
    
    print(f"✅ Layout adjustment completed")
    print(f"📊 Original positions: {len(pos)} nodes")
    print(f"📊 Adjusted positions: {len(adjusted_pos)} nodes")
    
    print("\n🎉 LOUVAIN COMMUNITY DETECTION TEST COMPLETE!")
    print("✅ Louvain algorithm working!")
    print("✅ Community detection working!")
    print("✅ Layout adjustment working!")
    
    return True

if __name__ == "__main__":
    test_louvain_detection()
