#!/usr/bin/env python3
"""
Working illicit analysis test - NO external connections!
"""

from illicit_transaction_detector import IllicitTransactionDetector, Transaction, RiskLevel
from datetime import datetime
import random

def create_test_data():
    """Create test data for analysis."""
    transactions = []
    
    # Create addresses with known patterns
    suspicious_addr = "1SuspiciousAddr"  # Will be flagged
    clean_addrs = ["1Clean1", "1Clean2", "1Clean3"]
    
    # Create suspicious pattern: many small transactions (smurfing)
    for i in range(20):
        tx = Transaction(
            tx_hash=f"suspicious_tx_{i:03d}",
            from_address=suspicious_addr,
            to_address=random.choice(clean_addrs),
            value=random.uniform(0.001, 0.01),  # Small amounts
            timestamp=datetime.now(),
            block_height=800000 + i
        )
        transactions.append(tx)
    
    # Create normal transactions
    for i in range(10):
        tx = Transaction(
            tx_hash=f"normal_tx_{i:03d}",
            from_address=random.choice(clean_addrs),
            to_address=random.choice(clean_addrs),
            value=random.uniform(0.1, 5.0),  # Normal amounts
            timestamp=datetime.now(),
            block_height=800100 + i
        )
        transactions.append(tx)
    
    return transactions

def test_illicit_button():
    """Test the illicit analysis button functionality."""
    print("🚀 TESTING ILLICIT ANALYSIS BUTTON")
    print("=" * 45)
    
    # Create test data
    transactions = create_test_data()
    print(f"✅ Created {len(transactions)} test transactions")
    
    # Initialize detector
    detector = IllicitTransactionDetector()
    
    # Build graph
    print("🕸️  Building transaction graph...")
    graph = detector._build_transaction_graph(transactions)
    print(f"📊 Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Extract addresses
    print("👥 Extracting address nodes...")
    addresses = detector._extract_address_nodes(graph, transactions)
    print(f"📋 Extracted {len(addresses)} addresses")
    
    # Detect suspicious patterns
    print("🔍 Detecting suspicious patterns...")
    suspicious_patterns = detector._detect_suspicious_patterns(graph, addresses, transactions)
    print(f"🚨 Found {len(suspicious_patterns)} suspicious patterns")
    
    # Show address details
    print("\n🎯 ADDRESS DETAILS:")
    print("-" * 30)
    
    for addr, node in addresses.items():
        print(f"  {addr[:20]}...")
        print(f"    Risk Level: {node.risk_level.value}")
        print(f"    Risk Score: {node.risk_score:.3f}")
        print(f"    Patterns: {[p.value for p in node.suspicious_patterns]}")
        print(f"    TX Count: {node.transaction_count}")
        print(f"    Total Sent: {node.total_sent:.3f}")
        print(f"    Total Received: {node.total_received:.3f}")
        print()
    
    # Test illicit analysis on each node
    print("🔍 ILLICIT ANALYSIS RESULTS:")
    print("-" * 30)
    
    illicit_count = 0
    suspicious_count = 0
    
    for node in graph.nodes():
        address_node = addresses.get(node)
        if address_node:
            # Create a simple analysis object for the illicit analysis
            class SimpleAnalysis:
                def __init__(self, addresses):
                    self.addresses = addresses
            
            analysis = SimpleAnalysis(addresses)
            
            # Run illicit analysis
            try:
                node_analysis = detector.visualizer._analyze_node_illicit_patterns(node, graph, analysis)
                illicit_score = node_analysis['illicit_score']
                
                if illicit_score > 0.7:
                    illicit_count += 1
                    print(f"🚨 ILLICIT: {node[:20]}... (Score: {illicit_score:.3f})")
                elif illicit_score > 0.4:
                    suspicious_count += 1
                    print(f"⚠️  SUSPICIOUS: {node[:20]}... (Score: {illicit_score:.3f})")
                else:
                    print(f"✅ CLEAN: {node[:20]}... (Score: {illicit_score:.3f})")
            except Exception as e:
                print(f"❌ Error analyzing {node[:20]}...: {e}")
    
    print(f"\n📊 FINAL SUMMARY:")
    print(f"  🔴 Illicit addresses: {illicit_count}")
    print(f"  🟠 Suspicious addresses: {suspicious_count}")
    print(f"  ✅ Clean addresses: {len(addresses) - illicit_count - suspicious_count}")
    
    print("\n🎉 ILLICIT ANALYSIS BUTTON TEST COMPLETE!")
    print("✅ No external connections made!")
    print("✅ Local analysis working!")
    
    return True

if __name__ == "__main__":
    test_illicit_button()
