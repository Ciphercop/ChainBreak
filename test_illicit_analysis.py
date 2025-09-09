#!/usr/bin/env python3
"""
Test script for illicit analysis button functionality.
This demonstrates how to use the comprehensive illicit analysis.
"""

from illicit_transaction_detector import run_illicit_analysis_button, Transaction
from datetime import datetime
import random

def create_test_transactions():
    """Create sample transaction data for testing."""
    transactions = []
    
    # Create some suspicious patterns
    suspicious_addresses = [
        "1SuspiciousAddr1",  # High illicit activity
        "1SuspiciousAddr2",  # Medium illicit activity
        "1SuspiciousAddr3",  # Low illicit activity
    ]
    
    clean_addresses = [
        "1CleanAddr1",
        "1CleanAddr2", 
        "1CleanAddr3",
        "1CleanAddr4",
    ]
    
    # Create transactions with suspicious patterns
    for i in range(50):
        if i < 20:  # Suspicious transactions
            from_addr = random.choice(suspicious_addresses)
            to_addr = random.choice(clean_addresses)
            value = random.uniform(0.001, 0.1)  # Small amounts (smurfing pattern)
        else:  # Normal transactions
            from_addr = random.choice(clean_addresses)
            to_addr = random.choice(clean_addresses)
            value = random.uniform(0.1, 10.0)  # Normal amounts
            
        tx = Transaction(
            tx_hash=f"test_tx_{i:03d}",
            from_address=from_addr,
            to_address=to_addr,
            value=value,
            timestamp=datetime.now(),
            block_height=800000 + i
        )
        transactions.append(tx)
    
    # Add some mixing patterns (rapid transfers)
    for i in range(10):
        tx = Transaction(
            tx_hash=f"mixing_tx_{i:03d}",
            from_address="1SuspiciousAddr1",  # Known suspicious
            to_address=f"1MixAddr{i}",
            value=random.uniform(0.01, 0.05),
            timestamp=datetime.now(),
            block_height=800100 + i
        )
        transactions.append(tx)
    
    return transactions

def main():
    """Main function to run the illicit analysis test."""
    print("🚀 TESTING ILLICIT ANALYSIS BUTTON")
    print("=" * 50)
    
    # Create test data
    print("📊 Creating test transaction data...")
    transactions = create_test_transactions()
    print(f"✅ Created {len(transactions)} test transactions")
    
    # Run illicit analysis
    print("\n🔍 Running comprehensive illicit analysis...")
    results = run_illicit_analysis_button(transactions, "test_illicit_results")
    
    print("\n🎉 ANALYSIS COMPLETE!")
    print(f"📁 Check the 'test_illicit_results' folder for:")
    print(f"   • illicit_analysis_graph.html - Interactive visualization")
    print(f"   • ILLICIT_DETECTION_REPORT.txt - Detailed report")
    print(f"   • illicit_cluster_analysis.html - Cluster analysis")
    print(f"   • illicit_risk_heatmap.html - Risk heatmap")
    
    return results

if __name__ == "__main__":
    main()
