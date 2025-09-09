#!/usr/bin/env python3
"""
Test script for illicit detection system
"""

import requests
import json

def test_illicit_detection():
    """Test the illicit detection system with comprehensive data"""
    
    # Test with multiple addresses including known suspicious ones
    test_addresses = [
        '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa',  # Genesis block
        '1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2',  # Test address
        '1FfmbHfnpaZjKFvyi1okTjJJusN455paPH',  # Known exchange
        '3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy',  # Known mixer
        'bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh'  # Bech32 address
    ]

    print('🔍 Testing Illicit Detection System...')
    print('=' * 50)

    response = requests.post('http://localhost:5000/api/illicit-analysis', 
        json={
            'addresses': test_addresses,
            'max_transactions': 50,
            'include_visualization': True
        })

    if response.status_code == 200:
        data = response.json()
        analysis = data['analysis']
        
        print('✅ Analysis completed successfully!')
        print(f'📊 Total addresses analyzed: {analysis["total_addresses"]}')
        print(f'📊 Total transactions processed: {analysis["total_transactions"]}')
        print(f'🚨 High-risk addresses found: {len(analysis["high_risk_addresses"])}')
        print(f'🔍 Suspicious patterns detected: {len(analysis["suspicious_patterns"])}')
        clusters_count = len(analysis.get("clusters", {}))
        print(f'🏘️ Clusters identified: {clusters_count}')
        
        print('\n🚨 HIGH-RISK ADDRESSES:')
        for addr in analysis.get('high_risk_addresses', []):
            print(f'  • {addr}')
        
        print('\n🔍 SUSPICIOUS PATTERNS:')
        for pattern in analysis.get('suspicious_patterns', []):
            print(f'  • {pattern["pattern_type"]}: {pattern["description"]} (confidence: {pattern["confidence"]:.2f})')
        
        print('\n📈 RISK DISTRIBUTION:')
        for risk_level, count in analysis.get('risk_distribution', {}).items():
            print(f'  • {risk_level}: {count} addresses')
        
        if 'clusters' in analysis:
            print('\n🏘️ CLUSTER ANALYSIS:')
            for cluster_id, cluster_data in analysis['clusters'].items():
                print(f'  • Cluster {cluster_id}: {cluster_data["size"]} addresses, modularity: {cluster_data["modularity"]:.3f}')
        
        if 'detection_summary' in analysis:
            print('\n📊 DETECTION SUMMARY:')
            summary = analysis['detection_summary']
            print(f'  • Total patterns detected: {summary.get("total_patterns", 0)}')
            print(f'  • High confidence patterns: {summary.get("high_confidence_patterns", 0)}')
            print(f'  • Medium confidence patterns: {summary.get("medium_confidence_patterns", 0)}')
            print(f'  • Low confidence patterns: {summary.get("low_confidence_patterns", 0)}')
        
        # Test threat intelligence
        print('\n🔍 THREAT INTELLIGENCE TEST:')
        test_addr = test_addresses[0]
        ti_response = requests.get(f'http://localhost:5000/api/illicit-analysis/threat-intel/{test_addr}')
        if ti_response.status_code == 200:
            ti_data = ti_response.json()
            print(f'  • BitcoinWhosWho score: {ti_data["threat_intelligence"]["bitcoinwhoswho"]["score"] if ti_data["threat_intelligence"]["bitcoinwhoswho"] else "N/A"}')
            print(f'  • Scam reports: {ti_data["threat_intelligence"]["bitcoinwhoswho"]["scam_reports"] if ti_data["threat_intelligence"]["bitcoinwhoswho"] else "N/A"}')
        
        return data
        
    else:
        print(f'❌ Error: {response.status_code}')
        print(response.text)
        return None

if __name__ == '__main__':
    test_illicit_detection()
