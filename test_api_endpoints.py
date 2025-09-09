#!/usr/bin/env python3
"""
Test script for API endpoints
"""

import requests
import json

def test_api_endpoints():
    """Test all illicit analysis API endpoints"""
    
    print('🔍 Testing Individual API Endpoints...')
    print('=' * 50)

    # Test 1: System Status
    print('1. Testing System Status...')
    try:
        response = requests.get('http://localhost:5000/api/status')
        print(f'   Status: {response.status_code}')
        if response.status_code == 200:
            data = response.json()
            if data.get("success") and "data" in data:
                status_data = data["data"]
                print(f'   Backend: {status_data.get("backend_mode", "unknown")}')
                print(f'   System: {status_data.get("system_status", "unknown")}')
                print(f'   Connection: {status_data.get("backend_connection", "unknown")}')
            else:
                print(f'   Response: {data}')
        else:
            print(f'   Error: {response.text}')
    except Exception as e:
        print(f'   Error: {e}')

    print()

    # Test 2: Pattern Types
    print('2. Testing Pattern Types...')
    try:
        response = requests.get('http://localhost:5000/api/illicit-analysis/pattern-types')
        print(f'   Status: {response.status_code}')
        if response.status_code == 200:
            data = response.json()
            print(f'   Available patterns: {len(data.get("pattern_types", []))}')
            for pattern in data.get('pattern_types', []):
                print(f'     • {pattern}')
        else:
            print(f'   Error: {response.text}')
    except Exception as e:
        print(f'   Error: {e}')

    print()

    # Test 3: Risk Levels
    print('3. Testing Risk Levels...')
    try:
        response = requests.get('http://localhost:5000/api/illicit-analysis/risk-levels')
        print(f'   Status: {response.status_code}')
        if response.status_code == 200:
            data = response.json()
            print(f'   Risk levels: {data.get("risk_levels", [])}')
        else:
            print(f'   Error: {response.text}')
    except Exception as e:
        print(f'   Error: {e}')

    print()

    # Test 4: Threat Intelligence
    print('4. Testing Threat Intelligence...')
    test_address = '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa'
    try:
        response = requests.get(f'http://localhost:5000/api/illicit-analysis/threat-intel/{test_address}')
        print(f'   Status: {response.status_code}')
        if response.status_code == 200:
            data = response.json()
            ti = data.get('threat_intelligence', {})
            bww = ti.get('bitcoinwhoswho', {})
            if bww:
                print(f'   BitcoinWhosWho Score: {bww.get("score", "N/A")}')
                print(f'   Scam Reports: {bww.get("scam_reports", "N/A")}')
            else:
                print('   No BitcoinWhosWho data available')
        else:
            print(f'   Error: {response.text}')
    except Exception as e:
        print(f'   Error: {e}')

    print()

    # Test 5: Run Detection on Existing Graph
    print('5. Testing Run Detection...')
    try:
        sample_graph_data = {
            "nodes": [
                {"id": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa", "type": "address"},
                {"id": "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2", "type": "address"}
            ],
            "edges": [
                {"source": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa", "target": "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2", "value": 1.0}
            ]
        }
        
        response = requests.post('http://localhost:5000/api/illicit-analysis/run-detection', 
                               json=sample_graph_data)
        print(f'   Status: {response.status_code}')
        if response.status_code == 200:
            data = response.json()
            print(f'   Detection completed: {data.get("success", False)}')
            analysis = data.get('analysis', {})
            print(f'   Patterns detected: {len(analysis.get("suspicious_patterns", []))}')
        else:
            print(f'   Error: {response.text}')
    except Exception as e:
        print(f'   Error: {e}')

    print()
    print('✅ API Endpoint Testing Complete!')

if __name__ == '__main__':
    test_api_endpoints()
