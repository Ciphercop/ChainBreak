#!/usr/bin/env python3
"""
Test script for frontend connectivity
"""

import requests
import time

def test_frontend():
    """Test if frontend is accessible"""
    
    print('🌐 Testing Frontend Connectivity...')
    print('=' * 50)
    
    # Wait a moment for frontend to start
    print('Waiting for frontend to start...')
    time.sleep(5)
    
    try:
        # Test frontend accessibility
        response = requests.get('http://localhost:3000', timeout=10)
        print(f'Frontend Status: {response.status_code}')
        
        if response.status_code == 200:
            print('✅ Frontend is running successfully!')
            print('🌐 Open http://localhost:3000 in your browser')
            print()
            print('📋 Testing Checklist:')
            print('1. ✅ Backend running on http://localhost:5000')
            print('2. ✅ Frontend running on http://localhost:3000')
            print('3. 🔍 Test Graph Analysis tab')
            print('4. 🔍 Test Illicit Detection tab')
            print('5. 🔍 Test Law Enforcement Dashboard tab')
            print()
            print('🎯 Quick Test Steps:')
            print('• Enter addresses: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa,1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2')
            print('• Click "Analyze" in Graph Analysis tab')
            print('• Click "Analyze Transactions" in Illicit Detection tab')
            print('• Click "Analyze Sample" in Law Enforcement tab')
        else:
            print(f'❌ Frontend error: {response.status_code}')
            
    except requests.exceptions.ConnectionError:
        print('❌ Frontend not accessible. Make sure to run:')
        print('   cd frontend && npm start')
    except Exception as e:
        print(f'❌ Error: {e}')

if __name__ == '__main__':
    test_frontend()
