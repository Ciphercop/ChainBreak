#!/usr/bin/env python3
"""
Simple test to verify server can start
"""

from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    return jsonify({'success': True, 'message': 'Server is working'})

if __name__ == '__main__':
    print('Starting simple test server...')
    app.run(host='0.0.0.0', port=5000, debug=True)
