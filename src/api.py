"""
Flask API Layer for ChainBreak
Provides RESTful interface for system interaction
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import logging
import json
from typing import Dict, Any, List
import traceback
from .chainbreak import ChainBreak
from .api_frontend import bp as frontend_bp

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
chainbreak_instance = None
neo4j_available = False

app.register_blueprint(frontend_bp)


def get_chainbreak():
    global chainbreak_instance, neo4j_available
    if chainbreak_instance is None:
        try:
            chainbreak_instance = ChainBreak()
            neo4j_available = True
            logger.info("ChainBreak instance created for API")
        except Exception as e:
            logger.error(f"Error creating ChainBreak instance: {str(e)}")
            neo4j_available = False
            # Don't raise the exception, let the API handle it gracefully
            return None
    return chainbreak_instance


def check_neo4j_status():
    """Check if Neo4j is available"""
    global neo4j_available
    try:
        chainbreak = get_chainbreak()
        if chainbreak is None:
            return False
        # Try a simple query to test connection
        with chainbreak.data_ingestor.driver.session() as session:
            session.run("RETURN 1")
        neo4j_available = True
        return True
    except Exception as e:
        logger.warning(f"Neo4j connection check failed: {str(e)}")
        neo4j_available = False
        return False


def get_mock_data():
    """Return mock data when Neo4j is unavailable"""
    return {
        'status': 'offline',
        'message': 'Neo4j database is currently unavailable. Some features may be limited.',
        'mock_mode': True
    }


@app.route('/')
def index():
    html_template = """
    <!DOCTYPE html><html><head><meta charset='utf-8'/><title>ChainBreak API</title></head>
    <body>
      <h1>ChainBreak API</h1>
      <p><a href="/frontend/index.html">Open Frontend</a></p>
      <p><a href="/api/status">Check System Status</a></p>
    </body></html>
    """
    return html_template


@app.route('/api/status', methods=['GET'])
def get_status():
    try:
        neo4j_status = check_neo4j_status()
        if not neo4j_status:
            return jsonify({
                'success': True, 
                'data': get_mock_data(),
                'timestamp': get_current_timestamp(),
                'message': 'System status retrieved (Neo4j offline)'
            })
        
        chainbreak = get_chainbreak()
        if chainbreak is None:
            return jsonify({
                'success': True,
                'data': get_mock_data(),
                'timestamp': get_current_timestamp(),
                'message': 'System status retrieved (ChainBreak unavailable)'
            })
            
        status = chainbreak.get_system_status()
        return jsonify({
            'success': True, 
            'data': status, 
            'timestamp': status.get('timestamp'), 
            'message': 'System status retrieved successfully'
        })
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        return jsonify({
            'success': True,
            'data': get_mock_data(),
            'error': str(e), 
            'timestamp': get_current_timestamp()
        })


@app.route('/api/analyze', methods=['POST'])
def analyze_address():
    try:
        if not check_neo4j_status():
            return jsonify({
                'success': False, 
                'error': 'Neo4j database is currently unavailable. Please try again later.',
                'timestamp': get_current_timestamp()
            }), 503
            
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data provided', 'timestamp': get_current_timestamp()}), 400
        address = data.get('address')
        blockchain = data.get('blockchain', 'btc')
        generate_visualizations = data.get('generate_visualizations', True)
        if not address:
            return jsonify({'success': False, 'error': 'Address parameter is required', 'timestamp': get_current_timestamp()}), 400
        logger.info(f"API: Analyzing address {address}")
        chainbreak = get_chainbreak()
        if chainbreak is None:
            return jsonify({
                'success': False, 
                'error': 'System is currently unavailable. Please try again later.',
                'timestamp': get_current_timestamp()
            }), 503
            
        results = chainbreak.analyze_address(
            address, blockchain, generate_visualizations)
        if 'error' in results:
            return jsonify({'success': False, 'error': results['error'], 'timestamp': get_current_timestamp()}), 400
        return jsonify({'success': True, 'data': results, 'timestamp': results.get('analysis_timestamp'), 'message': f'Analysis completed for address {address}'})
    except Exception as e:
        logger.error(f"Error in analyze_address API: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': f'Analysis failed: {str(e)}', 'timestamp': get_current_timestamp()}), 500


@app.route('/api/analyze/batch', methods=['POST'])
def analyze_multiple_addresses():
    try:
        if not check_neo4j_status():
            return jsonify({
                'success': False, 
                'error': 'Neo4j database is currently unavailable. Please try again later.',
                'timestamp': get_current_timestamp()
            }), 503
            
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data provided', 'timestamp': get_current_timestamp()}), 400
        addresses = data.get('addresses', [])
        blockchain = data.get('blockchain', 'btc')
        if not addresses or not isinstance(addresses, list):
            return jsonify({'success': False, 'error': 'Addresses parameter must be a non-empty list', 'timestamp': get_current_timestamp()}), 400
        if len(addresses) > 100:
            return jsonify({'success': False, 'error': 'Maximum 100 addresses allowed per batch', 'timestamp': get_current_timestamp()}), 400
        logger.info(f"API: Analyzing {len(addresses)} addresses")
        chainbreak = get_chainbreak()
        if chainbreak is None:
            return jsonify({
                'success': False, 
                'error': 'System is currently unavailable. Please try again later.',
                'timestamp': get_current_timestamp()
            }), 503
            
        results = chainbreak.analyze_multiple_addresses(addresses, blockchain)
        if 'error' in results:
            return jsonify({'success': False, 'error': results['error'], 'timestamp': get_current_timestamp()}), 400
        return jsonify({'success': True, 'data': results, 'timestamp': results.get('analysis_timestamp'), 'message': f'Batch analysis completed for {len(addresses)} addresses'})
    except Exception as e:
        logger.error(f"Error in analyze_multiple_addresses API: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': f'Batch analysis failed: {str(e)}', 'timestamp': get_current_timestamp()}), 500


@app.route('/api/export/gephi', methods=['GET'])
def export_to_gephi():
    try:
        if not check_neo4j_status():
            return jsonify({
                'success': False, 
                'error': 'Neo4j database is currently unavailable. Please try again later.',
                'timestamp': get_current_timestamp()
            }), 503
            
        address = request.args.get('address')
        output_file = request.args.get('output_file')
        logger.info(f"API: Exporting to Gephi (address: {address})")
        chainbreak = get_chainbreak()
        if chainbreak is None:
            return jsonify({
                'success': False, 
                'error': 'System is currently unavailable. Please try again later.',
                'timestamp': get_current_timestamp()
            }), 503
            
        export_file = chainbreak.export_network_to_gephi(address, output_file)
        if not export_file:
            return jsonify({'success': False, 'error': 'Export failed - no data available', 'timestamp': get_current_timestamp()}), 400
        return jsonify({'success': True, 'data': {'export_file': export_file, 'address': address}, 'timestamp': get_current_timestamp(), 'message': f'Network exported to {export_file}'})
    except Exception as e:
        logger.error(f"Error in export_to_gephi API: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': f'Export failed: {str(e)}', 'timestamp': get_current_timestamp()}), 500


@app.route('/api/report/risk', methods=['POST'])
def generate_risk_report():
    try:
        if not check_neo4j_status():
            return jsonify({
                'success': False, 
                'error': 'Neo4j database is currently unavailable. Please try again later.',
                'timestamp': get_current_timestamp()
            }), 503
            
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data provided', 'timestamp': get_current_timestamp()}), 400
        addresses = data.get('addresses', [])
        output_file = data.get('output_file')
        if not addresses or not isinstance(addresses, list):
            return jsonify({'success': False, 'error': 'Addresses parameter must be a non-empty list', 'timestamp': get_current_timestamp()}), 400
        logger.info(
            f"API: Generating risk report for {len(addresses)} addresses")
        chainbreak = get_chainbreak()
        if chainbreak is None:
            return jsonify({
                'success': False, 
                'error': 'System is currently unavailable. Please try again later.',
                'timestamp': get_current_timestamp()
            }), 503
            
        report_content = chainbreak.generate_risk_report(
            addresses, output_file)
        return jsonify({'success': True, 'data': {'report_content': report_content, 'addresses_count': len(addresses), 'output_file': output_file}, 'timestamp': get_current_timestamp(), 'message': f'Risk report generated for {len(addresses)} addresses'})
    except Exception as e:
        logger.error(f"Error in generate_risk_report API: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': f'Report generation failed: {str(e)}', 'timestamp': get_current_timestamp()}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found', 'timestamp': get_current_timestamp()}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error', 'timestamp': get_current_timestamp()}), 500


def get_current_timestamp():
    from datetime import datetime
    return datetime.now().isoformat()


def create_app(config=None):
    if config:
        app.config.update(config)
    logging.basicConfig(level=logging.INFO)
    return app


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
