"""
Flask API Layer for ChainBreak
Provides RESTful interface for system interaction
"""

from flask import Flask, jsonify, request, render_template_string, send_from_directory
from flask_cors import CORS
import logging
import traceback
from datetime import datetime
from .chainbreak import ChainBreak
from .api_frontend import bp as frontend_bp
from pathlib import Path
import json

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.register_blueprint(frontend_bp)

GRAPH_DIR = Path("data/graphs")
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


def get_chainbreak():
    """Get or create ChainBreak instance"""
    if not hasattr(app, 'chainbreak_instance'):
        try:
            app.chainbreak_instance = ChainBreak()
        except Exception as e:
            logger.error(f"Failed to initialize ChainBreak: {e}")
            app.chainbreak_instance = None
    return app.chainbreak_instance


@app.route("/")
def index_new():
    return send_from_directory("frontend", "index.html")


@app.route('/api/')
def api_index():
    """Main API documentation page"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ChainBreak API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { background: #007bff; color: white; padding: 5px 10px; border-radius: 3px; display: inline-block; margin-right: 10px; }
            .url { font-family: monospace; font-weight: bold; }
            .description { margin-top: 10px; }
        </style>
    </head>
    <body>
        <h1>🔗 ChainBreak Blockchain Forensic Analysis API</h1>
        <p>Comprehensive blockchain analysis and anomaly detection API</p>
        
        <h2>📊 System Status</h2>
        <div class="endpoint">
            <span class="method">GET</span>
            <span class="url">/api/status</span>
            <div class="description">Get system status and health information</div>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span>
            <span class="url">/api/mode</span>
            <div class="description">Get current backend mode (Neo4j or JSON)</div>
        </div>
        
        <h2>🔍 Analysis Endpoints</h2>
        <div class="endpoint">
            <span class="method">POST</span>
            <span class="url">/api/analyze</span>
            <div class="description">Analyze a single Bitcoin address</div>
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span>
            <span class="url">/api/analyze/batch</span>
            <div class="description">Analyze multiple Bitcoin addresses</div>
        </div>
        
        <h2>📤 Export Endpoints</h2>
        <div class="endpoint">
            <span class="method">GET</span>
            <span class="url">/api/export/gephi</span>
            <div class="description">Export network to Gephi format</div>
        </div>
        
        <h2>📋 Reporting Endpoints</h2>
        <div class="endpoint">
            <span class="method">POST</span>
            <span class="url">/api/report/risk</span>
            <div class="description">Generate comprehensive risk report</div>
        </div>
        
        <h2>📊 Statistics Endpoints</h2>
        <div class="endpoint">
            <span class="method">GET</span>
            <span class="url">/api/addresses</span>
            <div class="description">Get list of analyzed addresses</div>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span>
            <span class="url">/api/statistics</span>
            <div class="description">Get system statistics and metrics</div>
        </div>
        
        <h2>🌐 Frontend</h2>
        <div class="endpoint">
            <span class="method">GET</span>
            <span class="url">/frontend/index.html</span>
            <div class="description">Interactive graph visualization interface</div>
        </div>
        
        <h2>📖 Usage Examples</h2>
        <h3>Analyze Address</h3>
        <pre>
curl -X POST http://localhost:5000/api/analyze \\
  -H "Content-Type: application/json" \\
  -d '{"address": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"}'
        </pre>
        
        <h3>Check Status</h3>
        <pre>curl http://localhost:5000/api/status</pre>
        
        <h3>Check Backend Mode</h3>
        <pre>curl http://localhost:5000/api/mode</pre>
        
        <p><strong>Note:</strong> The system automatically falls back to JSON mode if Neo4j is unavailable.</p>
    </body>
    </html>
    """
    return html_template


@app.route("/api/graph/list", methods=["GET"])
def list_graphs():
    try:
        files = [f.name for f in GRAPH_DIR.glob("*.json")]
        return jsonify({"success": True, "files": files})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/graph/address", methods=["POST"])
def fetch_graph_address():
    try:
        data = request.get_json()
        address = data.get("address")
        tx_limit = data.get("tx_limit", 50)
        if not address:
            return jsonify({"success": False, "error": "Address required"}), 400
        chainbreak = get_chainbreak()
        if not chainbreak:
            return jsonify({"success": False, "error": "ChainBreak not initialized"}), 500
        graph_json = chainbreak.export_graph_json(address, tx_limit)
        file_path = GRAPH_DIR / f"{address}.json"
        with open(file_path, "w") as f:
            json.dump(graph_json, f)
        return jsonify({"success": True, "file": file_path.name})
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/graph/get", methods=["GET"])
def get_graph():
    try:
        name = request.args.get("name")
        file_path = GRAPH_DIR / name
        if not file_path.exists():
            return jsonify({"success": False, "error": "Graph not found"}), 404
        with open(file_path, "r") as f:
            graph_json = json.load(f)
        return jsonify(graph_json)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status and health information"""
    try:
        chainbreak = get_chainbreak()
        if not chainbreak:
            return jsonify({
                'success': False,
                'error': 'ChainBreak not initialized',
                'timestamp': get_current_timestamp()
            }), 500

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
            'success': False,
            'error': str(e),
            'timestamp': get_current_timestamp()
        }), 500


@app.route('/api/mode', methods=['GET'])
def get_backend_mode():
    """Get current backend mode"""
    try:
        chainbreak = get_chainbreak()
        if not chainbreak:
            return jsonify({
                'success': False,
                'error': 'ChainBreak not initialized',
                'timestamp': get_current_timestamp()
            }), 500

        mode_info = {
            'backend_mode': chainbreak.get_backend_mode(),
            'neo4j_available': chainbreak.is_neo4j_available(),
            'use_json_backend': getattr(chainbreak, 'use_json_backend', False),
            'timestamp': get_current_timestamp()
        }

        return jsonify({
            'success': True,
            'data': mode_info,
            'message': f'Backend mode: {mode_info["backend_mode"]}'
        })

    except Exception as e:
        logger.error(f"Error getting backend mode: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': get_current_timestamp()
        }), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_address():
    """Analyze a single Bitcoin address"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided',
                'timestamp': get_current_timestamp()
            }), 400

        address = data.get('address')
        blockchain = data.get('blockchain', 'btc')
        generate_visualizations = data.get('generate_visualizations', True)

        if not address:
            return jsonify({
                'success': False,
                'error': 'Address parameter is required',
                'timestamp': get_current_timestamp()
            }), 400

        logger.info(f"API: Analyzing address {address}")
        chainbreak = get_chainbreak()

        if not chainbreak:
            return jsonify({
                'success': False,
                'error': 'ChainBreak not initialized',
                'timestamp': get_current_timestamp()
            }), 500

        results = chainbreak.analyze_address(
            address, blockchain, generate_visualizations)

        if 'error' in results:
            return jsonify({
                'success': False,
                'error': results['error'],
                'timestamp': get_current_timestamp()
            }), 400

        return jsonify({
            'success': True,
            'data': results,
            'timestamp': results.get('analysis_timestamp'),
            'message': f'Analysis completed for address {address}'
        })

    except Exception as e:
        logger.error(f"Error in analyze_address API: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Analysis failed: {str(e)}',
            'timestamp': get_current_timestamp()
        }), 500


@app.route('/api/analyze/batch', methods=['POST'])
def analyze_multiple_addresses():
    """Analyze multiple Bitcoin addresses"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided',
                'timestamp': get_current_timestamp()
            }), 400

        addresses = data.get('addresses', [])
        blockchain = data.get('blockchain', 'btc')

        if not addresses or not isinstance(addresses, list):
            return jsonify({
                'success': False,
                'error': 'Addresses parameter must be a non-empty list',
                'timestamp': get_current_timestamp()
            }), 400

        if len(addresses) > 100:
            return jsonify({
                'success': False,
                'error': 'Maximum 100 addresses allowed per batch',
                'timestamp': get_current_timestamp()
            }), 400

        logger.info(f"API: Analyzing {len(addresses)} addresses")
        chainbreak = get_chainbreak()

        if not chainbreak:
            return jsonify({
                'success': False,
                'error': 'ChainBreak not initialized',
                'timestamp': get_current_timestamp()
            }), 500

        results = chainbreak.analyze_multiple_addresses(addresses, blockchain)

        if 'error' in results:
            return jsonify({
                'success': False,
                'error': results['error'],
                'timestamp': get_current_timestamp()
            }), 400

        return jsonify({
            'success': True,
            'data': results,
            'timestamp': results.get('analysis_timestamp'),
            'message': f'Batch analysis completed for {len(addresses)} addresses'
        })

    except Exception as e:
        logger.error(f"Error in analyze_multiple_addresses API: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Batch analysis failed: {str(e)}',
            'timestamp': get_current_timestamp()
        }), 500


@app.route('/api/export/gephi', methods=['GET'])
def export_to_gephi():
    """Export network to Gephi format"""
    try:
        address = request.args.get('address')
        output_file = request.args.get('output_file')

        if not address:
            return jsonify({
                'success': False,
                'error': 'Address parameter is required',
                'timestamp': get_current_timestamp()
            }), 400

        logger.info(f"API: Exporting to Gephi for address {address}")
        chainbreak = get_chainbreak()

        if not chainbreak:
            return jsonify({
                'success': False,
                'error': 'ChainBreak not initialized',
                'timestamp': get_current_timestamp()
            }), 500

        export_file = chainbreak.export_network_to_gephi(address, output_file)

        if not export_file:
            return jsonify({
                'success': False,
                'error': 'Export failed - no data available',
                'timestamp': get_current_timestamp()
            }), 400

        return jsonify({
            'success': True,
            'data': {
                'export_file': export_file,
                'address': address
            },
            'timestamp': get_current_timestamp(),
            'message': f'Network exported to {export_file}'
        })

    except Exception as e:
        logger.error(f"Error in export_to_gephi API: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Export failed: {str(e)}',
            'timestamp': get_current_timestamp()
        }), 500


@app.route('/api/report/risk', methods=['POST'])
def generate_risk_report():
    """Generate comprehensive risk report"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided',
                'timestamp': get_current_timestamp()
            }), 400

        addresses = data.get('addresses', [])
        output_file = data.get('output_file')

        if not addresses or not isinstance(addresses, list):
            return jsonify({
                'success': False,
                'error': 'Addresses parameter must be a non-empty list',
                'timestamp': get_current_timestamp()
            }), 400

        logger.info(
            f"API: Generating risk report for {len(addresses)} addresses")
        chainbreak = get_chainbreak()

        if not chainbreak:
            return jsonify({
                'success': False,
                'error': 'ChainBreak not initialized',
                'timestamp': get_current_timestamp()
            }), 500

        report_content = chainbreak.generate_risk_report(
            addresses, output_file)

        return jsonify({
            'success': True,
            'data': {
                'report_content': report_content,
                'addresses_count': len(addresses),
                'output_file': output_file
            },
            'timestamp': get_current_timestamp(),
            'message': f'Risk report generated for {len(addresses)} addresses'
        })

    except Exception as e:
        logger.error(f"Error in generate_risk_report API: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Report generation failed: {str(e)}',
            'timestamp': get_current_timestamp()
        }), 500


@app.route('/api/addresses', methods=['GET'])
def get_analyzed_addresses():
    """Get list of analyzed addresses"""
    try:
        chainbreak = get_chainbreak()
        if not chainbreak:
            return jsonify({
                'success': False,
                'error': 'ChainBreak not initialized',
                'timestamp': get_current_timestamp()
            }), 500

        if chainbreak.is_neo4j_available():
            # Get addresses from Neo4j
            with chainbreak.data_ingestor.driver.session() as session:
                result = session.run(
                    "MATCH (a:Address) RETURN a.address as address")
                addresses = [record['address'] for record in result]
        else:
            # Get addresses from JSON files
            addresses = []
            data_dir = Path("data")
            if data_dir.exists():
                for json_file in data_dir.glob("*.json"):
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                            if 'meta' in data and 'address' in data['meta']:
                                addresses.append(data['meta']['address'])
                    except:
                        continue

        return jsonify({
            'success': True,
            'data': {
                'addresses': addresses,
                'count': len(addresses),
                'backend_mode': chainbreak.get_backend_mode()
            },
            'timestamp': get_current_timestamp()
        })

    except Exception as e:
        logger.error(f"Error getting analyzed addresses: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': get_current_timestamp()
        }), 500


@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get system statistics and metrics"""
    try:
        chainbreak = get_chainbreak()
        if not chainbreak:
            return jsonify({
                'success': False,
                'error': 'ChainBreak not initialized',
                'timestamp': get_current_timestamp()
            }), 500

        status = chainbreak.get_system_status()

        return jsonify({
            'success': True,
            'data': status,
            'timestamp': get_current_timestamp()
        })

    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': get_current_timestamp()
        }), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'timestamp': get_current_timestamp()
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'timestamp': get_current_timestamp()
    }), 500


def get_current_timestamp():
    from datetime import datetime
    return datetime.now().isoformat()


def create_app(config=None):
    if config:
        app.config.update(config)
    return app


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
