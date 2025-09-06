from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import logging
import traceback
import json
from pathlib import Path
from .chainbreak import ChainBreak
from .api_frontend import bp as frontend_bp

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Enable CORS before registering blueprints
CORS(app, resources={
    r"/api/*": {"origins": ["http://localhost:3000", "http://localhost:5000"]},
    r"/frontend/*": {"origins": ["http://localhost:3000", "http://localhost:5000"]}
})

app.register_blueprint(frontend_bp)

# Unified data directory - use Data/graph (case-sensitive)
GRAPH_DIR = Path("Data/graph")
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"Graph directory initialized: {GRAPH_DIR.resolve()}")


def get_chainbreak():
    try:
        from .chainbreak import ChainBreak
        return ChainBreak()
    except Exception as e:
        logger.error(f"Failed to initialize ChainBreak: {e}")
        return None


@app.route("/")
def index_new():
    try:
        # Try to serve the React build index.html
        frontend_build = Path("frontend/build").resolve()
        if frontend_build.exists():
            logger.info(f"Serving React frontend from {frontend_build}")
            return send_from_directory(str(frontend_build), "index.html")
        else:
            logger.info(
                "React build not found, falling back to static frontend")
            static_frontend = Path("frontend").resolve()
            return send_from_directory(str(static_frontend), "index.html")
    except Exception as e:
        logger.error(f"Error serving frontend: {e}")
        return jsonify({"error": "Frontend not available"}), 404


@app.route("/static/<path:filename>")
def serve_static(filename):
    try:
        frontend_build = Path("frontend/build/static").resolve()
        if frontend_build.exists():
            logger.info(f"Serving static file from React build: {filename}")
            return send_from_directory(str(frontend_build), filename)
        else:
            logger.info(
                f"React static not found, falling back to static: {filename}")
            static_dir = Path("frontend/static").resolve()
            return send_from_directory(str(static_dir), filename)
    except Exception as e:
        logger.error(f"Error serving static file {filename}: {e}")
        return jsonify({"error": "Static file not found"}), 404


@app.route("/<path:filename>")
def serve_frontend_files(filename):
    try:
        # Skip API routes
        if filename.startswith('api/'):
            return jsonify({"error": "API endpoint not found"}), 404

        frontend_build = Path("frontend/build").resolve()
        if frontend_build.exists():
            file_path = frontend_build / filename
            if file_path.exists() and file_path.is_file():
                logger.info(f"Serving React file: {filename}")
                return send_from_directory(str(frontend_build), filename)

        # Fallback to old static files
        logger.info(
            f"React file not found, falling back to static: {filename}")
        static_dir = Path("frontend").resolve()
        return send_from_directory(str(static_dir), filename)
    except Exception as e:
        logger.error(f"Error serving frontend file {filename}: {e}")
        return jsonify({"error": "File not found"}), 404


@app.route("/api/mode", methods=["GET"])
def get_backend_mode():
    """Get current backend mode"""
    try:
        chainbreak = get_chainbreak()
        if chainbreak:
            return jsonify({
                "success": True,
                "data": {
                    "backend_mode": chainbreak.get_backend_mode(),
                    "neo4j_available": chainbreak.is_neo4j_available()
                }
            })
        else:
            return jsonify({
                "success": False,
                "error": "ChainBreak not initialized"
            }), 500
    except Exception as e:
        logger.error(f"Error getting backend mode: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/status", methods=["GET"])
def get_system_status():
    """Get system status"""
    try:
        chainbreak = get_chainbreak()
        if chainbreak:
            status = chainbreak.get_system_status()
            return jsonify({
                "success": True,
                "data": status
            })
        else:
            return jsonify({
                "success": False,
                "error": "ChainBreak not initialized"
            }), 500
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/graph/list", methods=["GET"])
def list_graphs():
    """List available graph files"""
    try:
        files = [f.name for f in GRAPH_DIR.glob("*.json")]
        return jsonify({"success": True, "files": files})
    except Exception as e:
        logger.error(f"Error listing graphs: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/graph/address", methods=["POST"])
def fetch_graph_address():
    """Fetch and save graph for an address"""
    try:
        data = request.get_json()
        address = data.get("address")
        tx_limit = data.get("tx_limit", 50)

        if not address:
            return jsonify({"success": False, "error": "Address required"}), 400

        chainbreak = get_chainbreak()
        if not chainbreak:
            return jsonify({"success": False, "error": "ChainBreak not initialized"}), 500

        # Import here to avoid circular imports
        from .fetch_blockchain_com import BlockchainComFetcher

        fetcher = BlockchainComFetcher()
        graph = fetcher.build_graph_for_address(address, tx_limit=tx_limit)

        # Sanitize filename - only allow alphanumeric, underscore, hyphen
        import re
        safe_address = re.sub(r'[^A-Za-z0-9_\-]', '_', address)
        filename = f"graph_{safe_address}.json"

        file_path = GRAPH_DIR / filename
        with open(file_path, "w") as f:
            json.dump(graph, f, indent=2)

        logger.info(
            f"Graph saved: {filename} with {len(graph.get('nodes', []))} nodes")

        return jsonify({
            "success": True,
            "file": filename,
            "meta": graph.get("meta", {})
        })

    except Exception as e:
        logger.error(f"Error fetching graph: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/graph/get", methods=["GET"])
def get_graph():
    """Get a specific graph by name"""
    try:
        name = request.args.get("name")
        if not name:
            return jsonify({"success": False, "error": "Name parameter required"}), 400

        file_path = GRAPH_DIR / name
        if not file_path.exists():
            return jsonify({"success": False, "error": "Graph not found"}), 404

        with open(file_path, "r") as f:
            graph_json = json.load(f)

        return jsonify(graph_json)

    except Exception as e:
        logger.error(f"Error getting graph: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/analyze", methods=["POST"])
def analyze_address():
    """Analyze a single address"""
    try:
        data = request.get_json()
        address = data.get("address")
        blockchain = data.get("blockchain", "btc")
        generate_visualizations = data.get("generate_visualizations", True)

        if not address:
            return jsonify({"success": False, "error": "Address required"}), 400

        chainbreak = get_chainbreak()
        if not chainbreak:
            return jsonify({"success": False, "error": "ChainBreak not initialized"}), 500

        result = chainbreak.analyze_address(
            address, blockchain, generate_visualizations)
        return jsonify({"success": True, "data": result})

    except Exception as e:
        logger.error(f"Error analyzing address: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/analyze/batch", methods=["POST"])
def analyze_multiple_addresses():
    """Analyze multiple addresses"""
    try:
        data = request.get_json()
        addresses = data.get("addresses", [])
        blockchain = data.get("blockchain", "btc")

        if not addresses:
            return jsonify({"success": False, "error": "Addresses array required"}), 400

        chainbreak = get_chainbreak()
        if not chainbreak:
            return jsonify({"success": False, "error": "ChainBreak not initialized"}), 500

        result = chainbreak.analyze_multiple_addresses(addresses, blockchain)
        return jsonify({"success": True, "data": result})

    except Exception as e:
        logger.error(f"Error analyzing addresses: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/export/gephi", methods=["GET"])
def export_to_gephi():
    """Export network to Gephi format"""
    try:
        address = request.args.get("address")
        output_file = request.args.get("output_file")

        if not address:
            return jsonify({"success": False, "error": "Address parameter required"}), 400

        chainbreak = get_chainbreak()
        if not chainbreak:
            return jsonify({"success": False, "error": "ChainBreak not initialized"}), 500

        result = chainbreak.export_network_to_gephi(address, output_file)
        return jsonify({"success": True, "data": {"file": result}})

    except Exception as e:
        logger.error(f"Error exporting to Gephi: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/report/risk", methods=["POST"])
def generate_risk_report():
    """Generate risk report for addresses"""
    try:
        data = request.get_json()
        addresses = data.get("addresses", [])
        output_file = data.get("output_file")

        if not addresses:
            return jsonify({"success": False, "error": "Addresses array required"}), 400

        chainbreak = get_chainbreak()
        if not chainbreak:
            return jsonify({"success": False, "error": "ChainBreak not initialized"}), 500

        result = chainbreak.generate_risk_report(addresses, output_file)
        return jsonify({"success": True, "data": {"report": result}})

    except Exception as e:
        logger.error(f"Error generating risk report: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/addresses", methods=["GET"])
def get_analyzed_addresses():
    """Get list of analyzed addresses"""
    try:
        chainbreak = get_chainbreak()
        if not chainbreak:
            return jsonify({"success": False, "error": "ChainBreak not initialized"}), 500

        # This would need to be implemented in ChainBreak class
        return jsonify({"success": True, "data": {"addresses": []}})

    except Exception as e:
        logger.error(f"Error getting addresses: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/statistics", methods=["GET"])
def get_statistics():
    """Get system statistics"""
    try:
        chainbreak = get_chainbreak()
        if not chainbreak:
            return jsonify({"success": False, "error": "ChainBreak not initialized"}), 500

        # This would need to be implemented in ChainBreak class
        return jsonify({"success": True, "data": {"statistics": {}}})

    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    logger.warning(f"404 error: {request.url}")
    return jsonify({"error": "Not found", "path": request.path}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 error: {error}")
    return jsonify({"error": "Internal server error"}), 500


def get_current_timestamp():
    from datetime import datetime
    return datetime.now().isoformat()


def create_app(config=None):
    if config:
        app.config.update(config)
    return app


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
