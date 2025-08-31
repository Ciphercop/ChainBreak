from flask import Blueprint, jsonify, request, send_from_directory
import logging
from pathlib import Path
from .fetch_blockchain_com import BlockchainComFetcher, DATA_DIR

logger = logging.getLogger(__name__)

bp = Blueprint("frontend_api", __name__)


@bp.route("/api/graph/address", methods=["POST"])
def build_graph_address():
    try:
        payload = request.get_json(force=True)
        address = payload.get("address", "").strip()
        tx_limit = int(payload.get("tx_limit", 50))
        if not address:
            return jsonify({"success": False, "error": "address required"}), 400
        fetcher = BlockchainComFetcher()
        graph = fetcher.build_graph_for_address(address, tx_limit=tx_limit)
        saved = fetcher.save_graph(graph)
        return jsonify({"success": True, "file": saved, "meta": graph.get("meta", {})})
    except Exception as e:
        logger.error(f"build_graph_error err={e}")
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route("/api/graph/list", methods=["GET"])
def list_graphs():
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        files = sorted([p.name for p in DATA_DIR.glob("*.json")])
        return jsonify({"success": True, "files": files})
    except Exception as e:
        logger.error(f"list_graphs_error err={e}")
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route("/api/graph/get", methods=["GET"])
def get_graph():
    try:
        name = request.args.get("name", "").strip()
        if not name:
            return jsonify({"success": False, "error": "name required"}), 400
        
        # Ensure DATA_DIR is absolute
        data_dir = DATA_DIR.resolve()
        file_path = data_dir / name
        
        # Check if file exists
        if not file_path.exists():
            return jsonify({"success": False, "error": f"File {name} not found"}), 404
        
        # Read and return the file content directly
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        from flask import Response
        return Response(content, mimetype="application/json")
    except Exception as e:
        logger.error(f"get_graph_error err={e}")
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route("/frontend/<path:filename>")
def serve_frontend(filename):
    root = Path("frontend").resolve()
    return send_from_directory(root.as_posix(), filename)
