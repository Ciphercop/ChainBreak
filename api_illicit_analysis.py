"""
API endpoints for illicit transaction analysis
"""

from flask import Blueprint, request, jsonify
import logging
from datetime import datetime
import json
import os
import sys

# Add the parent directory to the path to import the detector
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from illicit_transaction_detector import (
    IllicitTransactionDetector, 
    Transaction, 
    GraphVisualizer
)

logger = logging.getLogger(__name__)

# Create Blueprint
illicit_analysis_bp = Blueprint('illicit_analysis', __name__, url_prefix='/api/illicit-analysis')

# Initialize detector with Chainalysis API key
CHAINALYSIS_API_KEY = "db373a00f1f63693d7ccf144ee781787865310acda3870ca8abfb09135cbfc58"
detector = None

def get_detector():
    """Get or create detector instance."""
    global detector
    if detector is None:
        try:
            detector = IllicitTransactionDetector(chainalysis_api_key=CHAINALYSIS_API_KEY)
        except Exception as e:
            logger.error(f"Failed to initialize detector: {e}")
            # Initialize without API key as fallback
            detector = IllicitTransactionDetector(chainalysis_api_key=None)
    return detector

@illicit_analysis_bp.route('/', methods=['POST'])
def analyze_illicit_transactions():
    """
    Analyze Bitcoin addresses for illicit transaction patterns.
    
    Expected JSON payload:
    {
        "addresses": ["1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa", "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2"],
        "include_visualization": true,
        "max_transactions": 1000
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'addresses' not in data:
            return jsonify({
                'success': False,
                'error': 'Addresses are required'
            }), 400
        
        addresses = data['addresses']
        include_visualization = data.get('include_visualization', True)
        max_transactions = data.get('max_transactions', 1000)
        
        logger.info(f"Starting illicit transaction analysis for {len(addresses)} addresses")
        
        # For demo purposes, create sample transactions
        # In production, this would fetch real transaction data
        transactions = create_sample_transactions(addresses, max_transactions)
        
        # Get detector instance
        detector_instance = get_detector()
        
        # Perform analysis
        analysis = detector_instance.analyze_transactions(transactions)
        
        # Build transaction graph for visualization
        graph = detector_instance._build_transaction_graph(transactions)
        
        # Prepare response data
        response_data = {
            'success': True,
            'analysis': {
                'total_addresses': analysis.total_addresses,
                'total_transactions': analysis.total_transactions,
                'high_risk_addresses': analysis.high_risk_addresses,
                'risk_distribution': analysis.risk_distribution,
                'suspicious_patterns': [
                    {
                        'pattern_type': pattern.pattern_type.value,
                        'description': pattern.description,
                        'confidence': pattern.confidence,
                        'risk_score': pattern.risk_score,
                        'addresses': pattern.addresses,
                        'transaction_count': len(pattern.transactions),
                        'metadata': pattern.metadata
                    }
                    for pattern in analysis.suspicious_patterns
                ],
                'detection_summary': analysis.detection_summary,
                'analysis_timestamp': analysis.analysis_timestamp.isoformat()
            },
            'addresses': {
                address: {
                    'risk_level': node.risk_level.value,
                    'risk_score': node.risk_score,
                    'transaction_count': node.transaction_count,
                    'total_sent': node.total_sent,
                    'total_received': node.total_received,
                    'suspicious_patterns': [p.value for p in node.suspicious_patterns],
                    'centrality_measures': node.centrality_measures,
                    'threat_intel_data': node.threat_intel_data,
                    'sir_model_state': node.sir_model_state,
                    'sir_probability': node.sir_probability
                }
                for address, node in analysis.addresses.items()
            }
        }
        
        # Add visualization data if requested
        if include_visualization:
            try:
                graph_data = detector.get_visualization_data(analysis, graph)
                response_data['graph_data'] = graph_data
                
                # Create visualizations
                viz_files = detector.create_visualizations(analysis, graph, "static/visualizations")
                response_data['visualization_files'] = viz_files
                
            except Exception as e:
                logger.warning(f"Visualization creation failed: {e}")
                response_data['visualization_error'] = str(e)
        
        logger.info(f"Analysis completed successfully: {len(analysis.suspicious_patterns)} patterns detected")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return jsonify({
            'success': False,
            'error': f'Analysis failed: {str(e)}'
        }), 500


@illicit_analysis_bp.route('/risk-levels', methods=['GET'])
def get_risk_levels():
    """Get available risk levels."""
    from illicit_transaction_detector import RiskLevel
    
    levels = [
        {
            'value': level.value,
            'name': level.value,
            'description': get_risk_level_description(level.value)
        }
        for level in RiskLevel
    ]
    
    return jsonify({
        'success': True,
        'risk_levels': levels
    })

@illicit_analysis_bp.route('/threat-intel/<address>', methods=['GET'])
def get_threat_intelligence(address):
    """Get threat intelligence data for a specific address."""
    try:
        # Get detector instance
        detector_instance = get_detector()
        
        # Check with BitcoinWhosWho
        bw_result = detector_instance.threat_intel_scraper.search_address(address)
        
        # Check with Chainalysis if available
        chainalysis_result = None
        if detector_instance.chainalysis_api:
            chainalysis_result = detector_instance.chainalysis_api.check_address(address)
        
        threat_data = {
            'address': address,
            'bitcoinwhoswho': {
                'score': getattr(bw_result, 'score', None) if bw_result else None,
                'tags': getattr(bw_result, 'tags', None) if bw_result else None,
                'scam_reports': len(getattr(bw_result, 'scam_reports', []) or []) if bw_result else 0,
                'confidence': getattr(bw_result, 'confidence', None) if bw_result else None
            } if bw_result else None,
            'chainalysis': chainalysis_result if chainalysis_result and 'error' not in chainalysis_result else None
        }
        
        return jsonify({
            'success': True,
            'threat_intelligence': threat_data
        })
        
    except Exception as e:
        logger.error(f"Threat intelligence lookup failed for {address}: {e}")
        return jsonify({
            'success': False,
            'error': f'Threat intelligence lookup failed: {str(e)}'
        }), 500

@illicit_analysis_bp.route('/run-detection', methods=['POST'])
def run_illicit_detection():
    """
    Run illicit detection on an existing graph.
    Expects graph data in the request body.
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract graph data from request (handle both formats)
        if 'graph_data' in data:
            # Handle nested graph_data format
            graph_data = data['graph_data']
            nodes = graph_data.get('nodes', [])
            edges = graph_data.get('edges', [])
        else:
            # Handle direct format
            nodes = data.get('nodes', [])
            edges = data.get('edges', [])
        
        if not nodes or not edges:
            return jsonify({'error': 'Graph data required (nodes and edges)'}), 400
        
        # Convert graph data to Transaction objects
        transactions = []
        for edge in edges:
            # Find source and target nodes
            source_node = next((n for n in nodes if n['id'] == edge['source']), None)
            target_node = next((n for n in nodes if n['id'] == edge['target']), None)
            
            if source_node and target_node:
                # Create transaction from edge data
                transaction = Transaction(
                    tx_hash=edge.get('id', f"tx_{len(transactions)}"),
                    from_address=edge['source'],
                    to_address=edge['target'],
                    value=edge.get('value', 0.0),
                    timestamp=datetime.fromisoformat(edge.get('timestamp', datetime.now().isoformat())),
                    block_height=edge.get('block_height'),
                    fee=edge.get('fee'),
                    confirmations=edge.get('confirmations')
                )
                transactions.append(transaction)
        
        if not transactions:
            return jsonify({'error': 'No valid transactions found'}), 400
        
        logger.info(f"Running illicit detection on {len(transactions)} transactions from graph")
        
        # Get detector instance
        detector_instance = get_detector()
        
        # Run analysis
        analysis = detector_instance.analyze_transactions(transactions)
        
        # Convert analysis to JSON-serializable format
        analysis_data = {
            'addresses': {},
            'suspicious_patterns': [],
            'clusters': analysis.clusters,
            'risk_distribution': analysis.risk_distribution,
            'high_risk_addresses': analysis.high_risk_addresses,
            'analysis_timestamp': analysis.analysis_timestamp.isoformat(),
            'total_transactions': analysis.total_transactions,
            'total_addresses': analysis.total_addresses,
            'detection_summary': analysis.detection_summary,
            'sir_model_results': analysis.sir_model_results,
            'exchange_paths': analysis.exchange_paths
        }
        
        # Convert addresses to serializable format
        for addr, node in analysis.addresses.items():
            analysis_data['addresses'][addr] = {
                'address': node.address,
                'total_received': node.total_received,
                'total_sent': node.total_sent,
                'transaction_count': node.transaction_count,
                'first_seen': node.first_seen.isoformat() if node.first_seen else None,
                'last_seen': node.last_seen.isoformat() if node.last_seen else None,
                'risk_score': node.risk_score,
                'risk_level': node.risk_level.value,
                'suspicious_patterns': [p.value for p in node.suspicious_patterns],
                'cluster_id': node.cluster_id,
                'centrality_measures': node.centrality_measures,
                'threat_intel_data': node.threat_intel_data,
                'sir_model_state': node.sir_model_state,
                'sir_probability': node.sir_probability
            }
        
        # Convert suspicious patterns to serializable format
        for pattern in analysis.suspicious_patterns:
            analysis_data['suspicious_patterns'].append({
                'pattern_type': pattern.pattern_type.value,
                'addresses': pattern.addresses,
                'transactions': [
                    {
                        'tx_hash': tx.tx_hash,
                        'from_address': tx.from_address,
                        'to_address': tx.to_address,
                        'value': tx.value,
                        'timestamp': tx.timestamp.isoformat(),
                        'block_height': tx.block_height,
                        'fee': tx.fee,
                        'confirmations': tx.confirmations
                    } for tx in pattern.transactions
                ],
                'confidence': pattern.confidence,
                'description': pattern.description,
                'risk_score': pattern.risk_score,
                'metadata': pattern.metadata
            })
        
        return jsonify({
            'success': True,
            'analysis': analysis_data,
            'message': f'Illicit detection completed for {len(transactions)} transactions'
        })
        
    except Exception as e:
        logger.error(f"Error running illicit detection: {e}")
        return jsonify({'error': str(e)}), 500

def create_sample_transactions(addresses, max_transactions=1000):
    """Create sample transactions for demonstration purposes."""
    from datetime import datetime, timedelta
    import random
    
    transactions = []
    base_time = datetime.now() - timedelta(days=30)
    
    # Create various transaction patterns
    pattern_types = ['normal', 'mixing', 'peel_chain', 'smurfing', 'rapid_transfers']
    
    for i in range(min(max_transactions, len(addresses) * 50)):
        # Select pattern type
        pattern = random.choice(pattern_types)
        
        # Select addresses
        from_addr = random.choice(addresses)
        other_addresses = [addr for addr in addresses if addr != from_addr]
        to_addr = random.choice(other_addresses) if other_addresses else from_addr
        
        # Generate transaction based on pattern
        if pattern == 'mixing':
            # Mixing: many small transactions
            value = random.uniform(0.001, 0.1)
            timestamp = base_time + timedelta(minutes=random.randint(0, 60))
        elif pattern == 'peel_chain':
            # Peel chain: decreasing values
            value = max(0.001, 1.0 - (i * 0.1))
            timestamp = base_time + timedelta(minutes=i * 5)
        elif pattern == 'smurfing':
            # Smurfing: many small transactions
            value = random.uniform(0.001, 0.01)
            timestamp = base_time + timedelta(minutes=random.randint(0, 120))
        elif pattern == 'rapid_transfers':
            # Rapid transfers: quick succession
            value = random.uniform(0.1, 1.0)
            timestamp = base_time + timedelta(seconds=random.randint(0, 300))
        else:
            # Normal transactions
            value = random.uniform(0.001, 10.0)
            timestamp = base_time + timedelta(hours=random.randint(0, 720))
        
        tx = Transaction(
            tx_hash=f"sample_tx_{i:06d}",
            from_address=from_addr,
            to_address=to_addr,
            value=value,
            timestamp=timestamp,
            block_height=100000 + i,
            fee=random.uniform(0.0001, 0.01),
            confirmations=random.randint(1, 6)
        )
        
        transactions.append(tx)
    
    return transactions

def get_pattern_description(pattern_type):
    """Get description for pattern type."""
    descriptions = {
        'peel_chain': 'Series of transactions with decreasing values, often used to obscure the final destination',
        'mixing': 'Many-to-many transaction pattern used to obfuscate transaction trails',
        'chain_hopping': 'Moving funds through multiple exchanges to break transaction chains',
        'smurfing': 'Breaking large amounts into many small transactions to avoid detection',
        'layering': 'Multiple intermediate transactions to create complex transaction paths',
        'rapid_transfers': 'High-frequency transactions within short time periods',
        'round_amounts': 'Transactions with suspiciously round amounts',
        'sudden_bursts': 'Unusual spikes in transaction activity'
    }
    return descriptions.get(pattern_type, 'Suspicious transaction pattern detected')

def get_risk_level_description(risk_level):
    """Get description for risk level."""
    descriptions = {
        'CRITICAL': 'Highest risk - immediate attention required',
        'HIGH': 'High risk - significant suspicious activity detected',
        'MEDIUM': 'Medium risk - some suspicious patterns present',
        'LOW': 'Low risk - minimal suspicious activity',
        'CLEAN': 'No suspicious activity detected'
    }
    return descriptions.get(risk_level, 'Unknown risk level')

@illicit_analysis_bp.route('/pattern-types', methods=['GET'])
def get_pattern_types():
    """Get available pattern types."""
    try:
        pattern_types = [
            'peel_chain',
            'mixing',
            'chain_hopping',
            'smurfing',
            'layering',
            'rapid_transfers',
            'round_amounts',
            'sudden_bursts'
        ]
        
        return jsonify({
            'success': True,
            'pattern_types': pattern_types
        })
    except Exception as e:
        logger.error(f"Error getting pattern types: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Register the blueprint
def register_illicit_analysis_routes(app):
    """Register illicit analysis routes with the Flask app."""
    app.register_blueprint(illicit_analysis_bp)
    logger.info("Illicit transaction analysis routes registered")
