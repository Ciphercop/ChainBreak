"""
Main ChainBreak Application Class
Integrates all components for comprehensive blockchain analysis
"""

import logging
import yaml
import os
from typing import Dict, Any, List, Optional
from pathlib import Path

from .data_ingestion import BlockchainDataIngestor
from .anomaly_detection import (
    LayeringDetector, 
    SmurfingDetector, 
    VolumeAnomalyDetector,
    TemporalAnomalyDetector
)
from .risk_scoring import RiskScorer
from .visualization import NetworkVisualizer, GephiExporter, ChartGenerator

logger = logging.getLogger(__name__)


class ChainBreak:
    """Main ChainBreak application class integrating all components"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize ChainBreak with configuration"""
        self.config = self._load_config(config_path)
        self._setup_logging()
        
        # Initialize Neo4j connection
        neo4j_config = self.config.get('neo4j', {})
        # Use environment variables if available (for Docker), otherwise use config
        self.neo4j_uri = os.environ.get('NEO4J_URI', neo4j_config.get('uri', 'bolt://localhost:7687'))
        self.neo4j_user = os.environ.get('NEO4J_USERNAME', neo4j_config.get('username', 'neo4j'))
        self.neo4j_password = os.environ.get('NEO4J_PASSWORD', neo4j_config.get('password', 'password'))
        
        # Debug logging
        logger.info(f"Neo4j URI: {self.neo4j_uri}")
        logger.info(f"Neo4j User: {self.neo4j_user}")
        logger.info(f"Neo4j Password: {'*' * len(self.neo4j_password) if self.neo4j_password else 'None'}")
        
        # Initialize components
        self._initialize_components()
        
        logger.info("ChainBreak initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Error loading config from {config_path}: {str(e)}")
            logger.info("Using default configuration")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'neo4j': {
                'uri': 'bolt://localhost:7687',
                'username': 'neo4j',
                'password': 'password'
            },
            'blockcypher': {
                'api_key': 'your_api_key_here',
                'timeout': 30
            },
            'analysis': {
                'time_window_hours': 24,
                'min_transactions': 5,
                'volume_threshold': 1000000
            },
            'risk_scoring': {
                'volume_weight': 0.3,
                'frequency_weight': 0.2,
                'layering_weight': 0.3,
                'smurfing_weight': 0.2
            }
        }
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('chainbreak.log'),
                logging.StreamHandler()
            ]
        )
    
    def _initialize_components(self):
        """Initialize all ChainBreak components"""
        try:
            # Check if Neo4j should be skipped
            skip_neo4j = os.environ.get('CHAINBREAK_NO_NEO4J', '0') == '1'
            
            if skip_neo4j:
                logger.warning("Neo4j initialization skipped due to CHAINBREAK_NO_NEO4J environment variable")
                self.data_ingestor = None
                self.layering_detector = None
                self.smurfing_detector = None
                self.volume_detector = None
                self.temporal_detector = None
                self.risk_scorer = None
                self.visualizer = None
                self.gephi_exporter = None
                self.chart_generator = None
                logger.info("Components initialized in limited mode (no Neo4j)")
                return
            
            # Initialize data ingestor
            self.data_ingestor = BlockchainDataIngestor(
                self.neo4j_uri, 
                self.neo4j_user, 
                self.neo4j_password
            )
            
            # Initialize anomaly detectors
            self.layering_detector = LayeringDetector(self.data_ingestor.driver)
            self.smurfing_detector = SmurfingDetector(self.data_ingestor.driver)
            self.volume_detector = VolumeAnomalyDetector(self.data_ingestor.driver)
            self.temporal_detector = TemporalAnomalyDetector(self.data_ingestor.driver)
            
            # Initialize risk scorer
            self.risk_scorer = RiskScorer(self.data_ingestor.driver, self.config)
            
            # Initialize visualization components
            self.visualizer = NetworkVisualizer(self.data_ingestor.driver)
            self.gephi_exporter = GephiExporter(self.data_ingestor.driver)
            self.chart_generator = ChartGenerator(self.data_ingestor.driver)
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            # Don't raise the exception, allow the system to run in limited mode
            logger.warning("System will run in limited mode due to initialization errors")
            self.data_ingestor = None
            self.layering_detector = None
            self.smurfing_detector = None
            self.volume_detector = None
            self.temporal_detector = None
            self.risk_scorer = None
            self.visualizer = None
            self.gephi_exporter = None
            self.chart_generator = None
    
    def analyze_address(self, address: str, blockchain: str = 'btc', 
                       generate_visualizations: bool = True) -> Dict[str, Any]:
        """Comprehensive analysis of a single address"""
        try:
            logger.info(f"Starting comprehensive analysis for address: {address}")
            
            # Step 1: Ingest data
            logger.info("Step 1: Ingesting blockchain data...")
            ingestion_success = self.data_ingestor.ingest_address_data(address, blockchain)
            
            if not ingestion_success:
                logger.warning(f"Data ingestion failed for address {address}")
                return self._get_analysis_error_result(address, "Data ingestion failed")
            
            # Step 2: Detect anomalies
            logger.info("Step 2: Detecting anomalies...")
            
            # Layering detection
            layering_patterns = self.layering_detector.detect_layering_patterns(address)
            complex_layering = self.layering_detector.detect_complex_layering(address)
            
            # Smurfing detection
            smurfing_patterns = self.smurfing_detector.detect_smurfing_patterns()
            structured_smurfing = self.smurfing_detector.detect_structured_smurfing(address)
            
            # Volume anomalies
            volume_anomalies = self.volume_detector.detect_volume_anomalies()
            value_pattern_anomalies = self.volume_detector.detect_value_pattern_anomalies(address)
            
            # Temporal anomalies
            timing_anomalies = self.temporal_detector.detect_timing_anomalies(address)
            
            # Step 3: Calculate risk score
            logger.info("Step 3: Calculating risk score...")
            risk_score = self.risk_scorer.calculate_address_risk_score(address)
            
            # Step 4: Generate visualizations if requested
            visualizations = {}
            if generate_visualizations:
                logger.info("Step 4: Generating visualizations...")
                try:
                    # Network visualization
                    network_graph = self.visualizer.visualize_address_network(address)
                    visualizations['network_graph'] = network_graph
                    
                    # Transaction timeline
                    self.visualizer.create_transaction_timeline(address)
                    visualizations['timeline_created'] = True
                    
                    # Transaction volume chart
                    self.chart_generator.create_transaction_volume_chart(address)
                    visualizations['volume_chart_created'] = True
                    
                except Exception as e:
                    logger.warning(f"Error generating visualizations: {str(e)}")
                    visualizations['error'] = str(e)
            
            # Compile results
            analysis_results = {
                'address': address,
                'blockchain': blockchain,
                'analysis_timestamp': self._get_current_timestamp(),
                'ingestion_success': ingestion_success,
                'anomalies': {
                    'layering_patterns': layering_patterns,
                    'complex_layering': complex_layering,
                    'smurfing_patterns': smurfing_patterns,
                    'structured_smurfing': structured_smurfing,
                    'volume_anomalies': volume_anomalies,
                    'value_pattern_anomalies': value_pattern_anomalies,
                    'timing_anomalies': timing_anomalies
                },
                'risk_score': risk_score,
                'visualizations': visualizations,
                'summary': self._generate_analysis_summary(
                    layering_patterns, smurfing_patterns, volume_anomalies, risk_score
                )
            }
            
            logger.info(f"Analysis completed successfully for address {address}")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing address {address}: {str(e)}")
            return self._get_analysis_error_result(address, str(e))
    
    def analyze_multiple_addresses(self, addresses: List[str], blockchain: str = 'btc') -> Dict[str, Any]:
        """Analyze multiple addresses and provide comparative analysis"""
        try:
            logger.info(f"Starting analysis of {len(addresses)} addresses")
            
            individual_results = []
            for address in addresses:
                logger.info(f"Analyzing address: {address}")
                result = self.analyze_address(address, blockchain, generate_visualizations=False)
                individual_results.append(result)
            
            # Generate comparative visualizations
            risk_scores = [result['risk_score']['total_risk_score'] for result in individual_results]
            self.visualizer.create_risk_heatmap(addresses, risk_scores)
            
            # Generate risk summary
            risk_summary = self.risk_scorer.get_risk_summary(addresses)
            self.chart_generator.create_risk_distribution_chart(risk_summary)
            
            # Compile comparative results
            comparative_results = {
                'total_addresses': len(addresses),
                'blockchain': blockchain,
                'analysis_timestamp': self._get_current_timestamp(),
                'individual_results': individual_results,
                'risk_summary': risk_summary,
                'comparative_analysis': self._generate_comparative_analysis(individual_results)
            }
            
            logger.info(f"Comparative analysis completed for {len(addresses)} addresses")
            return comparative_results
            
        except Exception as e:
            logger.error(f"Error in comparative analysis: {str(e)}")
            return {'error': str(e), 'total_addresses': len(addresses)}
    
    def export_network_to_gephi(self, address: str = None, output_file: str = None) -> str:
        """Export transaction network to Gephi format"""
        try:
            if address:
                # Export specific address subgraph
                output_file = output_file or f"subgraph_{address[:8]}.gexf"
                return self.gephi_exporter.export_address_subgraph(address, output_file=output_file)
            else:
                # Export entire network
                output_file = output_file or "transaction_network.gexf"
                return self.gephi_exporter.export_to_gephi(output_file)
                
        except Exception as e:
            logger.error(f"Error exporting to Gephi: {str(e)}")
            return ""
    
    def generate_risk_report(self, addresses: List[str], output_file: str = None) -> str:
        """Generate comprehensive risk analysis report"""
        try:
            return self.risk_scorer.export_risk_report(addresses, output_file)
        except Exception as e:
            logger.error(f"Error generating risk report: {str(e)}")
            return f"Error generating report: {str(e)}"
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and health information"""
        try:
            # Check if components are initialized
            if self.data_ingestor is None:
                return {
                    'system_status': 'limited',
                    'neo4j_connection': 'unavailable',
                    'database_statistics': {'error': 'Neo4j not initialized'},
                    'configuration': {
                        'neo4j_uri': self.neo4j_uri,
                        'blockchain': 'btc',
                        'analysis_window': self.config['analysis']['time_window_hours']
                    },
                    'timestamp': self._get_current_timestamp(),
                    'message': 'System running in limited mode - Neo4j unavailable'
                }
            
            # Test Neo4j connection
            neo4j_status = "healthy"
            try:
                with self.data_ingestor.driver.session() as session:
                    result = session.run("RETURN 1 as test")
                    result.single()
            except Exception as e:
                neo4j_status = f"error: {str(e)}"
            
            # Get database statistics
            db_stats = self._get_database_statistics()
            
            return {
                'system_status': 'operational',
                'neo4j_connection': neo4j_status,
                'database_statistics': db_stats,
                'configuration': {
                    'neo4j_uri': self.neo4j_uri,
                    'blockchain': 'btc',
                    'analysis_window': self.config['analysis']['time_window_hours']
                },
                'timestamp': self._get_current_timestamp()
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {
                'system_status': 'error',
                'error': str(e),
                'timestamp': self._get_current_timestamp()
            }
    
    def _get_database_statistics(self) -> Dict[str, Any]:
        """Get database statistics from Neo4j"""
        try:
            if self.data_ingestor is None:
                return {'error': 'Neo4j not initialized'}
                
            with self.data_ingestor.driver.session() as session:
                # Count nodes by type
                node_counts = session.run("""
                    MATCH (n)
                    RETURN labels(n)[0] as node_type, count(n) as count
                    ORDER BY count DESC
                """)
                
                # Count relationships by type
                rel_counts = session.run("""
                    MATCH ()-[r]->()
                    RETURN type(r) as relationship_type, count(r) as count
                    ORDER BY count DESC
                """)
                
                return {
                    'node_counts': {record['node_type']: record['count'] for record in node_counts},
                    'relationship_counts': {record['relationship_type']: record['count'] for record in rel_counts}
                }
                
        except Exception as e:
            logger.warning(f"Error getting database statistics: {str(e)}")
            return {'error': str(e)}
    
    def _generate_analysis_summary(self, layering_patterns: List, smurfing_patterns: List, 
                                  volume_anomalies: List, risk_score: Dict) -> Dict[str, Any]:
        """Generate summary of analysis results"""
        return {
            'total_anomalies': len(layering_patterns) + len(smurfing_patterns) + len(volume_anomalies),
            'layering_count': len(layering_patterns),
            'smurfing_count': len(smurfing_patterns),
            'volume_anomaly_count': len(volume_anomalies),
            'risk_level': risk_score.get('risk_level', 'UNKNOWN'),
            'risk_score': risk_score.get('total_risk_score', 0),
            'recommendations': self._generate_recommendations(risk_score)
        }
    
    def _generate_comparative_analysis(self, individual_results: List[Dict]) -> Dict[str, Any]:
        """Generate comparative analysis of multiple addresses"""
        risk_scores = [result['risk_score']['total_risk_score'] for result in individual_results]
        
        return {
            'average_risk_score': sum(risk_scores) / len(risk_scores) if risk_scores else 0,
            'highest_risk_address': max(individual_results, key=lambda x: x['risk_score']['total_risk_score']),
            'lowest_risk_address': min(individual_results, key=lambda x: x['risk_score']['total_risk_score']),
            'risk_distribution': {
                'high_risk': len([r for r in individual_results if r['risk_score']['risk_level'] in ['HIGH', 'CRITICAL']]),
                'medium_risk': len([r for r in individual_results if r['risk_score']['risk_level'] == 'MEDIUM']),
                'low_risk': len([r for r in individual_results if r['risk_score']['risk_level'] in ['LOW', 'VERY_LOW']])
            }
        }
    
    def _generate_recommendations(self, risk_score: Dict) -> List[str]:
        """Generate recommendations based on risk score"""
        recommendations = []
        risk_level = risk_score.get('risk_level', 'UNKNOWN')
        
        if risk_level in ['CRITICAL', 'HIGH']:
            recommendations.extend([
                "Immediate investigation required",
                "Consider freezing associated accounts",
                "Report to relevant authorities",
                "Monitor all related addresses"
            ])
        elif risk_level == 'MEDIUM':
            recommendations.extend([
                "Enhanced monitoring recommended",
                "Review transaction patterns",
                "Consider additional verification"
            ])
        elif risk_level in ['LOW', 'VERY_LOW']:
            recommendations.extend([
                "Standard monitoring sufficient",
                "No immediate action required"
            ])
        
        return recommendations
    
    def _get_analysis_error_result(self, address: str, error_message: str) -> Dict[str, Any]:
        """Generate error result for failed analysis"""
        return {
            'address': address,
            'error': error_message,
            'analysis_timestamp': self._get_current_timestamp(),
            'status': 'failed'
        }
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def close(self):
        """Clean up resources and close connections"""
        try:
            if hasattr(self, 'data_ingestor'):
                self.data_ingestor.close()
            logger.info("ChainBreak shutdown completed")
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
