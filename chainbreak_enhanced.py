#!/usr/bin/env python3
"""
ChainBreak Enhanced Integration Module
Integrates all enhanced components for comprehensive illicit transaction detection

This module provides:
- Unified API for all enhanced features
- Real-time monitoring capabilities
- Comprehensive reporting
- Law enforcement dashboard integration
- Multi-blockchain support
- Advanced analytics and insights
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import json
import networkx as nx
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

# Import enhanced components
from enhanced_money_laundering_detector import EnhancedMoneyLaunderingDetector, LaunderingPatternResult, AddressRiskProfile
from enhanced_network_visualizer import EnhancedNetworkVisualizer, VisualizationConfig, VisualizationMode
from enhanced_risk_scorer import EnhancedRiskScorer, RiskAssessment, RiskLevel

logger = logging.getLogger(__name__)

@dataclass
class AnalysisConfig:
    """Configuration for comprehensive analysis"""
    enable_ml_prediction: bool = True
    enable_real_time_monitoring: bool = True
    enable_cross_blockchain: bool = True
    risk_threshold: float = 0.6
    max_addresses: int = 10000
    max_transactions: int = 100000
    analysis_timeout: int = 300  # seconds
    visualization_mode: VisualizationMode = VisualizationMode.NETWORK_3D
    export_formats: List[str] = None

    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = ['json', 'html', 'pdf']

@dataclass
class ComprehensiveAnalysisResult:
    """Comprehensive analysis result"""
    analysis_id: str
    timestamp: datetime
    addresses_analyzed: int
    transactions_analyzed: int
    laundering_patterns: List[LaunderingPatternResult]
    risk_assessments: List[RiskAssessment]
    network_graph: nx.DiGraph
    communities: Dict[str, int]
    visualizations: Dict[str, str]  # filename -> path
    reports: Dict[str, str]  # report_type -> content
    recommendations: List[str]
    summary: Dict[str, Any]
    processing_time: float
    success: bool
    errors: List[str]

class ChainBreakEnhanced:
    """
    Enhanced ChainBreak system integrating all advanced features
    """
    
    def __init__(self, config: AnalysisConfig = None):
        """Initialize the enhanced ChainBreak system"""
        self.config = config or AnalysisConfig()
        
        # Initialize components
        self.laundering_detector = EnhancedMoneyLaunderingDetector()
        self.network_visualizer = EnhancedNetworkVisualizer(
            VisualizationConfig(mode=self.config.visualization_mode)
        )
        self.risk_scorer = EnhancedRiskScorer()
        
        # Analysis state
        self.active_analyses = {}
        self.analysis_history = []
        
        logger.info("ChainBreak Enhanced system initialized")

    async def analyze_transactions_comprehensive(self, 
                                               addresses: List[str],
                                               transactions: List[Dict],
                                               blockchain: str = "btc") -> ComprehensiveAnalysisResult:
        """
        Perform comprehensive analysis of cryptocurrency transactions
        """
        analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        logger.info(f"Starting comprehensive analysis {analysis_id} for {len(addresses)} addresses")
        
        try:
            # Store analysis state
            self.active_analyses[analysis_id] = {
                'status': 'running',
                'start_time': start_time,
                'addresses': addresses,
                'progress': 0.0
            }
            
            # Step 1: Build transaction graph
            logger.info("Building transaction graph...")
            graph = self._build_transaction_graph(transactions)
            self._update_progress(analysis_id, 0.1)
            
            # Step 2: Detect money laundering patterns
            logger.info("Detecting money laundering patterns...")
            laundering_patterns = await self._detect_laundering_patterns(graph, transactions)
            self._update_progress(analysis_id, 0.3)
            
            # Step 3: Calculate risk assessments
            logger.info("Calculating risk assessments...")
            risk_assessments = await self._calculate_risk_assessments(addresses, graph, transactions, laundering_patterns)
            self._update_progress(analysis_id, 0.5)
            
            # Step 4: Community detection
            logger.info("Detecting communities...")
            communities = await self._detect_communities(graph)
            self._update_progress(analysis_id, 0.7)
            
            # Step 5: Generate visualizations
            logger.info("Generating visualizations...")
            visualizations = await self._generate_visualizations(graph, risk_assessments, communities)
            self._update_progress(analysis_id, 0.9)
            
            # Step 6: Generate reports
            logger.info("Generating reports...")
            reports = await self._generate_reports(laundering_patterns, risk_assessments, communities)
            self._update_progress(analysis_id, 1.0)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create comprehensive result
            result = ComprehensiveAnalysisResult(
                analysis_id=analysis_id,
                timestamp=start_time,
                addresses_analyzed=len(addresses),
                transactions_analyzed=len(transactions),
                laundering_patterns=laundering_patterns,
                risk_assessments=risk_assessments,
                network_graph=graph,
                communities=communities,
                visualizations=visualizations,
                reports=reports,
                recommendations=self._generate_recommendations(laundering_patterns, risk_assessments),
                summary=self._generate_summary(laundering_patterns, risk_assessments),
                processing_time=processing_time,
                success=True,
                errors=[]
            )
            
            # Store in history
            self.analysis_history.append(result)
            
            # Clean up active analysis
            del self.active_analyses[analysis_id]
            
            logger.info(f"Comprehensive analysis {analysis_id} completed in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Analysis {analysis_id} failed: {e}")
            
            # Clean up on error
            if analysis_id in self.active_analyses:
                del self.active_analyses[analysis_id]
            
            return ComprehensiveAnalysisResult(
                analysis_id=analysis_id,
                timestamp=start_time,
                addresses_analyzed=len(addresses),
                transactions_analyzed=len(transactions),
                laundering_patterns=[],
                risk_assessments=[],
                network_graph=nx.DiGraph(),
                communities={},
                visualizations={},
                reports={},
                recommendations=[],
                summary={},
                processing_time=(datetime.now() - start_time).total_seconds(),
                success=False,
                errors=[str(e)]
            )

    async def monitor_addresses_real_time(self, addresses: List[str], 
                                        monitoring_duration: int = 3600) -> Dict[str, Any]:
        """
        Real-time monitoring of addresses for suspicious activity
        """
        logger.info(f"Starting real-time monitoring for {len(addresses)} addresses")
        
        monitoring_results = {
            'start_time': datetime.now(),
            'addresses': addresses,
            'alerts': [],
            'risk_changes': [],
            'new_patterns': [],
            'monitoring_active': True
        }
        
        # Start monitoring loop
        end_time = datetime.now() + timedelta(seconds=monitoring_duration)
        
        while datetime.now() < end_time and monitoring_results['monitoring_active']:
            try:
                # Check for new transactions (simplified - in production, this would poll blockchain APIs)
                new_transactions = await self._get_new_transactions(addresses)
                
                if new_transactions:
                    # Analyze new transactions
                    analysis_result = await self.analyze_transactions_comprehensive(
                        addresses, new_transactions
                    )
                    
                    # Check for alerts
                    alerts = self._check_for_alerts(analysis_result)
                    monitoring_results['alerts'].extend(alerts)
                    
                    # Check for risk changes
                    risk_changes = self._check_risk_changes(analysis_result)
                    monitoring_results['risk_changes'].extend(risk_changes)
                    
                    # Check for new patterns
                    new_patterns = self._check_new_patterns(analysis_result)
                    monitoring_results['new_patterns'].extend(new_patterns)
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(60)
        
        monitoring_results['end_time'] = datetime.now()
        monitoring_results['monitoring_active'] = False
        
        logger.info("Real-time monitoring completed")
        return monitoring_results

    def create_law_enforcement_dashboard(self, analysis_result: ComprehensiveAnalysisResult) -> str:
        """
        Create comprehensive law enforcement dashboard
        """
        logger.info("Creating law enforcement dashboard")
        
        # Prepare data for dashboard
        dashboard_data = {
            'analysis_id': analysis_result.analysis_id,
            'timestamp': analysis_result.timestamp.isoformat(),
            'summary': analysis_result.summary,
            'critical_addresses': [
                {
                    'address': assessment.address,
                    'risk_level': assessment.risk_level.name,
                    'risk_score': assessment.overall_risk_score,
                    'patterns': len(assessment.risk_factors),
                    'recommendations': assessment.recommendations
                }
                for assessment in analysis_result.risk_assessments
                if assessment.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]
            ],
            'laundering_patterns': [
                {
                    'pattern_type': pattern.pattern_type.value,
                    'confidence': pattern.confidence,
                    'risk_score': pattern.risk_score,
                    'description': pattern.description,
                    'addresses_count': len(pattern.addresses_involved)
                }
                for pattern in analysis_result.laundering_patterns
            ],
            'visualizations': analysis_result.visualizations,
            'recommendations': analysis_result.recommendations
        }
        
        # Create dashboard using the network visualizer
        dashboard_app = self.network_visualizer.create_law_enforcement_dashboard(dashboard_data)
        
        # Export dashboard
        dashboard_filename = f"law_enforcement_dashboard_{analysis_result.analysis_id}.html"
        dashboard_app.run_server(debug=False, port=8050)
        
        logger.info(f"Law enforcement dashboard created: {dashboard_filename}")
        return dashboard_filename

    def export_analysis_results(self, analysis_result: ComprehensiveAnalysisResult, 
                              export_dir: str = "exports") -> Dict[str, str]:
        """
        Export analysis results in multiple formats
        """
        logger.info(f"Exporting analysis results for {analysis_result.analysis_id}")
        
        import os
        os.makedirs(export_dir, exist_ok=True)
        
        exported_files = {}
        
        # Export JSON data
        if 'json' in self.config.export_formats:
            json_filename = os.path.join(export_dir, f"analysis_{analysis_result.analysis_id}.json")
            with open(json_filename, 'w') as f:
                json.dump(asdict(analysis_result), f, indent=2, default=str)
            exported_files['json'] = json_filename
        
        # Export risk assessments
        for assessment in analysis_result.risk_assessments:
            if assessment.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
                risk_filename = self.risk_scorer.export_risk_assessment(assessment, 'json')
                exported_files[f'risk_{assessment.address}'] = risk_filename
        
        # Export visualizations
        for viz_type, viz_path in analysis_result.visualizations.items():
            exported_files[f'viz_{viz_type}'] = viz_path
        
        # Export reports
        for report_type, report_content in analysis_result.reports.items():
            report_filename = os.path.join(export_dir, f"report_{report_type}_{analysis_result.analysis_id}.txt")
            with open(report_filename, 'w') as f:
                f.write(report_content)
            exported_files[f'report_{report_type}'] = report_filename
        
        logger.info(f"Exported {len(exported_files)} files for analysis {analysis_result.analysis_id}")
        return exported_files

    def get_analysis_status(self, analysis_id: str) -> Dict[str, Any]:
        """Get status of a running analysis"""
        if analysis_id in self.active_analyses:
            return self.active_analyses[analysis_id]
        else:
            # Check if analysis is in history
            for analysis in self.analysis_history:
                if analysis.analysis_id == analysis_id:
                    return {
                        'status': 'completed',
                        'success': analysis.success,
                        'processing_time': analysis.processing_time,
                        'timestamp': analysis.timestamp.isoformat()
                    }
            return {'status': 'not_found'}

    def get_analysis_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get analysis history"""
        recent_analyses = self.analysis_history[-limit:]
        return [
            {
                'analysis_id': analysis.analysis_id,
                'timestamp': analysis.timestamp.isoformat(),
                'addresses_analyzed': analysis.addresses_analyzed,
                'transactions_analyzed': analysis.transactions_analyzed,
                'success': analysis.success,
                'processing_time': analysis.processing_time,
                'summary': analysis.summary
            }
            for analysis in recent_analyses
        ]

    async def _detect_laundering_patterns(self, graph: nx.DiGraph, transactions: List[Dict]) -> List[LaunderingPatternResult]:
        """Detect money laundering patterns"""
        patterns = []
        
        # Run all detection algorithms
        patterns.extend(self.laundering_detector.detect_advanced_layering(graph, transactions))
        patterns.extend(self.laundering_detector.detect_mixing_services(graph, transactions))
        patterns.extend(self.laundering_detector.detect_structured_transactions(transactions))
        patterns.extend(self.laundering_detector.detect_round_tripping(graph, transactions))
        patterns.extend(self.laundering_detector.detect_chain_hopping(transactions))
        
        return patterns

    async def _calculate_risk_assessments(self, addresses: List[str], graph: nx.DiGraph, 
                                       transactions: List[Dict], 
                                       laundering_patterns: List[LaunderingPatternResult]) -> List[RiskAssessment]:
        """Calculate comprehensive risk assessments"""
        assessments = []
        
        for address in addresses:
            assessment = self.risk_scorer.calculate_comprehensive_risk_score(
                address, graph, transactions, laundering_patterns
            )
            assessments.append(assessment)
        
        return assessments

    async def _detect_communities(self, graph: nx.DiGraph) -> Dict[str, int]:
        """Detect communities in the transaction graph"""
        try:
            # Convert to undirected graph for community detection
            undirected_graph = graph.to_undirected()
            
            # Use Louvain algorithm
            import community as community_louvain
            communities = community_louvain.best_partition(undirected_graph)
            
            return communities
        except Exception as e:
            logger.warning(f"Community detection failed: {e}")
            return {}

    async def _generate_visualizations(self, graph: nx.DiGraph, 
                                     risk_assessments: List[RiskAssessment],
                                     communities: Dict[str, int]) -> Dict[str, str]:
        """Generate visualizations"""
        visualizations = {}
        
        # Convert risk assessments to format expected by visualizer
        risk_profiles = {
            assessment.address: {
                'risk_score': assessment.overall_risk_score,
                'risk_level': assessment.risk_level.name,
                'transaction_count': assessment.behavioral_profile.get('transaction_count', 0),
                'transaction_volume': assessment.behavioral_profile.get('total_volume', 0.0)
            }
            for assessment in risk_assessments
        }
        
        try:
            # 3D Network visualization
            fig_3d = self.network_visualizer.create_interactive_3d_network(
                graph, risk_profiles, communities
            )
            viz_3d_path = self.network_visualizer.export_visualization(
                fig_3d, f"network_3d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            )
            visualizations['network_3d'] = viz_3d_path
            
            # Risk heatmap
            fig_heatmap = self.network_visualizer.create_risk_heatmap(risk_profiles)
            viz_heatmap_path = self.network_visualizer.export_visualization(
                fig_heatmap, f"risk_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            )
            visualizations['risk_heatmap'] = viz_heatmap_path
            
            # Community analysis
            if communities:
                fig_community = self.network_visualizer.create_community_analysis(
                    graph, communities, risk_profiles
                )
                viz_community_path = self.network_visualizer.export_visualization(
                    fig_community, f"community_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                )
                visualizations['community_analysis'] = viz_community_path
            
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
        
        return visualizations

    async def _generate_reports(self, laundering_patterns: List[LaunderingPatternResult],
                              risk_assessments: List[RiskAssessment],
                              communities: Dict[str, int]) -> Dict[str, str]:
        """Generate comprehensive reports"""
        reports = {}
        
        # Executive summary
        reports['executive_summary'] = self._generate_executive_summary(
            laundering_patterns, risk_assessments
        )
        
        # Technical analysis report
        reports['technical_analysis'] = self._generate_technical_report(
            laundering_patterns, risk_assessments, communities
        )
        
        # Law enforcement report
        reports['law_enforcement'] = self._generate_law_enforcement_report(
            laundering_patterns, risk_assessments
        )
        
        # Compliance report
        reports['compliance'] = self._generate_compliance_report(risk_assessments)
        
        return reports

    def _build_transaction_graph(self, transactions: List[Dict]) -> nx.DiGraph:
        """Build transaction graph from transaction data"""
        graph = nx.DiGraph()
        
        for tx in transactions:
            from_addr = tx.get('from_address')
            to_addr = tx.get('to_address')
            
            if from_addr and to_addr:
                graph.add_edge(from_addr, to_addr, **tx)
        
        return graph

    def _update_progress(self, analysis_id: str, progress: float):
        """Update analysis progress"""
        if analysis_id in self.active_analyses:
            self.active_analyses[analysis_id]['progress'] = progress

    def _generate_recommendations(self, laundering_patterns: List[LaunderingPatternResult],
                                risk_assessments: List[RiskAssessment]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Critical risk addresses
        critical_addresses = [a for a in risk_assessments if a.risk_level == RiskLevel.CRITICAL]
        if critical_addresses:
            recommendations.append(f"Immediate investigation required for {len(critical_addresses)} critical risk addresses")
        
        # High-confidence laundering patterns
        high_confidence_patterns = [p for p in laundering_patterns if p.confidence > 0.8]
        if high_confidence_patterns:
            recommendations.append(f"Monitor {len(high_confidence_patterns)} high-confidence laundering patterns")
        
        # Mixing service detection
        mixing_patterns = [p for p in laundering_patterns if p.pattern_type.value == 'mixing_service']
        if mixing_patterns:
            recommendations.append(f"Investigate {len(mixing_patterns)} potential mixing services")
        
        # Layering patterns
        layering_patterns = [p for p in laundering_patterns if p.pattern_type.value == 'layering']
        if layering_patterns:
            recommendations.append(f"Analyze {len(layering_patterns)} layering chains for money laundering")
        
        return recommendations

    def _generate_summary(self, laundering_patterns: List[LaunderingPatternResult],
                         risk_assessments: List[RiskAssessment]) -> Dict[str, Any]:
        """Generate analysis summary"""
        critical_count = len([a for a in risk_assessments if a.risk_level == RiskLevel.CRITICAL])
        high_count = len([a for a in risk_assessments if a.risk_level == RiskLevel.HIGH])
        
        pattern_types = {}
        for pattern in laundering_patterns:
            pattern_type = pattern.pattern_type.value
            pattern_types[pattern_type] = pattern_types.get(pattern_type, 0) + 1
        
        return {
            'total_addresses': len(risk_assessments),
            'critical_risk_addresses': critical_count,
            'high_risk_addresses': high_count,
            'total_patterns': len(laundering_patterns),
            'pattern_distribution': pattern_types,
            'average_risk_score': np.mean([a.overall_risk_score for a in risk_assessments]),
            'highest_risk_score': max([a.overall_risk_score for a in risk_assessments]) if risk_assessments else 0
        }

    def _check_for_alerts(self, analysis_result: ComprehensiveAnalysisResult) -> List[Dict[str, Any]]:
        """Check for alerts in analysis results"""
        alerts = []
        
        # Critical risk alerts
        critical_addresses = [a for a in analysis_result.risk_assessments if a.risk_level == RiskLevel.CRITICAL]
        for address in critical_addresses:
            alerts.append({
                'type': 'critical_risk',
                'address': address.address,
                'risk_score': address.overall_risk_score,
                'timestamp': datetime.now().isoformat(),
                'message': f"Critical risk address detected: {address.address}"
            })
        
        # High-confidence pattern alerts
        high_confidence_patterns = [p for p in analysis_result.laundering_patterns if p.confidence > 0.9]
        for pattern in high_confidence_patterns:
            alerts.append({
                'type': 'high_confidence_pattern',
                'pattern_type': pattern.pattern_type.value,
                'confidence': pattern.confidence,
                'timestamp': datetime.now().isoformat(),
                'message': f"High-confidence {pattern.pattern_type.value} pattern detected"
            })
        
        return alerts

    def _check_risk_changes(self, analysis_result: ComprehensiveAnalysisResult) -> List[Dict[str, Any]]:
        """Check for risk level changes"""
        # Simplified implementation - in production, this would compare with historical data
        return []

    def _check_new_patterns(self, analysis_result: ComprehensiveAnalysisResult) -> List[Dict[str, Any]]:
        """Check for new suspicious patterns"""
        # Simplified implementation
        return []

    async def _get_new_transactions(self, addresses: List[str]) -> List[Dict]:
        """Get new transactions for addresses (simplified implementation)"""
        # In production, this would poll blockchain APIs or use webhooks
        return []

    def _generate_executive_summary(self, laundering_patterns: List[LaunderingPatternResult],
                                   risk_assessments: List[RiskAssessment]) -> str:
        """Generate executive summary report"""
        critical_count = len([a for a in risk_assessments if a.risk_level == RiskLevel.CRITICAL])
        high_count = len([a for a in risk_assessments if a.risk_level == RiskLevel.HIGH])
        
        summary = f"""
EXECUTIVE SUMMARY - CRYPTOCURRENCY RISK ANALYSIS
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERVIEW:
- Total addresses analyzed: {len(risk_assessments)}
- Critical risk addresses: {critical_count}
- High risk addresses: {high_count}
- Suspicious patterns detected: {len(laundering_patterns)}

KEY FINDINGS:
"""
        
        if critical_count > 0:
            summary += f"- {critical_count} addresses require immediate investigation\n"
        
        if high_count > 0:
            summary += f"- {high_count} addresses require enhanced monitoring\n"
        
        pattern_types = {}
        for pattern in laundering_patterns:
            pattern_type = pattern.pattern_type.value
            pattern_types[pattern_type] = pattern_types.get(pattern_type, 0) + 1
        
        for pattern_type, count in pattern_types.items():
            summary += f"- {count} {pattern_type.replace('_', ' ')} patterns detected\n"
        
        summary += "\nRECOMMENDATIONS:\n"
        summary += "- Implement enhanced due diligence for high-risk addresses\n"
        summary += "- Monitor suspicious patterns for further investigation\n"
        summary += "- Consider regulatory reporting requirements\n"
        
        return summary

    def _generate_technical_report(self, laundering_patterns: List[LaunderingPatternResult],
                                 risk_assessments: List[RiskAssessment],
                                 communities: Dict[str, int]) -> str:
        """Generate technical analysis report"""
        report = f"""
TECHNICAL ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RISK ASSESSMENT DETAILS:
"""
        
        for assessment in risk_assessments:
            if assessment.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
                report += f"""
Address: {assessment.address}
Risk Level: {assessment.risk_level.name}
Risk Score: {assessment.overall_risk_score:.3f}
Risk Factors: {len(assessment.risk_factors)}
Recommendations: {', '.join(assessment.recommendations)}
"""
        
        report += f"""
LAUNDERING PATTERN ANALYSIS:
"""
        
        for pattern in laundering_patterns:
            report += f"""
Pattern Type: {pattern.pattern_type.value}
Confidence: {pattern.confidence:.3f}
Risk Score: {pattern.risk_score:.3f}
Addresses Involved: {len(pattern.addresses_involved)}
Description: {pattern.description}
"""
        
        report += f"""
COMMUNITY ANALYSIS:
- Total communities detected: {len(set(communities.values()))}
- Community distribution: {dict(pd.Series(list(communities.values())).value_counts())}
"""
        
        return report

    def _generate_law_enforcement_report(self, laundering_patterns: List[LaunderingPatternResult],
                                        risk_assessments: List[RiskAssessment]) -> str:
        """Generate law enforcement report"""
        report = f"""
LAW ENFORCEMENT INVESTIGATION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

INVESTIGATION SUMMARY:
"""
        
        critical_addresses = [a for a in risk_assessments if a.risk_level == RiskLevel.CRITICAL]
        high_addresses = [a for a in risk_assessments if a.risk_level == RiskLevel.HIGH]
        
        report += f"- Critical risk addresses requiring immediate attention: {len(critical_addresses)}\n"
        report += f"- High risk addresses requiring investigation: {len(high_addresses)}\n"
        report += f"- Suspicious patterns requiring analysis: {len(laundering_patterns)}\n"
        
        report += "\nCRITICAL ADDRESSES:\n"
        for address in critical_addresses:
            report += f"- {address.address} (Risk Score: {address.overall_risk_score:.3f})\n"
        
        report += "\nHIGH CONFIDENCE PATTERNS:\n"
        high_confidence_patterns = [p for p in laundering_patterns if p.confidence > 0.8]
        for pattern in high_confidence_patterns:
            report += f"- {pattern.pattern_type.value}: {pattern.description}\n"
            report += f"  Confidence: {pattern.confidence:.3f}, Risk Score: {pattern.risk_score:.3f}\n"
        
        report += "\nINVESTIGATION RECOMMENDATIONS:\n"
        report += "- Prioritize investigation of critical risk addresses\n"
        report += "- Analyze high-confidence laundering patterns\n"
        report += "- Consider coordination with other law enforcement agencies\n"
        report += "- Document all findings for potential legal proceedings\n"
        
        return report

    def _generate_compliance_report(self, risk_assessments: List[RiskAssessment]) -> str:
        """Generate compliance report"""
        report = f"""
COMPLIANCE REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

COMPLIANCE SUMMARY:
"""
        
        sar_threshold_exceeded = 0
        ctr_threshold_exceeded = 0
        structuring_detected = 0
        
        for assessment in risk_assessments:
            compliance_status = assessment.compliance_status
            if compliance_status.get('sar_threshold_exceeded', False):
                sar_threshold_exceeded += 1
            if compliance_status.get('ctr_threshold_exceeded', False):
                ctr_threshold_exceeded += 1
            if compliance_status.get('structuring_detected', False):
                structuring_detected += 1
        
        report += f"- SAR threshold exceeded: {sar_threshold_exceeded} addresses\n"
        report += f"- CTR threshold exceeded: {ctr_threshold_exceeded} addresses\n"
        report += f"- Structuring detected: {structuring_detected} addresses\n"
        
        report += "\nCOMPLIANCE VIOLATIONS:\n"
        for assessment in risk_assessments:
            violations = assessment.compliance_status.get('violations', [])
            if violations:
                report += f"- {assessment.address}: {', '.join(violations)}\n"
        
        report += "\nCOMPLIANCE RECOMMENDATIONS:\n"
        report += "- Review SAR filing requirements for threshold exceedances\n"
        report += "- Implement enhanced monitoring for structuring patterns\n"
        report += "- Update compliance procedures based on findings\n"
        
        return report

# Example usage
if __name__ == "__main__":
    # Initialize enhanced ChainBreak system
    config = AnalysisConfig(
        enable_ml_prediction=True,
        enable_real_time_monitoring=True,
        visualization_mode=VisualizationMode.NETWORK_3D
    )
    
    chainbreak = ChainBreakEnhanced(config)
    
    # Sample data
    sample_addresses = ['addr1', 'addr2', 'addr3']
    sample_transactions = [
        {
            'tx_hash': 'tx1',
            'from_address': 'addr1',
            'to_address': 'addr2',
            'value': 100000000,
            'timestamp': datetime.now().isoformat()
        },
        {
            'tx_hash': 'tx2',
            'from_address': 'addr2',
            'to_address': 'addr3',
            'value': 50000000,
            'timestamp': datetime.now().isoformat()
        }
    ]
    
    # Run comprehensive analysis
    async def run_analysis():
        result = await chainbreak.analyze_transactions_comprehensive(
            sample_addresses, sample_transactions
        )
        
        print(f"Analysis completed: {result.success}")
        print(f"Addresses analyzed: {result.addresses_analyzed}")
        print(f"Patterns detected: {len(result.laundering_patterns)}")
        print(f"Risk assessments: {len(result.risk_assessments)}")
        
        # Export results
        exported_files = chainbreak.export_analysis_results(result)
        print(f"Exported {len(exported_files)} files")
    
    # Run the analysis
    asyncio.run(run_analysis())
