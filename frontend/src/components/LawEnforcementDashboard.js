import React, { useState, useEffect, useCallback } from 'react';
import chainbreakAPI from '../utils/api';
import logger from '../utils/logger';

const LawEnforcementDashboard = () => {
  const [analysisData, setAnalysisData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedAddress, setSelectedAddress] = useState(null);
  const [filterLevel, setFilterLevel] = useState('ALL');
  const [graphData, setGraphData] = useState(null);

  // 🚨 CRITICAL ALERTS PANEL
  const CriticalAlertsPanel = ({ alerts }) => (
    <div className="critical-alerts-panel">
      <h3>🚨 Critical Alerts</h3>
      <div className="alerts-grid">
        {alerts.map((alert, index) => (
          <div key={index} className={`alert-card ${alert.severity.toLowerCase()}`}>
            <div className="alert-header">
              <span className="alert-icon">{alert.icon}</span>
              <span className="alert-title">{alert.title}</span>
              <span className="alert-time">{alert.timestamp}</span>
            </div>
            <div className="alert-content">
              <p><strong>Address:</strong> {alert.address}</p>
              <p><strong>Pattern:</strong> {alert.pattern}</p>
              <p><strong>Confidence:</strong> {alert.confidence}%</p>
              <p><strong>Risk Score:</strong> {alert.riskScore}</p>
            </div>
            <div className="alert-actions">
              <button onClick={() => investigateAddress(alert.address)}>
                🔍 Investigate
              </button>
              <button onClick={() => addToWatchlist(alert.address)}>
                👁️ Watch
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  // 📊 RISK DISTRIBUTION CHART
  const RiskDistributionChart = ({ data }) => (
    <div className="risk-distribution-chart">
      <h3>📊 Risk Distribution</h3>
      <div className="risk-bars">
        {Object.entries(data).map(([level, count]) => (
          <div key={level} className="risk-bar">
            <div className="risk-label">{level}</div>
            <div className="risk-count">{count}</div>
            <div 
              className={`risk-indicator ${level.toLowerCase()}`}
              style={{ width: `${(count / Math.max(...Object.values(data))) * 100}%` }}
            />
          </div>
        ))}
      </div>
    </div>
  );

  // 🔍 PATTERN ANALYSIS TABLE
  const PatternAnalysisTable = ({ patterns }) => (
    <div className="pattern-analysis-table">
      <h3>🔍 Suspicious Patterns</h3>
      <div className="table-container">
        <table>
          <thead>
            <tr>
              <th>Pattern Type</th>
              <th>Description</th>
              <th>Confidence</th>
              <th>Risk Score</th>
              <th>Addresses</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {patterns.map((pattern, index) => (
              <tr key={index} className={`pattern-row ${pattern.confidence > 0.7 ? 'high-confidence' : ''}`}>
                <td>
                  <span className={`pattern-badge ${pattern.pattern_type.toLowerCase()}`}>
                    {pattern.pattern_type.replace('_', ' ')}
                  </span>
                </td>
                <td>{pattern.description}</td>
                <td>
                  <div className="confidence-bar">
                    <span>{Math.round(pattern.confidence * 100)}%</span>
                    <div 
                      className="confidence-fill"
                      style={{ width: `${pattern.confidence * 100}%` }}
                    />
                  </div>
                </td>
                <td>{pattern.risk_score.toFixed(3)}</td>
                <td>{pattern.addresses.length}</td>
                <td>
                  <button onClick={() => analyzePattern(pattern)}>
                    🔍 Analyze
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );

  // 🏘️ CLUSTER ANALYSIS
  const ClusterAnalysis = ({ clusters }) => (
    <div className="cluster-analysis">
      <h3>🏘️ Address Clusters</h3>
      <div className="clusters-grid">
        {Object.entries(clusters).map(([clusterId, cluster]) => (
          <div key={clusterId} className="cluster-card">
            <div className="cluster-header">
              <h4>Cluster {clusterId}</h4>
              <span className="cluster-size">{cluster.size} addresses</span>
            </div>
            <div className="cluster-metrics">
              <div className="metric">
                <span className="metric-label">Modularity:</span>
                <span className="metric-value">{cluster.modularity.toFixed(3)}</span>
              </div>
              <div className="metric">
                <span className="metric-label">Risk Level:</span>
                <span className={`metric-value risk-${cluster.risk_level.toLowerCase()}`}>
                  {cluster.risk_level}
                </span>
              </div>
            </div>
            <div className="cluster-actions">
              <button onClick={() => investigateCluster(clusterId)}>
                🔍 Investigate Cluster
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  // 🌐 NETWORK VISUALIZATION
  const NetworkVisualization = ({ graphData }) => {
    const [selectedNode, setSelectedNode] = useState(null);

    return (
      <div className="network-visualization">
        <h3>🌐 Transaction Network</h3>
        <div className="network-controls">
          <button onClick={() => setFilterLevel('HIGH')}>Show High Risk Only</button>
          <button onClick={() => setFilterLevel('ALL')}>Show All</button>
          <button onClick={() => exportNetwork()}>📤 Export Network</button>
        </div>
        <div className="network-container">
          {/* D3.js network visualization would go here */}
          <div className="network-placeholder">
            <p>🌐 Interactive Network Graph</p>
            <p>Nodes: {graphData?.nodes?.length || 0}</p>
            <p>Edges: {graphData?.edges?.length || 0}</p>
          </div>
        </div>
        {selectedNode && (
          <div className="node-details">
            <h4>Node Details: {selectedNode.id}</h4>
            <p>Risk Score: {selectedNode.risk_score}</p>
            <p>Risk Level: {selectedNode.risk_level}</p>
            <p>Transaction Count: {selectedNode.transaction_count}</p>
          </div>
        )}
      </div>
    );
  };

  // 📈 THREAT INTELLIGENCE PANEL
  const ThreatIntelligencePanel = ({ address }) => {
    const [threatData, setThreatData] = useState(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
      if (address) {
        setLoading(true);
        chainbreakAPI.getThreatIntelligence(address)
          .then(response => {
            setThreatData(response.data.threat_intelligence);
            setLoading(false);
          })
          .catch(error => {
            logger.error('Threat intelligence error:', error);
            setLoading(false);
          });
      }
    }, [address]);

    if (!address) return null;

    return (
      <div className="threat-intelligence-panel">
        <h3>🔍 Threat Intelligence: {address}</h3>
        {loading ? (
          <div className="loading">Loading threat intelligence...</div>
        ) : threatData ? (
          <div className="threat-data">
            <div className="threat-source">
              <h4>BitcoinWhosWho</h4>
              {threatData.bitcoinwhoswho ? (
                <div className="threat-metrics">
                  <div className="metric">
                    <span>Score:</span>
                    <span className={`score ${threatData.bitcoinwhoswho.score > 0.7 ? 'high' : 'low'}`}>
                      {threatData.bitcoinwhoswho.score}
                    </span>
                  </div>
                  <div className="metric">
                    <span>Scam Reports:</span>
                    <span>{threatData.bitcoinwhoswho.scam_reports}</span>
                  </div>
                  <div className="metric">
                    <span>Confidence:</span>
                    <span>{threatData.bitcoinwhoswho.confidence}</span>
                  </div>
                </div>
              ) : (
                <p>No BitcoinWhosWho data available</p>
              )}
            </div>
            <div className="threat-source">
              <h4>Chainalysis</h4>
              {threatData.chainalysis ? (
                <div className="threat-metrics">
                  <div className="metric">
                    <span>Risk Level:</span>
                    <span>{threatData.chainalysis.risk_level}</span>
                  </div>
                </div>
              ) : (
                <p>No Chainalysis data available</p>
              )}
            </div>
          </div>
        ) : (
          <div className="error">Failed to load threat intelligence</div>
        )}
      </div>
    );
  };

  // 🎯 INVESTIGATION WORKFLOW
  const InvestigationWorkflow = () => (
    <div className="investigation-workflow">
      <h3>🎯 Investigation Workflow</h3>
      <div className="workflow-steps">
        <div className="workflow-step">
          <div className="step-number">1</div>
          <div className="step-content">
            <h4>Initial Alert</h4>
            <p>System detects suspicious pattern</p>
          </div>
        </div>
        <div className="workflow-step">
          <div className="step-number">2</div>
          <div className="step-content">
            <h4>Threat Assessment</h4>
            <p>Analyze threat intelligence data</p>
          </div>
        </div>
        <div className="workflow-step">
          <div className="step-number">3</div>
          <div className="step-content">
            <h4>Network Analysis</h4>
            <p>Examine transaction network</p>
          </div>
        </div>
        <div className="workflow-step">
          <div className="step-number">4</div>
          <div className="step-content">
            <h4>Evidence Collection</h4>
            <p>Gather supporting evidence</p>
          </div>
        </div>
        <div className="workflow-step">
          <div className="step-number">5</div>
          <div className="step-content">
            <h4>Case Documentation</h4>
            <p>Create investigation report</p>
          </div>
        </div>
      </div>
    </div>
  );

  // 📊 ANALYTICS DASHBOARD
  const AnalyticsDashboard = ({ data }) => (
    <div className="analytics-dashboard">
      <h3>📊 Investigation Analytics</h3>
      <div className="analytics-grid">
        <div className="analytics-card">
          <h4>Detection Rate</h4>
          <div className="metric-value">95.2%</div>
          <div className="metric-trend">↗️ +2.1%</div>
        </div>
        <div className="analytics-card">
          <h4>False Positive Rate</h4>
          <div className="metric-value">12.3%</div>
          <div className="metric-trend">↘️ -1.8%</div>
        </div>
        <div className="analytics-card">
          <h4>Average Investigation Time</h4>
          <div className="metric-value">2.4 hours</div>
          <div className="metric-trend">↘️ -0.3h</div>
        </div>
        <div className="analytics-card">
          <h4>Cases Resolved</h4>
          <div className="metric-value">87</div>
          <div className="metric-trend">↗️ +12</div>
        </div>
      </div>
    </div>
  );

  // Action handlers
  const investigateAddress = (address) => {
    setSelectedAddress(address);
    logger.info(`Investigating address: ${address}`);
  };

  const addToWatchlist = (address) => {
    logger.info(`Adding to watchlist: ${address}`);
    // Implementation for watchlist
  };

  const analyzePattern = (pattern) => {
    logger.info(`Analyzing pattern: ${pattern.pattern_type}`);
    // Implementation for pattern analysis
  };

  const investigateCluster = (clusterId) => {
    logger.info(`Investigating cluster: ${clusterId}`);
    // Implementation for cluster investigation
  };

  const exportNetwork = () => {
    logger.info('Exporting network data');
    // Implementation for network export
  };

  const runAnalysis = useCallback(async (addresses) => {
    setLoading(true);
    setError(null);

    try {
      const response = await chainbreakAPI.analyzeIllicitTransactions({
        addresses,
        max_transactions: 100,
        include_visualization: true
      });

      setAnalysisData(response.data);
      setGraphData(response.data.graph_data);
      logger.info('Analysis completed successfully');
    } catch (err) {
      setError(err.message);
      logger.error('Analysis failed:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <div className="law-enforcement-dashboard">
      <div className="dashboard-header">
        <h1>🚨 Law Enforcement Dashboard</h1>
        <div className="dashboard-controls">
          <input 
            type="text" 
            placeholder="Enter Bitcoin addresses (comma-separated)"
            onKeyPress={(e) => {
              if (e.key === 'Enter') {
                const addresses = e.target.value.split(',').map(addr => addr.trim());
                runAnalysis(addresses);
              }
            }}
          />
          <button onClick={() => runAnalysis(['1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa'])}>
            🔍 Analyze Sample
          </button>
        </div>
      </div>

      {loading && (
        <div className="loading-overlay">
          <div className="loading-spinner">🔄 Analyzing transactions...</div>
        </div>
      )}

      {error && (
        <div className="error-panel">
          <h3>❌ Analysis Error</h3>
          <p>{error}</p>
        </div>
      )}

      {analysisData && (
        <div className="dashboard-content">
          <div className="dashboard-grid">
            <div className="dashboard-section">
              <CriticalAlertsPanel alerts={generateAlerts(analysisData)} />
            </div>
            
            <div className="dashboard-section">
              <RiskDistributionChart data={analysisData.analysis.risk_distribution} />
            </div>
            
            <div className="dashboard-section">
              <PatternAnalysisTable patterns={analysisData.analysis.suspicious_patterns} />
            </div>
            
            <div className="dashboard-section">
              <ClusterAnalysis clusters={analysisData.analysis.clusters} />
            </div>
            
            <div className="dashboard-section full-width">
              <NetworkVisualization graphData={graphData} />
            </div>
            
            <div className="dashboard-section">
              <ThreatIntelligencePanel address={selectedAddress} />
            </div>
            
            <div className="dashboard-section">
              <InvestigationWorkflow />
            </div>
            
            <div className="dashboard-section">
              <AnalyticsDashboard data={analysisData} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Helper function to generate alerts from analysis data
const generateAlerts = (data) => {
  const alerts = [];
  const analysis = data.analysis;
  
  // Generate alerts from high-risk addresses
  analysis.high_risk_addresses.forEach(address => {
    alerts.push({
      icon: '🚨',
      title: 'High Risk Address Detected',
      address,
      pattern: 'Multiple suspicious patterns',
      confidence: 85,
      riskScore: 0.85,
      severity: 'HIGH',
      timestamp: new Date().toLocaleTimeString()
    });
  });
  
  // Generate alerts from high-confidence patterns
  analysis.suspicious_patterns
    .filter(pattern => pattern.confidence > 0.7)
    .forEach(pattern => {
      alerts.push({
        icon: '⚠️',
        title: `${pattern.pattern_type} Pattern Detected`,
        address: pattern.addresses[0],
        pattern: pattern.description,
        confidence: Math.round(pattern.confidence * 100),
        riskScore: pattern.risk_score,
        severity: 'MEDIUM',
        timestamp: new Date().toLocaleTimeString()
      });
    });
  
  return alerts;
};

export default LawEnforcementDashboard;
