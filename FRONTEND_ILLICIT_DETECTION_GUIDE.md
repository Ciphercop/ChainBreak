# Frontend Illicit Detection Integration Guide

## 🚀 **Complete Integration Flow**

### **1. Backend API Endpoint**
The backend now has a new endpoint: `/api/illicit-analysis/run-detection`

**Purpose**: Runs illicit detection on existing graph data
**Method**: POST
**Input**: Graph data (nodes and edges)
**Output**: Complete illicit analysis results

### **2. Frontend API Integration**
The frontend API utility now includes: `chainbreakAPI.runIllicitDetection(graphData)`

## 🔄 **Step-by-Step Integration**

### **Step 1: After Graph Generation**

Once your graph is generated in the frontend, you'll have graph data in this format:

```javascript
const graphData = {
  nodes: [
    {
      id: "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
      x: 0.5,
      y: 0.3,
      // ... other node properties
    },
    // ... more nodes
  ],
  edges: [
    {
      source: "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
      target: "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2",
      value: 1.5,
      id: "tx_001",
      timestamp: "2024-01-15T10:30:00Z",
      // ... other edge properties
    },
    // ... more edges
  ]
};
```

### **Step 2: Run Illicit Detection**

```javascript
import { chainbreakAPI } from '../utils/api';

const runIllicitDetection = async (graphData) => {
  try {
    setLoading(true);
    
    // Run illicit detection on the graph
    const result = await chainbreakAPI.runIllicitDetection(graphData);
    
    if (result.success) {
      // Process the analysis results
      const analysis = result.analysis;
      
      // Update your state with illicit detection results
      setIllicitAnalysis(analysis);
      
      // Update graph visualization with risk colors
      updateGraphWithRiskData(analysis);
      
      // Show analysis summary
      showAnalysisSummary(analysis);
      
    } else {
      console.error('Illicit detection failed:', result.error);
    }
    
  } catch (error) {
    console.error('Error running illicit detection:', error);
  } finally {
    setLoading(false);
  }
};
```

### **Step 3: Process Analysis Results**

The analysis results contain comprehensive illicit detection data:

```javascript
const analysis = {
  // Address-level analysis
  addresses: {
    "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa": {
      risk_level: "HIGH",
      risk_score: 0.85,
      transaction_count: 150,
      total_sent: 25.5,
      total_received: 30.2,
      suspicious_patterns: ["mixing", "rapid_transfers"],
      centrality_measures: {
        degree_centrality: 0.75,
        betweenness_centrality: 0.60,
        eigenvector_centrality: 0.45,
        closeness_centrality: 0.55
      },
      threat_intel_data: {
        bitcoinwhoswho: {
          score: 0.8,
          tags: ["scam", "fraudulent"],
          scam_reports: 3,
          confidence: 0.9
        },
        chainalysis: {
          risk_score: 0.7,
          category: "exchange",
          confidence: 0.8
        }
      },
      sir_model_state: "I", // Susceptible, Infected, or Recovered
      sir_probability: 0.75
    }
    // ... more addresses
  },
  
  // Pattern-level analysis
  suspicious_patterns: [
    {
      pattern_type: "mixing",
      addresses: ["1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa", "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2"],
      transactions: [
        {
          tx_hash: "tx_001",
          from_address: "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
          to_address: "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2",
          value: 1.5,
          timestamp: "2024-01-15T10:30:00Z"
        }
        // ... more transactions
      ],
      confidence: 0.85,
      description: "Mixing pattern with 5 inputs and 8 outputs",
      risk_score: 0.9,
      metadata: {
        input_count: 5,
        output_count: 8,
        mixing_score: 0.75
      }
    }
    // ... more patterns
  ],
  
  // Community analysis
  clusters: {
    0: ["1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa", "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2"],
    1: ["1CvBMSEYstWetqTFn5Au4m4GFg7xJaNVN3", "1DvBMSEYstWetqTFn5Au4m4GFg7xJaNVN4"]
  },
  
  // Risk distribution
  risk_distribution: {
    "CRITICAL": 2,
    "HIGH": 5,
    "MEDIUM": 8,
    "LOW": 15,
    "CLEAN": 20
  },
  
  // High-risk addresses
  high_risk_addresses: [
    "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
    "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2"
  ],
  
  // Summary statistics
  total_transactions: 150,
  total_addresses: 50,
  analysis_timestamp: "2024-01-15T10:35:00Z",
  
  // Detection summary
  detection_summary: {
    pattern_counts: {
      "mixing": 3,
      "peel_chain": 2,
      "rapid_transfers": 5,
      "round_amounts": 8,
      "smurfing": 1,
      "layering": 2
    },
    anomaly_count: 12,
    clusters_found: 5,
    exchange_paths_found: 8
  },
  
  // SIR model results
  sir_model_results: {
    final_states: {
      "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa": "I",
      "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2": "S"
    },
    final_probabilities: {
      "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa": 0.75,
      "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2": 0.25
    },
    infected_addresses: ["1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"],
    high_risk_addresses: ["1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"]
  },
  
  // Exchange paths
  exchange_paths: {
    "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa": [
      ["1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa", "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2", "exchange_address"]
    ]
  }
};
```

### **Step 4: Update Graph Visualization**

```javascript
const updateGraphWithRiskData = (analysis) => {
  // Update node colors based on risk levels
  const riskColors = {
    'CRITICAL': '#FF0000',  // Red
    'HIGH': '#FF6600',      // Orange
    'MEDIUM': '#FFCC00',    // Yellow
    'LOW': '#00CC00',       // Green
    'CLEAN': '#0066CC'      // Blue
  };
  
  // Update node colors
  graphData.nodes.forEach(node => {
    const addressData = analysis.addresses[node.id];
    if (addressData) {
      node.color = riskColors[addressData.risk_level] || '#CCCCCC';
      node.risk_score = addressData.risk_score;
      node.risk_level = addressData.risk_level;
      node.suspicious_patterns = addressData.suspicious_patterns;
    }
  });
  
  // Update edge colors based on illicit patterns
  graphData.edges.forEach(edge => {
    // Check if this edge is part of any suspicious pattern
    const isSuspicious = analysis.suspicious_patterns.some(pattern => 
      pattern.transactions.some(tx => 
        tx.from_address === edge.source && tx.to_address === edge.target
      )
    );
    
    if (isSuspicious) {
      edge.color = '#FF0000'; // Red for suspicious
      edge.width = 3; // Thicker for suspicious
    } else {
      edge.color = '#CCCCCC'; // Gray for normal
      edge.width = 1;
    }
  });
  
  // Re-render the graph
  renderGraph(graphData);
};
```

### **Step 5: Display Analysis Summary**

```javascript
const showAnalysisSummary = (analysis) => {
  const summary = {
    totalAddresses: analysis.total_addresses,
    totalTransactions: analysis.total_transactions,
    highRiskAddresses: analysis.high_risk_addresses.length,
    suspiciousPatterns: analysis.suspicious_patterns.length,
    riskDistribution: analysis.risk_distribution,
    patternCounts: analysis.detection_summary.pattern_counts,
    clustersFound: analysis.detection_summary.clusters_found,
    anomaliesDetected: analysis.detection_summary.anomaly_count
  };
  
  // Display in your UI
  setAnalysisSummary(summary);
};
```

## 🎨 **Complete React Component Example**

```javascript
import React, { useState, useEffect } from 'react';
import { chainbreakAPI } from '../utils/api';
import * as d3 from 'd3';

const IllicitDetectionComponent = ({ graphData }) => {
  const [loading, setLoading] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const [analysisSummary, setAnalysisSummary] = useState(null);
  const [error, setError] = useState(null);

  const runIllicitDetection = async () => {
    if (!graphData || !graphData.nodes || !graphData.edges) {
      setError('No graph data available');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      
      const result = await chainbreakAPI.runIllicitDetection(graphData);
      
      if (result.success) {
        setAnalysis(result.analysis);
        updateGraphWithRiskData(result.analysis);
        showAnalysisSummary(result.analysis);
      } else {
        setError(result.error);
      }
      
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const updateGraphWithRiskData = (analysis) => {
    const riskColors = {
      'CRITICAL': '#FF0000',
      'HIGH': '#FF6600',
      'MEDIUM': '#FFCC00',
      'LOW': '#00CC00',
      'CLEAN': '#0066CC'
    };
    
    // Update nodes
    graphData.nodes.forEach(node => {
      const addressData = analysis.addresses[node.id];
      if (addressData) {
        node.color = riskColors[addressData.risk_level] || '#CCCCCC';
        node.risk_score = addressData.risk_score;
        node.risk_level = addressData.risk_level;
        node.suspicious_patterns = addressData.suspicious_patterns;
      }
    });
    
    // Update edges
    graphData.edges.forEach(edge => {
      const isSuspicious = analysis.suspicious_patterns.some(pattern => 
        pattern.transactions.some(tx => 
          tx.from_address === edge.source && tx.to_address === edge.target
        )
      );
      
      edge.color = isSuspicious ? '#FF0000' : '#CCCCCC';
      edge.width = isSuspicious ? 3 : 1;
    });
    
    // Re-render graph (implement your graph rendering logic)
    renderGraph(graphData);
  };

  const showAnalysisSummary = (analysis) => {
    const summary = {
      totalAddresses: analysis.total_addresses,
      totalTransactions: analysis.total_transactions,
      highRiskAddresses: analysis.high_risk_addresses.length,
      suspiciousPatterns: analysis.suspicious_patterns.length,
      riskDistribution: analysis.risk_distribution,
      patternCounts: analysis.detection_summary.pattern_counts
    };
    
    setAnalysisSummary(summary);
  };

  const renderGraph = (data) => {
    // Implement your D3.js graph rendering logic here
    // This is where you'd update the visual representation
    console.log('Rendering graph with risk data:', data);
  };

  return (
    <div className="illicit-detection-component">
      <h2>Illicit Transaction Detection</h2>
      
      <button 
        onClick={runIllicitDetection}
        disabled={loading || !graphData}
        className="run-detection-btn"
      >
        {loading ? 'Running Analysis...' : 'Run Illicit Detection'}
      </button>
      
      {error && (
        <div className="error-message">
          Error: {error}
        </div>
      )}
      
      {analysisSummary && (
        <div className="analysis-summary">
          <h3>Analysis Summary</h3>
          <div className="summary-stats">
            <div className="stat">
              <span className="label">Total Addresses:</span>
              <span className="value">{analysisSummary.totalAddresses}</span>
            </div>
            <div className="stat">
              <span className="label">Total Transactions:</span>
              <span className="value">{analysisSummary.totalTransactions}</span>
            </div>
            <div className="stat">
              <span className="label">High Risk Addresses:</span>
              <span className="value">{analysisSummary.highRiskAddresses}</span>
            </div>
            <div className="stat">
              <span className="label">Suspicious Patterns:</span>
              <span className="value">{analysisSummary.suspiciousPatterns}</span>
            </div>
          </div>
          
          <div className="risk-distribution">
            <h4>Risk Distribution</h4>
            {Object.entries(analysisSummary.riskDistribution).map(([level, count]) => (
              <div key={level} className="risk-level">
                <span className="level">{level}:</span>
                <span className="count">{count}</span>
              </div>
            ))}
          </div>
          
          <div className="pattern-counts">
            <h4>Pattern Types Detected</h4>
            {Object.entries(analysisSummary.patternCounts).map(([pattern, count]) => (
              <div key={pattern} className="pattern-type">
                <span className="pattern">{pattern}:</span>
                <span className="count">{count}</span>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {analysis && (
        <div className="detailed-analysis">
          <h3>Detailed Analysis</h3>
          
          <div className="high-risk-addresses">
            <h4>High Risk Addresses</h4>
            {analysis.high_risk_addresses.map(address => {
              const addressData = analysis.addresses[address];
              return (
                <div key={address} className="address-card">
                  <div className="address">{address}</div>
                  <div className="risk-level">{addressData.risk_level}</div>
                  <div className="risk-score">Score: {addressData.risk_score.toFixed(3)}</div>
                  <div className="patterns">
                    Patterns: {addressData.suspicious_patterns.join(', ')}
                  </div>
                </div>
              );
            })}
          </div>
          
          <div className="suspicious-patterns">
            <h4>Suspicious Patterns</h4>
            {analysis.suspicious_patterns.map((pattern, index) => (
              <div key={index} className="pattern-card">
                <div className="pattern-type">{pattern.pattern_type}</div>
                <div className="description">{pattern.description}</div>
                <div className="confidence">Confidence: {pattern.confidence.toFixed(3)}</div>
                <div className="risk-score">Risk Score: {pattern.risk_score.toFixed(3)}</div>
                <div className="addresses">
                  Addresses: {pattern.addresses.join(', ')}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default IllicitDetectionComponent;
```

## 🎯 **Key Features**

### **1. Real-time Analysis**
- Runs illicit detection on existing graph data
- No need to regenerate the graph
- Fast analysis using pre-computed graph structure

### **2. Comprehensive Results**
- **Address-level analysis**: Risk scores, patterns, threat intelligence
- **Pattern detection**: 7 different suspicious pattern types
- **Community analysis**: Cluster detection and analysis
- **Risk distribution**: Statistical breakdown by risk level
- **SIR modeling**: Activity propagation simulation
- **Exchange paths**: Path finding to known exchanges

### **3. Visual Integration**
- **Risk-based coloring**: Nodes colored by risk level
- **Pattern highlighting**: Edges highlighted for suspicious patterns
- **Interactive exploration**: Hover for detailed information
- **Real-time updates**: Graph updates with analysis results

### **4. User Experience**
- **Loading states**: Clear feedback during analysis
- **Error handling**: Graceful error management
- **Summary display**: Quick overview of results
- **Detailed analysis**: In-depth exploration of findings

## 🚀 **Usage Flow**

1. **Generate Graph**: Create your transaction graph
2. **Run Detection**: Click "Run Illicit Detection" button
3. **View Results**: See analysis summary and detailed results
4. **Explore Graph**: Interact with risk-colored graph
5. **Investigate**: Click on high-risk addresses for details

This integration provides a complete illicit transaction detection system that works seamlessly with your existing graph visualization! 🎯
