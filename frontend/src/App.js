import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Toaster, toast } from 'react-hot-toast';
import { 
  Bitcoin, 
  Database, 
  FileText, 
  Settings, 
  BarChart3, 
  Network,
  Zap,
  AlertTriangle,
  CheckCircle,
  Info
} from 'lucide-react';
import GraphRenderer from './components/GraphRenderer';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import LogViewer from './components/LogViewer';
import { chainbreakAPI } from './utils/api';
import logger from './utils/logger';
import './App.css';

function App() {
  const [currentGraph, setCurrentGraph] = useState(null);
  const [availableGraphs, setAvailableGraphs] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [backendMode, setBackendMode] = useState('unknown');
  const [systemStatus, setSystemStatus] = useState(null);
  const [showLogs, setShowLogs] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    window.toast = toast;
    initializeApp();
  }, []);

  const initializeApp = useCallback(async () => {
    try {
      logger.info('Initializing ChainBreak application');
      
      const [modeResponse, statusResponse, graphsResponse] = await Promise.all([
        chainbreakAPI.getBackendMode(),
        chainbreakAPI.getSystemStatus(),
        chainbreakAPI.listGraphs()
      ]);

      setBackendMode(modeResponse.data?.backend_mode || 'unknown');
      setSystemStatus(statusResponse.data);
      setAvailableGraphs(graphsResponse.files || []);
      
      logger.info('Application initialized successfully', {
        backendMode: modeResponse.data?.backend_mode,
        availableGraphs: graphsResponse.files?.length || 0
      });
      
      toast.success('ChainBreak loaded successfully');
    } catch (err) {
      const errorMessage = 'Failed to initialize application';
      logger.error(errorMessage, err);
      setError(errorMessage);
      toast.error(errorMessage);
    }
  }, []);

  const handleFetchGraph = useCallback(async (address, txLimit = 50) => {
    if (!address.trim()) {
      toast.error('Please enter a valid Bitcoin address');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      logger.info('Fetching graph data', { address, txLimit });
      
      const response = await chainbreakAPI.fetchAndSaveGraph(address, txLimit);
      
      if (response.success) {
        toast.success(`Graph saved as ${response.file}`);
        await refreshGraphList();
        logger.info('Graph fetched and saved successfully', { address, file: response.file });
      } else {
        throw new Error(response.error || 'Failed to fetch graph');
      }
    } catch (err) {
      const errorMessage = 'Failed to fetch graph data';
      logger.error(errorMessage, err, { address, txLimit });
      setError(errorMessage);
      toast.error(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleLoadGraph = useCallback(async (graphName) => {
    if (!graphName) return;

    setIsLoading(true);
    setError(null);

    try {
      logger.info('Loading graph', { graphName });
      
      const response = await chainbreakAPI.getGraph(graphName);
      
      if (response.nodes && response.edges) {
        setCurrentGraph(response);
        toast.success(`Graph loaded: ${graphName}`);
        logger.info('Graph loaded successfully', { 
          graphName, 
          nodeCount: response.nodes.length, 
          edgeCount: response.edges.length 
        });
      } else {
        throw new Error('Invalid graph data format');
      }
    } catch (err) {
      const errorMessage = 'Failed to load graph';
      logger.error(errorMessage, err, { graphName });
      setError(errorMessage);
      toast.error(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const refreshGraphList = useCallback(async () => {
    try {
      const response = await chainbreakAPI.listGraphs();
      setAvailableGraphs(response.files || []);
      logger.info('Graph list refreshed', { count: response.files?.length || 0 });
    } catch (err) {
      logger.error('Failed to refresh graph list', err);
    }
  }, []);

  const handleNodeClick = useCallback((node) => {
    logger.info('Node clicked', { nodeId: node.id, nodeType: node.type });
    toast.success(`Selected: ${node.label || node.id}`);
  }, []);

  const handleGraphError = useCallback((error) => {
    logger.error('Graph rendering error', error);
    setError('Graph rendering failed');
    toast.error('Graph rendering failed');
  }, []);

  const runLouvainAlgorithm = useCallback(() => {
    if (!currentGraph) {
      toast.error('No graph loaded');
      return;
    }

    try {
      logger.info('Running Louvain community detection');
      
      const graph = new window.graphology.Graph();
      
      currentGraph.nodes.forEach((node, index) => {
        graph.addNode(node.id, {
          ...node,
          x: Math.cos((index * 2 * Math.PI) / currentGraph.nodes.length) * 100,
          y: Math.sin((index * 2 * Math.PI) / currentGraph.nodes.length) * 100
        });
      });

      currentGraph.edges.forEach((edge) => {
        if (graph.hasNode(edge.source) && graph.hasNode(edge.target)) {
          graph.addEdgeWithKey(edge.id, edge.source, edge.target, edge);
        }
      });

      const communities = window.graphologyCommunitiesLouvain.louvain(graph);
      const communityColors = {};
      const colors = [
        '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF',
        '#800000', '#008000', '#000080', '#808000', '#800080', '#008080',
        '#C0C0C0', '#808080', '#9999FF', '#993366', '#FFFFCC', '#CCFFFF'
      ];

      let colorIndex = 0;
      graph.forEachNode((node, attributes) => {
        const communityId = attributes.community;
        if (!communityColors[communityId]) {
          communityColors[communityId] = colors[colorIndex % colors.length];
          colorIndex++;
        }
        graph.setNodeAttribute(node, 'color', communityColors[communityId]);
      });

      const updatedGraph = {
        ...currentGraph,
        nodes: graph.nodes().map(id => ({
          ...graph.getNodeAttributes(id),
          id
        })),
        edges: graph.edges().map(id => ({
          ...graph.getEdgeAttributes(id),
          id
        }))
      };

      setCurrentGraph(updatedGraph);
      toast.success(`Detected ${Object.keys(communityColors).length} communities`);
      logger.info('Louvain algorithm completed', { 
        communityCount: Object.keys(communityColors).length 
      });
    } catch (err) {
      const errorMessage = 'Failed to run Louvain algorithm';
      logger.error(errorMessage, err);
      toast.error(errorMessage);
    }
  }, [currentGraph]);

  const getBackendStatusColor = () => {
    switch (backendMode) {
      case 'neo4j':
        return 'text-green-400';
      case 'json':
        return 'text-yellow-400';
      default:
        return 'text-red-400';
    }
  };

  const getBackendStatusIcon = () => {
    switch (backendMode) {
      case 'neo4j':
        return <Database className="w-4 h-4" />;
      case 'json':
        return <FileText className="w-4 h-4" />;
      default:
        return <AlertTriangle className="w-4 h-4" />;
    }
  };

  if (error && !currentGraph) {
    return (
      <div className="min-h-screen bg-dark-900 flex items-center justify-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="text-center p-8"
        >
          <AlertTriangle className="w-20 h-20 text-red-500 mx-auto mb-6" />
          <h1 className="text-3xl font-bold text-red-400 mb-4">Application Error</h1>
          <p className="text-dark-300 mb-6 text-lg">{error}</p>
          <button
            onClick={initializeApp}
            className="btn-primary text-lg px-8 py-3"
          >
            Retry Initialization
          </button>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-dark-900 text-dark-100">
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#1e293b',
            color: '#e2e8f0',
            border: '1px solid #334155'
          },
          success: {
            iconTheme: {
              primary: '#10b981',
              secondary: '#1e293b'
            }
          },
          error: {
            iconTheme: {
              primary: '#ef4444',
              secondary: '#1e293b'
            }
          }
        }}
      />

      <Header
        backendMode={backendMode}
        systemStatus={systemStatus}
        onShowLogs={() => setShowLogs(true)}
      />

      <div className="flex h-screen pt-16">
        <Sidebar
          availableGraphs={availableGraphs}
          onFetchGraph={handleFetchGraph}
          onLoadGraph={handleLoadGraph}
          onRefreshGraphs={refreshGraphList}
          onRunLouvain={runLouvainAlgorithm}
          isLoading={isLoading}
          backendMode={backendMode}
        />

        <main className="flex-1 relative">
          <AnimatePresence mode="wait">
            {currentGraph ? (
              <motion.div
                key="graph"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="w-full h-full"
              >
                <GraphRenderer
                  graphData={currentGraph}
                  onNodeClick={handleNodeClick}
                  onError={handleGraphError}
                />
              </motion.div>
            ) : (
              <motion.div
                key="welcome"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="flex items-center justify-center h-full"
              >
                <div className="text-center max-w-2xl mx-auto p-8">
                  <div className="mb-8">
                    <Bitcoin className="w-24 h-24 text-primary-500 mx-auto mb-6 animate-float" />
                    <h1 className="text-5xl font-bold text-gradient mb-4">
                      ChainBreak
                    </h1>
                    <p className="text-xl text-dark-300">
                      Advanced Blockchain Transaction Analysis
                    </p>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                    <div className="card text-center">
                      <Network className="w-12 h-12 text-blue-400 mx-auto mb-4" />
                      <h3 className="text-lg font-semibold mb-2">Network Analysis</h3>
                      <p className="text-dark-400">Visualize transaction networks and patterns</p>
                    </div>
                    <div className="card text-center">
                      <BarChart3 className="w-12 h-12 text-green-400 mx-auto mb-4" />
                      <h3 className="text-lg font-semibold mb-2">Risk Assessment</h3>
                      <p className="text-dark-400">Detect anomalies and calculate risk scores</p>
                    </div>
                    <div className="card text-center">
                      <Zap className="w-12 h-12 text-yellow-400 mx-auto mb-4" />
                      <h3 className="text-lg font-semibold mb-2">Real-time Processing</h3>
                      <p className="text-dark-400">Process blockchain data efficiently</p>
                    </div>
                  </div>

                  <div className="flex items-center justify-center space-x-4 text-sm">
                    <div className={`flex items-center space-x-2 ${getBackendStatusColor()}`}>
                      {getBackendStatusIcon()}
                      <span>Backend: {backendMode.toUpperCase()}</span>
                    </div>
                    {systemStatus && (
                      <div className="flex items-center space-x-2 text-green-400">
                        <CheckCircle className="w-4 h-4" />
                        <span>System: {systemStatus.system_status}</span>
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </main>
      </div>

      <AnimatePresence>
        {showLogs && (
          <LogViewer
            onClose={() => setShowLogs(false)}
          />
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;
