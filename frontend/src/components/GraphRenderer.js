import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Loader2, AlertCircle, Info, Zap, Users, GitBranch } from 'lucide-react';
import logger from '../utils/logger';

const GraphRenderer = ({ graphData, onNodeClick, onError, className = '' }) => {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [graphStats, setGraphStats] = useState({ nodes: 0, edges: 0 });
  const [layoutProgress, setLayoutProgress] = useState(0);
  const [isLayoutComplete, setIsLayoutComplete] = useState(false);
  const containerRef = useRef(null);
  const sigmaRef = useRef(null);
  const graphRef = useRef(null);

  const processedGraphData = useMemo(() => {
    if (!graphData) return null;
    
    try {
      const startTime = performance.now();
      
      const nodes = graphData.nodes || [];
      const edges = graphData.edges || [];
      
      const processedNodes = nodes.map((node, index) => ({
        ...node,
        x: Math.cos((index * 2 * Math.PI) / nodes.length) * 100,
        y: Math.sin((index * 2 * Math.PI) / nodes.length) * 100,
        size: node.type === 'transaction' ? 3 : 5,
        color: node.type === 'transaction' ? '#60a5fa' : '#34d399',
        label: node.label || node.id.substring(0, 10),
        originalLabel: node.label || node.id
      }));
      
      const processedEdges = edges.map((edge, index) => ({
        ...edge,
        id: edge.id || `${edge.source}->${edge.target}-${index}`,
        size: 1,
        color: edge.type === 'SENT_FROM' ? '#ef4444' : '#3b82f6',
        label: edge.type || ''
      }));
      
      const endTime = performance.now();
      logger.logPerformance('Graph data processing', startTime, endTime, {
        nodeCount: nodes.length,
        edgeCount: edges.length
      });
      
      return {
        nodes: processedNodes,
        edges: processedEdges
      };
    } catch (err) {
      logger.error('Error processing graph data', err, { graphData });
      throw new Error('Failed to process graph data');
    }
  }, [graphData]);

  const initializeGraph = useCallback(async () => {
    if (!processedGraphData || !containerRef.current) return;

    try {
      const { Sigma, Graph } = await import('sigma');
      const { forceAtlas2 } = await import('graphology-layout-forceatlas2');
      
      const graph = new Graph();
      const { nodes, edges } = processedGraphData;
      
      nodes.forEach(node => {
        graph.addNode(node.id, node);
      });
      
      edges.forEach(edge => {
        if (graph.hasNode(edge.source) && graph.hasNode(edge.target)) {
          graph.addEdgeWithKey(edge.id, edge.source, edge.target, edge);
        }
      });
      
      graphRef.current = graph;
      
      const sigma = new Sigma(graph, containerRef.current, {
        renderEdgeLabels: true,
        labelSize: 12,
        labelWeight: 'bold',
        labelColor: '#e2e8f0',
        zIndex: true,
        minCameraRatio: 0.1,
        maxCameraRatio: 10,
        enableEdgeHovering: true,
        enableNodeHovering: true,
        enableEdgeLabels: true,
        enableHovering: true,
        enableCamera: true,
        enableMouse: true,
        enableWheel: true,
        enableTouch: true,
        enableKeyboard: true,
        enableResize: true,
        enableFullscreen: true,
        enableMinimap: true,
        enableSearch: true,
        enableLabels: true
      });
      
      sigmaRef.current = sigma;
      
      if (onNodeClick) {
        sigma.on('clickNode', ({ node }) => {
          onNodeClick(graph.getNodeAttributes(node));
        });
      }
      
      setGraphStats({ nodes: nodes.length, edges: edges.length });
      
      const startTime = performance.now();
      
      forceAtlas2.assign(graph, {
        iterations: 100,
        adjustSizes: true,
        slowDown: 10,
        gravity: 1,
        scaling: 1,
        strongGravityMode: true,
        slowDownIterations: 0,
        outboundAttractionDistribution: false,
        linLogMode: false,
        edgeWeightInfluence: 0,
        jitterTolerance: 1,
        barnesHutOptimize: false,
        barnesHutTheta: 0.5
      });
      
      const endTime = performance.now();
      logger.logPerformance('ForceAtlas2 layout', startTime, endTime);
      
      setIsLayoutComplete(true);
      setLayoutProgress(100);
      logger.info('Graph layout completed', { 
        nodeCount: nodes.length, 
        edgeCount: edges.length 
      });
      
    } catch (err) {
      const errorMessage = 'Failed to initialize graph';
      logger.error(errorMessage, err);
      setError(errorMessage);
      onError?.(err);
    }
  }, [processedGraphData, onNodeClick, onError]);

  useEffect(() => {
    if (!processedGraphData) return;
    
    try {
      setIsLoading(true);
      setError(null);
      setLayoutProgress(0);
      setIsLayoutComplete(false);
      
      initializeGraph();
      setIsLoading(false);
    } catch (err) {
      const errorMessage = 'Failed to initialize graph';
      logger.error(errorMessage, err, { processedGraphData });
      setError(errorMessage);
      onError?.(err);
    }
  }, [processedGraphData, initializeGraph, onError]);

  useEffect(() => {
    return () => {
      if (sigmaRef.current) {
        sigmaRef.current.kill();
      }
    };
  }, []);

  if (error) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className={`flex flex-col items-center justify-center p-8 text-center ${className}`}
      >
        <AlertCircle className="w-16 h-16 text-red-500 mb-4" />
        <h3 className="text-xl font-semibold text-red-400 mb-2">Graph Loading Error</h3>
        <p className="text-dark-300 mb-4">{error}</p>
        <button
          onClick={() => window.location.reload()}
          className="btn-primary"
        >
          Retry
        </button>
      </motion.div>
    );
  }

  if (!processedGraphData) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className={`flex items-center justify-center p-8 ${className}`}
      >
        <div className="text-center">
          <Info className="w-12 h-12 text-dark-400 mx-auto mb-4" />
          <p className="text-dark-300">No graph data available</p>
        </div>
      </motion.div>
    );
  }

  return (
    <div className={`relative w-full h-full ${className}`} ref={containerRef}>
      <AnimatePresence>
        {isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-dark-900/80 backdrop-blur-sm z-10 flex items-center justify-center"
          >
            <div className="text-center">
              <Loader2 className="w-12 h-12 text-primary-500 animate-spin mx-auto mb-4" />
              <p className="text-dark-200 mb-2">Initializing Graph</p>
              <div className="w-48 bg-dark-700 rounded-full h-2">
                <div 
                  className="bg-primary-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${layoutProgress}%` }}
                />
              </div>
              <p className="text-dark-400 text-sm mt-2">{layoutProgress}%</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="absolute top-4 left-4 z-20">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="glass rounded-lg p-3"
        >
          <div className="flex items-center space-x-4 text-sm">
            <div className="flex items-center space-x-2">
              <Users className="w-4 h-4 text-blue-400" />
              <span className="text-dark-200">{graphStats.nodes}</span>
            </div>
            <div className="flex items-center space-x-2">
              <GitBranch className="w-4 h-4 text-green-400" />
              <span className="text-dark-200">{graphStats.edges}</span>
            </div>
            {isLayoutComplete && (
              <div className="flex items-center space-x-2">
                <Zap className="w-4 h-4 text-yellow-400" />
                <span className="text-dark-200">Ready</span>
              </div>
            )}
          </div>
        </motion.div>
      </div>

      <div className="w-full h-full" />
    </div>
  );
};

export default GraphRenderer;
