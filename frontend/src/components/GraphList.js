import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FileText, RefreshCw, Download, Trash2, Eye } from 'lucide-react';

const GraphList = ({ graphs, onGraphSelect, onRefresh, isLoading }) => {
  const handleGraphSelect = (graphName) => {
    onGraphSelect(graphName);
  };

  const formatGraphName = (filename) => {
    if (filename.startsWith('graph_')) {
      const address = filename.replace('graph_', '').replace('.json', '');
      return `${address.substring(0, 8)}...${address.substring(address.length - 8)}`;
    }
    return filename;
  };

  const getGraphType = (filename) => {
    if (filename.includes('bitcoin') || filename.includes('btc')) return 'Bitcoin';
    if (filename.includes('ethereum') || filename.includes('eth')) return 'Ethereum';
    return 'Blockchain';
  };

  return (
    <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-gray-700/50 p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <FileText className="w-6 h-6 text-green-500" />
          <h3 className="text-lg font-semibold text-white">Available Graphs</h3>
        </div>
        <button
          onClick={onRefresh}
          disabled={isLoading}
          className="flex items-center space-x-2 px-3 py-2 bg-gray-700 text-gray-300 rounded-lg hover:bg-gray-600 disabled:opacity-50 transition-colors"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>

      <div className="space-y-3">
        {graphs.length === 0 ? (
          <div className="text-center py-8">
            <FileText className="w-12 h-12 text-gray-600 mx-auto mb-3" />
            <p className="text-gray-400 text-sm">No graphs available</p>
            <p className="text-gray-500 text-xs">Fetch a Bitcoin address to create your first graph</p>
          </div>
        ) : (
          <AnimatePresence>
            {graphs.map((graph, index) => (
              <motion.div
                key={graph}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ delay: index * 0.1 }}
                className="group p-4 bg-gray-700/30 rounded-lg border border-gray-600/30 hover:border-gray-500/50 hover:bg-gray-700/50 transition-all cursor-pointer"
                onClick={() => handleGraphSelect(graph)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                      <FileText className="w-5 h-5 text-blue-400" />
                      <h4 className="font-medium text-white group-hover:text-blue-300 transition-colors">
                        {formatGraphName(graph)}
                      </h4>
                      <span className="px-2 py-1 bg-blue-900/30 text-blue-300 text-xs rounded-full">
                        {getGraphType(graph)}
                      </span>
                    </div>
                    <p className="text-sm text-gray-400">
                      {graph.replace('.json', '')}
                    </p>
                  </div>
                  
                  <div className="flex items-center space-x-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleGraphSelect(graph);
                      }}
                      className="p-2 text-gray-400 hover:text-blue-400 hover:bg-blue-900/20 rounded-lg transition-colors"
                      title="View Graph"
                    >
                      <Eye className="w-4 h-4" />
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        // Download functionality would go here
                      }}
                      className="p-2 text-gray-400 hover:text-green-400 hover:bg-green-900/20 rounded-lg transition-colors"
                      title="Download"
                    >
                      <Download className="w-4 h-4" />
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        // Delete functionality would go here
                      }}
                      className="p-2 text-gray-400 hover:text-red-400 hover:bg-red-900/20 rounded-lg transition-colors"
                      title="Delete"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        )}
      </div>

      {graphs.length > 0 && (
        <div className="mt-4 p-3 bg-green-900/20 border border-green-500/30 rounded-lg">
          <p className="text-sm text-green-300">
            <strong>Tip:</strong> Click on any graph to load and visualize it in the main view.
          </p>
        </div>
      )}
    </div>
  );
};

export default GraphList;
