import React from 'react';
import { motion } from 'framer-motion';
import { X, Info, Hash, Calendar, DollarSign, Network, MapPin } from 'lucide-react';

const NodeDetails = ({ node, onClose }) => {
  if (!node) return null;

  const formatValue = (value) => {
    if (typeof value === 'number') {
      if (value > 1000000) return `${(value / 1000000).toFixed(2)}M`;
      if (value > 1000) return `${(value / 1000).toFixed(2)}K`;
      return value.toString();
    }
    return value;
  };

  const getNodeType = (node) => {
    if (node.type) return node.type;
    if (node.id && node.id.length > 40) return 'Transaction';
    return 'Address';
  };

  const getNodeIcon = (nodeType) => {
    switch (nodeType) {
      case 'transaction':
        return <Hash className="w-5 h-5 text-blue-400" />;
      case 'address':
        return <MapPin className="w-5 h-5 text-green-400" />;
      default:
        return <Info className="w-5 h-5 text-gray-400" />;
    }
  };

  const nodeType = getNodeType(node);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
      className="bg-gray-800/90 backdrop-blur-sm rounded-lg border border-gray-700/50 p-6 shadow-2xl"
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          {getNodeIcon(nodeType)}
          <h3 className="text-lg font-semibold text-white">Node Details</h3>
        </div>
        <button
          onClick={onClose}
          className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      <div className="space-y-4">
        {/* Node ID */}
        <div className="p-3 bg-gray-700/30 rounded-lg">
          <div className="flex items-center space-x-2 mb-2">
            <Hash className="w-4 h-4 text-blue-400" />
            <span className="text-sm font-medium text-gray-300">Node ID</span>
          </div>
          <p className="text-sm text-gray-200 font-mono break-all">
            {node.id}
          </p>
        </div>

      </div>

      <div className="mt-6 p-3 bg-blue-900/20 border border-blue-500/30 rounded-lg">
        <p className="text-sm text-blue-300">
          <strong>Tip:</strong> Click on different nodes in the graph to view their details.
        </p>
      </div>
    </motion.div>
  );
};

export default NodeDetails;
