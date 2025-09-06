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

        {/* Node Type */}
        <div className="p-3 bg-gray-700/30 rounded-lg">
          <div className="flex items-center space-x-2 mb-2">
            <Info className="w-4 h-4 text-green-400" />
            <span className="text-sm font-medium text-gray-300">Type</span>
          </div>
          <p className="text-sm text-gray-200 capitalize">
            {nodeType}
          </p>
        </div>

        {/* Label */}
        {node.label && (
          <div className="p-3 bg-gray-700/30 rounded-lg">
            <div className="flex items-center space-x-2 mb-2">
              <Info className="w-4 h-4 text-purple-400" />
              <span className="text-sm font-medium text-gray-300">Label</span>
            </div>
            <p className="text-sm text-gray-200">
              {node.label}
            </p>
          </div>
        )}

        {/* Size */}
        {node.size && (
          <div className="p-3 bg-gray-700/30 rounded-lg">
            <div className="flex items-center space-x-2 mb-2">
              <Network className="w-4 h-4 text-yellow-400" />
              <span className="text-sm font-medium text-gray-300">Size</span>
            </div>
            <p className="text-sm text-gray-200">
              {formatValue(node.size)}
            </p>
          </div>
        )}

        {/* Color */}
        {node.color && (
          <div className="p-3 bg-gray-700/30 rounded-lg">
            <div className="flex items-center space-x-2 mb-2">
              <div 
                className="w-4 h-4 rounded-full border border-gray-600"
                style={{ backgroundColor: node.color }}
              />
              <span className="text-sm font-medium text-gray-300">Color</span>
            </div>
            <p className="text-sm text-gray-200 font-mono">
              {node.color}
            </p>
          </div>
        )}

        {/* Position */}
        {(node.x !== undefined || node.y !== undefined) && (
          <div className="p-3 bg-gray-700/30 rounded-lg">
            <div className="flex items-center space-x-2 mb-2">
              <MapPin className="w-4 h-4 text-red-400" />
              <span className="text-sm font-medium text-gray-300">Position</span>
            </div>
            <div className="text-sm text-gray-200 space-y-1">
              {node.x !== undefined && <div>X: {node.x.toFixed(2)}</div>}
              {node.y !== undefined && <div>Y: {node.y.toFixed(2)}</div>}
            </div>
          </div>
        )}

        {/* Additional Properties */}
        {Object.keys(node).some(key => !['id', 'label', 'size', 'color', 'x', 'y', 'type'].includes(key)) && (
          <div className="p-3 bg-gray-700/30 rounded-lg">
            <div className="flex items-center space-x-2 mb-2">
              <Info className="w-4 h-4 text-indigo-400" />
              <span className="text-sm font-medium text-gray-300">Additional Properties</span>
            </div>
            <div className="space-y-2">
              {Object.entries(node).map(([key, value]) => {
                if (['id', 'label', 'size', 'color', 'x', 'y', 'type'].includes(key)) return null;
                return (
                  <div key={key} className="flex justify-between text-sm">
                    <span className="text-gray-400 capitalize">{key.replace(/_/g, ' ')}:</span>
                    <span className="text-gray-200">
                      {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        )}
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
