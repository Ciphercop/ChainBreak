import React from 'react';
import { motion } from 'framer-motion';
import { 
  Bitcoin, 
  Database, 
  FileText, 
  AlertTriangle, 
  BarChart3,
  Settings,
  FileText as LogsIcon
} from 'lucide-react';

const Header = ({ backendMode, systemStatus, onShowLogs }) => {
  const getBackendStatusColor = () => {
    switch (backendMode) {
      case 'neo4j':
        return 'text-green-400 bg-green-400/10 border-green-400/20';
      case 'json':
        return 'text-yellow-400 bg-yellow-400/10 border-yellow-400/20';
      default:
        return 'text-red-400 bg-red-400/10 border-red-400/20';
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

  const getBackendStatusText = () => {
    switch (backendMode) {
      case 'neo4j':
        return 'Neo4j Database';
      case 'json':
        return 'JSON Backend';
      default:
        return 'Unknown Backend';
    }
  };

  return (
    <motion.header
      initial={{ y: -100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      className="fixed top-0 left-0 right-0 z-50 bg-dark-800/80 backdrop-blur-md border-b border-dark-700"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-3">
              <Bitcoin className="w-8 h-8 text-primary-500" />
              <h1 className="text-xl font-bold text-gradient">
                ChainBreak
              </h1>
            </div>
            
            <div className="hidden md:flex items-center space-x-2">
              <BarChart3 className="w-4 h-4 text-dark-400" />
              <span className="text-sm text-dark-400">Blockchain Analysis</span>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <div className={`flex items-center space-x-2 px-3 py-1.5 rounded-full border text-xs font-medium ${getBackendStatusColor()}`}>
              {getBackendStatusIcon()}
              <span className="hidden sm:inline">{getBackendStatusText()}</span>
            </div>

            {systemStatus && (
              <div className="hidden sm:flex items-center space-x-2 text-xs">
                <div className={`w-2 h-2 rounded-full ${
                  systemStatus.system_status === 'operational' ? 'bg-green-400' : 'bg-red-400'
                }`} />
                <span className="text-dark-300">
                  {systemStatus.system_status === 'operational' ? 'System Online' : 'System Offline'}
                </span>
              </div>
            )}

            <div className="flex items-center space-x-2">
              <button
                onClick={onShowLogs}
                className="p-2 text-dark-400 hover:text-dark-200 hover:bg-dark-700 rounded-lg transition-colors duration-200"
                title="View Logs"
              >
                <LogsIcon className="w-5 h-5" />
              </button>
              
              <button className="p-2 text-dark-400 hover:text-dark-200 hover:bg-dark-700 rounded-lg transition-colors duration-200">
                <Settings className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </motion.header>
  );
};

export default Header;
