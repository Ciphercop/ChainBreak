import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  X, 
  Download, 
  Trash2, 
  Filter, 
  Search,
  AlertTriangle,
  Info,
  CheckCircle,
  XCircle,
  AlertCircle
} from 'lucide-react';
import logger from '../utils/logger';

const LogViewer = ({ onClose }) => {
  const [logs, setLogs] = useState([]);
  const [filterLevel, setFilterLevel] = useState('ALL');
  const [searchTerm, setSearchTerm] = useState('');
  const [autoScroll, setAutoScroll] = useState(true);
  const logContainerRef = useRef(null);

  useEffect(() => {
    const handleLogEvent = (event) => {
      setLogs(prevLogs => [...prevLogs, event.detail]);
    };

    window.addEventListener('log', handleLogEvent);
    
    const interval = setInterval(() => {
      const currentLogs = logger.getLogs();
      setLogs(currentLogs);
    }, 1000);

    return () => {
      window.removeEventListener('log', handleLogEvent);
      clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    if (autoScroll && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs, autoScroll]);

  const filteredLogs = logs.filter(log => {
    if (filterLevel !== 'ALL' && log.level !== filterLevel) return false;
    if (searchTerm && !log.message.toLowerCase().includes(searchTerm.toLowerCase())) return false;
    return true;
  });

  const getLevelIcon = (level) => {
    switch (level) {
      case 'DEBUG':
        return <Info className="w-4 h-4 text-blue-400" />;
      case 'INFO':
        return <CheckCircle className="w-4 h-4 text-green-400" />;
      case 'WARN':
        return <AlertTriangle className="w-4 h-4 text-yellow-400" />;
      case 'ERROR':
        return <XCircle className="w-4 h-4 text-red-400" />;
      case 'CRITICAL':
        return <AlertCircle className="w-4 h-4 text-red-600" />;
      default:
        return <Info className="w-4 h-4 text-gray-400" />;
    }
  };

  const getLevelColor = (level) => {
    switch (level) {
      case 'DEBUG':
        return 'text-blue-400';
      case 'INFO':
        return 'text-green-400';
      case 'WARN':
        return 'text-yellow-400';
      case 'ERROR':
        return 'text-red-400';
      case 'CRITICAL':
        return 'text-red-600';
      default:
        return 'text-gray-400';
    }
  };

  const formatTimestamp = (timestamp) => {
    try {
      return new Date(timestamp).toLocaleTimeString();
    } catch {
      return timestamp;
    }
  };

  const exportLogs = () => {
    logger.exportLogs();
  };

  const clearLogs = () => {
    logger.clearLogs();
    setLogs([]);
  };

  const getLogCounts = () => {
    const counts = { DEBUG: 0, INFO: 0, WARN: 0, ERROR: 0, CRITICAL: 0 };
    logs.forEach(log => {
      if (counts[log.level] !== undefined) {
        counts[log.level]++;
      }
    });
    return counts;
  };

  const logCounts = getLogCounts();

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.95, opacity: 0 }}
        className="bg-dark-800 border border-dark-700 rounded-xl w-full max-w-6xl h-[80vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between p-6 border-b border-dark-700">
          <div className="flex items-center space-x-3">
            <h2 className="text-xl font-semibold text-dark-100">Application Logs</h2>
            <div className="flex items-center space-x-2 text-sm">
              {Object.entries(logCounts).map(([level, count]) => (
                <div key={level} className={`flex items-center space-x-1 ${getLevelColor(level)}`}>
                  {getLevelIcon(level)}
                  <span>{count}</span>
                </div>
              ))}
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <button
              onClick={exportLogs}
              className="btn-secondary flex items-center space-x-2"
            >
              <Download className="w-4 h-4" />
              <span>Export</span>
            </button>
            <button
              onClick={clearLogs}
              className="btn-secondary flex items-center space-x-2"
            >
              <Trash2 className="w-4 h-4" />
              <span>Clear</span>
            </button>
            <button
              onClick={onClose}
              className="p-2 text-dark-400 hover:text-dark-200 hover:bg-dark-700 rounded-lg transition-colors duration-200"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        <div className="flex-1 p-6 space-y-4">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Filter className="w-4 h-4 text-dark-400" />
              <select
                value={filterLevel}
                onChange={(e) => setFilterLevel(e.target.value)}
                className="input-field text-sm"
              >
                <option value="ALL">All Levels</option>
                <option value="DEBUG">Debug</option>
                <option value="INFO">Info</option>
                <option value="WARN">Warning</option>
                <option value="ERROR">Error</option>
                <option value="CRITICAL">Critical</option>
              </select>
            </div>

            <div className="flex items-center space-x-2 flex-1">
              <Search className="w-4 h-4 text-dark-400" />
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Search logs..."
                className="input-field flex-1 text-sm"
              />
            </div>

            <label className="flex items-center space-x-2 text-sm text-dark-300">
              <input
                type="checkbox"
                checked={autoScroll}
                onChange={(e) => setAutoScroll(e.target.checked)}
                className="rounded border-dark-600 bg-dark-700 text-primary-500 focus:ring-primary-500"
              />
              <span>Auto-scroll</span>
            </label>
          </div>

          <div
            ref={logContainerRef}
            className="bg-dark-900 border border-dark-600 rounded-lg p-4 h-full overflow-y-auto space-y-2"
          >
            <AnimatePresence>
              {filteredLogs.length === 0 ? (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="text-center py-8 text-dark-400"
                >
                  <Info className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>No logs available</p>
                </motion.div>
              ) : (
                filteredLogs.map((log, index) => (
                  <motion.div
                    key={log.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.01 }}
                    className="flex items-start space-x-3 p-3 hover:bg-dark-700 rounded-lg transition-colors duration-200"
                  >
                    <div className="flex-shrink-0 mt-0.5">
                      {getLevelIcon(log.level)}
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center space-x-2 mb-1">
                        <span className={`text-xs font-medium ${getLevelColor(log.level)}`}>
                          {log.level}
                        </span>
                        <span className="text-xs text-dark-400">
                          {formatTimestamp(log.timestamp)}
                        </span>
                      </div>
                      
                      <p className="text-sm text-dark-200 break-words">
                        {log.message}
                      </p>
                      
                      {log.data && (
                        <details className="mt-2">
                          <summary className="text-xs text-dark-400 cursor-pointer hover:text-dark-300">
                            View Details
                          </summary>
                          <pre className="mt-2 text-xs text-dark-300 bg-dark-800 p-2 rounded border border-dark-600 overflow-x-auto">
                            {JSON.stringify(log.data, null, 2)}
                          </pre>
                        </details>
                      )}
                    </div>
                  </motion.div>
                ))
              )}
            </AnimatePresence>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default LogViewer;
