import React, { useState, useEffect } from 'react';
import { Shield, AlertTriangle, CheckCircle, XCircle, Loader2, Info } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import logger from '../utils/logger';
import toast from 'react-hot-toast';

const ThreatIntelligencePanel = ({ graphData, onThreatIntelUpdate }) => {
  const [threatIntelStatus, setThreatIntelStatus] = useState(null);
  const [illicitAddresses, setIllicitAddresses] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [lastChecked, setLastChecked] = useState(null);

  // Check threat intelligence status on component mount
  useEffect(() => {
    checkThreatIntelStatus();
  }, []);

  // Check for illicit addresses when graph data changes
  useEffect(() => {
    if (graphData && graphData.nodes && graphData.nodes.length > 0) {
      checkIllicitAddresses();
    }
  }, [graphData]);

  const checkThreatIntelStatus = async () => {
    try {
      const response = await fetch('/api/threat-intelligence/status');
      const result = await response.json();
      
      if (result.success) {
        setThreatIntelStatus(result.data);
        logger.info('Threat intelligence status loaded', result.data);
      } else {
        logger.error('Failed to load threat intelligence status', result.error);
        toast.error('Failed to load threat intelligence status');
      }
    } catch (error) {
      logger.error('Error checking threat intelligence status', error);
      toast.error('Error checking threat intelligence status');
    }
  };

  const checkIllicitAddresses = async () => {
    if (!graphData || !graphData.nodes) return;
    
    setIsLoading(true);
    try {
      const response = await fetch('/api/threat-intelligence/check-graph', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ graph_data: graphData }),
      });
      
      const result = await response.json();
      
      if (result.success) {
        const data = result.data;
        setIllicitAddresses(data.illicit_addresses || []);
        setLastChecked(new Date());
        
        if (data.illicit_addresses && data.illicit_addresses.length > 0) {
          toast.error(`🚨 Found ${data.illicit_addresses.length} illicit addresses!`);
          logger.warn('Illicit addresses detected', data.illicit_addresses);
          
          // Notify parent component
          if (onThreatIntelUpdate) {
            onThreatIntelUpdate({
              illicitAddresses: data.illicit_addresses,
              totalAddresses: data.total_addresses,
              illicitPercentage: data.illicit_percentage
            });
          }
        } else {
          toast.success('✅ All addresses verified clean');
          logger.info('All addresses verified clean by threat intelligence');
          
          if (onThreatIntelUpdate) {
            onThreatIntelUpdate({
              illicitAddresses: [],
              totalAddresses: data.total_addresses,
              illicitPercentage: 0
            });
          }
        }
      } else {
        logger.error('Failed to check illicit addresses', result.error);
        toast.error('Failed to check addresses for illicit activity');
      }
    } catch (error) {
      logger.error('Error checking illicit addresses', error);
      toast.error('Error checking addresses for illicit activity');
    } finally {
      setIsLoading(false);
    }
  };

  const getRiskLevelColor = (riskLevel) => {
    switch (riskLevel?.toLowerCase()) {
      case 'critical':
        return 'text-red-500';
      case 'high':
        return 'text-orange-500';
      case 'medium':
        return 'text-yellow-500';
      case 'low':
        return 'text-green-500';
      default:
        return 'text-gray-500';
    }
  };

  const getRiskLevelIcon = (riskLevel) => {
    switch (riskLevel?.toLowerCase()) {
      case 'critical':
        return <XCircle className="w-4 h-4 text-red-500" />;
      case 'high':
        return <AlertTriangle className="w-4 h-4 text-orange-500" />;
      case 'medium':
        return <AlertTriangle className="w-4 h-4 text-yellow-500" />;
      case 'low':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      default:
        return <Info className="w-4 h-4 text-gray-500" />;
    }
  };

  if (!threatIntelStatus) {
    return (
      <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <div className="flex items-center gap-2 text-gray-400">
          <Loader2 className="w-4 h-4 animate-spin" />
          <span>Loading threat intelligence status...</span>
        </div>
      </div>
    );
  }

  if (!threatIntelStatus.available) {
    return (
      <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <div className="flex items-center gap-2 text-gray-400">
          <XCircle className="w-4 h-4" />
          <span>Threat intelligence not available</span>
        </div>
        <p className="text-sm text-gray-500 mt-2">
          {threatIntelStatus.error || 'Threat intelligence services are not configured'}
        </p>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700">
      {/* Header */}
      <div 
        className="p-4 cursor-pointer hover:bg-gray-750 transition-colors"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Shield className="w-5 h-5 text-blue-400" />
            <h3 className="text-lg font-semibold text-white">Threat Intelligence</h3>
            {illicitAddresses.length > 0 && (
              <span className="bg-red-500 text-white text-xs px-2 py-1 rounded-full">
                {illicitAddresses.length} illicit
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            {isLoading && <Loader2 className="w-4 h-4 animate-spin text-blue-400" />}
            <motion.div
              animate={{ rotate: isExpanded ? 180 : 0 }}
              transition={{ duration: 0.2 }}
            >
              <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </motion.div>
          </div>
        </div>
        
        {/* Status summary */}
        <div className="mt-2 flex items-center gap-4 text-sm">
          <div className="flex items-center gap-1">
            <CheckCircle className="w-4 h-4 text-green-500" />
            <span className="text-gray-300">Active</span>
          </div>
          {lastChecked && (
            <div className="text-gray-500">
              Last checked: {lastChecked.toLocaleTimeString()}
            </div>
          )}
        </div>
      </div>

      {/* Expandable content */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 space-y-4">
              {/* Source status */}
              <div>
                <h4 className="text-sm font-medium text-gray-300 mb-2">Threat Intelligence Sources</h4>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {threatIntelStatus.sources && Object.entries(threatIntelStatus.sources).map(([source, info]) => (
                    <div key={source} className="flex items-center gap-2 p-2 bg-gray-700 rounded">
                      <div className={`w-2 h-2 rounded-full ${info.enabled ? 'bg-green-500' : 'bg-gray-500'}`} />
                      <span className="text-gray-300 capitalize">{source}</span>
                      <span className="text-gray-500 text-xs">{info.type}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Illicit addresses */}
              {illicitAddresses.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium text-red-400 mb-2">
                    🚨 Illicit Addresses Detected ({illicitAddresses.length})
                  </h4>
                  <div className="space-y-2 max-h-48 overflow-y-auto">
                    {illicitAddresses.map((addr, index) => (
                      <motion.div
                        key={addr.address}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className="p-3 bg-red-900/20 border border-red-500/30 rounded-lg"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            {getRiskLevelIcon(addr.risk_level)}
                            <span className="font-mono text-sm text-red-300">
                              {addr.address.substring(0, 20)}...
                            </span>
                          </div>
                          <span className={`text-xs font-medium ${getRiskLevelColor(addr.risk_level)}`}>
                            {addr.risk_level?.toUpperCase()}
                          </span>
                        </div>
                        
                        <div className="text-xs text-gray-400 space-y-1">
                          <div>Confidence: {(addr.confidence * 100).toFixed(1)}%</div>
                          <div>Sources: {addr.sources.join(', ')}</div>
                          {addr.illicit_activity_analysis && (
                            <div className="mt-2 space-y-1">
                              <div className="text-yellow-400">
                                <strong>Primary Activity:</strong> {addr.illicit_activity_analysis.primary_activity_type?.replace(/_/g, ' ').toUpperCase()}
                              </div>
                              {addr.illicit_activity_analysis.secondary_activities?.length > 0 && (
                                <div className="text-orange-400">
                                  <strong>Secondary:</strong> {addr.illicit_activity_analysis.secondary_activities.map(sa => sa.type.replace(/_/g, ' ')).join(', ')}
                                </div>
                              )}
                              {addr.illicit_activity_analysis.risk_indicators?.length > 0 && (
                                <div className="text-gray-300">
                                  <strong>Evidence:</strong> {addr.illicit_activity_analysis.risk_indicators.slice(0, 2).join(', ')}
                                  {addr.illicit_activity_analysis.risk_indicators.length > 2 && '...'}
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                        
                        {addr.details && addr.details.detailed_results && (
                          <div className="mt-2 text-xs text-gray-500">
                            <details>
                              <summary className="cursor-pointer hover:text-gray-400">
                                View detailed results
                              </summary>
                              <pre className="mt-1 p-2 bg-gray-800 rounded text-xs overflow-x-auto">
                                {JSON.stringify(addr.details.detailed_results, null, 2)}
                              </pre>
                            </details>
                          </div>
                        )}
                      </motion.div>
                    ))}
                  </div>
                </div>
              )}

              {/* Clean addresses summary */}
              {illicitAddresses.length === 0 && lastChecked && (
                <div className="text-center py-4">
                  <CheckCircle className="w-8 h-8 text-green-500 mx-auto mb-2" />
                  <p className="text-green-400 font-medium">All addresses verified clean</p>
                  <p className="text-gray-500 text-sm">
                    No illicit activity detected by threat intelligence sources
                  </p>
                </div>
              )}

              {/* Manual check button */}
              <div className="pt-2 border-t border-gray-700">
                <button
                  onClick={checkIllicitAddresses}
                  disabled={isLoading || !graphData}
                  className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Checking...
                    </>
                  ) : (
                    <>
                      <Shield className="w-4 h-4" />
                      Re-check Addresses
                    </>
                  )}
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default ThreatIntelligencePanel;
