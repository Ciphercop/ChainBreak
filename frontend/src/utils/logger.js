class Logger {
  constructor() {
    this.logs = [];
    this.maxLogs = 1000;
    this.levels = {
      DEBUG: 0,
      INFO: 1,
      WARN: 2,
      ERROR: 3,
      CRITICAL: 4
    };
    this.currentLevel = this.levels.INFO;
  }

  setLevel(level) {
    if (this.levels[level] !== undefined) {
      this.currentLevel = this.levels[level];
      this.info(`Log level set to: ${level}`);
    }
  }

  formatMessage(level, message, data = null) {
    const timestamp = new Date().toISOString();
    const logEntry = {
      timestamp,
      level,
      message,
      data,
      id: Date.now() + Math.random()
    };
    
    this.logs.push(logEntry);
    
    if (this.logs.length > this.maxLogs) {
      this.logs.shift();
    }
    
    return logEntry;
  }

  shouldLog(level) {
    return this.levels[level] >= this.currentLevel;
  }

  debug(message, data = null) {
    if (this.shouldLog('DEBUG')) {
      const logEntry = this.formatMessage('DEBUG', message, data);
      console.debug(`[DEBUG] ${message}`, data);
      this.emitLogEvent(logEntry);
    }
  }

  info(message, data = null) {
    if (this.shouldLog('INFO')) {
      const logEntry = this.formatMessage('INFO', message, data);
      console.info(`[INFO] ${message}`, data);
      this.emitLogEvent(logEntry);
    }
  }

  warn(message, data = null) {
    if (this.shouldLog('WARN')) {
      const logEntry = this.formatMessage('WARN', message, data);
      console.warn(`[WARN] ${message}`, data);
      this.emitLogEvent(logEntry);
    }
  }

  error(message, error = null, data = null) {
    if (this.shouldLog('ERROR')) {
      const logEntry = this.formatMessage('ERROR', message, { error, data });
      console.error(`[ERROR] ${message}`, error, data);
      this.emitLogEvent(logEntry);
    }
  }

  critical(message, error = null, data = null) {
    if (this.shouldLog('CRITICAL')) {
      const logEntry = this.formatMessage('CRITICAL', message, { error, data });
      console.error(`[CRITICAL] ${message}`, error, data);
      this.emitLogEvent(logEntry);
      
      if (typeof window !== 'undefined' && window.toast) {
        window.toast.error(`Critical Error: ${message}`);
      }
    }
  }

  emitLogEvent(logEntry) {
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent('log', { detail: logEntry }));
    }
  }

  getLogs(level = null, limit = 100) {
    let filteredLogs = this.logs;
    
    if (level) {
      filteredLogs = this.logs.filter(log => log.level === level);
    }
    
    return filteredLogs.slice(-limit);
  }

  clearLogs() {
    this.logs = [];
    this.info('Logs cleared');
  }

  exportLogs() {
    const logData = {
      exportTime: new Date().toISOString(),
      totalLogs: this.logs.length,
      logs: this.logs
    };
    
    const blob = new Blob([JSON.stringify(logData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chainbreak-logs-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
    
    this.info('Logs exported successfully');
  }

  logPerformance(operation, startTime, endTime, metadata = {}) {
    const duration = endTime - startTime;
    this.info(`Performance: ${operation} completed in ${duration.toFixed(2)}ms`, {
      operation,
      duration,
      startTime,
      endTime,
      ...metadata
    });
  }

  logGraphOperation(operation, nodeCount, edgeCount, metadata = {}) {
    this.info(`Graph Operation: ${operation}`, {
      operation,
      nodeCount,
      edgeCount,
      timestamp: new Date().toISOString(),
      ...metadata
    });
  }

  logAPIRequest(method, endpoint, status, duration, metadata = {}) {
    const level = status >= 400 ? 'WARN' : 'INFO';
    this[level.toLowerCase()](`API Request: ${method} ${endpoint}`, {
      method,
      endpoint,
      status,
      duration: `${duration.toFixed(2)}ms`,
      timestamp: new Date().toISOString(),
      ...metadata
    });
  }
}

const logger = new Logger();

if (typeof window !== 'undefined') {
  window.logger = logger;
}

export default logger;
export { logger };
