import React, { useState, useEffect, useRef, useCallback } from 'react';
import * as d3 from 'd3';
import { chainbreakAPI } from '../utils/api';
import { logger } from '../utils/logger';
// Temporarily using basic HTML elements until Material-UI is properly installed
const Box = ({ children, sx, ...props }) => <div style={sx} {...props}>{children}</div>;
const Card = ({ children, sx, ...props }) => <div style={{...sx, border: '1px solid #ccc', borderRadius: '4px', padding: '16px', margin: '8px'}} {...props}>{children}</div>;
const CardContent = ({ children, sx, ...props }) => <div style={sx} {...props}>{children}</div>;
const Typography = ({ variant, children, sx, gutterBottom, ...props }) => {
  const styles = {
    h4: { fontSize: '2rem', fontWeight: 'bold', marginBottom: gutterBottom ? '16px' : '0' },
    h6: { fontSize: '1.25rem', fontWeight: 'bold', marginBottom: gutterBottom ? '8px' : '0' },
    subtitle2: { fontSize: '0.875rem', fontWeight: 'bold', marginBottom: gutterBottom ? '8px' : '0' },
    body2: { fontSize: '0.875rem', marginBottom: gutterBottom ? '8px' : '0' }
  };
  return <div style={{...styles[variant], ...sx}} {...props}>{children}</div>;
};
const Button = ({ children, variant, onClick, disabled, startIcon, sx, ...props }) => (
  <button 
    style={{
      ...sx,
      padding: '8px 16px',
      border: variant === 'contained' ? 'none' : '1px solid #ccc',
      backgroundColor: variant === 'contained' ? '#1976d2' : 'transparent',
      color: variant === 'contained' ? 'white' : 'black',
      borderRadius: '4px',
      cursor: disabled ? 'not-allowed' : 'pointer',
      opacity: disabled ? 0.6 : 1
    }}
    onClick={onClick}
    disabled={disabled}
    {...props}
  >
    {startIcon} {children}
  </button>
);
const TextField = ({ label, value, onChange, placeholder, variant, fullWidth, SelectProps, size, ...props }) => (
  <div style={{ marginBottom: '16px' }}>
    <label style={{ display: 'block', marginBottom: '4px' }}>{label}</label>
    <input
      type="text"
      value={value}
      onChange={(e) => onChange(e)}
      placeholder={placeholder}
      style={{
        width: fullWidth ? '100%' : 'auto',
        padding: '8px',
        border: '1px solid #ccc',
        borderRadius: '4px'
      }}
      {...props}
    />
  </div>
);
const Grid = ({ container, item, children, xs, sm, md, spacing, alignItems, sx, ...props }) => (
  <div style={{
    ...sx,
    display: container ? 'flex' : 'block',
    flexWrap: container ? 'wrap' : 'nowrap',
    gap: spacing ? `${spacing * 8}px` : '0',
    alignItems: alignItems || 'stretch',
    flex: item ? `0 0 ${(xs || 12) * 8.33}%` : 'none'
  }} {...props}>{children}</div>
);
const Chip = ({ label, icon, sx, size, color, ...props }) => (
  <span style={{
    ...sx,
    display: 'inline-flex',
    alignItems: 'center',
    padding: '4px 8px',
    backgroundColor: color === 'primary' ? '#1976d2' : color === 'secondary' ? '#dc004e' : '#f5f5f5',
    color: color === 'primary' || color === 'secondary' ? 'white' : 'black',
    borderRadius: '16px',
    fontSize: size === 'small' ? '0.75rem' : '0.875rem',
    margin: '2px'
  }} {...props}>
    {icon} {label}
  </span>
);
const Alert = ({ severity, children, sx, ...props }) => (
  <div style={{
    ...sx,
    padding: '12px',
    backgroundColor: severity === 'error' ? '#ffebee' : '#e3f2fd',
    color: severity === 'error' ? '#c62828' : '#1565c0',
    border: `1px solid ${severity === 'error' ? '#ffcdd2' : '#bbdefb'}`,
    borderRadius: '4px',
    marginBottom: '16px'
  }} {...props}>{children}</div>
);
const CircularProgress = ({ size, ...props }) => <span {...props}>⏳</span>;
const Tabs = ({ children, value, onChange, ...props }) => (
  <div style={{ borderBottom: '1px solid #ccc' }} {...props}>
    {children}
  </div>
);
const Tab = ({ label, ...props }) => (
  <button style={{ padding: '12px 24px', border: 'none', background: 'none', cursor: 'pointer' }} {...props}>
    {label}
  </button>
);
const Paper = ({ children, sx, ...props }) => (
  <div style={{...sx, backgroundColor: 'white', padding: '16px', borderRadius: '4px'}} {...props}>{children}</div>
);
const Table = ({ children, ...props }) => <table style={{ width: '100%', borderCollapse: 'collapse' }} {...props}>{children}</table>;
const TableBody = ({ children, ...props }) => <tbody {...props}>{children}</tbody>;
const TableCell = ({ children, ...props }) => <td style={{ padding: '8px', border: '1px solid #ccc' }} {...props}>{children}</td>;
const TableContainer = ({ children, component, ...props }) => <div {...props}>{children}</div>;
const TableHead = ({ children, ...props }) => <thead {...props}>{children}</thead>;
const TableRow = ({ children, ...props }) => <tr {...props}>{children}</tr>;
const Dialog = ({ open, onClose, children, maxWidth, fullWidth, ...props }) => 
  open ? <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0,0,0,0.5)', zIndex: 1000 }} onClick={onClose} {...props}>{children}</div> : null;
const DialogTitle = ({ children, ...props }) => <h2 style={{ margin: 0, padding: '16px' }} {...props}>{children}</h2>;
const DialogContent = ({ children, ...props }) => <div style={{ padding: '16px' }} {...props}>{children}</div>;
const DialogActions = ({ children, ...props }) => <div style={{ padding: '16px', textAlign: 'right' }} {...props}>{children}</div>;
const List = ({ children, ...props }) => <ul style={{ listStyle: 'none', padding: 0 }} {...props}>{children}</ul>;
const ListItem = ({ children, ...props }) => <li style={{ padding: '8px 0', borderBottom: '1px solid #eee' }} {...props}>{children}</li>;
const ListItemText = ({ primary, secondary, ...props }) => (
  <div {...props}>
    <div style={{ fontWeight: 'bold' }}>{primary}</div>
    {secondary && <div style={{ color: '#666', fontSize: '0.875rem' }}>{secondary}</div>}
  </div>
);
const ListItemIcon = ({ children, ...props }) => <span style={{ marginRight: '8px' }} {...props}>{children}</span>;
const Divider = ({ ...props }) => <hr style={{ border: 'none', borderTop: '1px solid #eee', margin: '16px 0' }} {...props} />;
// const Tooltip = ({ children, title, ...props }) => <div title={title} {...props}>{children}</div>;
const IconButton = ({ children, onClick, size, ...props }) => (
  <button style={{ border: 'none', background: 'none', cursor: 'pointer', padding: '4px' }} onClick={onClick} {...props}>
    {children}
  </button>
);
// const Badge = ({ children, badgeContent, ...props }) => (
//   <div style={{ position: 'relative' }} {...props}>
//     {children}
//     {badgeContent && <span style={{ position: 'absolute', top: '-8px', right: '-8px', backgroundColor: 'red', color: 'white', borderRadius: '50%', width: '16px', height: '16px', fontSize: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>{badgeContent}</span>}
//   </div>
// );
// Temporarily using simple icons until Material-UI is properly installed
const Security = () => <span>🛡️</span>;
const Warning = () => <span>⚠️</span>;
const CheckCircle = () => <span>✅</span>;
const Error = () => <span>❌</span>;
const Info = () => <span>ℹ️</span>;
const Visibility = () => <span>👁️</span>;
// const Download = () => <span>⬇️</span>;
// const Refresh = () => <span>🔄</span>;
// const FilterList = () => <span>🔍</span>;
const Timeline = () => <span>📈</span>;
const AccountBalance = () => <span>💰</span>;
const TrendingUp = () => <span>📊</span>;
const Group = () => <span>👥</span>;
const Assessment = () => <span>📋</span>;

const IllicitTransactionAnalyzer = () => {
  const [analysisData, setAnalysisData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedTab, setSelectedTab] = useState(0);
  const [selectedAddress, setSelectedAddress] = useState(null);
  const [addressInput, setAddressInput] = useState('');
  const [filterRisk, setFilterRisk] = useState('ALL');
  const [filterPattern, setFilterPattern] = useState('ALL');
  const [graphData, setGraphData] = useState(null);
  const svgRef = useRef();
  const graphContainerRef = useRef();

  // Risk level colors
  const riskColors = {
    CRITICAL: '#FF0000',
    HIGH: '#FF6600',
    MEDIUM: '#FFCC00',
    LOW: '#00CC00',
    CLEAN: '#0066CC'
  };

  // Pattern colors
  // const patternColors = {
  //   mixing: '#FF00FF',
  //   peel_chain: '#00FFFF',
  //   smurfing: '#FFFF00',
  //   rapid_transfers: '#FF9900',
  //   round_amounts: '#99FF00',
  //   sudden_bursts: '#FF0099',
  //   layering: '#0099FF',
  //   chain_hopping: '#CC00CC'
  // };

  const analyzeTransactions = async () => {
    if (!addressInput.trim()) {
      setError('Please enter at least one Bitcoin address');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      logger.info('Starting illicit transaction analysis', { address: addressInput });
      
      const response = await chainbreakAPI.analyzeIllicitTransactions(
        addressInput.split(',').map(addr => addr.trim()),
        true
      );

      if (response.success) {
        setAnalysisData(response.analysis);
        setGraphData(response.graph_data);
        logger.info('Analysis completed successfully', { 
          patterns: response.analysis.suspicious_patterns.length,
          highRisk: response.analysis.high_risk_addresses.length
        });
      } else {
        setError(response.error || 'Analysis failed');
      }
    } catch (err) {
      logger.error('Analysis failed', err);
      setError('Failed to analyze transactions: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const renderGraph = useCallback(() => {
    if (!graphData || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const width = 800;
    const height = 600;
    const margin = { top: 20, right: 20, bottom: 20, left: 20 };

    svg.attr("width", width).attr("height", height);

    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Create force simulation
    const simulation = d3.forceSimulation(graphData.nodes)
      .force("link", d3.forceLink(graphData.edges).id(d => d.id).distance(100))
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(width / 2, height / 2));

    // Create links
    const link = g.append("g")
      .attr("class", "links")
      .selectAll("line")
      .data(graphData.edges)
      .enter().append("line")
      .attr("stroke", d => d.color)
      .attr("stroke-width", d => d.width)
      .attr("stroke-opacity", 0.6);

    // Create nodes
    const node = g.append("g")
      .attr("class", "nodes")
      .selectAll("circle")
      .data(graphData.nodes)
      .enter().append("circle")
      .attr("r", d => d.size)
      .attr("fill", d => d.color)
      .attr("stroke", "#000")
      .attr("stroke-width", 2)
      .call(d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended))
      .on("click", (event, d) => {
        setSelectedAddress(d);
      });

    // Add labels
    const label = g.append("g")
      .attr("class", "labels")
      .selectAll("text")
      .data(graphData.nodes)
      .enter().append("text")
      .text(d => d.id.substring(0, 8) + "...")
      .attr("font-size", "10px")
      .attr("text-anchor", "middle")
      .attr("dy", "0.35em");

    // Add tooltips
    node.append("title")
      .text(d => `
        Address: ${d.id}
        Risk Level: ${d.risk_level}
        Risk Score: ${d.risk_score.toFixed(3)}
        Transactions: ${d.transaction_count}
        Volume: ${d.total_volume.toFixed(2)}
        Patterns: ${d.patterns.join(', ')}
      `);

    // Update positions on simulation tick
    simulation.on("tick", () => {
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

      node
        .attr("cx", d => d.x)
        .attr("cy", d => d.y);

      label
        .attr("x", d => d.x)
        .attr("y", d => d.y);
    });

    function dragstarted(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(event, d) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }
  }, [graphData]);

  useEffect(() => {
    if (graphData) {
      renderGraph();
    }
  }, [graphData, renderGraph]);

  const getRiskIcon = (riskLevel) => {
    switch (riskLevel) {
      case 'CRITICAL': return <Error color="error" />;
      case 'HIGH': return <Warning color="warning" />;
      case 'MEDIUM': return <Info color="info" />;
      case 'LOW': return <CheckCircle color="success" />;
      default: return <Security color="primary" />;
    }
  };

  const getPatternIcon = (pattern) => {
    switch (pattern) {
      case 'mixing': return <Group />;
      case 'peel_chain': return <Timeline />;
      case 'smurfing': return <AccountBalance />;
      case 'rapid_transfers': return <TrendingUp />;
      default: return <Assessment />;
    }
  };

  const filteredAddresses = analysisData ? Object.entries(analysisData.addresses).filter(([_, node]) => {
    if (filterRisk !== 'ALL' && node.risk_level !== filterRisk) return false;
    if (filterPattern !== 'ALL' && !node.suspicious_patterns.includes(filterPattern)) return false;
    return true;
  }) : [];

  const TabPanel = ({ children, value, index, ...other }) => (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tabpanel-${index}`}
      aria-labelledby={`tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Illicit Transaction Analyzer
      </Typography>
      
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={8}>
              <TextField
                fullWidth
                label="Bitcoin Addresses (comma-separated)"
                value={addressInput}
                onChange={(e) => setAddressInput(e.target.value)}
                placeholder="1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa, 1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2"
                variant="outlined"
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <Button
                fullWidth
                variant="contained"
                onClick={analyzeTransactions}
                disabled={loading}
                startIcon={loading ? <CircularProgress size={20} /> : <Assessment />}
                sx={{ height: '56px' }}
              >
                {loading ? 'Analyzing...' : 'Analyze Transactions'}
              </Button>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {analysisData && (
        <>
          {/* Summary Cards */}
          <Grid container spacing={3} sx={{ mb: 3 }}>
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center">
                    <AccountBalance color="primary" sx={{ mr: 2 }} />
                    <Box>
                      <Typography variant="h6">{analysisData.total_addresses}</Typography>
                      <Typography color="textSecondary">Total Addresses</Typography>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center">
                    <Timeline color="primary" sx={{ mr: 2 }} />
                    <Box>
                      <Typography variant="h6">{analysisData.total_transactions}</Typography>
                      <Typography color="textSecondary">Total Transactions</Typography>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center">
                    <Warning color="warning" sx={{ mr: 2 }} />
                    <Box>
                      <Typography variant="h6">{analysisData.high_risk_addresses.length}</Typography>
                      <Typography color="textSecondary">High Risk Addresses</Typography>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center">
                    <Security color="error" sx={{ mr: 2 }} />
                    <Box>
                      <Typography variant="h6">{analysisData.suspicious_patterns.length}</Typography>
                      <Typography color="textSecondary">Suspicious Patterns</Typography>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Tabs */}
          <Paper sx={{ width: '100%' }}>
            <Tabs value={selectedTab} onChange={(e, v) => setSelectedTab(v)}>
              <Tab label="Graph Visualization" />
              <Tab label="Address Analysis" />
              <Tab label="Pattern Detection" />
              <Tab label="Risk Assessment" />
            </Tabs>

            <TabPanel value={selectedTab} index={0}>
              <Box ref={graphContainerRef}>
                <Typography variant="h6" gutterBottom>
                  Transaction Network Graph
                </Typography>
                <Box sx={{ border: '1px solid #ccc', borderRadius: 1, p: 2 }}>
                  <svg ref={svgRef}></svg>
                </Box>
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>Legend:</Typography>
                  <Grid container spacing={1}>
                    {Object.entries(riskColors).map(([level, color]) => (
                      <Grid item key={level}>
                        <Chip
                          icon={getRiskIcon(level)}
                          label={level}
                          sx={{ backgroundColor: color, color: 'white' }}
                          size="small"
                        />
                      </Grid>
                    ))}
                  </Grid>
                </Box>
              </Box>
            </TabPanel>

            <TabPanel value={selectedTab} index={1}>
              <Box sx={{ mb: 2 }}>
                <Grid container spacing={2} alignItems="center">
                  <Grid item>
                    <TextField
                      select
                      label="Filter by Risk Level"
                      value={filterRisk}
                      onChange={(e) => setFilterRisk(e.target.value)}
                      SelectProps={{ native: true }}
                      size="small"
                    >
                      <option value="ALL">All Risk Levels</option>
                      <option value="CRITICAL">Critical</option>
                      <option value="HIGH">High</option>
                      <option value="MEDIUM">Medium</option>
                      <option value="LOW">Low</option>
                      <option value="CLEAN">Clean</option>
                    </TextField>
                  </Grid>
                  <Grid item>
                    <TextField
                      select
                      label="Filter by Pattern"
                      value={filterPattern}
                      onChange={(e) => setFilterPattern(e.target.value)}
                      SelectProps={{ native: true }}
                      size="small"
                    >
                      <option value="ALL">All Patterns</option>
                      <option value="mixing">Mixing</option>
                      <option value="peel_chain">Peel Chain</option>
                      <option value="smurfing">Smurfing</option>
                      <option value="rapid_transfers">Rapid Transfers</option>
                      <option value="round_amounts">Round Amounts</option>
                      <option value="sudden_bursts">Sudden Bursts</option>
                      <option value="layering">Layering</option>
                    </TextField>
                  </Grid>
                </Grid>
              </Box>

              <TableContainer component={Paper}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Address</TableCell>
                      <TableCell>Risk Level</TableCell>
                      <TableCell>Risk Score</TableCell>
                      <TableCell>Transactions</TableCell>
                      <TableCell>Volume</TableCell>
                      <TableCell>Patterns</TableCell>
                      <TableCell>Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {filteredAddresses.map(([address, node]) => (
                      <TableRow key={address}>
                        <TableCell>
                          <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                            {address}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Chip
                            icon={getRiskIcon(node.risk_level)}
                            label={node.risk_level}
                            sx={{ 
                              backgroundColor: riskColors[node.risk_level], 
                              color: 'white' 
                            }}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>{node.risk_score.toFixed(3)}</TableCell>
                        <TableCell>{node.transaction_count}</TableCell>
                        <TableCell>{(node.total_sent + node.total_received).toFixed(2)}</TableCell>
                        <TableCell>
                          {node.suspicious_patterns.map(pattern => (
                            <Chip
                              key={pattern}
                              icon={getPatternIcon(pattern)}
                              label={pattern}
                              size="small"
                              sx={{ mr: 0.5, mb: 0.5 }}
                            />
                          ))}
                        </TableCell>
                        <TableCell>
                          <IconButton
                            onClick={() => setSelectedAddress({ id: address, ...node })}
                            size="small"
                          >
                            <Visibility />
                          </IconButton>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </TabPanel>

            <TabPanel value={selectedTab} index={2}>
              <Typography variant="h6" gutterBottom>
                Detected Suspicious Patterns
              </Typography>
              <Grid container spacing={2}>
                {analysisData.suspicious_patterns.map((pattern, index) => (
                  <Grid item xs={12} md={6} key={index}>
                    <Card>
                      <CardContent>
                        <Box display="flex" alignItems="center" sx={{ mb: 2 }}>
                          {getPatternIcon(pattern.pattern_type)}
                          <Typography variant="h6" sx={{ ml: 1 }}>
                            {pattern.pattern_type.replace('_', ' ').toUpperCase()}
                          </Typography>
                          <Chip
                            label={`${(pattern.confidence * 100).toFixed(1)}%`}
                            color="primary"
                            size="small"
                            sx={{ ml: 'auto' }}
                          />
                        </Box>
                        <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                          {pattern.description}
                        </Typography>
                        <Typography variant="body2" sx={{ mb: 1 }}>
                          <strong>Risk Score:</strong> {pattern.risk_score.toFixed(3)}
                        </Typography>
                        <Typography variant="body2" sx={{ mb: 1 }}>
                          <strong>Addresses Involved:</strong> {pattern.addresses.length}
                        </Typography>
                        <Typography variant="body2">
                          <strong>Transactions:</strong> {pattern.transactions.length}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </TabPanel>

            <TabPanel value={selectedTab} index={3}>
              <Typography variant="h6" gutterBottom>
                Risk Distribution
              </Typography>
              <Grid container spacing={2}>
                {Object.entries(analysisData.risk_distribution).map(([level, count]) => (
                  <Grid item xs={12} sm={6} md={4} key={level}>
                    <Card>
                      <CardContent>
                        <Box display="flex" alignItems="center">
                          {getRiskIcon(level)}
                          <Box sx={{ ml: 2 }}>
                            <Typography variant="h4">{count}</Typography>
                            <Typography color="textSecondary">{level}</Typography>
                          </Box>
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </TabPanel>

          </Paper>

          {/* Address Detail Dialog */}
          <Dialog
            open={selectedAddress !== null}
            onClose={() => setSelectedAddress(null)}
            maxWidth="md"
            fullWidth
          >
            <DialogTitle>
              Address Details: {selectedAddress?.id}
            </DialogTitle>
            <DialogContent>
              {selectedAddress && (
                <List>
                  <ListItem>
                    <ListItemIcon>
                      {getRiskIcon(selectedAddress.risk_level)}
                    </ListItemIcon>
                    <ListItemText
                      primary="Risk Level"
                      secondary={selectedAddress.risk_level}
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <Assessment />
                    </ListItemIcon>
                    <ListItemText
                      primary="Risk Score"
                      secondary={selectedAddress.risk_score.toFixed(3)}
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <Timeline />
                    </ListItemIcon>
                    <ListItemText
                      primary="Transaction Count"
                      secondary={selectedAddress.transaction_count}
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <AccountBalance />
                    </ListItemIcon>
                    <ListItemText
                      primary="Total Volume"
                      secondary={`${(selectedAddress.total_sent + selectedAddress.total_received).toFixed(2)} BTC`}
                    />
                  </ListItem>
                  <Divider />
                  <ListItem>
                    <ListItemText
                      primary="Suspicious Patterns"
                      secondary={selectedAddress.patterns.length > 0 
                        ? selectedAddress.patterns.join(', ')
                        : 'None detected'
                      }
                    />
                  </ListItem>
                </List>
              )}
            </DialogContent>
            <DialogActions>
              <Button onClick={() => setSelectedAddress(null)}>Close</Button>
            </DialogActions>
          </Dialog>
        </>
      )}
    </Box>
  );
};

export default IllicitTransactionAnalyzer;
