# ChainBreak Illicit Transaction Detector

A comprehensive cryptocurrency transaction analysis tool that detects suspicious patterns and provides advanced visualization capabilities for blockchain forensics.

## 🚀 Features

### Pattern Detection
- **Peel Chains**: Decreasing-value transaction sequences
- **Mixing**: Many-to-many transaction patterns
- **Smurfing**: Large amounts split into small transactions
- **Rapid Transfers**: High-frequency transactions
- **Round Amounts**: Suspiciously round transaction values
- **Sudden Bursts**: Unusual spikes in activity
- **Layering**: Multi-hop transaction paths
- **Chain Hopping**: Cross-exchange fund movement

### Risk Assessment
- **Multi-factor Risk Scoring**: Combines threat intelligence, patterns, centrality, and volume
- **SIR Model**: Susceptible-Infected-Recovery simulation for illicit activity propagation
- **Anomaly Detection**: Isolation Forest and Local Outlier Factor algorithms
- **Community Detection**: Louvain algorithm for clustering analysis

### Visualization
- **Interactive Graphs**: Plotly-based network visualizations
- **Edge Colors**: Pattern-based edge coloring
- **Node Highlighting**: Risk-level based node colors and sizes
- **Cluster Visualization**: Multi-panel dashboard
- **Risk Heatmaps**: Address interaction matrices
- **Export Capabilities**: JSON export for frontend consumption

### Threat Intelligence
- **BitcoinWhosWho Integration**: Scam reports and address scoring
- **Chainalysis API**: Professional blockchain analysis
- **Secure Key Storage**: Encrypted API key management

## 📦 Installation

### Quick Install (Linux/macOS)
```bash
chmod +x install.sh
./install.sh
```

### Quick Install (Windows)
```cmd
install.bat
```

### Manual Installation

1. **Python Dependencies**
```bash
pip install -r requirements.txt
pip install python-louvain plotly seaborn matplotlib
```

2. **Frontend Dependencies**
```bash
cd frontend
npm install
npm install @mui/material @mui/icons-material @emotion/react @emotion/styled
```

3. **Environment Setup**
```bash
# Create .env file
cat > .env << EOF
FLASK_ENV=development
FLASK_DEBUG=True
API_BASE_URL=http://localhost:5000
CHAINALYSIS_API_KEY=db373a00f1f63693d7ccf144ee781787865310acda3870ca8abfb09135cbfc58
ENABLE_BITCOINWHOSWHO=true
ENABLE_CHAINABUSE=false
ENABLE_CROPTY=false
LOG_LEVEL=INFO
EOF
```

## 🚀 Usage

### 1. Start the Backend
```bash
# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate.bat  # Windows

# Start the API server
python -m src.api
```

### 2. Start the Frontend
```bash
cd frontend
npm start
```

### 3. Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000
- **Illicit Detection**: Switch to "Illicit Detection" tab

### 4. Run Tests
```bash
# Test the detector
python test_illicit_detector.py

# Test installation
python test_installation.py
```

## 🔧 API Endpoints

### Illicit Analysis
- `POST /api/illicit-analysis` - Analyze addresses for suspicious patterns
- `GET /api/illicit-analysis/patterns` - Get available pattern types
- `GET /api/illicit-analysis/risk-levels` - Get risk level descriptions
- `GET /api/illicit-analysis/threat-intel/{address}` - Get threat intelligence

### Example API Usage
```python
import requests

# Analyze addresses
response = requests.post('http://localhost:5000/api/illicit-analysis', json={
    'addresses': ['1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa'],
    'include_visualization': True,
    'max_transactions': 1000
})

analysis = response.json()
print(f"Detected {len(analysis['analysis']['suspicious_patterns'])} patterns")
```

## 📊 Analysis Output

### Risk Levels
- **CRITICAL**: Highest risk - immediate attention required
- **HIGH**: High risk - significant suspicious activity
- **MEDIUM**: Medium risk - some suspicious patterns
- **LOW**: Low risk - minimal suspicious activity
- **CLEAN**: No suspicious activity detected

### Pattern Types
- **mixing**: Many-to-many transaction obfuscation
- **peel_chain**: Decreasing-value transaction sequences
- **smurfing**: Large amounts split into small transactions
- **rapid_transfers**: High-frequency transactions
- **round_amounts**: Suspiciously round transaction values
- **sudden_bursts**: Unusual spikes in activity
- **layering**: Multi-hop transaction paths
- **chain_hopping**: Cross-exchange fund movement

### Visualization Features
- **Color-coded nodes** by risk level
- **Pattern-colored edges** for different suspicious activities
- **Size-based nodes** proportional to transaction volume
- **Interactive tooltips** with detailed information
- **Drag-and-drop** graph manipulation

## 🔒 Security Features

### API Key Protection
- **Fernet Encryption**: Chainalysis API keys are encrypted at rest
- **Environment Variables**: Secure configuration management
- **Input Validation**: Comprehensive input sanitization

### Data Privacy
- **Local Processing**: All analysis performed locally
- **No Data Storage**: Transaction data not permanently stored
- **Secure Communication**: HTTPS endpoints for production

## 🧪 Testing

### Test Scripts
- `test_illicit_detector.py` - Comprehensive detector testing
- `test_installation.py` - Installation verification

### Test Data
The test script creates realistic transaction patterns:
- Peel chain with decreasing amounts
- Mixing pattern with many-to-many transactions
- Smurfing with small transaction amounts
- Rapid transfers in short time periods
- Round amount transactions

### Running Tests
```bash
# Run all tests
python test_illicit_detector.py

# Test specific components
python -c "from illicit_transaction_detector import IllicitTransactionDetector; print('✅ Import successful')"
```

## 📁 Project Structure

```
ChainBreak/
├── illicit_transaction_detector.py    # Main detector class
├── api_illicit_analysis.py            # API endpoints
├── test_illicit_detector.py          # Test script
├── requirements.txt                   # Python dependencies
├── install.sh                        # Linux/macOS installer
├── install.bat                       # Windows installer
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   └── IllicitTransactionAnalyzer.js
│   │   ├── utils/
│   │   │   └── api.js
│   │   └── App.js
│   └── package.json
├── data/
│   └── graph/                        # Graph data storage
├── static/
│   └── visualizations/               # Generated visualizations
└── logs/                             # Application logs
```

## 🔧 Configuration

### Environment Variables
- `CHAINALYSIS_API_KEY`: Your Chainalysis API key
- `ENABLE_BITCOINWHOSWHO`: Enable BitcoinWhosWho scraper (true/false)
- `ENABLE_CHAINABUSE`: Enable ChainAbuse scraper (true/false)
- `ENABLE_CROPTY`: Enable Cropty scraper (true/false)
- `LOG_LEVEL`: Logging level (DEBUG/INFO/WARNING/ERROR)

### Detection Thresholds
```python
thresholds = {
    'peel_chain_min_transactions': 5,
    'peel_chain_value_decrease_threshold': 0.1,
    'mixing_min_addresses': 10,
    'mixing_time_window_hours': 24,
    'rapid_transfer_time_window_minutes': 60,
    'round_amount_threshold': 0.01,
    'sudden_burst_multiplier': 5.0,
    'smurfing_min_transactions': 20,
    'smurfing_max_value_ratio': 0.1,
    'layering_min_layers': 3,
    'anomaly_contamination': 0.1,
    'lof_neighbors': 20,
    'cluster_min_size': 3
}
```

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure all dependencies are installed
   pip install -r requirements.txt
   pip install python-louvain plotly seaborn matplotlib
   ```

2. **Frontend Dependencies**
   ```bash
   cd frontend
   npm install
   npm install @mui/material @mui/icons-material @emotion/react @emotion/styled
   ```

3. **API Connection Issues**
   - Check if backend is running on port 5000
   - Verify CORS settings
   - Check firewall settings

4. **Visualization Issues**
   - Ensure plotly is installed: `pip install plotly`
   - Check browser console for JavaScript errors
   - Verify D3.js is loaded in frontend

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python -m src.api
```

## 📈 Performance

### Optimization Tips
- **Batch Processing**: Analyze multiple addresses together
- **Caching**: Results are cached for repeated analysis
- **Parallel Processing**: Multi-threaded pattern detection
- **Memory Management**: Efficient graph data structures

### Scalability
- **Large Datasets**: Handles thousands of transactions
- **Memory Efficient**: Streaming processing for large graphs
- **Distributed Processing**: Can be scaled across multiple servers

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install pytest pytest-cov black flake8 mypy

# Run tests
pytest tests/

# Format code
black .

# Lint code
flake8 .
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This tool is for educational and research purposes only. Always comply with local laws and regulations when analyzing blockchain data. The authors are not responsible for any misuse of this software.

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the test scripts for examples

---

**Happy analyzing! 🔍💰**
