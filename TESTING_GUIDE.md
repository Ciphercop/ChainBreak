# 🧪 Complete Testing Guide for Illicit Detection System

## **📋 Prerequisites**

### **1. Backend Server**
```bash
# Start the main server
python app.py --api
```
**Expected Output**: Server running on `http://localhost:5000`

### **2. Frontend Server**
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies (if not already done)
npm install

# Start React development server
npm start
```
**Expected Output**: Frontend running on `http://localhost:3000`

---

## **🔧 Backend Testing**

### **1. API Endpoint Tests**

#### **A. System Status Test**
```bash
python test_api_endpoints.py
```
**Expected Results**:
- ✅ Status: 200
- ✅ Backend: json
- ✅ System: operational

#### **B. Individual Endpoint Tests**
```bash
# Test pattern types
curl http://localhost:5000/api/illicit-analysis/pattern-types

# Test risk levels
curl http://localhost:5000/api/illicit-analysis/risk-levels

# Test threat intelligence
curl http://localhost:5000/api/illicit-analysis/threat-intel/1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa
```

### **2. Illicit Detection Test**
```bash
python test_illicit_system.py
```
**Expected Results**:
- ✅ Analysis completed successfully
- ✅ 5 addresses analyzed
- ✅ 50 transactions processed
- ✅ 14+ suspicious patterns detected
- ✅ BitcoinWhosWho integration working

### **3. Manual API Testing**

#### **A. Full Analysis Test**
```bash
curl -X POST http://localhost:5000/api/illicit-analysis \
  -H "Content-Type: application/json" \
  -d '{
    "addresses": ["1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa", "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2"],
    "max_transactions": 50,
    "include_visualization": true
  }'
```

#### **B. Run Detection Test**
```bash
curl -X POST http://localhost:5000/api/illicit-analysis/run-detection \
  -H "Content-Type: application/json" \
  -d '{
    "graph_data": {
      "nodes": [
        {"id": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa", "type": "address"},
        {"id": "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2", "type": "address"}
      ],
      "edges": [
        {"source": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa", "target": "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2", "value": 1.0}
      ]
    }
  }'
```

---

## **🌐 Frontend Testing**

### **1. Access the Application**
1. Open browser to `http://localhost:3000`
2. You should see the ChainBreak interface
3. Navigate between tabs: "Graph Analysis", "Illicit Detection", "Law Enforcement"

### **2. Graph Analysis Tab**
1. **Enter Addresses**: Input Bitcoin addresses (comma-separated)
2. **Click "Analyze"**: Should generate transaction graph
3. **Verify Graph**: Check if nodes and edges are displayed
4. **Node Interaction**: Click on nodes to see details

### **3. Illicit Detection Tab**
1. **Enter Addresses**: Input Bitcoin addresses
2. **Click "Analyze Transactions"**: Should run illicit detection
3. **Check Results**: Verify suspicious patterns are detected
4. **View Visualizations**: Check if charts and graphs appear

### **4. Law Enforcement Dashboard Tab**
1. **Enter Addresses**: Input Bitcoin addresses
2. **Click "Analyze Sample"**: Should run comprehensive analysis
3. **Review Critical Alerts**: Check high-risk address alerts
4. **Examine Patterns**: Review suspicious pattern table
5. **Check Threat Intelligence**: Verify BitcoinWhosWho data
6. **Investigate Clusters**: Review cluster analysis

---

## **🎯 Specific Test Cases**

### **1. Money Laundering Detection**
**Test Addresses**:
```
1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa,1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2,1FfmbHfnpaZjKFvyi1okTjJJusN455paPH
```

**Expected Results**:
- ✅ Rapid transfer patterns detected
- ✅ High confidence scores (0.8+)
- ✅ Multiple suspicious patterns

### **2. Exchange Hopping**
**Test Addresses**:
```
3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy,bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh
```

**Expected Results**:
- ✅ Smurfing patterns detected
- ✅ Round amount transactions
- ✅ Medium confidence scores

### **3. Mixing Service Usage**
**Test Addresses**:
```
1FfmbHfnpaZjKFvyi1okTjJJusN455paPH,3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy
```

**Expected Results**:
- ✅ Mixing service patterns
- ✅ High risk scores
- ✅ Threat intelligence data

---

## **🔍 Debugging Guide**

### **1. Backend Issues**

#### **A. Server Not Starting**
```bash
# Check if port 5000 is in use
netstat -an | findstr :5000

# Kill existing processes
taskkill /f /im python.exe

# Restart server
python app.py --api
```

#### **B. API Errors**
```bash
# Check server logs
# Look for error messages in terminal

# Test basic connectivity
curl http://localhost:5000/api/status
```

### **2. Frontend Issues**

#### **A. React Server Not Starting**
```bash
# Check if port 3000 is in use
netstat -an | findstr :3000

# Clear npm cache
npm cache clean --force

# Reinstall dependencies
rm -rf node_modules
npm install
npm start
```

#### **B. API Connection Issues**
1. **Check Browser Console**: Open Developer Tools (F12)
2. **Look for Errors**: Check Console tab for API errors
3. **Network Tab**: Check if API calls are being made
4. **CORS Issues**: Verify backend CORS is enabled

### **3. Common Error Solutions**

#### **A. "Internal Server Error"**
- Check backend server is running
- Verify API endpoints are registered
- Check Python dependencies are installed

#### **B. "Failed to Load Resource"**
- Verify server URLs are correct
- Check network connectivity
- Ensure ports are not blocked

#### **C. "Module Not Found"**
- Run `npm install` in frontend directory
- Check package.json dependencies
- Verify file paths are correct

---

## **📊 Performance Testing**

### **1. Load Testing**
```bash
# Test with multiple addresses
python -c "
import requests
addresses = ['1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa'] * 10
response = requests.post('http://localhost:5000/api/illicit-analysis', 
    json={'addresses': addresses, 'max_transactions': 100})
print(f'Status: {response.status_code}')
print(f'Response time: {response.elapsed.total_seconds():.2f}s')
"
```

### **2. Memory Usage**
```bash
# Monitor memory usage
python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"
```

---

## **✅ Success Criteria**

### **Backend Tests Pass When**:
- ✅ Server starts without errors
- ✅ All API endpoints return 200 status
- ✅ Illicit detection completes successfully
- ✅ Threat intelligence data is retrieved
- ✅ Response times < 5 seconds

### **Frontend Tests Pass When**:
- ✅ React app loads without errors
- ✅ All tabs are accessible
- ✅ API calls complete successfully
- ✅ Data visualizations render
- ✅ User interactions work smoothly

### **Integration Tests Pass When**:
- ✅ Frontend can communicate with backend
- ✅ Data flows correctly between components
- ✅ Visualizations update with new data
- ✅ Error handling works properly
- ✅ Performance is acceptable

---

## **🚀 Quick Start Commands**

### **Start Everything**
```bash
# Terminal 1: Backend
python app.py --api

# Terminal 2: Frontend
cd frontend && npm start
```

### **Run All Tests**
```bash
# Backend tests
python test_illicit_system.py
python test_api_endpoints.py

# Frontend tests (manual)
# Open http://localhost:3000 and test each tab
```

### **Check System Status**
```bash
# Backend status
curl http://localhost:5000/api/status

# Frontend status
curl http://localhost:3000
```

---

**🎉 You're now ready to test the complete illicit detection system!**
