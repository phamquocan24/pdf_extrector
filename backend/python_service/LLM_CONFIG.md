# 🤖 LLM Configuration Guide

## 🚨 Current Issue: Invalid Groq API Key

### ❌ **Error:**
```
Groq LLM processing error: Error code: 401 - {'error': {'message': 'Invalid API Key', 'type': 'invalid_request_error', 'code': 'invalid_api_key'}}
```

### ✅ **Solution: Configure Valid API Key**

## 🔧 Option 1: Get Free Groq API Key (Recommended)

### **Step 1: Get Groq API Key**
1. Visit: https://console.groq.com/
2. Sign up/login with Google/GitHub account
3. Navigate to "API Keys" section
4. Click "Create API Key"
5. Copy the generated key (starts with `gsk_`)

### **Step 2: Set Environment Variable**

**Windows (PowerShell):**
```powershell
# Temporary (current session only)
$env:GROQ_API_KEY="gsk_your_actual_api_key_here"

# Permanent (all sessions)
[Environment]::SetEnvironmentVariable("GROQ_API_KEY", "gsk_your_actual_api_key_here", "User")
```

**Windows (Command Prompt):**
```cmd
set GROQ_API_KEY=gsk_your_actual_api_key_here
```

**Linux/Mac:**
```bash
export GROQ_API_KEY="gsk_your_actual_api_key_here"

# Add to ~/.bashrc or ~/.zshrc for persistence
echo 'export GROQ_API_KEY="gsk_your_actual_api_key_here"' >> ~/.bashrc
```

### **Step 3: Create .env File (Alternative)**
```bash
# Create .env file in backend/python_service/
cd backend/python_service
echo "GROQ_API_KEY=gsk_your_actual_api_key_here" > .env
```

## 🔧 Option 2: Disable LLM Processing

If you don't want to use LLM processing, the system will fall back to basic text cleaning:

```bash
# Remove or comment out API key
# GROQ_API_KEY=
# OPENAI_API_KEY=
```

**Result:**
- ✅ System still works
- ✅ Basic text cleaning applied
- ⚠️  No advanced LLM text enhancement

## 🔧 Option 3: Use OpenAI Instead

If you have OpenAI API key:

```bash
# Set OpenAI API key instead
export OPENAI_API_KEY="sk-your_openai_key_here"
```

## 🧪 Test LLM Configuration

```bash
cd backend/python_service
python -c "
import os
from groq import Groq

api_key = os.getenv('GROQ_API_KEY')
if api_key:
    try:
        client = Groq(api_key=api_key)
        print('✅ Groq API key valid')
    except Exception as e:
        print(f'❌ Groq API error: {e}')
else:
    print('⚠️  No Groq API key found')
"
```

## 🎯 Expected Behavior After Fix

### ✅ **With Valid API Key:**
```
📋 Text Extraction Methods Available:
  1. ✅ PyMuPDF (text-based PDFs)
  2. ⚠️  EasyOCR (disabled - compatibility issues)
  3. ✅ LLM text processing (Groq)

Text processing: "2002\\n \\n2001" -> "2002 2001"
```

### ✅ **Without API Key (Fallback):**
```
📋 Text Extraction Methods Available:
  1. ✅ PyMuPDF (text-based PDFs)  
  2. ⚠️  EasyOCR (disabled - compatibility issues)
  3. ✅ Basic text cleaning (LLM disabled)

Basic cleaning: "2002\\n \\n2001" -> "2002 2001"
```

## 🚀 Restart Service After Configuration

```bash
cd backend/python_service
python app.py
```

## 💡 Free Groq API Limits

- **Rate Limit**: 30 requests/minute
- **Daily Limit**: 14,400 requests/day
- **Model**: Llama 3.1 8B (fast & free)
- **Perfect for**: Table text cleaning & processing

## 🐛 Troubleshooting

### **Issue: API key not working**
```bash
# Check if key is set
echo $GROQ_API_KEY

# Test API connectivity
curl -X POST "https://api.groq.com/openai/v1/chat/completions" \
  -H "Authorization: Bearer $GROQ_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-3.1-8b-instant","messages":[{"role":"user","content":"Hi"}],"max_tokens":10}'
```

### **Issue: Environment variable not loading**
```bash
# Restart terminal/IDE after setting env vars
# Or use .env file method
```

### **Issue: Still getting 401 error**
- ✅ Verify API key format (should start with `gsk_`)
- ✅ Check for extra spaces/characters
- ✅ Regenerate API key from Groq console
- ✅ Verify account is active and not suspended

## 🎉 Summary

**Quick Fix:**
1. Get Groq API key from https://console.groq.com/
2. Set environment variable: `GROQ_API_KEY=gsk_your_key`
3. Restart Python service
4. ✅ Enhanced text processing working!
