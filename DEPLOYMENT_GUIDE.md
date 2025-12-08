# 🚀 Deployment Guide - GARCH Algo Intelligence Platform

## Quick Deploy to Streamlit Cloud (FREE - 5 Minutes!)

### Step 1: Push to GitHub

First, make sure your code is on GitHub:

```bash
# In your project folder
cd c:\Users\-PC-\Documents\garch_algos

# Initialize git if not already done
git init

# Add all files
git add .

# Commit
git commit -m "Ready for deployment"

# Add your GitHub repo (create one first on github.com)
git remote add origin https://github.com/YOUR_USERNAME/garch_algos.git

# Push to GitHub
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Click **"New app"**
3. Connect your GitHub account (if not already)
4. Select:
   - **Repository**: `garch_algos`
   - **Branch**: `main`
   - **Main file path**: `main.py`
5. Click **"Deploy!"**

### Step 3: Get Your URL!

After ~2-5 minutes, you'll get a URL like:
```
https://your-app-name.streamlit.app
```

**Share this URL with your client!** 🎉

---

## Alternative: Railway (Fast & Reliable)

### Step 1: Create railway.json

```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "streamlit run main.py --server.port $PORT --server.address 0.0.0.0"
  }
}
```

### Step 2: Deploy

1. Go to **[railway.app](https://railway.app)**
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Choose your repo
5. Railway auto-detects Streamlit and deploys!

---

## Alternative: Render (Free Tier Available)

### Step 1: Create render.yaml

```yaml
services:
  - type: web
    name: garch-algo-platform
    env: python
    buildCommand: pip install -r requirements-deploy.txt
    startCommand: streamlit run main.py --server.port $PORT --server.address 0.0.0.0
```

### Step 2: Deploy

1. Go to **[render.com](https://render.com)**
2. Click **"New" → "Web Service"**
3. Connect GitHub
4. Select your repo
5. It will auto-deploy!

---

## Before Deploying: Checklist

### ✅ Files Required

- [x] `main.py` - Your Streamlit app
- [x] `requirements.txt` or `requirements-deploy.txt` - Dependencies
- [x] `.streamlit/config.toml` - Theme & settings (optional but recommended)
- [x] `src/` folder - All your modules

### ✅ For Streamlit Cloud Specifically

Rename `requirements-deploy.txt` to `requirements.txt` before pushing:

```bash
# Use the lighter requirements for deployment
copy requirements-deploy.txt requirements.txt
```

Or create a simple `requirements.txt` with just essential packages.

---

## 🔐 API Keys (Important!)

For production, **don't hardcode API keys!** Use Streamlit secrets:

### Step 1: Create secrets on Streamlit Cloud

In your Streamlit Cloud dashboard:
1. Go to your app
2. Click **"Settings"**
3. Find **"Secrets"**
4. Add your secrets:

```toml
TWELVEDATA_API_KEY = "your_key_here"
ALPHAVANTAGE_API_KEY = "your_key_here"
```

### Step 2: Access in code

```python
import streamlit as st

# Access secrets
api_key = st.secrets["TWELVEDATA_API_KEY"]
```

---

## 🎨 Custom Domain (Optional)

### Streamlit Cloud
- Go to Settings → Custom Domain
- Add your domain (e.g., `trading.yourcompany.com`)
- Add a CNAME record in your DNS

### Railway / Render
- Both support custom domains in settings
- Usually free to add!

---

## 📊 Expected Performance

| Metric | Streamlit Cloud | Railway | Render |
|--------|----------------|---------|--------|
| **Cold Start** | 5-10 sec | 2-5 sec | 5-10 sec |
| **Memory** | 1GB free | 512MB free | 512MB free |
| **Uptime** | 99%+ | 99%+ | 99%+ |
| **SSL** | ✅ Auto | ✅ Auto | ✅ Auto |
| **Custom Domain** | ✅ Free | ✅ Free | ✅ Free |

---

## 🚀 Quick Commands

### Push to GitHub
```bash
git add .
git commit -m "Deploy update"
git push
```

### After pushing, Streamlit Cloud auto-deploys!

---

## 🔧 Troubleshooting

### "Module not found" error
- Check `requirements.txt` includes all dependencies
- Make sure module names are correct (e.g., `scikit-learn` not `sklearn`)

### "App not loading" 
- Check Streamlit logs in dashboard
- Reduce dependencies (use `requirements-deploy.txt`)

### "Memory exceeded"
- Reduce data size
- Use `@st.cache_data` for expensive operations
- Consider paid tier

### "API rate limited"
- Add caching
- Reduce refresh frequency
- Use secrets for API keys

---

## 📱 Mobile Access

Your deployed app works on mobile too! Share the URL and clients can access from:
- 📱 iPhone / Android
- 💻 Desktop
- 📱 Tablet

---

## 🎉 Share With Your Client

Once deployed, send your client:

**Email Template:**
```
Hi [Client Name],

Your Bloomberg-style Portfolio Analytics Dashboard is live!

🔗 Access it here: https://your-app.streamlit.app

Features:
- Real-time market data
- GARCH volatility modeling
- ARIMA price forecasting
- Portfolio analytics
- Risk metrics

Let me know if you have any questions!

Best regards,
[Your Name]
```

---

## 📞 Support

If you run into issues:
1. Check Streamlit Cloud logs
2. Review this guide
3. Ask for help!

---

**Good luck with your deployment!** 🚀

Last Updated: December 8, 2025
