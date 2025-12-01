# financial_dashboard_with_plotly_dash
Building a Live Updating Financial Dashboard

# 📈 Financial Dashboard with ML Sentiment Analysis

A real-time financial dashboard built with **Python Dash** and **Plotly**. This application visualizes stock market data, calculates risk metrics (volatility), and uses a **Hugging Face** Machine Learning model to analyze the sentiment of the latest news headlines.

## 🚀 Features

- **Intraday Candlestick Charts:** Visualizes 15-minute interval price data (Open, High, Low, Close) for the current trading day.
- **Volatility Analysis:** Calculates **Annualized 30-Day Rolling Volatility** to assess historical risk.
- **AI-Powered Sentiment:** Fetches the latest news headline for a ticker and runs it through a Transformer model (DistilBERT) to determine if the news is Positive or Negative.
- **Recent Data Table:** Displays the raw pricing data for the last 10 minutes.
- **Live Updates:** Charts refresh every 15 minutes; tables refresh every minute.

---

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| **`app.py`** | The main Dash application. Handles the UI layout, callbacks, `yfinance` data fetching, and `newsapi` integration. |
| **`ml_model_api.py`** | A helper module that handles the connection to the Hugging Face Inference API for sentiment analysis. |
| **`test_api.py`** | A utility script to verify your Hugging Face API token and model connection before running the main app. |

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have Python installed (3.8+ recommended).

### 2. Install Dependencies
Install the required Python packages:

```bash
pip install dash plotly yfinance pandas requests newsapi-python