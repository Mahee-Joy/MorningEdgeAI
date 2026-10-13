# MorningEdge AI — Web Version

A deployable Streamlit web app for the MorningEdge AI project. It has a browser dashboard, CSV-based technical screener, position sizing, 1:2 target calculations, CSV export, and an official NSE MCP tool explorer.

## Publish it online (free-tier deployment)

This ZIP is the source code. To create a public website, you must deploy it using your own hosting account:

1. Download and extract this ZIP.
2. Create a GitHub repository and upload the files from this folder.
3. Open Streamlit Community Cloud: https://share.streamlit.io/
4. Sign in with GitHub, select **Create app**, choose your repository and branch, and set the main file path to `app.py`.
5. Deploy. Streamlit will build the environment from `requirements.txt` and provide a public `*.streamlit.app` address.

The site address is created by the hosting service; this package does not already have a public URL. Keep the GitHub repository public if using the free deployment flow and don't upload secrets or account credentials.

## Local run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Features
- Responsive browser dashboard
- Official NSE MCP endpoint tool discovery and invocation
- CSV OHLCV screener with EMA20/50, RSI(14), MACD, relative volume, 20-session breakout and liquidity checks
- Top 10 ranked list, risk flags, quantity sizing, 1:2 target, downloadable CSV
- Risk settings for capital, planned risk %, minimum price and liquidity threshold

## CSV format
Required fields: `symbol,date,open,high,low,close,volume`. Use authorized data and at least 30 trading sessions per symbol; 60+ is recommended.

## Important
NSE describes its MCP service as informational/educational, not for commercial use, and market snapshot data may be delayed. Check the current official terms at https://www.nseindia.com/nse-mcp. No live pre-open fields are assumed unless the endpoint exposes them. This app does not place trades and does not guarantee outcomes. The score is a heuristic, not a probability of profit. Backtest and paper-trade before real-money use.
