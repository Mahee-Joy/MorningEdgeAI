import asyncio, json, math
import pandas as pd
import streamlit as st
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

st.set_page_config(page_title="MorningEdge AI", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root { --ink:#eaf0ff; --muted:#91a0bc; --panel:#111a2b; --line:#25334c; }
.stApp { background: radial-gradient(ellipse at 10% 0%, #172a49 0%, #0a1020 48%, #070b14 100%); color:var(--ink); }
section[data-testid="stSidebar"] { background:#0b1220; border-right:1px solid var(--line); }
.block-container { padding-top:1.8rem; max-width:1500px; }
h1,h2,h3 { letter-spacing:-.03em; }
.eyebrow { color:#77e3c2; text-transform:uppercase; letter-spacing:.16em; font-size:.75rem; font-weight:800; }
.hero { padding:1.5rem 1.7rem; border:1px solid #2b3c59; border-radius:20px; background:linear-gradient(120deg,rgba(29,51,83,.86),rgba(13,23,40,.9)); margin-bottom:1rem; }
.hero h1 { font-size:2.5rem; margin:.2rem 0 .4rem 0; }
.hero p { color:#a9b7d0; margin-bottom:0; }
.metric-card { border:1px solid #293954; background:rgba(17,26,43,.85); border-radius:15px; padding:1rem 1.1rem; min-height:105px; }
.metric-label { color:#92a4c1; font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; }
.metric-value { font-size:1.7rem; font-weight:800; margin-top:.2rem; color:#f5f8ff; }
.small-muted { color:#91a0bc; font-size:.85rem; }
div[data-testid="stMetric"] { background:rgba(17,26,43,.78); border:1px solid #293954; padding:14px 16px; border-radius:14px; }
.stButton > button { border-radius:10px; font-weight:700; }
[data-testid="stDataFrame"] { border:1px solid #293954; border-radius:12px; overflow:hidden; }
</style>
<div class="hero">
  <div class="eyebrow">NSE research workspace · Option A</div>
  <h1>MorningEdge <span style="color:#77e3c2">AI</span></h1>
  <p>A disciplined morning stock-screening workspace with technical ranking, risk sizing and an official NSE MCP tool explorer.</p>
</div>
""", unsafe_allow_html=True)

st.warning("Research/educational use only. NSE MCP data may be delayed and NSE's published terms restrict use to informational/educational purposes, not commercial use. This app does not place orders.")

ENDPOINTS = {
    "Historical Bhavcopy": "https://mcp.nseindia.in/bhavcopy/cm/mcp",
    "Current CM market snapshot": "https://mcp.nseindia.in/cmmkt/mcp",
}

def run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try: return loop.run_until_complete(coro)
        finally: loop.close()

async def list_tools(url):
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return [t.model_dump() for t in result.tools]

async def call_tool(url, name, args):
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments=args)
            items = []
            structured = getattr(result, "structuredContent", None)
            if structured is not None: items.append(json.dumps(structured, ensure_ascii=False, default=str))
            for c in (result.content or []):
                items.append(getattr(c, "text", str(c)))
            return {"error": getattr(result, "isError", False), "content": items}

def normalize(df):
    norms = {str(c).strip().lower().replace("_"," ").replace("-"," "): c for c in df.columns}
    aliases = {
        "symbol":["symbol","ticker","tradingsymbol","security","stock"],
        "date":["date","datetime","timestamp","trading date"],
        "open":["open","open price","openprice"], "high":["high","high price","highprice"],
        "low":["low","low price","lowprice"], "close":["close","close price","closeprice","last","ltp"],
        "volume":["volume","total traded quantity","traded volume","qty","shares traded"]
    }
    ren = {}
    for target, options in aliases.items():
        for opt in options:
            if opt in norms: ren[norms[opt]] = target; break
    df = df.rename(columns=ren).copy()
    missing = [x for x in ["symbol","date","open","high","low","close","volume"] if x not in df.columns]
    if missing: raise ValueError("Missing columns: "+", ".join(missing))
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",",""), errors="coerce")
    return df.dropna(subset=["symbol","date","open","high","low","close","volume"]).sort_values(["symbol","date"])

def rsi(s, n=14):
    d=s.diff()
    up=d.clip(lower=0).ewm(alpha=1/n,min_periods=n,adjust=False).mean()
    down=(-d.clip(upper=0)).ewm(alpha=1/n,min_periods=n,adjust=False).mean()
    rs=up/down.replace(0,float("nan"))
    return 100-100/(1+rs)

def score(g, capital, risk_pct, min_price, min_value):
    g=g.sort_values("date")
    if len(g)<30: return None
    c,h,o,l,v=[g[x] for x in ["close","high","open","low","volume"]]
    e20=c.ewm(span=20,adjust=False).mean(); e50=c.ewm(span=50,adjust=False).mean()
    rv=rsi(c); macd=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean()
    sig=macd.ewm(span=9,adjust=False).mean()
    av=v.rolling(20).mean(); traded=(c*v).rolling(20).mean()
    last=g.iloc[-1]; avgvol=float(av.iloc[-1]) if pd.notna(av.iloc[-1]) else 0
    rel=float(last.volume/avgvol) if avgvol else 0
    value=float(traded.iloc[-1]) if pd.notna(traded.iloc[-1]) else 0
    prior_high=h.shift(1).rolling(20).max().iloc[-1]
    r=float(rv.iloc[-1]) if pd.notna(rv.iloc[-1]) else 50
    pts=0; reasons=[]; flags=[]
    if last.close>e20.iloc[-1]>e50.iloc[-1]: pts+=15; reasons.append("EMA20 > EMA50 trend")
    elif last.close>e20.iloc[-1]: pts+=7; reasons.append("Above EMA20")
    else: flags.append("Below EMA20")
    if rel>=2: pts+=15; reasons.append(f"Relative volume {rel:.1f}x")
    elif rel>=1.3: pts+=10; reasons.append(f"Volume rising {rel:.1f}x")
    elif rel>=1: pts+=5
    else: flags.append("Below-average volume")
    if last.close>last.open: pts+=5; reasons.append("Positive candle")
    if pd.notna(prior_high) and last.close>prior_high: pts+=15; reasons.append("20-session breakout")
    elif pd.notna(prior_high) and last.close>=prior_high*.98: pts+=7; reasons.append("Near 20-session high")
    if 50<=r<=68: pts+=10; reasons.append(f"RSI supportive ({r:.0f})")
    elif 45<=r<50 or 68<r<=75: pts+=5
    elif r>75: flags.append(f"RSI extended ({r:.0f})")
    else: flags.append(f"Weak RSI ({r:.0f})")
    if macd.iloc[-1]>sig.iloc[-1]: pts+=10; reasons.append("MACD above signal")
    else: flags.append("MACD below signal")
    if value>=min_value: pts+=10; reasons.append("Liquidity passed")
    else: flags.append("Low traded value")
    if last.close>=min_price: pts+=5
    else: flags.append("Below price floor")
    if last.close>last.low: pts+=5
    if r>80: pts-=15
    if last.close<e50.iloc[-1]: pts-=15
    pts=max(0,min(100,pts))
    entry=float(last.close); stop=min(float(last.low),entry*.985)
    if stop>=entry: stop=entry*.985
    per=entry-stop; maxrisk=capital*risk_pct/100
    qrisk=math.floor(maxrisk/per) if per>0 else 0
    qcap=math.floor(capital/entry) if entry>0 else 0
    qty=max(0,min(qrisk,qcap)); target=entry+2*per
    status="STRONG WATCH" if pts>=80 and not flags else ("WATCH" if pts>=65 else "AVOID")
    return {"Symbol":last.symbol,"Date":last.date.date().isoformat(),"Reference Entry (₹)":round(entry,2),
      "Stop Loss (₹)":round(stop,2),"Target 1:2 (₹)":round(target,2),"Score":pts,"Status":status,
      "RSI":round(r,1),"Rel Volume":round(rel,2),"20D Avg Traded Value (₹)":round(value,0),
      "Quantity":qty,"Planned Risk (₹)":round(qty*per,2),"Potential Gross Reward (₹)":round(qty*(target-entry),2),
      "Positive Factors":"; ".join(reasons),"Risk Flags":"; ".join(flags) or "None"}

with st.sidebar:
    st.markdown("### ⚙️ Risk controls")
    capital=st.number_input("Capital allocation (₹)",min_value=1000,value=50000,step=5000)
    risk_pct=st.slider("Maximum planned risk (%)",.25,2.0,1.0,.25)
    min_price=st.number_input("Minimum price (₹)",min_value=1.0,value=20.0,step=5.0)
    min_value=st.number_input("Minimum 20D average traded value (₹)",min_value=0,value=5000000,step=1000000)
    st.caption("Position size is limited by both allocated capital and planned risk. Fees and slippage are excluded.")

scan, mcp_tab, guide = st.tabs(["✦ Stock Screener","↔ NSE MCP Connection","ⓘ Guide"])

with scan:
    st.subheader("Ranked candidates")
    st.caption("Upload historical OHLCV data to calculate indicators. For stable EMA50 values, 60+ sessions per symbol are recommended.")
    file=st.file_uploader("Upload OHLCV CSV",type=["csv"],help="Required: symbol, date, open, high, low, close, volume")
    if file:
        try:
            df=normalize(pd.read_csv(file))
            rows=[]
            for sym,g in df.groupby("symbol"):
                item=score(g,capital,risk_pct,min_price,min_value)
                if item: rows.append(item)
            if rows:
                ranked=pd.DataFrame(rows).sort_values(["Score","20D Avg Traded Value (₹)"],ascending=[False,False])
                a,b,c=st.columns(3)
                a.metric("Stocks scored",len(ranked))
                b.metric("Strong watch",int((ranked.Status=="STRONG WATCH").sum()))
                c.metric("Watch list",int((ranked.Status=="WATCH").sum()))
                st.dataframe(ranked.head(10),use_container_width=True,hide_index=True)
                st.download_button("Download complete ranked list (CSV)",ranked.to_csv(index=False).encode(), "morningedge_ranked.csv","text/csv")
                with st.expander("View all candidates"): st.dataframe(ranked,use_container_width=True,hide_index=True)
            else: st.info("Each symbol needs at least 30 valid trading sessions. Please upload more history.")
        except Exception as e: st.error(f"Unable to score file: {e}")
    else:
        st.info("No sample prices are shown as live data. Upload an authorized historical OHLCV CSV to run the screener.")
        st.markdown("#### Ranking factors")
        c1,c2,c3=st.columns(3)
        c1.markdown("**Trend & momentum**\n\nEMA20/50 · RSI14 · MACD")
        c2.markdown("**Volume & breakout**\n\nRelative volume · 20-session highs")
        c3.markdown("**Risk discipline**\n\nLiquidity floor · position sizing · 1:2 target")

with mcp_tab:
    st.subheader("Official NSE MCP connector")
    st.caption("Discover the tools and schemas the official endpoint actually exposes; fields are not assumed.")
    endpoint_label=st.selectbox("Data endpoint",list(ENDPOINTS))
    endpoint=ENDPOINTS[endpoint_label]
    if st.button("Test connection and discover tools",type="primary"):
        try:
            with st.spinner("Connecting to NSE MCP..."):
                tools=run_async(list_tools(endpoint))
            st.session_state["tools"]=tools; st.session_state["endpoint"]=endpoint
            st.success(f"Connected — discovered {len(tools)} tools.")
        except Exception as e:
            st.error(f"Connection failed: {type(e).__name__}: {e}")
            st.info("Check internet connectivity and retry. The public endpoint may be temporarily unavailable.")
    tools=st.session_state.get("tools",[])
    if tools and st.session_state.get("endpoint")==endpoint:
        labels=[f"{t.get('name','tool')} — {t.get('description','')[:90]}" for t in tools]
        idx=st.selectbox("Available tool",range(len(tools)),format_func=lambda i:labels[i])
        tool=tools[idx]
        st.write(tool.get("description","No description"))
        schema=tool.get("inputSchema") or tool.get("input_schema") or {}
        st.markdown("**Input schema**"); st.json(schema)
        props=schema.get("properties",{}) if isinstance(schema,dict) else {}
        required=schema.get("required",[]) if isinstance(schema,dict) else []
        args={k:(v.get("default","") if isinstance(v,dict) else "") for k,v in props.items() if k in required}
        args_text=st.text_area("Arguments (JSON)",json.dumps(args,indent=2),height=150)
        if st.button("Call selected NSE tool"):
            try:
                with st.spinner("Calling tool..."):
                    result=run_async(call_tool(endpoint,tool["name"],json.loads(args_text or "{}")))
                st.session_state["last_result"]=result; st.session_state["last_tool"]=tool["name"]
            except Exception as e: st.error(f"Tool call failed: {e}")
        if st.session_state.get("last_tool")==tool["name"]:
            res=st.session_state["last_result"]
            if res["error"]: st.error("The server returned a tool error.")
            for i,content in enumerate(res["content"]):
                st.text_area(f"Response {i+1}",content,height=220)
                st.download_button(f"Download response {i+1}",content,file_name=f"nse_response_{i+1}.txt",key=f"response_dl_{i}")
    else: st.info("Test the connection to load the available tools.")

with guide:
    st.subheader("Start here")
    st.markdown("""
**1. Connect:** Open NSE MCP Connection and discover available tools.  
**2. Screen:** Upload a CSV with columns `symbol,date,open,high,low,close,volume`.  
**3. Review:** Inspect the ranked list, reasons and risk flags.  
**4. Validate:** Backtest and paper-trade before risking money.

### Risk model
- Target = entry + 2 × (entry − stop loss)
- Quantity is capped by both available capital and maximum planned loss.
- A 1:2 target is a calculation, not a promise that the target will be reached.

### Limitations
- Daily OHLCV cannot validate live pre-open imbalance, intraday VWAP or opening confirmation.
- Current MCP market data may be delayed; do not use it as an execution feed.
- Scoring is a heuristic, not a probability of profit or validated investment advice.
- No broker connection, order placement or automatic trading.
- Review the current NSE MCP terms before use: https://www.nseindia.com/nse-mcp
""")
st.markdown("<div class='small-muted'>MorningEdge AI · Research workspace · No trading orders are placed by this application.</div>",unsafe_allow_html=True)
