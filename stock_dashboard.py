import streamlit as st
import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import time

st.set_page_config(page_title="📈 Real-Time Stock Prediction & Analytics", layout="wide")

# --- Auto NSE Converter ---
def convert_symbol(symbol):
    symbol = symbol.strip().upper()
    if "." in symbol:
        return symbol
    return symbol + ".NS"   # Auto convert Indian stocks


# --- Sidebar Settings ---
st.sidebar.header("⚙️ Settings")
user_input = st.sidebar.text_input("Enter Stock Symbol", "TCS")
stock_symbol = convert_symbol(user_input)

update_interval = st.sidebar.slider("Update Interval (seconds):", 10, 120, 30)

# --- Page Title ---
st.title("📊 Real-Time Stock Price Monitoring, Analytics & Prediction")


# --- Session State (Stores Streaming Data) ---
if "prices" not in st.session_state:
    st.session_state.prices = []
if "predicted" not in st.session_state:
    st.session_state.predicted = []
if "time_index" not in st.session_state:
    st.session_state.time_index = []


# --- ML Model ---
model = LinearRegression()

# --- Fetch Live Data ---
try:
    df = yf.download(
        tickers=stock_symbol,
        period="1d",
        interval="1m",
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        st.error("⚠ No data found for this stock!")
        st.stop()

    latest_price = float(df["Close"].iloc[-1])
    st.session_state.prices.append(latest_price)
    st.session_state.time_index.append(len(st.session_state.prices))

    # --- Prediction ---
    if len(st.session_state.prices) >= 5:
        X = np.arange(len(st.session_state.prices)).reshape(-1, 1)
        y = np.array(st.session_state.prices)
        model.fit(X, y)
        pred = model.predict([[len(st.session_state.prices)]])[0]
        st.session_state.predicted.append(pred)
    else:
        st.session_state.predicted.append(latest_price)

    # --- Trend ---
    trend = "➡️ Stable"
    if len(st.session_state.prices) > 1:
        if st.session_state.prices[-1] > st.session_state.prices[-2]:
            trend = "📈 Up"
        elif st.session_state.prices[-1] < st.session_state.prices[-2]:
            trend = "📉 Down"

except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()


# -------------------------------------------------------
#                     PRICE ANALYTICS
# -------------------------------------------------------
st.header("📌 Price Analytics")

info = yf.Ticker(stock_symbol).info

current_price = info.get("currentPrice", latest_price)
prev_close = info.get("previousClose", None)
day_high = info.get("dayHigh", None)
day_low = info.get("dayLow", None)
week52_high = info.get("fiftyTwoWeekHigh", None)
week52_low = info.get("fiftyTwoWeekLow", None)
volume = info.get("volume", None)
market_cap = info.get("marketCap", None)

# % change
pct_change = None
if prev_close:
    pct_change = ((current_price - prev_close) / prev_close) * 100

# Moving Averages
daily_df = yf.download(stock_symbol, period="1y", interval="1d", auto_adjust=True, progress=False)
daily_df["MA20"] = daily_df["Close"].rolling(20).mean()
daily_df["MA50"] = daily_df["Close"].rolling(50).mean()
daily_df["MA200"] = daily_df["Close"].rolling(200).mean()

# ------- Display Cards -------
col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"₹{current_price}")
col2.metric("Previous Close", f"₹{prev_close}")
col3.metric("Daily % Change", f"{pct_change:.2f}%" if pct_change else "---")

col4, col5, col6 = st.columns(3)
col4.metric("Day High", f"₹{day_high}")
col5.metric("Day Low", f"₹{day_low}")
col6.metric("Volume", f"{volume:,}" if volume else "---")

col7, col8, col9 = st.columns(3)
col7.metric("52-Week High", f"₹{week52_high}")
col8.metric("52-Week Low", f"₹{week52_low}")
col9.metric("Market Cap", f"{market_cap:,}" if market_cap else "---")

# -------------------------------------------------------
#                 REAL-TIME PRICE & PREDICTION
# -------------------------------------------------------

st.subheader(f"Trend: {trend}")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=st.session_state.time_index,
    y=st.session_state.prices,
    mode="lines+markers",
    name="Actual Price"
))
fig.add_trace(go.Scatter(
    x=st.session_state.time_index,
    y=st.session_state.predicted,
    mode="lines+markers",
    name="Predicted Price"
))
fig.update_layout(
    title=f"{stock_symbol} - Real-Time Price & Prediction",
    height=500,
    xaxis_title="Update Count",
    yaxis_title="Price"
)
st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------
#              MOVING AVERAGE ANALYTICS CHART
# -------------------------------------------------------

st.header("📈 Moving Average Chart (20 / 50 / 200 Days)")

ma_fig = go.Figure()
ma_fig.add_trace(go.Scatter(x=daily_df.index, y=daily_df["Close"], name="Close Price"))
ma_fig.add_trace(go.Scatter(x=daily_df.index, y=daily_df["MA20"], name="MA20"))
ma_fig.add_trace(go.Scatter(x=daily_df.index, y=daily_df["MA50"], name="MA50"))
ma_fig.add_trace(go.Scatter(x=daily_df.index, y=daily_df["MA200"], name="MA200"))

ma_fig.update_layout(height=500)
st.plotly_chart(ma_fig, use_container_width=True)

# -------------------------------------------------------
#                      AUTO REFRESH
# -------------------------------------------------------
time.sleep(update_interval)
st.rerun()
