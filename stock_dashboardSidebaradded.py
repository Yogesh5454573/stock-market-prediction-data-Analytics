import streamlit as st
import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import time

st.set_page_config(page_title="📈 Real-Time Stock Prediction", layout="wide")

# --- Auto NSE Converter ---
def convert_symbol(symbol):
    symbol = symbol.strip().upper()
    if "." in symbol:
        return symbol
    return symbol + ".NS"   # convert Indian stock automatically

# --- Sidebar Settings ---
st.sidebar.header("⚙️ Settings")
user_input = st.sidebar.text_input("Enter Stock Symbol", "TCS")
stock_symbol = convert_symbol(user_input)

update_interval = st.sidebar.slider("Update Interval (seconds):", 10, 120, 30)

# --- Page Title ---
st.title("📊 Real-Time Stock Price Monitoring & Prediction")

# Data containers
if "prices" not in st.session_state:
    st.session_state.prices = []
if "predicted" not in st.session_state:
    st.session_state.predicted = []
if "time_index" not in st.session_state:
    st.session_state.time_index = []

# Model
model = LinearRegression()

# Fetch Live Data
try:
    df = yf.download(
        tickers=stock_symbol,
        period="1d",
        interval="1m",
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        st.error("⚠ No data available for this stock!")
        st.stop()

    latest_price = float(df["Close"].iloc[-1])
    st.session_state.prices.append(latest_price)
    st.session_state.time_index.append(len(st.session_state.prices))

    # Prediction
    if len(st.session_state.prices) >= 5:
        X = np.arange(len(st.session_state.prices)).reshape(-1, 1)
        y = np.array(st.session_state.prices)
        model.fit(X, y)
        next_pred = model.predict([[len(st.session_state.prices)]])[0]
        st.session_state.predicted.append(next_pred)
    else:
        st.session_state.predicted.append(latest_price)

    # Trend detection
    trend = "➡️ Stable"
    if len(st.session_state.prices) > 1:
        if st.session_state.prices[-1] > st.session_state.prices[-2]:
            trend = "📈 Up"
        elif st.session_state.prices[-1] < st.session_state.prices[-2]:
            trend = "📉 Down"

except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# --- UI Layout ---
col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="Current Price",
        value=f"₹{latest_price:.2f}"
    )

with col2:
    st.metric(
        label="Predicted Next Price",
        value=f"₹{st.session_state.predicted[-1]:.2f}"
    )

st.subheader(f"Trend: {trend}")

# --- Chart ---
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
    title=f"{stock_symbol} - Price & Prediction",
    height=500,
    xaxis_title="Time (updates)",
    yaxis_title="Price"
)

st.plotly_chart(fig, use_container_width=True)

# Auto-Refresh
time.sleep(update_interval)
st.rerun()

