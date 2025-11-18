import streamlit as st
import yfinance as yf
import numpy as np
import time
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

st.set_page_config(page_title="Simple Stock Prediction", layout="wide")
st.title("📈 Real-Time Stock Price Prediction")

stock_symbol = st.text_input("Enter Stock Symbol:", "AAPL")

prices = []
predicted_prices = []
time_index = []

model = LinearRegression()
chart = st.empty()
status_text = st.empty()
trend_text = st.empty()

update_interval = st.slider("Update Interval (seconds):", 10, 120, 60, 10)

while True:
    try:
        data = yf.download(
            tickers=stock_symbol,
            period="1d",
            interval="1m",
            auto_adjust=True,
            progress=False
        )

        if data.empty:
            status_text.warning("Waiting for data...")
            time.sleep(update_interval)
            continue
        else:
            status_text.success(f"Latest data fetched for {stock_symbol}")
        latest_price = float(data['Close'].iloc[-1])
        prices.append(latest_price)
        time_index.append(len(prices))

        if len(prices) >= 5:
            X = np.arange(len(prices)).reshape(-1, 1)
            y = np.array(prices)
            model.fit(X, y)
            next_pred = model.predict([[len(prices)]])[0]
            predicted_prices.append(next_pred)
        else:
            predicted_prices.append(latest_price)

        if len(prices) > 1:
            if prices[-1] > prices[-2]:
                trend = "📈 Up"
            elif prices[-1] < prices[-2]:
                trend = "📉 Down"
            else:
                trend = "➡️ Stable"
            trend_text.markdown(f"**Trend:** {trend} | Latest Price: {latest_price:.2f} | Predicted Next Price: {predicted_prices[-1]:.2f}")
        else:
            trend_text.markdown(f"Latest Price: {latest_price:.2f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_index, y=prices,
            mode="lines+markers", name="Actual Price",
            line=dict(color="blue")
        ))
        fig.add_trace(go.Scatter(
            x=time_index, y=predicted_prices,
            mode="lines+markers", name="Predicted Price",
            line=dict(color="red", dash="dash")
        ))
        fig.update_layout(
            title=f"{stock_symbol} Price & Prediction",
            xaxis_title="Time (minutes)",
            yaxis_title="Price",
            height=500
        )

        chart.plotly_chart(fig, use_container_width=True)

        time.sleep(update_interval)

    except Exception as e:
        status_text.error(f"Error fetching data: {e}")
        time.sleep(update_interval)
