from kafka import KafkaProducer
import yfinance as yf
import json
import time

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def get_price(stock="AAPL"):
    data = yf.download(tickers=stock, period='1d', interval='1m')
    price = float(data['Close'][-1])
    return price

while True:
    price = get_price("AAPL")
    payload = {"symbol": "AAPL", "price": price}
    producer.send("stock_prices", payload)
    print("Sent:", payload)
    time.sleep(2)
