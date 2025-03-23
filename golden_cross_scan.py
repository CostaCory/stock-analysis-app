
import yfinance as yf
import pandas as pd

# Define the stock list (you can expand this list)
stock_list = ['AAPL', 'TSLA', 'GOOG', 'META', 'MSFT', 'NVDA', 'AMZN', 'AMD', 'NFLX', 'INTC']

golden_cross_stocks = []

for symbol in stock_list:
    data = yf.download(symbol, period='6mo', interval='1d')
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    
    if data['MA20'].iloc[-2] < data['MA50'].iloc[-2] and data['MA20'].iloc[-1] > data['MA50'].iloc[-1]:
        golden_cross_stocks.append(symbol)

if golden_cross_stocks:
    print("✅ 出現 Golden Cross 股票：")
    for stock in golden_cross_stocks:
        print("•", stock)
else:
    print("❌ 暫時無 Golden Cross 股票")
