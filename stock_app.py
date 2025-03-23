import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import ta

st.set_page_config(page_title="📈 股票走勢分析工具", layout="wide")

st.title("📊 股票走勢分析工具")
stock_symbol = st.text_input("請輸入股票代號（例如：AAPL, TSLA, GOOG）", value="TSLA")

if stock_symbol:
    data = yf.download(stock_symbol, period="1y", interval="1d")
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()

    # RSI 修正
    close_price = data['Close']
    if isinstance(close_price, pd.DataFrame):
        close_price = close_price.iloc[:, 0]
    data['RSI'] = ta.momentum.RSIIndicator(close=close_price, window=14).rsi()

    # MA 買賣訊號標註
    data['Signal'] = 0
    data.loc[data['MA20'] > data['MA50'], 'Signal'] = 1
    data.loc[data['MA20'] < data['MA50'], 'Signal'] = -1