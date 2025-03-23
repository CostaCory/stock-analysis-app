import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas_ta as ta

st.set_page_config(page_title="📈 AI 股票分析工具", layout="wide")
st.title("📊 股票走勢分析工具")
stock_symbol = st.text_input("請輸入股票代號（例如：AAPL, TSLA, GOOG）", "TSLA")

if stock_symbol:
    data = yf.download(stock_symbol, period="1y", interval="1d")
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_signal'] = macd.signal()

    # Signal 判斷
    data['signal'] = np.where(data['MA20'] > data['MA50'], 1, 0)
    data['Prediction'] = data['Close'].shift(-1)
    data.dropna(inplace=True)

    st.subheader(f"{stock_symbol} 最近一年數據")
    st.dataframe(data.tail(5))

    # 📈 價格走勢圖
    st.subheader(f"{stock_symbol} 股票價格走勢")
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Close'], label="Close Price")
    ax.plot(data.index, data['MA20'], label="20-day MA", linestyle="--")
    ax.plot(data.index, data['MA50'], label="50-day MA", linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.set_title("Stock Price with Moving Averages")
    ax.legend()
    st.pyplot(fig)

    # 📉 RSI 圖
    st.markdown("📉 **RSI 技術指標圖**")
    fig_rsi, ax_rsi = plt.subplots()
    ax_rsi.plot(data.index, data['RSI'], color='purple')
    ax_rsi.axhline(70, color='red', linestyle='--')
    ax_rsi.axhline(30, color='green', linestyle='--')
    ax_rsi.set_title("RSI 指標")
    ax_rsi.set_ylabel("RSI")
    st.pyplot(fig_rsi)

    # 📉 MACD 圖
    st.markdown("📉 **MACD 技術指標圖**")
    fig_macd, ax_macd = plt.subplots()
    ax_macd.plot(data.index, data['MACD'], label='MACD', color='blue')
    ax_macd.plot(data.index, data['MACD_signal'], label='Signal Line', color='orange')
    ax_macd.set_title("MACD 指標")
    ax_macd.legend()
    st.pyplot(fig_macd)

    # 📊 預測模型
    X = np.array(data[['Close']])
    y = np.array(data['Prediction'])
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)

    # 🔍 顯示 MSE
    st.subheader("📉 AI 預測誤差分析")
    st.info(f"預測誤差（MSE）：{mse:.2f}")

    # 🔮 預測下一日收盤價
    last_closing_price = data['Close'].iloc[-1]
    next_day_prediction = model.predict(np.array([[last_closing_price]]))[0]
    st.subheader("📌 AI 預測下一日收盤價")
    st.success(f"{stock_symbol} 下一日預測收盤價：${next_day_prediction:.2f}")
