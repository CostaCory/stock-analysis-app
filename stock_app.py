
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import ta

st.set_page_config(page_title="股票走勢分析工具", page_icon="📈")

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

    # MA 買入賣出訊號
    data['Signal'] = 0
    data.loc[data['MA20'] > data['MA50'], 'Signal'] = 1
    data.loc[data['MA20'] < data['MA50'], 'Signal'] = -1

    # 預測下一日收盤價
    data['Prediction'] = data['Close'].shift(-1)
    data.dropna(inplace=True)

    X = np.array(data['Close']).reshape(-1, 1)
    y = np.array(data['Prediction'])

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)

    st.subheader(f"{stock_symbol} 最近一年數據")
    st.dataframe(data.tail(5))

    st.subheader(f"{stock_symbol} 股票價格走勢")
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Close'], label='Close Price')
    ax.plot(data.index, data['MA20'], label='20-day MA', linestyle='--')
    ax.plot(data.index, data['MA50'], label='50-day MA', linestyle='-.')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.set_title(f"{stock_symbol} Stock Price with Moving Averages")
    ax.legend()
    st.pyplot(fig)

    st.subheader("📉 RSI 技術指標分析")
    st.write(f"RSI = {round(data['RSI'].iloc[-1], 2)}")

    st.subheader("📌 MA 買賣訊號")
    signal_value = data['Signal'].iloc[-1]
    if signal_value == 1:
        st.success("出現買入訊號（黃金交叉）")
    elif signal_value == -1:
        st.error("出現賣出訊號（死亡交叉）")
    else:
        st.info("暫時未出現明顯買賣訊號")

    st.subheader("🎯 預測誤差 MSE")
    st.write(f"MSE（預測誤差）: {round(mse, 4)}")
