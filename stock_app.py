
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta  # 替代 pandas_ta，計算 RSI

# 輸入股票代號
stock_symbol = st.text_input("請輸入股票代號（例如：AAPL, TSLA, GOOG）", "TSLA")
data = yf.download(stock_symbol, period="1y")

# 計算移動平均線
data['MA20'] = data['Close'].rolling(window=20).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()

# 計算 RSI
close_series = data['Close'].squeeze()  # 確保係一維
data['RSI'] = ta.momentum.RSIIndicator(close_series).rsi()

# 判斷買賣訊號
data['Signal'] = np.where(
    (data['MA20'] > data['MA50']) & (data['MA20'].shift(1) <= data['MA50'].shift(1)), 1,
    np.where((data['MA20'] < data['MA50']) & (data['MA20'].shift(1) >= data['MA50'].shift(1)), -1, 0)
)

# 預測下一日收盤價（簡單 AI 模型）
data['Prediction'] = data['Close'].shift(-1)
data.dropna(inplace=True)

# 顯示數據表
st.subheader(f"{stock_symbol} 最近一年數據")
st.dataframe(data.tail())

# 畫圖：股價走勢 + MA
st.subheader(f"{stock_symbol} 股票價格走勢")
fig, ax = plt.subplots()
ax.plot(data.index, data['Close'], label="Close Price")
ax.plot(data.index, data['MA20'], label="20-day MA", linestyle='--')
ax.plot(data.index, data['MA50'], label="50-day MA", linestyle='-.')
ax.set_title(f"{stock_symbol} Stock Price with Moving Averages")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend()
st.pyplot(fig)

# 顯示 RSI 結果
st.subheader("📉 RSI 技術指標分析")
latest_rsi = data['RSI'].iloc[-1]
if latest_rsi > 70:
    st.warning(f"🔺 RSI = {latest_rsi:.2f}，處於超買區，可能出現回調")
elif latest_rsi < 30:
    st.success(f"🔻 RSI = {latest_rsi:.2f}，處於超賣區，可能反彈")
else:
    st.info(f"⚪ RSI = {latest_rsi:.2f}，處於正常範圍")

# 顯示買賣訊號
st.subheader("📌 MA 買賣訊號")
if data['Signal'].iloc[-1] == 1:
    st.success("📈 Golden Cross（買入訊號）")
elif data['Signal'].iloc[-1] == -1:
    st.error("📉 Death Cross（賣出訊號）")
else:
    st.info("💤 暫時未出現明確買賣訊號")

# 預測收盤價
st.subheader("🔮 AI 預測下一日收盤價")
predicted_price = data['Prediction'].iloc[-1]
st.write(f"預測 {stock_symbol} 下一個交易日收盤價為：**{predicted_price:.2f} USD** 🚀")
