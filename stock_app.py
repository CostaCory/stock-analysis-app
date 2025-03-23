import ta
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# 網頁標題
st.title("📈 AI股票價格預測工具")

# 用戶輸入股票代號
stock_symbol = st.text_input("請輸入股票代號（例如：AAPL, TSLA, GOOG）：", "TSLA")

# 下載股票數據
data = yf.download(stock_symbol, period="1y", interval="1d")

# 計算移動平均線 (MA)
data['MA20'] = data['Close'].rolling(window=20).mean()  # 20日移動平均線
data['MA50'] = data['Close'].rolling(window=50).mean()  # 50日移動平均線

# 找出買入和賣出訊號（黃金交叉與死亡交叉）
data['Signal'] = 0.0
data['Signal'][20:] = np.where(data['MA20'][20:] > data['MA50'][20:], 1.0, 0.0)
data['Position'] = data['Signal'].diff()

# 取得最近一次買賣訊號
latest_signal = data['Position'].dropna().iloc[-1]
latest_signal_date = data['Position'].dropna().index[-1]

if latest_signal == 1.0:
    signal_text = f"🟢 黃金交叉（買入訊號）於 {latest_signal_date.date()} 出現，股票短期可能繼續上升！"
elif latest_signal == -1.0:
    signal_text = f"🔴 死亡交叉（賣出訊號）於 {latest_signal_date.date()} 出現，股票短期可能轉弱下跌！"
else:
    signal_text = "⚠️ 目前冇明顯買入或賣出訊號。"

# 顯示訊號喺網頁上
st.subheader("📌 股票買賣訊號提示")
st.markdown(signal_text)

# 顯示最近幾日數據
st.subheader(f"{stock_symbol} 最近股票數據")
st.dataframe(data.tail(5))

# 畫股票價格走勢圖
st.subheader(f"{stock_symbol} 股票價格走勢")
fig, ax = plt.subplots()
ax.plot(data.index, data["Close"], label="Close Price")
ax.plot(data.index, data["MA20"], label="20-day MA", linestyle='--')
ax.plot(data.index, data["MA50"], label="50-day MA", linestyle='-.')
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.set_title(f"{stock_symbol} Stock Price with Moving Averages")
ax.legend()
st.pyplot(fig)

ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.set_title(f"{stock_symbol} Stock Price - Past Year")
ax.legend()
st.pyplot(fig)

# --- 新增：AI預測股票價格功能 ---
st.subheader("🔮 AI 預測下一日收盤價")

# 準備數據 (利用過去收盤價預測未來收盤價)
data['Prediction'] = data['Close'].shift(-1)
data.dropna(inplace=True)

X = np.array(data[['Close']])
y = np.array(data['Prediction'])

# 使用 Random Forest 模型訓練
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# 預測下一日股票價格
last_close_price = X[-1].reshape(1, -1)
predicted_price = model.predict(last_close_price)[0]

# 顯示預測結果
st.markdown(f"#### 預測 {stock_symbol} 下一個交易日嘅收盤價為： **{predicted_price:.2f} USD** 🚀")

# ➕ 新增 RSI 技術指標區塊
st.subheader(f"{stock_symbol} RSI 技術指標")

# 計算 RSI（預設為 14 日）
data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()


# 畫 RSI 圖表
fig, ax = plt.subplots()
ax.plot(data.index, data["RSI"], label="RSI (14)", color="purple")
ax.axhline(70, linestyle='--', color='red', label="Overbought (70)")
ax.axhline(30, linestyle='--', color='green', label="Oversold (30)")
ax.set_ylabel("RSI")
ax.set_xlabel("Date")
ax.set_title(f"{stock_symbol} RSI Indicator")
ax.legend()
st.pyplot(fig)
