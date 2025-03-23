import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import ta

# 設定 Streamlit 頁面標題與圖示
st.set_page_config(page_title="股票走勢分析工具", page_icon="📈")

# 主標題
st.title("📊 股票走勢分析工具")

# 讓使用者輸入想查詢的股票代號，預設為 "TSLA"
stock_symbol = st.text_input("請輸入股票代號（例如：AAPL, TSLA, GOOG）", value="TSLA")

# 只有在使用者輸入了股票代號時才執行以下邏輯
if stock_symbol:
    # 下載該股票最近一年的資料（日線）
    data = yf.download(stock_symbol, period="1y", interval="1d")

    # 計算 20 日與 50 日移動平均線
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()

    # 計算 RSI 指標
    close_price = data['Close']
    # 若 close_price 是 DataFrame（很少見，但做個保險判斷）
    if isinstance(close_price, pd.DataFrame):
        close_price = close_price.iloc[:, 0]
    data['RSI'] = ta.momentum.RSIIndicator(close=close_price, window=14).rsi()

    # RSI 圖表區塊
    st.subheader(f"📉 {stock_symbol} RSI 指標圖表")
    fig_rsi, ax_rsi = plt.subplots(figsize=(10, 3))
    ax_rsi.plot(data.index, data['RSI'], label='RSI', color='purple')
    ax_rsi.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    ax_rsi.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    ax_rsi.set_title(f"{stock_symbol} RSI Indicator")
    ax_rsi.set_ylabel("RSI")
    ax_rsi.legend()
    st.pyplot(fig_rsi)

    # MA 買入賣出訊號
    data['Signal'] = 0
    data.loc[data['MA20'] > data['MA50'], 'Signal'] = 1
    data.loc[data['MA20'] < data['MA50'], 'Signal'] = -1

    # 建立預測欄位：把明日收盤價往上移一格
    data['Prediction'] = data['Close'].shift(-1)

    # 移除空值
    data.dropna(inplace=True)

    # 準備訓練資料
    X = np.array(data['Close']).reshape(-1, 1)
    y = np.array(data['Prediction'])

    split = int(len(X) * 0.8)  # 80% 當作訓練資料，20% 當作測試資料
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 建立並訓練隨機森林模型
    model = RandomForestRegressor(n_estimators=_
