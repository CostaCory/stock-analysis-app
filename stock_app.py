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

    # 若抓不到任何資料，提醒使用者
    if data.empty:
        st.warning("無法下載到此股票的資料，請確認股票代號或網路連線是否正確。")
    else:
        # 計算 20 日與 50 日移動平均線
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()

        # 計算 RSI 指標
        close_price = data['Close']
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

        # 如果資料量太少，模型可能無法訓練，做個判斷
        if len(data) < 2:
            st.warning("資料量不足，無法進行預測。")
        else:
            # 準備訓練資料
            X = np.array(data['Close']).reshape(-1, 1)
            y = np.array(data['Prediction'])

            split = int(len(X) * 0.8)  # 80% 當作訓練資料，20% 當作測試資料
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            # 建立並訓練隨機森林模型
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # 預測測試集
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)

            # 預測下一日收盤價：以最後一天的收盤價做預測
            last_close = data['Close'].iloc[-1]
            next_day_prediction = model.predict(np.array([[last_close]]))[0]

            # 顯示預測結果（下一日收盤價）
            st.subheader("📈 預測結果")
            st.metric(label="預測明日收盤價", value=f"{next_day_prediction:.2f}")

            # 顯示最近幾筆資料
            st.subheader(f"{stock_symbol} 最近一年數據（最後 5 筆）")
            st.dataframe(data.tail(5))

            # 畫出股票價格與 MA 走勢圖
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

            # 顯示 RSI 最新數值
            st.subheader("📉 RSI 技術指標分析")
            st.write(f"RSI = {round(data['RSI'].iloc[-1], 2)}")

            # 顯示 MA 買賣訊號
            st.subheader("📌 MA 買賣訊號")
            signal_value = data['Signal'].iloc[-1]
            if signal_value == 1:
                st.success("出現買入訊號（黃金交叉）")
            elif signal_value == -1:
                st.error("出現賣出訊號（死亡交叉）")
            else:
                st.info("暫時未出現明顯買賣訊號")

            # 顯示預測誤差
            st.subheader("🎯 預測誤差 MSE")
            st.write(f"MSE（預測誤差）: {round(mse, 4)}")


# ------------------------------------------------------------
# 下面是 Golden Cross 股票掃描功能
def scan_golden_cross_stocks():
    """
    掃描預先設定的股票清單，若出現 MA20 上穿 MA50 則視為 Golden Cross。
    回傳出現 Golden Cross 的股票代號清單。
    """
    import yfinance as yf
    import pandas as pd

    stock_list = ['AAPL', 'GOOG', 'META', 'AMZN', 'MSFT', 'TSLA', 'NVDA', 'NFLX', 'INTC', 'AMD']
    golden_cross_stocks = []

    for symbol in stock_list:
        try:
            df = yf.download(symbol, period="6mo", interval="1d")
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()

            # 檢查最後兩天 MA20 與 MA50 的相對關係
            if (df['MA20'].iloc[-1] > df['MA50'].iloc[-1]) and (df['MA20'].iloc[-2] <= df['MA50'].iloc[-2]):
                golden_cross_stocks.append(symbol)
        except Exception as e:
            print(f"Error checking {symbol}: {e}")

    return golden_cross_stocks

# 顯示 Golden Cross 股票掃描結果
st.subheader("📈 Golden Cross 股票掃描")
with st.spinner("掃描中，請稍候..."):
    gc_stocks = scan_golden_cross_stocks()

if gc_stocks:
    st.success("✅ 出現 Golden Cross 訊號的股票：")
    st.table(gc_stocks)
else:
    st.warning("暫時未發現 Golden Cross 股票")
