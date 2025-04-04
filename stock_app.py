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

    st.markdown("""
## ℹ️ RSI 技術指標圖解說明

**RSI（Relative Strength Index，相對強弱指標）** 係一種常見技術分析工具，用嚟衡量股票喺一段時間內嘅「漲跌強度」，幫助判斷市場係咪過熱（超買）或者過冷（超賣）。

---

### 📌 指標解讀：
- **RSI > 70** 👉 可能「超買」，即價格升咗好多，有機會短期回調。
- **RSI < 30** 👉 可能「超賣」，即價格跌咗好多，有機會短期反彈。
- **RSI 介乎 30-70** 👉 屬於正常波動區，無明顯超買／超賣信號。

---

### 📊 圖表說明：
- **紫色線**：RSI 變化走勢
- **紅色虛線（70）**：超買區上限
- **綠色虛線（30）**：超賣區下限

---

### 💡 應用建議：
- RSI **配合其他指標使用會更準確**（例如：MA、MACD）
- 可用嚟判斷短線轉勢點或作為買賣參考
- RSI 只係參考，**唔等於必定漲／跌**，請結合市場消息同基本面一齊判斷

---

### ⚠️ 注意事項：
- RSI 唔適用於橫盤整理市，容易出現假信號
- 部分股票會長期處於高 RSI 區域（例如強勢股），要小心判斷

""")

    # 📉 RSI 指標圖表
    fig_rsi, ax_rsi = plt.subplots(figsize=(10, 3))
    ax_rsi.plot(data.index, data['RSI'], label='RSI', color='purple')
    ax_rsi.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    ax_rsi.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    ax_rsi.set_title(f"{stock_symbol} RSI Indicator")
    ax_rsi.set_ylabel("RSI")
    ax_rsi.legend()
    st.pyplot(fig_rsi)

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

# 🔍 Golden Cross 股票掃描功能
def scan_golden_cross_stocks():
    import yfinance as yf
    import pandas as pd

    stock_list = ['AAPL', 'GOOG', 'META', 'AMZN', 'MSFT', 'TSLA', 'NVDA', 'NFLX', 'INTC', 'AMD']
    golden_cross_stocks = []

    for symbol in stock_list:
        try:
            df = yf.download(symbol, period="6mo", interval="1d")
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()

            if df['MA20'].iloc[-1] > df['MA50'].iloc[-1] and df['MA20'].iloc[-2] <= df['MA50'].iloc[-2]:
                golden_cross_stocks.append(symbol)
        except Exception as e:
            print(f"Error checking {symbol}: {e}")

    return golden_cross_stocks

# 📈 Golden Cross 股票掃描
st.subheader("📈 Golden Cross 股票掃描")
with st.spinner("掃描中，請稍候..."):
    gc_stocks = scan_golden_cross_stocks()
if gc_stocks:
    st.success("✅ 出現 Golden Cross 訊號的股票：")
    st.table(gc_stocks)
else:
    st.warning("暫時未發現 Golden Cross 股票")

# 🔮 AI 預測下一日收盤價顯示
try:
    last_close = data['Close'].iloc[-1]

    # 嘗試轉為單一值（scalar）
    if isinstance(last_close, pd.Series):
        last_close = last_close.item()

    if pd.isna(last_close):
        st.warning("⚠️ 無法預測下一日收盤價（收盤價為空值或無效）")
    else:
        next_day_prediction = model.predict(np.array([[last_close]]))[0]
        st.subheader("🔮 AI 預測下一日收盤價")
        st.write(f"預測下一交易日收盤價：${round(next_day_prediction, 2)}")

except Exception as e:
    st.error("⚠️ 無法計算預測下一日收盤價。請確認資料是否完整。")
    st.exception(e)
