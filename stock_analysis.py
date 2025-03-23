import yfinance as yf

# 設定要下載的股票代號（例如蘋果公司係「TSLA」）
stock_symbol = "TSLA"

# 下載過去一年嘅股票數據
data = yf.download(stock_symbol, period="1y", interval="1d")

# 印出數據（頭5日）
print("下載到嘅股票數據：")
print(data.head())

import matplotlib.pyplot as plt

# 用 matplotlib 繪製股票收盤價
data['Close'].plot(title='Tesla Stock Price - Past Year', figsize=(10, 6))

# 設定圖表標籤
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')

# 顯示圖表
plt.grid(True)
plt.show()

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 準備數據（用過去股票收盤價預測下一日嘅收盤價）
data['Prediction'] = data['Close'].shift(-1)

# 移除有缺失嘅數據
data.dropna(inplace=True)

# 製作特徵 (X) 同預測目標 (y)
X = np.array(data[['Close']])
y = np.array(data['Prediction'])

# 分割數據為訓練集同測試集（80%訓練, 20%測試）
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 使用 Random Forest 模型
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# 用測試集做預測
predictions = model.predict(X_test)

# 計算模型誤差
mse = mean_squared_error(y_test, predictions)
print(f"模型的均方誤差 (Mean Squared Error): {mse:.2f}")

# 印出未來一日的股票價格預測
last_closing_price = X[-1].reshape(1, -1)
next_day_prediction = model.predict(last_closing_price)
print(f"預測下一日嘅收盤價係：{next_day_prediction[0]:.2f} USD")
