import pandas as pd
import matplotlib
matplotlib.use('Agg')

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import yfinance as yf
import os
import uuid

def calculate_rsi(series, period=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def predict_stock_price(stock_ticker, start_date, end_date):
    try:
        stock_ticker = stock_ticker.upper()
        print(f"Trying to fetch {stock_ticker}...")

        # Try downloading the stock
        data = yf.download(stock_ticker, start=start_date, end=end_date)

        # If empty, try .NS
        if data.empty:
            print(f"No data for {stock_ticker}. Trying with .NS...")
            if not stock_ticker.endswith('.NS'):
                stock_ticker_ns = stock_ticker + '.NS'
                data = yf.download(stock_ticker_ns, start=start_date, end=end_date)
                if not data.empty:
                    print(f"Fetched data with {stock_ticker_ns}")
                    stock_ticker = stock_ticker_ns
                else:
                    print(f"No data for {stock_ticker_ns}. Trying with .BO...")
                    stock_ticker_bo = stock_ticker.replace('.NS', '') + '.BO'
                    data = yf.download(stock_ticker_bo, start=start_date, end=end_date)
                    if not data.empty:
                        print(f"Fetched data with {stock_ticker_bo}")
                        stock_ticker = stock_ticker_bo
                    else:
                        raise ValueError(f"No data found for {stock_ticker}, {stock_ticker_ns}, or {stock_ticker_bo} between {start_date} and {end_date}")
            else:
                raise ValueError(f"No data found for {stock_ticker} between {start_date} and {end_date}")
        else:
            print(f"Fetched data with {stock_ticker}")

        data['Price'] = data['Close']

        # Feature engineering
        data['High-Low'] = data['High'] - data['Low']
        data['Open-Close'] = data['Open'] - data['Close']
        data['MA_10'] = data['Close'].rolling(window=10).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        data['RSI'] = calculate_rsi(data['Close'])

        data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

        data['TR'] = data[['High', 'Low', 'Close']].max(axis=1) - data[['High', 'Low', 'Close']].min(axis=1)
        data['ATR'] = data['TR'].rolling(window=14).mean()

        for lag in range(1, 11):
            data[f'Lag_{lag}'] = data['Close'].shift(lag)

        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['Std_Dev_20'] = data['Close'].rolling(window=20).std()
        data['Bollinger_High'] = data['MA_20'] + 2 * data['Std_Dev_20']
        data['Bollinger_Low'] = data['MA_20'] - 2 * data['Std_Dev_20']

        data['Next_Day_Close'] = data['Close'].shift(-1)
        actual_next_day_price = data['Close'].iloc[-1]

        data = data.dropna()

        if data.empty:
            raise ValueError("After processing, no sufficient data available for training.")

        features = [
            'Open', 'High', 'Low', 'Volume', 'High-Low', 'Open-Close', 'MA_10', 'MA_50',
            'RSI', 'MACD', 'MACD_Signal', 'Bollinger_High', 'Bollinger_Low', 'ATR'
        ] + [f'Lag_{lag}' for lag in range(1, 11)]

        X = data[features]
        y = data['Next_Day_Close']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError("Training or testing set is empty after splitting.")

        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        plt.figure(figsize=(10, 5))
        plt.plot(y_test.values[:50], label="Actual Prices", color="blue")
        plt.plot(y_pred[:50], label="Predicted Prices", color="red")
        plt.title(f"{stock_ticker} Stock Price Prediction")
        plt.xlabel("Data Points")
        plt.ylabel("Stock Price")
        plt.legend()

        static_folder = 'static'
        os.makedirs(static_folder, exist_ok=True)
        plot_filename = f"{uuid.uuid4()}.png"
        plot_full_path = os.path.join(static_folder, plot_filename)
        plt.savefig(plot_full_path)
        plt.close()

        plot_url = plot_filename


        latest_features = scaler.transform([X.iloc[-1].values])
        predicted_next_day_price = model.predict(latest_features)[0]

        performance_metrics = {
            'rmse': round(rmse, 2),
            'r2': round(r2, 2),
            'mae': round(mae, 2),
            'mape': round(mape, 2)
        }

        return predicted_next_day_price, actual_next_day_price, performance_metrics, plot_url

    except Exception as e:
        print(f"[ERROR] predict_stock_price(): {e}")
        return None
