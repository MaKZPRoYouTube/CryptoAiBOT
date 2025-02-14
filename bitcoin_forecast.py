import telebot
import os
BOT_TOKEN = os.getenv('BOT_TOKEN')
bot = telebot.TeleBot(BOT_TOKEN)
@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    if message.text == "/start":
        bot.send_message(message.from_user.id, "Write '/forecast' to get forecast")
    elif message.text == "/forecast":
        import logging
        import ccxt
        import pandas as pd
        import numpy as np
        import ta
        import requests
        from datetime import datetime, timedelta, timezone
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        from sklearn.preprocessing import MinMaxScaler
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import Dataset, DataLoader
        import matplotlib.pyplot as plt

        # Для Google Trends – пытаемся импортировать pytrends
        try:
            from pytrends.request import TrendReq
        except ImportError:
            TrendReq = None
            logging.warning("pytrends не установлен, функция Google Trends будет использовать значение по умолчанию.")

        # Настройка логирования
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

        # ================================
        # Функции для дополнительных факторов
        # ================================

        def get_news_sentiment():
            bot.send_message(message.from_user.id, "Получение новостного сентимента через VADER")
            analyzer = SentimentIntensityAnalyzer()
            # Пример заголовков – для реального применения подключите API новостей
            news_headlines = [
                "Bitcoin surges to new all-time high amid strong institutional demand",
                "Crypto markets face uncertainty as regulators tighten controls",
                "Investors turn to Bitcoin as hedge against inflation"
            ]
            sentiments = [analyzer.polarity_scores(headline)['compound'] for headline in news_headlines]
            sentiment = np.mean(sentiments) if sentiments else 0
            bot.send_message(message.from_user.id, f"Новостной сентимент: {sentiment}")
            return sentiment

        def get_google_trends(keyword="Bitcoin", timeframe="today 12-m"):
            if TrendReq is None:
                logging.warning("pytrends не установлен, используем значение по умолчанию для Google Trends")
                return 50.0
            try:
                pytrend = TrendReq()
                pytrend.build_payload(kw_list=[keyword], timeframe=timeframe)
                data = pytrend.interest_over_time()
                if data.empty:
                    return 50.0
                trend_value = data[keyword].iloc[-1]
                bot.send_message(message.from_user.id, f"Google Trends для {keyword}: {trend_value}")
                return trend_value
            except Exception as e:
                logging.error(f"Ошибка при получении Google Trends: {e}")
                return 50.0

        def get_onchain_metrics():
            # Для реальных данных подключитесь к API (например, Glassnode, CryptoQuant). Здесь случайные значения.
            metrics = {
                "ActiveAddresses": np.random.uniform(500000, 1000000),
                "TransactionCount": np.random.uniform(200000, 500000),
                "HashRate": np.random.uniform(100, 200)
            }
            bot.send_message(message.from_user.id, f"On-chain метрики: {metrics}")
            return metrics

        def combine_features(df, news_sentiment, google_trends, onchain):
            bot.send_message(message.from_user.id, "Объединение дополнительных признаков")
            df_combined = df.copy()
            df_combined["News"] = news_sentiment
            df_combined["GoogleTrends"] = google_trends
            df_combined["ActiveAddresses"] = onchain["ActiveAddresses"]
            df_combined["TransactionCount"] = onchain["TransactionCount"]
            df_combined["HashRate"] = onchain["HashRate"]
            return df_combined

        # ================================
        # Функции для получения рыночных данных и вычисления технических индикаторов
        # ================================

        def fetch_market_data():
            exchange = ccxt.binance()
            symbol = 'BTC/USDT'
            timeframe = '1h'
            # Вычисляем "since" как текущий момент минус 1000 часов (1000 * 60 * 60 * 1000 миллисекунд)
            now_ms = exchange.milliseconds()
            since = now_ms - (1000 * 60 * 60 * 1000)
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df

        def add_technical_indicators(df):
            df['SMA_10'] = ta.trend.sma_indicator(df['close'], window=10)
            df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
            df['RSI'] = ta.momentum.rsi(df['close'], window=14)
            df['MACD'] = ta.trend.macd(df['close'])
            bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['BB_mid'] = bollinger.bollinger_mavg()
            df['BB_upper'] = bollinger.bollinger_hband()
            df['BB_lower'] = bollinger.bollinger_lband()
            df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
            df['ADX'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
            return df.dropna()

        # ================================
        # Подготовка данных для модели (технические индикаторы + дополнительные признаки)
        # ================================

        def prepare_data(df, seq_length=24):
            features = [
                'close', 'volume', 'SMA_10', 'SMA_50', 'RSI', 'MACD',
                'BB_mid', 'BB_upper', 'BB_lower', 'OBV', 'ADX',
                'News', 'GoogleTrends', 'ActiveAddresses', 'TransactionCount', 'HashRate'
            ]
            data = df[features].values
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(data)
            
            X, y = [], []
            for i in range(len(data_scaled) - seq_length):
                X.append(data_scaled[i:i+seq_length])
                y.append(data_scaled[i+seq_length, 0])  # прогнозируем close
            return np.array(X), np.array(y), scaler

        # ================================
        # Кастомный Dataset для PyTorch
        # ================================

        class TimeSeriesDataset(Dataset):
            def __init__(self, X, y):
                self.X = torch.tensor(X, dtype=torch.float32)
                self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
            def __len__(self):
                return len(self.X)
            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]

        # ================================
        # Продвинутая модель LSTM на PyTorch (двунаправленный LSTM с дополнительными слоями)
        # ================================

        class AdvancedLSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size=128, num_layers=2):
                super(AdvancedLSTMModel, self).__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                    batch_first=True, bidirectional=True)
                self.dropout = nn.Dropout(0.3)
                self.fc1 = nn.Linear(hidden_size * 2, 64)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(64, 1)
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                out = lstm_out[:, -1, :]  # используем выход последнего временного шага
                out = self.dropout(out)
                out = self.fc1(out)
                out = self.relu(out)
                out = self.fc2(out)
                return out

        # ================================
        # Обучение модели
        # ================================

        def train_model(model, dataloader, num_epochs=1000, learning_rate=0.0005, device='cpu'):
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            model.to(device)
            losses = []
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                model.train()
                for X_batch, y_batch in dataloader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * X_batch.size(0)
                epoch_loss /= len(dataloader.dataset)
                losses.append(epoch_loss)
                bot.send_message(message.from_user.id, f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")
            return losses

        # ================================
        # Итеративный прогноз на 48 часов
        # ================================

        def forecast_48_hours(model, scaler, last_sequence, rolling_closes, constant_volume, 
                                external_factors, n_steps=48, device='cpu'):
            predictions = []
            sequence = last_sequence.copy()
            for i in range(n_steps):
                model.eval()
                with torch.no_grad():
                    input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
                    pred_scaled = model(input_tensor).cpu().numpy()[0, 0]
                dummy = np.zeros((1, scaler.n_features_in_))
                dummy[0, 0] = pred_scaled
                pred_close = scaler.inverse_transform(dummy)[0, 0]
                predictions.append(pred_close)
                
                # Обновление rolling window для закрытий
                rolling_closes.append(pred_close)
                if len(rolling_closes) > 24:
                    rolling_closes.pop(0)
                # Пересчёт технических признаков (упрощённо)
                new_SMA_10 = np.mean(rolling_closes[-10:]) if len(rolling_closes) >= 10 else np.mean(rolling_closes)
                new_SMA_50 = np.mean(rolling_closes)
                new_RSI = 50      # нейтральное значение
                new_MACD = 0      # нейтральное значение
                new_BB_mid = np.mean(rolling_closes[-20:]) if len(rolling_closes) >= 20 else np.mean(rolling_closes)
                new_BB_upper = new_BB_mid * 1.02
                new_BB_lower = new_BB_mid * 0.98
                new_OBV = 0      # упрощено
                new_ADX = 25     # типичное значение
                
                new_volume = constant_volume
                new_News = external_factors['News']
                new_GoogleTrends = external_factors['GoogleTrends']
                new_ActiveAddresses = external_factors['ActiveAddresses']
                new_TransactionCount = external_factors['TransactionCount']
                new_HashRate = external_factors['HashRate']
                
                new_row = np.array([
                    pred_close, new_volume, new_SMA_10, new_SMA_50, new_RSI, new_MACD,
                    new_BB_mid, new_BB_upper, new_BB_lower, new_OBV, new_ADX,
                    new_News, new_GoogleTrends, new_ActiveAddresses, new_TransactionCount, new_HashRate
                ]).reshape(1, -1)
                new_row_scaled = scaler.transform(new_row)
                sequence = np.vstack([sequence[1:], new_row_scaled])
            return predictions

        # ================================
        # Основная функция
        # ================================

        def main():
            news_api_key = "33191d35b916471baf39bab55ad2611b"  # Если хотите заменить на реальный API-ключ
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            bot.send_message(message.from_user.id, "Получение рыночных данных с Binance...")
            df = fetch_market_data()
            bot.send_message(message.from_user.id, f"Данных получено: {len(df)}")
            
            bot.send_message(message.from_user.id, "Добавление технических индикаторов...")
            df = add_technical_indicators(df)
            
            # Получение дополнительных факторов
            news_sentiment = get_news_sentiment()
            google_trends_val = get_google_trends()
            onchain = get_onchain_metrics()
            
            # Объединение дополнительных признаков с историческими данными
            df = combine_features(df, news_sentiment, google_trends_val, onchain)
            
            df = df.dropna()
            seq_length = 24
            X, y, scaler = prepare_data(df, seq_length)
            bot.send_message(message.from_user.id, f"Размер данных X: {X.shape}, y: {y.shape}")
            
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            train_dataset = TimeSeriesDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            input_size = X.shape[2]
            model = AdvancedLSTMModel(input_size=input_size)
            bot.send_message(message.from_user.id, model)
            
            bot.send_message(message.from_user.id, "Обучение модели...")
            losses = train_model(model, train_loader, num_epochs=1000, learning_rate=0.0005, device=device)
            
            plt.figure(figsize=(10, 5))
            plt.plot(losses, label='Train Loss')
            plt.xlabel("Эпоха")
            plt.ylabel("Ошибка (MSE)")
            plt.title("График обучения модели")
            plt.legend()
            plt.show()
            
            # Для прогноза получаем текущие значения дополнительных факторов (будем считать их постоянными)
            external_factors = {
                'News': news_sentiment,
                'GoogleTrends': google_trends_val,
                'ActiveAddresses': onchain["ActiveAddresses"],
                'TransactionCount': onchain["TransactionCount"],
                'HashRate': onchain["HashRate"]
            }
            bot.send_message(message.from_user.id, f"Текущие внешние факторы: {external_factors}")
            
            last_sequence = X[-1]
            rolling_closes = list(df['close'].tail(seq_length).values)
            constant_volume = np.mean(df['volume'].tail(seq_length))
            
            bot.send_message(message.from_user.id, "Выполнение почасового прогноза на 48 часов...")
            predictions = forecast_48_hours(model, scaler, last_sequence, rolling_closes, 
                                            constant_volume, external_factors, n_steps=48, device=device)
            
            current_time = datetime.now(timezone.utc)
            forecast_times = [current_time + timedelta(hours=i+1) for i in range(48)]
            forecast_df = pd.DataFrame({
                "timestamp": forecast_times,
                "predicted_close": predictions
            })
            forecast_df.set_index("timestamp", inplace=True)
            
            bot.send_message(message.from_user.id, "Прогноз на 48 часов:")
            bot.send_message(message.from_user.id, f"\n{forecast_df}")
            
            plt.figure(figsize=(12, 6))
            plt.plot(forecast_df.index, forecast_df['predicted_close'], marker='o', label="Прогноз цены")
            plt.xlabel("Время")
            plt.ylabel("Цена закрытия")
            plt.title("Почасовой прогноз цены закрытия BTC на 48 часов")
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        if __name__ == '__main__':
            main()

        bot.send_message(message.from_user.id, "Напиши привет")
bot.polling(none_stop=True, interval=0)
