# main.py
from fastapi import FastAPI
import yfinance as yf
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()
analyzer = SentimentIntensityAnalyzer()
NEWS_API_KEY = "YOUR_NEWSAPI_KEY"  # Replace with your key

def get_stock_data(symbol):
    df = yf.download(symbol, period="1mo", interval="1d")
    return df

def compute_indicators(df):
    df['MA10'] = df['Close'].rolling(10).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df = compute_atr(df)
    return df

def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -1*delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_atr(df, period=14):
    df['H-L'] = df['High'] - df['Low']
    df['H-Cp'] = abs(df['High'] - df['Close'].shift(1))
    df['L-Cp'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L','H-Cp','L-Cp']].max(axis=1)
    df['ATR'] = df['TR'].rolling(period).mean()
    return df

def get_news_sentiment(symbol):
    url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWS_API_KEY}"
    news_data = requests.get(url).json()
    headlines = [article['title'] for article in news_data['articles'][:5]]
    scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
    sentiment = sum(scores)/len(scores) if scores else 0
    return sentiment, headlines

def predict_trend(df, sentiment):
    df['Sentiment'] = sentiment
    X = df[['MA10','RSI','Sentiment']].dropna()
    y = (df['Close'].shift(-1) > df['Close']).astype(int).dropna()
    model = RandomForestClassifier()
    model.fit(X[:-1], y)
    pred = model.predict(X[-1:])
    prob = model.predict_proba(X[-1:])[0][1]
    return int(pred[0]), prob

def calculate_sl_tp(df, prob, risk_ratio=1.5):
    last_close = df['Close'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    if prob > 0.5:
        sl = last_close - atr
        tp = last_close + atr*risk_ratio
    else:
        sl = last_close + atr
        tp = last_close - atr*risk_ratio
    return round(sl,2), round(tp,2)

@app.get("/predict/{symbol}")
def get_prediction(symbol: str):
    df = get_stock_data(symbol)
    df = compute_indicators(df)
    sentiment, headlines = get_news_sentiment(symbol)
    trend, prob = predict_trend(df, sentiment)
    sl, tp = calculate_sl_tp(df, prob)
    return {
        "symbol": symbol,
        "trend": "Up" if trend==1 else "Down",
        "probability": round(prob*100,2),
        "sl": sl,
        "tp": tp,
        "headlines": headlines
    }
