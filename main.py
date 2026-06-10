import requests    # AI-powered sentiment tracker — fetches live financial news and generates BUY/HOLD/SELL signals
import os
from dotenv import load_dotenv
import yfinance as yf
from transformers import pipeline

load_dotenv()

API_KEY = os.getenv("NEWS_API_KEY")

analyser = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
# Converts sentiment score to trading signal on a 0-20 scale
def generate_signal(vader_score):
    converted = (vader_score + 1) / 2 * 20



    if converted <= 7.5:
        return "SELL"
    elif converted <= 14:
        return "HOLD"
    else:
        return "BUY"

def fetch_news(ticker):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": ticker + " stock earnings OR revenue OR profit OR loss",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 5,
        "apiKey": API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()
    articles = data.get("articles", [])
    seen = []
    unique = []
    for article in articles:
        if article["title"] not in seen:
            seen.append(article["title"])
            unique.append(article)
    return unique


def fetch_yahoo_news(ticker):
    stock = yf.Ticker(ticker)
    news = stock.news
    articles = []
    for item in news[:5]:
        if "title" in item.get("content", {}):
            articles.append({"title": item["content"]["title"]})
    return articles
    



# Selected stocks being tracked

watchlist = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"]

source_weights = {
    "reuters.com": 1.5,
    "bloomberg.com": 1.5,
    "ft.com": 1.4,
    "wsj.com": 1.3,
    "cnbc.com": 1.2,
    "bbc.co.uk": 1.1
}

for ticker in watchlist:
    print("=== " + ticker + " ===")
    articles = fetch_news(ticker) + fetch_yahoo_news(ticker)
    scores = []
    for article in articles:
        title = article["title"]
        result = analyser(title)[0]
        label = result["label"]
        confidence = result["score"]
        if label == "positive":
            score = confidence
        elif label == "negative":
            score = -confidence
        else:
            score = 0.0

        source = article["source"]["id"]
        weight = source_weights.get(source, 1.0)
        score = score * weight
    
        scores.append(score)
        average = sum(scores) / len(scores)
    converted = (average + 1) / 2 * 20
    signal = generate_signal(average)
    print("TICKER: " + ticker)
    print("AVERAGE SCORE: " + str(round(converted, 2)) + "/20")
    print("SIGNAL: " + signal)
    print("---")




from transformers import pipeline
analyser = pipeline("text-classification", model="ProsusAI/finbert")


for article in articles:
    title = article["title"]
    result = analyser(title)[0]
    label = result["label"]
    confidence = result["score"]
   
    if label == "positive":
        score = confidence
  
    elif label == "negative":
        score = -confidence
  
    else:
        score = 0.0
    
    signal = generate_signal(score) 
    converted = (score + 1) / 2 * 20
    print(title)
    print("score: " + str(round(converted, 2)) + "/20")
    print("signal: " + signal)
    print("---")


