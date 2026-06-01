import requests
import os
from dotenv import load_dotenv

from transformers import pipeline

load_dotenv()

API_KEY = os.getenv("NEWS_API_KEY")

analyser = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

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
    articles = response.json()["articles"]
    return articles

watchlist = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"]

for ticker in watchlist:
    print("=== " + ticker + " ===")
    articles = fetch_news(ticker)
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
        print("Score: " + str(round(converted, 2)) + "/20")
        print("Signal: " + signal)
        print("---")

for article in articles:
    print(article["title"])

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
    print("Score: " + str(round(converted, 2)) + "/20")
    print("signal: " + signal)
    print("---")




