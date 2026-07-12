import requests    # AI-powered sentiment tracker — fetches live financial news and generates BUY/HOLD/SELL signals
import os
from dotenv import load_dotenv
import yfinance as yf
from transformers import pipeline
import os
import feedparser
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")


load_dotenv()

API_KEY = os.getenv("NEWS_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

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

company_names = {
    "AAPL": "apple",
    "TSLA": "tesla",
    "MSFT": "microsoft",
    "GOOGL": "google",
    "AMZN": "amazon"
}

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
    company = company_names.get(ticker, ticker.lower())
    articles = [a for a in articles if company in a.get("title", "").lower() or ticker.lower() in a.get("title", "").lower()]
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
            title = item["content"]["title"]
            company = company_names.get(ticker, ticker.lower())
            if company in title.lower() or ticker.lower() in title.lower():
                articles.append({"title": title})
    return articles


def fetch_rss_news(ticker):
    feeds = [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}",
        "https://feeds.reuters.com/reuters/businessNews",
        "https://www.cnbc.com/id/10001147/device/rss/rss.html"
    ]
    articles = []
    for url in feeds:
        feed = feedparser.parse(url)
        for entry in feed.entries[:3]:
            company = company_names.get(ticker, ticker.lower())
            if company in entry.title.lower() or ticker.lower() in entry.title.lower():
                articles.append({"title": entry.title, "description": entry.get("summary", "")})
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


from transformers import pipeline
analyser = pipeline("text-classification", model="ProsusAI/finbert")

cache = {}
for ticker in watchlist:
    news_articles = fetch_news(ticker) or []
    yahoo_articles = fetch_yahoo_news(ticker) or []
    rss_articles = fetch_rss_news(ticker) or []
    cache[ticker] = news_articles + yahoo_articles + rss_articles


for ticker in watchlist:
    print("=== " + ticker + " ===")
    articles = cache[ticker]
    scores = []
    for article in articles:
        title = article.get("title", "")
        if not title:
            continue
        result = analyser(title)[0]
        label = result["label"]
        confidence = result["score"]
        if label == "positive":
            score = confidence
        elif label == "negative":
            score = -confidence
        else:
            score = 0.0
        source = article.get("source", {}).get("id", "")
        weight = source_weights.get(source, 1.0)
        score = score * weight
        score = max(-0.8, min(0.8, score))
        scores.append(score)
        print(title)
        print("score: " + str(round((score + 1) / 2 * 20, 2)) + "/20")
        print("signal: " + generate_signal(score))
        print("---")
    if scores:
        average = sum(scores) / len(scores)
        converted = (average + 1) / 2 * 20
        signal = generate_signal(average)
        print("TICKER: " + ticker)
        print("AVERAGE SCORE: " + str(round(converted, 2)) + "/20")
        print("SIGNAL: " + signal)
        print("===")

