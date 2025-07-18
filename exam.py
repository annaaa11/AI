import requests
import pinecone
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime, timedelta
import streamlit as st
import re
import logging
import os


# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация
API_KEY = os.getenv("API_KEY", "1679ef25f9e643d5a8a73d5e1aa3f93e")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY",
                             "pcsk_22597x_NH2uDbw3R8bgndiWyRJcjpWirjwdcZaG99FTHLwPLH7yQnoAQ9Gd3EfWicAUWaF")

# Инициализация Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "crypto-tweets"

    # Проверка существующих индексов
    if index_name not in pc.list_indexes().names():
        logger.info(f"Creating index {index_name} in AWS us-east-1")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        logger.info("Index created with ServerlessSpec (aws us-east-1)")
    index = pc.Index(index_name)
except Exception as e:
    logger.error(f"Error initializing Pinecone: {e}")
    raise

model = SentenceTransformer("all-MiniLM-L6-v2")



nlp = spacy.load("en_core_web_sm")

# Инициализация DistilBERT для анализа тональности без sentencepiece
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=tokenizer)


# Предобработка текста
def clean_text(text):
    text = re.sub(r"http\S+|@\w+|#\w+", "", text)
    return text.strip()


# Сбор твитов
def fetch_tweets(since_date, target_count=100):
    url = "https://api.twitterapi.io/twitter/tweet/advanced_search"
    params = {
        "q": (
            "crypto OR cryptocurrency OR blockchain OR token OR coin OR DeFi OR NFT OR Web3 OR metaverse OR GameFi OR DAO OR "
            "smart contract OR dapp OR wallet OR staking OR yield farming OR Layer 1 OR Layer 2 OR "
            "Bitcoin OR BTC OR Ethereum OR ETH OR Solana OR SOL OR Cardano OR ADA OR Polkadot OR DOT OR "
            "HODL OR moon OR bull OR bear OR rugpull OR FOMO OR FUD OR whale OR pump OR dump OR "
            "#crypto OR #DeFi OR #NFT OR #Web3 OR #metaverse OR #GameFi OR #DAO OR #BTC OR #ETH OR #SOL OR #ADA OR "
            "AI blockchain OR zk-rollup OR cross-chain OR interoperability OR tokenization OR RWA OR #AIcrypto OR #zkrollup OR #RWA "
            "-filter:retweets lang:en min_faves:50"
        ),
        "since": since_date,
        "count": 100,
        "api_key": API_KEY
    }
    tweets = []
    while len(tweets) < target_count:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise Exception(f"Error fetching tweets: {response.status_code}")
        data = response.json()
        tweets.extend(data["tweets"])
        if "next_token" not in data or len(tweets) >= target_count:
            break
        params["next_token"] = data["next_token"]
    return tweets[:target_count]


# Сохранение эмбеддингов в Pinecone
def save_to_pinecone(tweets):
    tweet_texts = [clean_text(tweet["text"]) for tweet in tweets]
    tweet_embeddings = model.encode(tweet_texts)
    sentiments = analyze_sentiment(tweet_texts)
    vectors = [
        (str(tweet["id"]), embedding.tolist(), {
            "text": tweet["text"],
            "author": tweet["user"]["screen_name"],
            "created_at": tweet["created_at"],
            "likes": tweet["favorite_count"],
            "retweets": tweet["retweet_count"],
            "sentiment": sentiment
        })
        for tweet, embedding, sentiment in zip(tweets, tweet_embeddings, sentiments)
    ]
    index.upsert(vectors=vectors)


# Семантический поиск
def semantic_search(query, top_n=100):
    query_embedding = model.encode(query, convert_to_tensor=True)
    results = index.query(vector=query_embedding.tolist(), top_k=top_n, include_metadata=True,
                          filter={"created_at": {"$gte": since_date}})
    return [(r["metadata"], r["score"]) for r in results["matches"]]


# Извлечение проектов
def extract_project_names(tweet_texts):
    projects = []
    for text in tweet_texts:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "ORG" or ent.text.startswith("#"):
                projects.append(ent.text.replace("#", ""))
    return list(set(projects))


# Анализ тональности
def analyze_sentiment(tweet_texts):
    results = sentiment_analyzer(tweet_texts)
    return [r["score"] if r["label"] == "POSITIVE" else -r["score"] for r in results]


# Анализ трендов
def analyze_trends(project_names, days=3):
    since_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    trends = []
    for project in project_names:
        query_embedding = model.encode(project)
        results = index.query(vector=query_embedding.tolist(), top_k=1000, include_metadata=True,
                              filter={"created_at": {"$gte": since_date}})
        mentions = len(results["matches"])
        avg_sentiment = np.mean([r["metadata"].get("sentiment", 0) for r in results["matches"]]) if results[
            "matches"] else 0
        trends.append({"project_name": project, "mentions": mentions, "avg_sentiment": avg_sentiment})

    df = pd.DataFrame(trends)
    df["growth"] = df["mentions"] / df["mentions"].shift(1) * 100
    return df[(df["growth"] > 100) & (df["avg_sentiment"] > 0.5)]


# Визуализация с Matplotlib
def visualize_trends(df):
    if df.empty:
        return None

    plt.figure(figsize=(10, 6))
    norm = plt.Normalize(-1, 1)
    cmap = cm.get_cmap("RdYlGn")
    colors = [cmap(norm(sentiment)) for sentiment in df["avg_sentiment"]]

    bars = plt.bar(df["project_name"], df["mentions"], color=colors)

    plt.xlabel("Project Name")
    plt.ylabel("Mentions")
    plt.title("Trending Crypto Projects (Last 3 Days)")
    plt.xticks(rotation=45, ha="right")

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, label="Average Sentiment")

    plt.tight_layout()
    return plt.gcf()


# Streamlit интерфейс
def main():
    st.title("Crypto Trends Analyzer")
    query = st.text_input("Enter your query:", "new crypto projects with growing interest")
    global since_date
    since_date = "2025-07-01"

    if st.button("Analyze"):
        try:
            tweets = fetch_tweets(since_date, target_count=100)
            save_to_pinecone(tweets)

            relevant_tweets = semantic_search(query, top_n=100)
            tweet_texts = [tweet["text"] for tweet, _ in relevant_tweets]

            project_names = extract_project_names(tweet_texts)
            trending_df = analyze_trends(project_names, days=3)

            fig = visualize_trends(trending_df)
            if fig:
                st.pyplot(fig)
            else:
                st.write("No trending projects found.")

            st.write("Trending Crypto Projects:")
            st.dataframe(trending_df[["project_name", "mentions", "avg_sentiment"]])

            top_project = trending_df.iloc[0]["project_name"] if not trending_df.empty else None
            if top_project:
                st.write(f"Sample Tweets for {top_project}:")
                results = index.query(vector=model.encode(top_project).tolist(), top_k=5, include_metadata=True)
                for r in results["matches"]:
                    st.write(
                        f"- {r['metadata']['text']} (Likes: {r['metadata']['likes']}, Sentiment: {r['metadata']['sentiment']:.2f})")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            logger.error(f"Streamlit error: {e}")


main()