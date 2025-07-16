import requests
import pinecone
from sentence_transformers import SentenceTransformer
import spacy
from transformers import pipeline
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import streamlit as st
#from supabase import create_client, Client
import re

# Инициализация
API_KEY = "1679ef25f9e643d5a8a73d5e1aa3f93e"
PINECONE_API_KEY = "pcsk_22597x_NH2uDbw3R8bgndiWyRJcjpWirjwdcZaG99FTHLwPLH7yQnoAQ9Gd3EfWicAUWaF"
SUPABASE_URL = "your_supabase_url"
SUPABASE_KEY = "your_supabase_key"

pinecone.init(api_key=PINECONE_API_KEY, environment="us-west1-gcp")
index_name = "crypto-tweets"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=384, metric="cosine")
index = pinecone.Index(index_name)
model = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
#supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# Предобработка текста
def clean_text(text):
    text = re.sub(r"http\S+|@\w+|#\w+", "", text)  # Удаление URL, упоминаний, хэштегов
    return text.strip()


# Сбор твитов с полным списком терминов
def fetch_tweets(since_date, target_count=1000):
    url = "https://api.twitterapi.io/v1/search/tweets"
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
    results = index.query(query_embedding.tolist(), top_k=top_n, include_metadata=True,
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
        results = index.query(query_embedding.tolist(), top_k=1000, include_metadata=True,
                              filter={"created_at": {"$gte": since_date}})
        mentions = len(results["matches"])
        avg_sentiment = np.mean([r["metadata"].get("sentiment", 0) for r in results["matches"]]) if results[
            "matches"] else 0
        trends.append({"project_name": project, "mentions": mentions, "avg_sentiment": avg_sentiment})

        # Сохранение в Supabase
        # supabase.table("trends").insert({
        #     "project_name": project,
        #     "mentions": mentions,
        #     "avg_sentiment": avg_sentiment,
        #     "date": since_date
        # }).execute()

    df = pd.DataFrame(trends)
    df["growth"] = df["mentions"] / df["mentions"].shift(1) * 100
    return df[(df["growth"] > 100) & (df["avg_sentiment"] > 0.5)]


# Визуализация
def visualize_trends(df):
    fig = px.bar(df, x="project_name", y="mentions", color="avg_sentiment", title="Trending Crypto Projects")
    return fig


# Streamlit интерфейс
def main():
    st.title("Crypto Trends Analyzer")
    query = st.text_input("Enter your query:", "new crypto projects with growing interest")
    global since_date
    since_date = "2025-07-10"

    if st.button("Analyze"):
        # Сбор твитов
        tweets = fetch_tweets(since_date, target_count=10000)
        save_to_pinecone(tweets)

        # Семантический поиск
        relevant_tweets = semantic_search(query, top_n=100)
        tweet_texts = [tweet["text"] for tweet, _ in relevant_tweets]

        # Извлечение проектов
        project_names = extract_project_names(tweet_texts)

        # Анализ трендов
        trending_df = analyze_trends(project_names, days=3)

        # Визуализация
        fig = visualize_trends(trending_df)
        st.plotly_chart(fig)

        # Вывод результатов
        st.write("Trending Crypto Projects:")
        st.dataframe(trending_df[["project_name", "mentions", "avg_sentiment"]])

        # Пример твитов для топ-проекта
        top_project = trending_df.iloc[0]["project_name"] if not trending_df.empty else None
        if top_project:
            st.write(f"Sample Tweets for {top_project}:")
            results = index.query(model.encode(top_project).tolist(), top_k=5, include_metadata=True)
            for r in results["matches"]:
                st.write(
                    f"- {r['metadata']['text']} (Likes: {r['metadata']['likes']}, Sentiment: {r['metadata']['sentiment']:.2f})")


if __name__ == "__main__":
    main()