# import requests
# import pinecone
# from pinecone import Pinecone, ServerlessSpec
# from sentence_transformers import SentenceTransformer
# import spacy
# from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from datetime import datetime, timedelta
# import streamlit as st
# import re
# import logging
# import os
#
#
# # Настройка логирования
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# # Инициализация
# API_KEY = os.getenv("API_KEY", "1679ef25f9e643d5a8a73d5e1aa3f93e")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY",
#                              "pcsk_22597x_NH2uDbw3R8bgndiWyRJcjpWirjwdcZaG99FTHLwPLH7yQnoAQ9Gd3EfWicAUWaF")
#
# # Инициализация Pinecone
# try:
#     pc = Pinecone(api_key=PINECONE_API_KEY)
#     index_name = "crypto-tweets"
#
#     # Проверка существующих индексов
#     if index_name not in pc.list_indexes().names():
#         logger.info(f"Creating index {index_name} in AWS us-east-1")
#         pc.create_index(
#             name=index_name,
#             dimension=384,
#             metric="cosine",
#             spec=ServerlessSpec(cloud="aws", region="us-east-1")
#         )
#         logger.info("Index created with ServerlessSpec (aws us-east-1)")
#     index = pc.Index(index_name)
# except Exception as e:
#     logger.error(f"Error initializing Pinecone: {e}")
#     raise
#
# model = SentenceTransformer("all-MiniLM-L6-v2")
#
#
#
# nlp = spacy.load("en_core_web_sm")
#
# # Инициализация DistilBERT для анализа тональности без sentencepiece
# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# sentiment_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
# sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=tokenizer)
#
#
# # Предобработка текста
# def clean_text(text):
#     text = re.sub(r"http\S+|@\w+|#\w+", "", text)
#     return text.strip()
#
#
# # Сбор твитов
# def fetch_tweets(since_date, target_count=100):
#     url = "https://api.twitterapi.io/twitter/tweet/advanced_search"
#     params = {
#         "q": (
#             "crypto OR cryptocurrency OR blockchain OR token OR coin OR DeFi OR NFT OR Web3 OR metaverse OR GameFi OR DAO OR "
#             "smart contract OR dapp OR wallet OR staking OR yield farming OR Layer 1 OR Layer 2 OR "
#             "Bitcoin OR BTC OR Ethereum OR ETH OR Solana OR SOL OR Cardano OR ADA OR Polkadot OR DOT OR "
#             "HODL OR moon OR bull OR bear OR rugpull OR FOMO OR FUD OR whale OR pump OR dump OR "
#             "#crypto OR #DeFi OR #NFT OR #Web3 OR #metaverse OR #GameFi OR #DAO OR #BTC OR #ETH OR #SOL OR #ADA OR "
#             "AI blockchain OR zk-rollup OR cross-chain OR interoperability OR tokenization OR RWA OR #AIcrypto OR #zkrollup OR #RWA "
#             "-filter:retweets lang:en min_faves:50"
#         ),
#         "since": since_date,
#         "count": 100,
#         "api_key": API_KEY
#     }
#     tweets = []
#     while len(tweets) < target_count:
#         response = requests.get(url, params=params)
#         if response.status_code != 200:
#             raise Exception(f"Error fetching tweets: {response.status_code}")
#         data = response.json()
#         tweets.extend(data["tweets"])
#         if "next_token" not in data or len(tweets) >= target_count:
#             break
#         params["next_token"] = data["next_token"]
#     return tweets[:target_count]
#
#
# # Сохранение эмбеддингов в Pinecone
# def save_to_pinecone(tweets):
#     tweet_texts = [clean_text(tweet["text"]) for tweet in tweets]
#     tweet_embeddings = model.encode(tweet_texts)
#     sentiments = analyze_sentiment(tweet_texts)
#     vectors = [
#         (str(tweet["id"]), embedding.tolist(), {
#             "text": tweet["text"],
#             "author": tweet["user"]["screen_name"],
#             "created_at": tweet["created_at"],
#             "likes": tweet["favorite_count"],
#             "retweets": tweet["retweet_count"],
#             "sentiment": sentiment
#         })
#         for tweet, embedding, sentiment in zip(tweets, tweet_embeddings, sentiments)
#     ]
#     index.upsert(vectors=vectors)
#
#
# # Семантический поиск
# def semantic_search(query, top_n=100):
#     query_embedding = model.encode(query, convert_to_tensor=True)
#     results = index.query(vector=query_embedding.tolist(), top_k=top_n, include_metadata=True,
#                           filter={"created_at": {"$gte": since_date}})
#     return [(r["metadata"], r["score"]) for r in results["matches"]]
#
#
# # Извлечение проектов
# def extract_project_names(tweet_texts):
#     projects = []
#     for text in tweet_texts:
#         doc = nlp(text)
#         for ent in doc.ents:
#             if ent.label_ == "ORG" or ent.text.startswith("#"):
#                 projects.append(ent.text.replace("#", ""))
#     return list(set(projects))
#
#
# # Анализ тональности
# def analyze_sentiment(tweet_texts):
#     results = sentiment_analyzer(tweet_texts)
#     return [r["score"] if r["label"] == "POSITIVE" else -r["score"] for r in results]
#
#
# # Анализ трендов
# def analyze_trends(project_names, days=3):
#     since_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
#     trends = []
#     for project in project_names:
#         query_embedding = model.encode(project)
#         results = index.query(vector=query_embedding.tolist(), top_k=1000, include_metadata=True,
#                               filter={"created_at": {"$gte": since_date}})
#         mentions = len(results["matches"])
#         avg_sentiment = np.mean([r["metadata"].get("sentiment", 0) for r in results["matches"]]) if results[
#             "matches"] else 0
#         trends.append({"project_name": project, "mentions": mentions, "avg_sentiment": avg_sentiment})
#
#     df = pd.DataFrame(trends)
#     df["growth"] = df["mentions"] / df["mentions"].shift(1) * 100
#     return df[(df["growth"] > 100) & (df["avg_sentiment"] > 0.5)]
#
#
# # Визуализация с Matplotlib
# def visualize_trends(df):
#     if df.empty:
#         return None
#
#     plt.figure(figsize=(10, 6))
#     norm = plt.Normalize(-1, 1)
#     cmap = cm.get_cmap("RdYlGn")
#     colors = [cmap(norm(sentiment)) for sentiment in df["avg_sentiment"]]
#
#     bars = plt.bar(df["project_name"], df["mentions"], color=colors)
#
#     plt.xlabel("Project Name")
#     plt.ylabel("Mentions")
#     plt.title("Trending Crypto Projects (Last 3 Days)")
#     plt.xticks(rotation=45, ha="right")
#
#     sm = cm.ScalarMappable(cmap=cmap, norm=norm)
#     plt.colorbar(sm, label="Average Sentiment")
#
#     plt.tight_layout()
#     return plt.gcf()
#
#
# # Streamlit интерфейс
# def main():
#     st.title("Crypto Trends Analyzer")
#     query = st.text_input("Enter your query:", "new crypto projects with growing interest")
#     global since_date
#     since_date = "2025-07-01"
#
#     if st.button("Analyze"):
#         try:
#             tweets = fetch_tweets(since_date, target_count=100)
#             save_to_pinecone(tweets)
#
#             relevant_tweets = semantic_search(query, top_n=100)
#             tweet_texts = [tweet["text"] for tweet, _ in relevant_tweets]
#
#             project_names = extract_project_names(tweet_texts)
#             trending_df = analyze_trends(project_names, days=3)
#
#             fig = visualize_trends(trending_df)
#             if fig:
#                 st.pyplot(fig)
#             else:
#                 st.write("No trending projects found.")
#
#             st.write("Trending Crypto Projects:")
#             st.dataframe(trending_df[["project_name", "mentions", "avg_sentiment"]])
#
#             top_project = trending_df.iloc[0]["project_name"] if not trending_df.empty else None
#             if top_project:
#                 st.write(f"Sample Tweets for {top_project}:")
#                 results = index.query(vector=model.encode(top_project).tolist(), top_k=5, include_metadata=True)
#                 for r in results["matches"]:
#                     st.write(
#                         f"- {r['metadata']['text']} (Likes: {r['metadata']['likes']}, Sentiment: {r['metadata']['sentiment']:.2f})")
#         except Exception as e:
#             st.error(f"An error occurred: {e}")
#             logger.error(f"Streamlit error: {e}")
#
#
# main()

# import requests
# import pinecone
# from pinecone import Pinecone, ServerlessSpec
# from sentence_transformers import SentenceTransformer
# import spacy
# from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from datetime import datetime, timedelta
# import streamlit as st
# import re
# import logging
# import os
#
# global since_date
# since_date = "2025-07-01"
#
# # Настройка логирования
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# # Инициализация
# API_KEY = os.getenv("API_KEY", "1679ef25f9e643d5a8a73d5e1aa3f93e")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY",
#                              "pcsk_22597x_NH2uDbw3R8bgndiWyRJcjpWirjwdcZaG99FTHLwPLH7yQnoAQ9Gd3EfWicAUWaF")
# TARGET_ACCOUNT = "elonmusk"  # Account to monitor
#
# # Инициализация Pinecone
# try:
#     pc = Pinecone(api_key=PINECONE_API_KEY)
#     index_name = "crypto-tweets"
#
#     if index_name not in pc.list_indexes().names():
#         logger.info(f"Creating index {index_name} in AWS us-east-1")
#         pc.create_index(
#             name=index_name,
#             dimension=384,
#             metric="cosine",
#             spec=ServerlessSpec(cloud="aws", region="us-east-1")
#         )
#         logger.info("Index created with ServerlessSpec (aws us-east-1)")
#     index = pc.Index(index_name)
# except Exception as e:
#     logger.error(f"Error initializing Pinecone: {e}")
#     raise
#
# model = SentenceTransformer("all-MiniLM-L6-v2")
# # Проверка доступных моделей spaCy
# logger.info(f"Available spaCy models: {spacy.util.get_installed_models()}")
# try:
#     nlp = spacy.load("en_core_web_sm")
# except Exception as e:
#     logger.error(f"Failed to load en_core_web_sm: {e}")
#     raise
#
# # Инициализация DistilBERT для анализа тональности без sentencepiece
# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# sentiment_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
# sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=tokenizer)
#
#
# # Предобработка текста
# def clean_text(text):
#     text = re.sub(r"http\S+|@\w+|#\w+", "", text)
#     return text.strip()
#
#
# # Сбор твитов с учетом документации
# def fetch_tweets(since_date, target_count=100):
#     url = "https://api.twitterapi.io/twitter/tweet/advanced_search"
#     since_time = datetime.strptime(since_date, "%Y-%m-%d")
#     until_time = datetime.utcnow()
#     since_str = since_time.strftime("%Y-%m-%d_%H:%M:%S_UTC")
#     until_str = until_time.strftime("%Y-%m-%d_%H:%M:%S_UTC")
#
#     query = f"crypto from:{TARGET_ACCOUNT} since:{since_str} until:{until_str} include:nativeretweets"
#     params = {
#         "query": query,
#         "queryType": "Latest"
#     }
#     headers = {
#         "X-API-Key": API_KEY
#     }
#
#     all_tweets = []
#     next_cursor = None
#
#     while len(all_tweets) < target_count:
#         if next_cursor:
#             params["cursor"] = next_cursor
#
#         response = requests.get(url, headers=headers, params=params)
#
#         if response.status_code == 200:
#             data = response.json()
#             tweets = data.get("tweets", [])
#             all_tweets.extend(tweets)
#
#             if data.get("has_next_page", False) and data.get("next_cursor", "") != "":
#                 next_cursor = data.get("next_cursor")
#                 continue
#             else:
#                 break
#         else:
#             raise Exception(f"Error fetching tweets: {response.status_code} - {response.text}")
#
#     logger.info(f"Found {len(all_tweets)} total tweets from @{TARGET_ACCOUNT}")
#     print(all_tweets[0])
#     return all_tweets[:target_count]
#
#
# # Сохранение эмбеддингов в Pinecone
# def save_to_pinecone(tweets):
#     tweet_texts = [clean_text(tweet["text"]) for tweet in tweets]
#     tweet_embeddings = model.encode(tweet_texts)
#     sentiments = analyze_sentiment(tweet_texts)
#     vectors = [
#         (str(tweet["id"]), embedding.tolist(), {
#             "text": tweet["text"],
#             "author": tweet["user"]["screen_name"],
#             "created_at": tweet["created_at"],
#             "likes": tweet["favorite_count"],
#             "retweets": tweet["retweet_count"],
#             "sentiment": sentiment
#         })
#         for tweet, embedding, sentiment in zip(tweets, tweet_embeddings, sentiments)
#     ]
#     index.upsert(vectors=vectors)
#
#
# # Семантический поиск
# def semantic_search(query, top_n=100):
#     query_embedding = model.encode(query, convert_to_tensor=True)
#     results = index.query(vector=query_embedding.tolist(), top_k=top_n, include_metadata=True,
#                           filter={"created_at": {"$gte": since_date}})
#     return [(r["metadata"], r["score"]) for r in results["matches"]]
#
#
# # Извлечение проектов
# def extract_project_names(tweet_texts):
#     projects = []
#     for text in tweet_texts:
#         doc = nlp(text)
#         for ent in doc.ents:
#             if ent.label_ == "ORG" or ent.text.startswith("#"):
#                 projects.append(ent.text.replace("#", ""))
#     return list(set(projects))
#
#
# # Анализ тональности
# def analyze_sentiment(tweet_texts):
#     results = sentiment_analyzer(tweet_texts)
#     return [r["score"] if r["label"] == "POSITIVE" else -r["score"] for r in results]
#
#
# # Анализ трендов
# def analyze_trends(project_names, days=3):
#     since_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
#     trends = []
#     for project in project_names:
#         query_embedding = model.encode(project)
#         results = index.query(vector=query_embedding.tolist(), top_k=1000, include_metadata=True,
#                               filter={"created_at": {"$gte": since_date}})
#         mentions = len(results["matches"])
#         avg_sentiment = np.mean([r["metadata"].get("sentiment", 0) for r in results["matches"]]) if results[
#             "matches"] else 0
#         trends.append({"project_name": project, "mentions": mentions, "avg_sentiment": avg_sentiment})
#
#     df = pd.DataFrame(trends)
#     df["growth"] = df["mentions"] / df["mentions"].shift(1) * 100
#     return df[(df["growth"] > 100) & (df["avg_sentiment"] > 0.5)]
#
#
# # Визуализация с Matplotlib
# def visualize_trends(df):
#     if df.empty:
#         return None
#
#     plt.figure(figsize=(10, 6))
#     norm = plt.Normalize(-1, 1)
#     cmap = cm.get_cmap("RdYlGn")
#     colors = [cmap(norm(sentiment)) for sentiment in df["avg_sentiment"]]
#
#     bars = plt.bar(df["project_name"], df["mentions"], color=colors)
#
#     plt.xlabel("Project Name")
#     plt.ylabel("Mentions")
#     plt.title(f"Trending Crypto Projects from @{TARGET_ACCOUNT} (Last 3 Days)")
#     plt.xticks(rotation=45, ha="right")
#
#     sm = cm.ScalarMappable(cmap=cmap, norm=norm)
#     plt.colorbar(sm, label="Average Sentiment")
#
#     plt.tight_layout()
#     return plt.gcf()
#
#
# # Streamlit интерфейс
# def main():
#     st.title(f"Crypto Trends Analyzer for @{TARGET_ACCOUNT}")
#     query = st.text_input("Enter your query:", "new crypto projects with growing interest")
#
#
#     if st.button("Analyze"):
#         try:
#             tweets = fetch_tweets(since_date, target_count=100)
#             if not tweets:
#                 st.write(f"No tweets found from @{TARGET_ACCOUNT} since {since_date}.")
#                 return
#
#             save_to_pinecone(tweets)
#
#             relevant_tweets = semantic_search(query, top_n=100)
#             tweet_texts = [tweet["text"] for tweet, _ in relevant_tweets]
#
#             project_names = extract_project_names(tweet_texts)
#             trending_df = analyze_trends(project_names, days=3)
#
#             fig = visualize_trends(trending_df)
#             if fig:
#                 st.pyplot(fig)
#             else:
#                 st.write("No trending projects found.")
#
#             st.write(f"Trending Crypto Projects from @{TARGET_ACCOUNT}:")
#             st.dataframe(trending_df[["project_name", "mentions", "avg_sentiment"]])
#
#             top_project = trending_df.iloc[0]["project_name"] if not trending_df.empty else None
#             if top_project:
#                 st.write(f"Sample Tweets for {top_project}:")
#                 results = index.query(vector=model.encode(top_project).tolist(), top_k=5, include_metadata=True)
#                 for r in results["matches"]:
#                     st.write(
#                         f"- {r['metadata']['text']} (Likes: {r['metadata']['likes']}, Sentiment: {r['metadata']['sentiment']:.2f})")
#         except Exception as e:
#             st.error(f"An error occurred: {e}")
#             logger.error(f"Streamlit error: {e}")
#
#
# #main()
#
# fetch_tweets(since_date)

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
import json
import torch

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
    logger.info(f"Pinecone index stats: {index.describe_index_stats()}")
except Exception as e:
    logger.error(f"Error initializing Pinecone: {e}")
    raise

# Инициализация SentenceTransformer с явным указанием устройства
device = torch.device("cpu")  # Явно задаем CPU как устройство
try:
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    logger.info(f"SentenceTransformer initialized on {device}")
except Exception as e:
    logger.error(f"Failed to initialize SentenceTransformer: {e}")
    raise

# Проверка доступных моделей spaCy
logger.info(f"Available spaCy models: {spacy.util.get_installed_models()}")
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.error(f"Failed to load en_core_web_sm: {e}")
    raise

# Инициализация DistilBERT для анализа тональности
try:
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    sentiment_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    sentiment_model.to(device)  # Явно перемещаем модель на CPU
    sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=tokenizer, device=device)
except Exception as e:
    logger.error(f"Failed to initialize DistilBERT: {e}")
    raise

# Предобработка текста
def clean_text(text):
    text = re.sub(r"http\S+|@\w+", "", text)  # Сохраняем хэштеги для извлечения
    return text.strip()

# Сбор твитов с расширенным списком ключевых слов
def fetch_tweets(since_date, target_count):
    url = "https://api.twitterapi.io/twitter/tweet/advanced_search"
    since_time = datetime.combine(since_date, datetime.min.time())
    until_time = datetime.utcnow()
    since_str = since_time.strftime("%Y-%m-%d_%H:%M:%S_UTC")
    until_str = until_time.strftime("%Y-%m-%d_%H:%M:%S_UTC")

    query = (
        "crypto OR cryptocurrency OR blockchain OR token OR coin OR DeFi OR NFT OR Web3 OR metaverse OR GameFi OR DAO OR "
        "smart contract OR dapp OR wallet OR staking OR yield farming OR Layer 1 OR Layer 2 OR "
        "Bitcoin OR BTC OR Ethereum OR ETH OR Solana OR SOL OR Cardano OR ADA OR Polkadot OR DOT OR "
        "HODL OR moon OR bull OR bear OR rugpull OR FOMO OR FUD OR whale OR pump OR dump OR "
        "#crypto OR #DeFi OR #NFT OR #Web3 OR #metaverse OR #GameFi OR #DAO OR #BTC OR #ETH OR #SOL OR #ADA OR "
        "AI blockchain OR zk-rollup OR cross-chain OR interoperability OR tokenization OR RWA OR #AIcrypto OR #zkrollup OR #RWA "
        "-filter:retweets lang:en"
    )
    params = {
        "query": query,
        "queryType": "Latest"
    }
    headers = {
        "X-API-Key": API_KEY
    }

    all_tweets = []
    next_cursor = None

    while len(all_tweets) < target_count:
        if next_cursor:
            params["cursor"] = next_cursor

        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            data = response.json()
            tweets = data.get("tweets", [])
            if tweets:
                logger.info(f"Tweet object structure: {json.dumps(tweets[0], indent=2)}")
            all_tweets.extend(tweets)

            if data.get("has_next_page", False) and data.get("next_cursor", "") != "":
                next_cursor = data.get("next_cursor")
                continue
            else:
                break
        else:
            raise Exception(f"Error fetching tweets: {response.status_code} - {response.text}")

    logger.info(f"Found {len(all_tweets)} tweets matching crypto keywords")
    return all_tweets[:target_count]

# Удаление старых записей из Pinecone
def clear_pinecone_index(index):
    try:
        stats = index.describe_index_stats()
        if stats["total_vector_count"] > 0:
            logger.info("Deleting all vectors from Pinecone index")
            index.delete(delete_all=True)
            logger.info("All vectors deleted successfully")
    except Exception as e:
        logger.error(f"Error clearing Pinecone index: {e}")
        raise

# Сохранение эмбеддингов в Pinecone
def save_to_pinecone(tweets):
    tweet_texts = [clean_text(tweet["text"]) for tweet in tweets]
    tweet_embeddings = model.encode(tweet_texts)
    sentiments = analyze_sentiment(tweet_texts)
    vectors = []
    for tweet, embedding, sentiment in zip(tweets, tweet_embeddings, sentiments):
        try:
            author = tweet.get("user", {}).get("screen_name", "unknown")
            if author == "unknown":
                logger.warning(f"No 'user.screen_name' in tweet: {tweet}")
        except Exception as e:
            logger.error(f"Error accessing user data: {e}")
            author = "unknown"

        vectors.append((
            str(tweet["id"]),
            embedding.tolist(),
            {
                "text": tweet["text"],
                "author": author,
                "created_at": tweet.get("created_at", ""),
                "likes": tweet.get("favorite_count", 0),
                "retweets": tweet.get("retweet_count", 0),
                "sentiment": sentiment
            }
        ))
    index.upsert(vectors=vectors)
    logger.info(f"Upserted {len(vectors)} vectors to Pinecone")

# Семантический поиск
def semantic_search(query, top_n=100):
    query_embedding = model.encode(query, convert_to_tensor=True)
    results = index.query(vector=query_embedding.tolist(), top_k=top_n, include_metadata=True)
    logger.info(f"Semantic search returned {len(results['matches'])} matches for query: {query}")
    return [(r["metadata"], r["score"]) for r in results["matches"]]

# Извлечение проектов
def extract_project_names(tweet_texts):
    projects = []
    known_projects = [
        "Bitcoin", "BTC", "Ethereum", "ETH", "Solana", "SOL", "Cardano", "ADA",
        "Polkadot", "DOT", "Dogecoin", "MoonCoin", "DeFi", "NFT", "Web3"
    ]

    for text in tweet_texts:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "ORG" or ent.text.startswith("#"):
                project = ent.text.replace("#", "")
                if project not in projects:
                    projects.append(project)
        for project in known_projects:
            if re.search(r'\b' + re.escape(project) + r'\b', text, re.IGNORECASE):
                if project not in projects:
                    projects.append(project)

    logger.info(f"Extracted project names: {projects}")
    return list(set(projects))

# Анализ тональности
def analyze_sentiment(tweet_texts):
    results = sentiment_analyzer(tweet_texts)
    return [r["score"] if r["label"] == "POSITIVE" else -r["score"] for r in results]

# Анализ трендов
def analyze_trends(project_names, days=3):
    trends = []
    for project in project_names:
        query_embedding = model.encode(project)
        results = index.query(vector=query_embedding.tolist(), top_k=1000, include_metadata=True)
        mentions = len(results["matches"])
        avg_sentiment = np.mean([r["metadata"].get("sentiment", 0) for r in results["matches"]]) if results[
            "matches"] else 0
        trends.append({"project_name": project, "mentions": mentions, "avg_sentiment": avg_sentiment})

    logger.info(f"Trends before filtering: {trends}")
    df = pd.DataFrame(trends)
    if df.empty:
        logger.warning("No trends found; DataFrame is empty")
        return df

    filtered_df = df[df["avg_sentiment"] > 0.3]
    logger.info(f"Filtered trends DataFrame: {filtered_df.to_dict()}")
    return filtered_df

# Визуализация с Matplotlib
def visualize_trends(df):
    if df.empty or "mentions" not in df.columns:
        logger.warning("Cannot visualize: DataFrame is empty or missing 'mentions' column")
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
    since_date = st.date_input("Select start date for tweet search:", value=datetime(2025, 7, 1))
    target_count = st.number_input("Enter number of tweets to fetch (1-1000):", min_value=1, max_value=1000, value=1000)

    if st.button("Analyze"):
        try:
            tweets = fetch_tweets(since_date, target_count=target_count)
            if not tweets:
                st.write(f"No tweets matching crypto keywords found since {since_date}.")
                return

            # Вывод твитов в Streamlit
            st.write("### Collected Tweets:")
            for tweet in tweets[:10]:  # Ограничим вывод первыми 10 твитами для удобства
                st.write(f"- {tweet['text']} (Author: {tweet.get('user', {}).get('screen_name', 'unknown')}, Likes: {tweet.get('favorite_count', 0)})")
            st.write(f"Total tweets collected: {len(tweets)}")

            # Удаление старых записей из Pinecone
            clear_pinecone_index(index)

            # Сохранение новых данных в Pinecone
            save_to_pinecone(tweets)

            relevant_tweets = semantic_search(query, top_n=100)
            tweet_texts = [tweet["text"] for tweet, _ in relevant_tweets]
            logger.info(f"Sample tweet texts: {tweet_texts[:5]}")

            project_names = extract_project_names(tweet_texts)
            if not project_names:
                st.write("No project names extracted from tweets.")
                return

            trending_df = analyze_trends(project_names, days=3)

            fig = visualize_trends(trending_df)
            if fig:
                st.pyplot(fig)
            else:
                st.write("No trending projects found.")

            if not trending_df.empty:
                st.write("Trending Crypto Projects:")
                st.dataframe(trending_df[["project_name", "mentions", "avg_sentiment"]])

                top_project = trending_df.iloc[0]["project_name"] if not trending_df.empty else None
                if top_project:
                    st.write(f"Sample Tweets for {top_project}:")
                    results = index.query(vector=model.encode(top_project).tolist(), top_k=5, include_metadata=True)
                    for r in results["matches"]:
                        st.write(
                            f"- {r['metadata']['text']} (Likes: {r['metadata']['likes']}, Sentiment: {r['metadata']['sentiment']:.2f})")
            else:
                st.write("No trending projects meet the criteria.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            logger.error(f"Streamlit error: {e}")

if __name__ == "__main__":
    main()