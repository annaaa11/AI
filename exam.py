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
TARGET_ACCOUNT = "elonmusk"  # Account to monitor
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
# from pinecone import Pinecone, ServerlessSpec
# from sentence_transformers import SentenceTransformer
# import spacy
# from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from datetime import datetime, timedelta
# import time
# import re
# import logging
# import os
# import json
# import torch
# import dotenv
#
#
# # завантажити api ключі з папки .env
# dotenv.load_dotenv()
#
# # Настройка логирования
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# # Инициализация переменных окружения
# API_KEY = os.getenv("API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
#
# # Проверка наличия ключей
# if not API_KEY or not PINECONE_API_KEY:
#     raise ValueError("API_KEY или PINECONE_API_KEY не установлены в переменных окружения")
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
#
#     index = pc.Index(index_name)
#     logger.info(f"Pinecone index stats: {index.describe_index_stats()}")
# except Exception as e:
#     logger.error(f"Error initializing Pinecone: {e}")
#     raise
#
# # Установка устройства
# device = torch.device("cpu")
#
# # Инициализация SentenceTransformer
# try:
#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     model._target_device = device
#     logger.info(f"SentenceTransformer initialized on {device}")
# except Exception as e:
#     logger.error(f"Failed to initialize SentenceTransformer: {e}")
#     raise
#
# # Загрузка модели spaCy
# try:
#     nlp = spacy.load("en_core_web_sm")
# except Exception as e:
#     logger.error(f"Failed to load en_core_web_sm: {e}")
#     raise
#
# # Инициализация модели анализа тональности
# try:
#     tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
#     sentiment_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
#     sentiment_model.to(device)
#     sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=tokenizer, device=-1)
# except Exception as e:
#     logger.error(f"Failed to initialize DistilBERT: {e}")
#     raise
#
# # Очистка текста
# def clean_text(text):
#     text = re.sub(r"http\S+|@\w+", "", text)
#     return text.strip()

# # Получение твитов
# def fetch_tweets(since_date, target_count, batch_size=100):
#     url = "https://api.twitterapi.io/twitter/tweet/advanced_search"
#     since_time = datetime.combine(since_date, datetime.min.time())
#     until_time = datetime.utcnow()
#     since_str = since_time.strftime("%Y-%m-%d_%H:%M:%S_UTC")
#     until_str = until_time.strftime("%Y-%m-%d_%H:%M:%S_UTC")
#
#     query = (
#         "((\"pump and coin\" OR memecoin OR \"pump and fun\" OR Solana) "
#         "min_retweets:10 min_replies:5 lang:en -filter:retweets)"
#     )
#
#     headers = {
#         "X-API-Key": API_KEY
#     }
#
#     collected = []
#     next_cursor = None
#     seen_ids = set()
#
#     while len(collected) < target_count:
#         params = {
#             "query": query,
#             "queryType": "Latest",
#             "limit": batch_size  # Запрашиваем сразу больше твитов
#         }
#         if next_cursor:
#             params["cursor"] = next_cursor
#
#         response = requests.get(url, headers=headers, params=params)
#         if response.status_code != 200:
#             logger.error(f"Failed to fetch tweets: {response.status_code} - {response.text}")
#             break
#
#         data = response.json()
#         tweets = data.get("tweets", [])
#
#         # Фильтруем сразу внутри итерации
#         for tweet in tweets:
#             tweet_id = tweet.get("id")
#             if not tweet_id or tweet_id in seen_ids:
#                 continue
#
#             retweets = tweet.get("retweet_count", 0)
#             replies = tweet.get("reply_count", 0)
#             author = tweet.get("user", {}).get("screen_name", "")
#
#             if retweets > 20 and replies > 10 and author and author != "unknown":
#                 collected.append(tweet)
#                 seen_ids.add(tweet_id)
#
#                 if len(collected) >= target_count:
#                     break
#
#         if not data.get("has_next_page") or not data.get("next_cursor"):
#             logger.info("No more pages to fetch.")
#             break
#
#         next_cursor = data["next_cursor"]
#
#     logger.info(f"Collected {len(collected)} high-quality tweets.")
#     return collected[:target_count]



# # Очистка Pinecone new
# def clear_pinecone_index(index):
#     try:
#         stats = index.describe_index_stats()
#         if stats["total_vector_count"] > 0:
#             logger.info("Deleting all vectors from Pinecone index")
#             index.delete(filter={})
#             logger.info("All vectors deleted successfully")
#     except Exception as e:
#         logger.error(f"Error clearing Pinecone index: {e}")
#         raise
#
# # Сохранение твитов в Pinecone
# def save_to_pinecone(tweets):
#     tweet_texts = [clean_text(tweet["text"]) for tweet in tweets]
#     tweet_embeddings = model.encode(tweet_texts, show_progress_bar=False)
#     sentiments = analyze_sentiment(tweet_texts)
#
#     vectors = []
#     for tweet, embedding, sentiment in zip(tweets, tweet_embeddings, sentiments):
#         author = tweet.get("user", {}).get("screen_name", "unknown")
#         if author == "unknown":
#             continue
#
#         vectors.append((
#             str(tweet["id"]),
#             embedding.tolist(),
#             {
#                 "text": tweet["text"],
#                 "author": author,
#                 "created_at": tweet.get("created_at", ""),
#                 "likes": tweet.get("favorite_count", 0),
#                 "retweets": tweet.get("retweet_count", 0),
#                 "replies": tweet.get("reply_count", 0),
#                 "sentiment": sentiment
#             }
#         ))
#
#     if vectors:
#         index.upsert(vectors=vectors)
#         logger.info(f"Upserted {len(vectors)} vectors to Pinecone")
#     else:
#         logger.warning("No valid tweets to upsert")
#
# # Семантический поиск
# def semantic_search(query, top_n=100):
#     query_embedding = model.encode(query, convert_to_tensor=True)
#     results = index.query(vector=query_embedding.tolist(), top_k=top_n, include_metadata=True)
#     return [(r["metadata"], r["score"]) for r in results["matches"]]
#
# # Извлечение названий проектов
# def extract_project_names(tweet_texts):
#     projects = []
#     known_projects = [
#         "Bitcoin", "BTC", "Ethereum", "ETH", "Solana", "SOL", "Cardano", "ADA",
#         "Polkadot", "DOT", "Dogecoin", "MoonCoin", "DeFi", "NFT", "Web3", "memecoin"
#     ]
#
#     for text in tweet_texts:
#         doc = nlp(text)
#         for ent in doc.ents:
#             if ent.label_ == "ORG" or ent.text.startswith("#"):
#                 project = ent.text.replace("#", "")
#                 if project not in projects:
#                     projects.append(project)
#
#         for project in known_projects:
#             if re.search(r'\b' + re.escape(project) + r'\b', text, re.IGNORECASE):
#                 if project not in projects:
#                     projects.append(project)
#
#     logger.info(f"Extracted project names: {projects}")
#     return list(set(projects))
#
# # Анализ тональности
# def analyze_sentiment(tweet_texts):
#     results = sentiment_analyzer(tweet_texts)
#     return [r["score"] if r["label"] == "POSITIVE" else -r["score"] for r in results]
#
# # Анализ трендов
# def analyze_trends(project_names, days=3):
#     trends = []
#     for project in project_names:
#         query_embedding = model.encode(project)
#         results = index.query(vector=query_embedding.tolist(), top_k=1000, include_metadata=True)
#         mentions = len(results["matches"])
#         avg_sentiment = np.mean([r["metadata"].get("sentiment", 0) for r in results["matches"]]) if results["matches"] else 0
#         trends.append({"project_name": project, "mentions": mentions, "avg_sentiment": avg_sentiment})
#
#     df = pd.DataFrame(trends)
#     filtered_df = df[df["avg_sentiment"] > 0.3]
#     return filtered_df
#
# # Визуализация трендов
# def visualize_trends(df):
#     if df.empty or "mentions" not in df.columns:
#         logger.warning("Cannot visualize: empty DataFrame")
#         return None
#
#     plt.figure(figsize=(10, 6))
#     norm = plt.Normalize(-1, 1)
#     cmap = cm.get_cmap("RdYlGn")
#     colors = [cmap(norm(sentiment)) for sentiment in df["avg_sentiment"]]
#
#     plt.bar(df["project_name"], df["mentions"], color=colors)
#     plt.xlabel("Project Name")
#     plt.ylabel("Mentions")
#     plt.title("Trending Crypto Projects (Last 3 Days)")
#     plt.xticks(rotation=45, ha="right")
#
#     sm = cm.ScalarMappable(cmap=cmap, norm=norm)
#     plt.colorbar(sm, label="Average Sentiment")
#
#     plt.tight_layout()
#     plt.show()
#     return plt.gcf()
#
# def fetch_tweets(since_date, target_count):
#     url = "https://api.twitterapi.io/twitter/tweet/advanced_search"
#     since_time = datetime.combine(since_date, datetime.min.time())
#     until_time = datetime.utcnow()
#     since_str = since_time.strftime("%Y-%m-%d_%H:%M:%S_UTC")
#     until_str = until_time.strftime("%Y-%m-%d_%H:%M:%S_UTC")
#
#     query = "(Solana min_retweets:20 min_replies:20 lang:en -filter:retweets)"
#     params = {
#         "query": query,
#         "queryType": "Latest",
#         "max_results": 100
#     }
#     headers = {
#         "X-API-Key": API_KEY
#     }
#
#     all_tweets = []
#     next_cursor = None
#     request_count = 0
#     max_requests = 10
#
#     while len(all_tweets) < target_count and request_count < max_requests:
#         if next_cursor:
#             params["cursor"] = next_cursor
#
#         try:
#             start_time = time.time()
#             response = requests.get(url, headers=headers, params=params)
#             request_count += 1
#             logger.info(f"API request {request_count} took {time.time() - start_time:.2f} seconds")
#
#             if response.status_code == 200:
#                 data = response.json()
#                 tweets = data.get("tweets", [])
#                 if tweets:
#                     logger.info(f"Received {len(tweets)} tweets in request {request_count}")
#                     # Вывод твитов с текстом, количеством ретвитов и ответов
#                     print(f"\nTweets from API request {request_count}:")
#                     for tweet in tweets:
#                         text = tweet.get("text", "No text available")
#                         retweets = tweet.get("retweet_count", 0)
#                         replies = tweet.get("reply_count", 0)
#                         # Проверка наличия поля user
#                         user = tweet.get("user", {})
#                         if not user or "screen_name" not in user:
#                             logger.warning(f"Tweet missing 'user' or 'screen_name': {text}")
#                         if retweets == 0:
#                             logger.warning(f"Tweet with 0 retweets found: {text}")
#                         print(f"- Text: {text}")
#                         print(f"  Retweets: {retweets}")
#                         print(f"  Replies: {replies}")
#
#                 all_tweets.extend(tweets)
#
#                 if len(all_tweets) >= target_count:
#                     break
#
#                 if data.get("has_next_page", False) and data.get("next_cursor", ""):
#                     next_cursor = data.get("next_cursor")
#                     logger.info(f"Next cursor: {next_cursor}")
#                 else:
#                     logger.info("No more pages available")
#                     break
#             elif response.status_code == 429:
#                 logger.warning("Rate limit exceeded. Waiting 60 seconds...")
#                 time.sleep(60)
#             else:
#                 raise Exception(f"Error fetching tweets: {response.status_code} - {response.text}")
#         except Exception as e:
#             logger.error(f"API request failed: {e}")
#             break
#
#     logger.info(f"Total API requests made: {request_count}")
#     logger.info(f"Found {len(all_tweets)} tweets matching query: {query}")
#     return all_tweets[:target_count]
#
# # Главная функция
# def main():
#     print("Crypto Trends Analyzer")
#     query = input("Enter your query (default: 'pump and coin or memecoin or pump and fun or Solana'): ") or \
#             "pump and coin or memecoin or pump and fun or Solana"
#     since_date_str = input("Enter start date (YYYY-MM-DD, default: 2025-07-01): ") or "2025-07-01"
#     try:
#         since_date = datetime.strptime(since_date_str, "%Y-%m-%d").date()
#     except ValueError:
#         print("Invalid date. Using default: 2025-07-01")
#         since_date = datetime(2025, 7, 1).date()
#
#     target_count = input("Enter number of tweets to fetch (1-1000, default: 1000): ") or "1000"
#     try:
#         target_count = int(target_count)
#         if not 1 <= target_count <= 1000:
#             raise ValueError("Invalid range")
#     except ValueError:
#         print("Invalid number. Using default: 1000")
#         target_count = 1000
#
#     try:
#         tweets = fetch_tweets(since_date, target_count=target_count)
#         if not tweets:
#             print("No tweets found.")
#             return
#
#         print("\nSample tweets:")
#         for tweet in tweets[:10]:
#             print(f"- {tweet['text']} (Author: {tweet['user']['screen_name']})")
#         #
#         return
#
#         #clear_pinecone_index(index)
#         save_to_pinecone(tweets)
#
#         relevant_tweets = semantic_search(query)
#         tweet_texts = [tweet["text"] for tweet, _ in relevant_tweets]
#         project_names = extract_project_names(tweet_texts)
#
#         if not project_names:
#             print("No projects found.")
#             return
#
#         trending_df = analyze_trends(project_names)
#
#         if not trending_df.empty:
#             print("\nTrending Projects:")
#             print(trending_df[["project_name", "mentions", "avg_sentiment"]].to_string(index=False))
#
#             top_project = trending_df.iloc[0]["project_name"]
#             print(f"\nSample tweets for: {top_project}")
#             results = index.query(vector=model.encode(top_project).tolist(), top_k=5, include_metadata=True)
#             for r in results["matches"]:
#                 print(f"- {r['metadata']['text']} (Sentiment: {r['metadata']['sentiment']:.2f})")
#         else:
#             print("No trending projects found.")
#
#         visualize_trends(trending_df)
#
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         logger.error(f"Fatal error: {e}")
#
# if __name__ == "__main__":
#     main()

# import os
# import json
# import dotenv
# import streamlit as st
# from uuid import uuid4
# from datetime import datetime
# from urllib.parse import urlparse
# import requests
# import nest_asyncio
#
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_core.documents import Document
# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# from langchain_pinecone import PineconeVectorStore
# from pinecone import Pinecone, ServerlessSpec
# from langgraph.prebuilt import create_react_agent
#
# # Apply nest_asyncio to allow nested event loops
# nest_asyncio.apply()
#
# dotenv.load_dotenv()
#
# # API keys
# api_key = os.getenv("GEMINI_API_KEY")
# pinecone_key = os.getenv("PINECONE_API_KEY")
# twitter_api_key = os.getenv("TWITTER_API_KEY") or "b45c33e1de7d49c2a761857d7ac9ec01"
#
# # Initialize embeddings and Pinecone
# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/text-embedding-004",
#     google_api_key=api_key
# )
#
# pc = Pinecone(api_key=pinecone_key)
# index_name = "task1"
#
# if not pc.has_index(index_name):
#     pc.create_index(
#         name=index_name,
#         dimension=768,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1")
#     )
#
# index = pc.Index(index_name)
# vector_store = PineconeVectorStore(index=index, embedding=embeddings)
#
# # JSON for storing IDs
# json_path = "data_ai.json"
# if os.path.exists(json_path):
#     with open(json_path, "r", encoding="utf-8") as f:
#         id_data = json.load(f)
# else:
#     id_data = {}
#
#
# # Function to search tweets
# def search_tweets_by_user_since(twitter_url: str, start_date: datetime, limit=20):
#     parsed_url = urlparse(twitter_url)
#     path_parts = parsed_url.path.strip("/").split("/")
#     if len(path_parts) < 1 or not path_parts[0]:
#         print("Неверный URL Twitter аккаунта")
#         return []
#
#     user = path_parts[0]
#     since_str = start_date.strftime("%Y-%m-%d")
#
#     url = "https://api.twitterapi.io/twitter/tweet/advanced_search"
#     headers = {"x-api-key": twitter_api_key}
#
#     query = f"from:{user} since:{since_str}"
#
#     params = {
#         "query": query,
#         "queryType": "Latest",
#         "limit": limit
#     }
#
#     response = requests.get(url, headers=headers, params=params)
#
#     if response.status_code == 200:
#         data = response.json()
#         tweets = data.get("tweets") or data.get("data") or []
#         return tweets
#     else:
#         print(f"❌ Ошибка {response.status_code}: {response.text}")
#         return []
#
#
# # LLM and agent
# llm = ChatGoogleGenerativeAI(
#     model='gemini-2.0-flash',
#     google_api_key=api_key
# )
#
#
# def doc_ser(user_text: str):
#     """
#     Search for documents in the vector database based on user query.
#
#     Args:
#         user_text (str): The user's query text to search for similar documents.
#
#     Returns:
#         List[Document]: A list of the top 3 most similar documents from the vector database.
#     """
#     docs = vector_store.similarity_search(user_text, k=3)
#     return docs
#
#
# agent = create_react_agent(
#     model=llm,
#     tools=[doc_ser]
# )
#
# # Streamlit interface
# st.title("Администрация Векторной Базы Данных")
#
# # Add tweets
# st.subheader("Добавить твиты в базу")
#
# twitter_url = st.text_input("Введите URL Twitter аккаунта:")
# start_date = st.date_input("Выберите начальную дату:", value=datetime(2025, 7, 1))
# limit = st.number_input("Количество твитов:", min_value=1, max_value=100, value=20)
#
# if st.button("Загрузить твиты"):
#     if twitter_url:
#         # Очистка базы перед загрузкой
#         index = pc.Index(index_name)
#         index.delete(delete_all=True)
#         with open(json_path, "w", encoding="utf-8") as jf:
#             json.dump({}, jf, ensure_ascii=False)
#         #
#         tweets = search_tweets_by_user_since(twitter_url, start_date, limit)
#
#         if tweets:
#             docs = []
#             doc_ids = []
#             new_id_data = {}
#             username = urlparse(twitter_url).path.strip("/").split("/")[0]
#
#             for tweet in tweets:
#                 text = tweet.get("text", "")
#                 if not text:
#                     continue
#
#                 doc = Document(
#                     page_content=text,
#                     metadata={
#                         "source": f"twitter_{username}",
#                         "tweet_id": tweet.get("id_str", str(uuid4())),
#                         "created_at": tweet.get("created_at", ""),
#                         "retweet_count": tweet.get("retweet_count", 0),
#                         "reply_count": tweet.get("reply_count", 0),
#                         "view_count": tweet.get("view_count", 0)
#                     }
#                 )
#                 docs.append(doc)
#
#                 new_id = str(uuid4())
#                 doc_ids.append(new_id)
#
#                 key_name = f"twitter_{username}_{tweet.get('id_str', new_id)}"
#                 new_id_data[key_name] = new_id
#
#             # Update JSON
#             if os.path.exists(json_path):
#                 with open(json_path, "r", encoding="utf-8") as jf:
#                     existing_data = json.load(jf)
#             else:
#                 existing_data = {}
#
#             existing_data.update(new_id_data)
#
#             with open(json_path, "w", encoding="utf-8") as jf:
#                 json.dump(existing_data, jf, ensure_ascii=False)
#
#             # Add to vector store
#             vector_store.add_documents(docs, ids=doc_ids)
#
#             st.success(f"Добавлено {len(docs)} твитов от @{username} в базу.")
#         else:
#             st.error("Не удалось получить твиты. Проверьте URL или дату.")
#     else:
#         st.error("Пожалуйста, введите URL Twitter аккаунта.")
#
# # Chat with search
# st.subheader("Чат с поиskom по векторной базе")
#
# if 'data' not in st.session_state:
#     st.session_state["data"] = {'messages': [
#         SystemMessage(
#             "Ти чат-бот, який відповідає на питання, використовуючи документи з бази. "
#             "Якщо немає відповіді у документах — відповідай самостійно."
#         )
#     ]}
#
# user_text = st.chat_input('Ваше повідомлення: ')
# if user_text:
#     user_text = HumanMessage(user_text)
#     st.session_state['data']['messages'].append(user_text)
#     response = agent.invoke(st.session_state["data"])
#     st.session_state['data'] = response
#
# for message in st.session_state['data']['messages']:
#     if isinstance(message, HumanMessage):
#         role = "user"
#     elif isinstance(message, AIMessage):
#         role = "bot"
#     else:
#         continue
#
#     with st.chat_message(role):
#         st.markdown(message.content)
#

#############


import os
import json
import dotenv
import streamlit as st
from uuid import uuid4
from datetime import datetime
from urllib.parse import urlparse
import requests
import nest_asyncio
from bs4 import BeautifulSoup

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langgraph.prebuilt import create_react_agent

# Apply nest_asyncio to handle async issues in Streamlit
nest_asyncio.apply()

dotenv.load_dotenv()

# API keys
api_key = os.getenv("GEMINI_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")
twitter_api_key = os.getenv("TWITTER_API_KEY") or "b45c33e1de7d49c2a761857d7ac9ec01"

# Initialize embeddings and Pinecone
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=api_key
)

pc = Pinecone(api_key=pinecone_key)
index_name = "task1"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# JSON for storing IDs
json_path = "data_ai.json"
if os.path.exists(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        id_data = json.load(f)
else:
    id_data = {}


# Function to parse CoinMarketCap project page
def parse_coinmarketcap_project(url):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Ошибка: {response.status_code} при запросе {url}")
        return None

    soup = BeautifulSoup(response.text, "lxml")

    # Name and symbol: parse from <h1>
    name = "?"
    symbol = "?"
    h1 = soup.find("h1")
    if h1:
        lines = [line.strip() for line in h1.stripped_strings if line.strip()]
        filtered = [line for line in lines if line.lower() != 'price' and line != '']
        if len(filtered) >= 2:
            name = filtered[0]
            symbol = filtered[1]

    # Twitter link
    twitter_link = None
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "twitter.com" in href:
            twitter_link = href.strip()
            break

    if twitter_link and twitter_link.startswith("//"):
        twitter_link = "https:" + twitter_link

    return {
        "name": name,
        "symbol": symbol,
        "twitter": twitter_link or "Not found",
        "url": url
    }


# Function to search tweets
def search_tweets_by_query(query: str, start_date: datetime, limit: int = 20, min_retweets: int = 0,
                           min_replies: int = 0):
    url = "https://api.twitterapi.io/twitter/tweet/advanced_search"
    headers = {"x-api-key": twitter_api_key}
    since_str = start_date.strftime("%Y-%m-%d")

    all_tweets = []
    remaining_limit = limit
    max_per_request = 20

    while remaining_limit > 0:
        current_limit = min(max_per_request, remaining_limit)
        params = {
            "query": f"{query} since:{since_str} min_retweets:{min_retweets} min_replies:{min_replies}",
            "queryType": "Latest",
            "limit": current_limit
        }

        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            data = response.json()
            tweets = data.get("tweets") or data.get("data") or []
            all_tweets.extend(tweets)

            # If we got fewer tweets than requested, stop further requests
            if len(tweets) < current_limit:
                break

            remaining_limit -= current_limit
        else:
            print(f"❌ Ошибка {response.status_code}: {response.text}")
            break

    return all_tweets


# LLM and agent
llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    google_api_key=api_key
)


def doc_ser(user_text: str):
    """
    Search for documents in the vector database based on user query.

    Args:
        user_text (str): The user's query text to search for similar documents.

    Returns:
        List[Document]: A list of the top 3 most similar documents from the vector database.
    """
    docs = vector_store.similarity_search(user_text, k=3)
    return docs


agent = create_react_agent(
    model=llm,
    tools=[doc_ser]
)

# Streamlit interface
st.title("Администрация Векторной Базы Данных")

# Clear database
st.subheader("Очистка базы данных")
if st.button("Очистить векторную базу и JSON"):
    index = pc.Index(index_name)
    index.delete(delete_all=True)
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump({}, jf, ensure_ascii=False)
    st.success("Векторная база и JSON-файл очищены.")

# Add tweets
st.subheader("Добавить твиты в базу")

coinmarketcap_url = st.text_input(
    "Введите URL CoinMarketCap (например, https://coinmarketcap.com/currencies/legends-of-elumia/):")
start_date = st.date_input("Выберите начальную дату:", value=datetime(2025, 7, 1))
limit = st.number_input("Количество твитов:", min_value=1, max_value=100, value=20)
min_retweets = st.number_input("Минимальное количество ретвитов:", min_value=0, value=0)
min_replies = st.number_input("Минимальное количество ответов:", min_value=0, value=0)

if st.button("Загрузить твиты"):
    if coinmarketcap_url:
        # Parse CoinMarketCap to get project info
        project_info = parse_coinmarketcap_project(coinmarketcap_url)
        if not project_info or project_info["twitter"] == "Not found":
            st.error("Не удалось найти Twitter URL на странице CoinMarketCap.")
        else:
            twitter_url = project_info["twitter"]
            project_name = project_info["name"]
            project_symbol = project_info["symbol"]
            st.info(f"Найден Twitter: {twitter_url} (Проект: {project_name}, Символ: {project_symbol})")

            # Clear database before loading new tweets
            index = pc.Index(index_name)
            index.delete(delete_all=True)
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump({}, jf, ensure_ascii=False)

            # Form query for tweet search
            query = f"({project_name}) OR ${project_symbol}"
            tweets = search_tweets_by_query(query, start_date, limit, min_retweets, min_replies)

            if tweets:
                docs = []
                doc_ids = []
                new_id_data = {}
                official_username = urlparse(twitter_url).path.strip("/").split("/")[0]

                for tweet in tweets:
                    text = tweet.get("text", "")
                    if not text:
                        continue

                    # Determine if tweet is from official account or external user
                    author_username = tweet.get("user", {}).get("screen_name", "unknown")
                    is_official = author_username.lower() == official_username.lower()
                    author_type = "official" if is_official else "external"

                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": f"twitter_{official_username}",
                            "tweet_id": tweet.get("id_str", str(uuid4())),
                            "created_at": tweet.get("created_at", ""),
                            "retweet_count": tweet.get("retweet_count", 0),
                            "reply_count": tweet.get("reply_count", 0),
                            "view_count": tweet.get("view_count", 0),
                            "project_name": project_name,
                            "project_symbol": project_symbol,
                            "coinmarketcap_url": coinmarketcap_url,
                            "author_username": author_username,
                            "author_type": author_type
                        }
                    )
                    docs.append(doc)

                    new_id = str(uuid4())
                    doc_ids.append(new_id)

                    key_name = f"twitter_{official_username}_{tweet.get('id_str', new_id)}"
                    new_id_data[key_name] = new_id

                # Update JSON
                if os.path.exists(json_path):
                    with open(json_path, "r", encoding="utf-8") as jf:
                        existing_data = json.load(jf)
                else:
                    existing_data = {}

                existing_data.update(new_id_data)

                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(existing_data, jf, ensure_ascii=False)

                # Add to vector store
                vector_store.add_documents(docs, ids=doc_ids)

                st.success(f"Добавлено {len(docs)} твитов, связанных с {project_name} (${project_symbol}).")
            else:
                st.error("Не удалось получить твиты. Проверьте параметры запроса.")
    else:
        st.error("Пожалуйста, введите URL CoinMarketCap.")

# Chat with search
st.subheader("Чат с поиском по векторной базе")

if 'data' not in st.session_state:
    st.session_state["data"] = {'messages': [
        SystemMessage(
            "Ти чат-бот, який відповідає на питання, використовуючи документи з бази. "
            "Якщо немає відповіді у документах — відповідай самостійно."
        )
    ]}

user_text = st.chat_input('Ваше повідомлення: ')
if user_text:
    user_text = HumanMessage(user_text)
    st.session_state['data']['messages'].append(user_text)
    response = agent.invoke(st.session_state["data"])
    st.session_state['data'] = response

for message in st.session_state['data']['messages']:
    if isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, AIMessage):
        role = "bot"
    else:
        continue

    with st.chat_message(role):
        st.markdown(message.content)