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
#
# import os
# import json
# import dotenv
# import streamlit as st
# from uuid import uuid4
# from datetime import datetime
# from urllib.parse import urlparse
# import requests
# import nest_asyncio
# from bs4 import BeautifulSoup
#
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_core.documents import Document
# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# from langchain_pinecone import PineconeVectorStore
# from pinecone import Pinecone, ServerlessSpec
# from langgraph.prebuilt import create_react_agent
#
# # Apply nest_asyncio to handle async issues in Streamlit
# nest_asyncio.apply()
#
# # Load environment variables
# dotenv.load_dotenv()  # Adjust path if needed: dotenv.load_dotenv('/mount/src/ai/.env')
#
# # API keys
# api_key = os.getenv("GEMINI_API_KEY")
# pinecone_key = os.getenv("PINECONE_API_KEY")
# twitter_api_key = os.getenv("TWITTER_API_KEY") or "b45c33e1de7d49c2a761857d7ac9ec01"
#
# # Validate API keys
# if not api_key:
#     st.error("GEMINI_API_KEY is not set. Please add it to the .env file or Streamlit Cloud Secrets.")
#     st.stop()
# if not pinecone_key:
#     st.error("PINECONE_API_KEY is not set. Please add it to the .env file or Streamlit Cloud Secrets.")
#     st.stop()
# if not twitter_api_key:
#     st.warning("TWITTER_API_KEY is not set, using default key.")
#
# # Debug print to verify API keys
# print(f"GEMINI_API_KEY: {api_key}")
# print(f"PINECONE_API_KEY: {pinecone_key}")
# print(f"TWITTER_API_KEY: {twitter_api_key}")
#
# # Initialize embeddings and Pinecone
# try:
#     embeddings = GoogleGenerativeAIEmbeddings(
#         model="models/text-embedding-004",
#         google_api_key=api_key
#     )
# except Exception as e:
#     st.error(f"Failed to initialize GoogleGenerativeAIEmbeddings: {str(e)}")
#     st.stop()
#
# try:
#     pc = Pinecone(api_key=pinecone_key)
# except Exception as e:
#     st.error(f"Failed to initialize Pinecone: {str(e)}")
#     st.stop()
#
# index_name = "task1"
#
# try:
#     if not pc.has_index(index_name):
#         pc.create_index(
#             name=index_name,
#             dimension=768,
#             metric="cosine",
#             spec=ServerlessSpec(cloud="aws", region="us-east-1")
#         )
# except Exception as e:
#     st.error(f"Failed to create Pinecone index: {str(e)}")
#     st.stop()
#
# index = pc.Index(index_name)
# vector_store = PineconeVectorStore(index=index, embedding=embeddings)
#
# # JSON for storing IDs
# json_path = "data_ai.json"
# if os.path.exists(json_path):
#     try:
#         with open(json_path, "r", encoding="utf-8") as f:
#             id_data = json.load(f)
#     except Exception as e:
#         st.error(f"Failed to read JSON file {json_path}: {str(e)}")
#         id_data = {}
# else:
#     id_data = {}
#
#
# # Function to parse CoinMarketCap project page
# def parse_coinmarketcap_project(url):
#     headers = {
#         "User-Agent": "Mozilla/5.0"
#     }
#
#     try:
#         response = requests.get(url, headers=headers)
#         response.raise_for_status()
#     except requests.RequestException as e:
#         print(f"Ошибка: {str(e)} при запросе {url}")
#         return None
#
#     soup = BeautifulSoup(response.text, "lxml")
#
#     # Name and symbol: parse from <h1>
#     name = "?"
#     symbol = "?"
#     h1 = soup.find("h1")
#     if h1:
#         lines = [line.strip() for line in h1.stripped_strings if line.strip()]
#         filtered = [line for line in lines if line.lower() != 'price' and line != '']
#         if len(filtered) >= 2:
#             name = filtered[0]
#             symbol = filtered[1]
#
#     # Twitter link
#     twitter_link = None
#     for a in soup.find_all("a", href=True):
#         href = a["href"]
#         if "twitter.com" in href:
#             twitter_link = href.strip()
#             break
#
#     if twitter_link and twitter_link.startswith("//"):
#         twitter_link = "https:" + twitter_link
#
#     return {
#         "name": name,
#         "symbol": symbol,
#         "twitter": twitter_link or "Not found",
#         "url": url
#     }
#
#
# # Function to search tweets
# def search_tweets_by_query(query: str, start_date: datetime, limit: int = 20, min_retweets: int = 0,
#                            min_replies: int = 0):
#     url = "https://api.twitterapi.io/twitter/tweet/advanced_search"
#     headers = {"x-api-key": twitter_api_key}
#     since_str = start_date.strftime("%Y-%m-%d")
#
#     all_tweets = []
#     remaining_limit = limit
#     max_per_request = 20
#
#     while remaining_limit > 0:
#         current_limit = min(max_per_request, remaining_limit)
#         params = {
#             "query": f"{query} since:{since_str} min_retweets:{min_retweets} min_replies:{min_replies}",
#             "queryType": "Latest",
#             "limit": current_limit
#         }
#
#         try:
#             response = requests.get(url, headers=headers, params=params)
#             response.raise_for_status()
#             data = response.json()
#             tweets = data.get("tweets") or data.get("data") or []
#             all_tweets.extend(tweets)
#
#             # If we got fewer tweets than requested, stop further requests
#             if len(tweets) < current_limit:
#                 break
#
#             remaining_limit -= current_limit
#         except requests.RequestException as e:
#             print(f"❌ Ошибка при запросе к Twitter API: {str(e)}")
#             break
#
#     return all_tweets
#
#
# # LLM and agent
# try:
#     llm = ChatGoogleGenerativeAI(
#         model='gemini-2.0-flash',
#         google_api_key=api_key
#     )
# except Exception as e:
#     st.error(f"Failed to initialize ChatGoogleGenerativeAI: {str(e)}")
#     st.stop()
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
#     try:
#         docs = vector_store.similarity_search(user_text, k=3)
#         return docs
#     except Exception as e:
#         st.error(f"Error during vector store search: {str(e)}")
#         return []
#
#
# try:
#     agent = create_react_agent(
#         model=llm,
#         tools=[doc_ser]
#     )
# except Exception as e:
#     st.error(f"Failed to create agent: {str(e)}")
#     st.stop()
#
# # Streamlit interface
# st.title("Администрация Векторной Базы Данных")
#
# # Clear database
# st.subheader("Очистка базы данных")
# if st.button("Очистить векторную базу и JSON"):
#     try:
#         index = pc.Index(index_name)
#         index.delete(delete_all=True)
#         with open(json_path, "w", encoding="utf-8") as jf:
#             json.dump({}, jf, ensure_ascii=False)
#         st.success("Векторная база и JSON-файл очищены.")
#     except Exception as e:
#         st.error(f"Failed to clear database: {str(e)}")
#
# # Add tweets
# st.subheader("Добавить твиты в базу")
#
# coinmarketcap_url = st.text_input(
#     "Введите URL CoinMarketCap (например, https://coinmarketcap.com/currencies/legends-of-elumia/):")
# start_date = st.date_input("Выберите начальную дату:", value=datetime(2025, 7, 1))
# limit = st.number_input("Количество твитов:", min_value=1, max_value=100, value=20)
# min_retweets = st.number_input("Минимальное количество ретвитов:", min_value=0, value=0)
# min_replies = st.number_input("Минимальное количество ответов:", min_value=0, value=0)
#
# if st.button("Загрузить твиты"):
#     if coinmarketcap_url:
#         # Parse CoinMarketCap to get project info
#         project_info = parse_coinmarketcap_project(coinmarketcap_url)
#         if not project_info or project_info["twitter"] == "Not found":
#             st.error("Не удалось найти Twitter URL на странице CoinMarketCap.")
#         else:
#             twitter_url = project_info["twitter"]
#             project_name = project_info["name"]
#             project_symbol = project_info["symbol"]
#             st.info(f"Найден Twitter: {twitter_url} (Проект: {project_name}, Символ: {project_symbol})")
#
#             # Clear database before loading new tweets
#
#
#             # Form query for tweet search
#             query = f"({project_name}) OR ${project_symbol}"
#             tweets = search_tweets_by_query(query, start_date, limit, min_retweets, min_replies)
#
#             if tweets:
#                 docs = []
#                 doc_ids = []
#                 new_id_data = {}
#                 official_username = urlparse(twitter_url).path.strip("/").split("/")[0]
#
#                 for tweet in tweets:
#                     text = tweet.get("text", "")
#                     if not text:
#                         continue
#
#                     # Determine if tweet is from official account or external user
#                     author_username = tweet.get("user", {}).get("screen_name", "unknown")
#                     is_official = author_username.lower() == official_username.lower()
#                     author_type = "official" if is_official else "external"
#
#                     doc = Document(
#                         page_content=text,
#                         metadata={
#                             "source": f"twitter_{official_username}",
#                             "tweet_id": tweet.get("id_str", str(uuid4())),
#                             "created_at": tweet.get("created_at", ""),
#                             "retweet_count": tweet.get("retweet_count", 0),
#                             "reply_count": tweet.get("reply_count", 0),
#                             "view_count": tweet.get("view_count", 0),
#                             "project_name": project_name,
#                             "project_symbol": project_symbol,
#                             "coinmarketcap_url": coinmarketcap_url,
#                             "author_username": author_username,
#                             "author_type": author_type
#                         }
#                     )
#                     docs.append(doc)
#
#                     new_id = str(uuid4())
#                     doc_ids.append(new_id)
#
#                     key_name = f"twitter_{official_username}_{tweet.get('id_str', new_id)}"
#                     new_id_data[key_name] = new_id
#
#                 # Update JSON
#                 try:
#                     if os.path.exists(json_path):
#                         with open(json_path, "r", encoding="utf-8") as jf:
#                             existing_data = json.load(jf)
#                     else:
#                         existing_data = {}
#
#                     existing_data.update(new_id_data)
#
#                     with open(json_path, "w", encoding="utf-8") as jf:
#                         json.dump(existing_data, jf, ensure_ascii=False)
#                 except Exception as e:
#                     st.error(f"Failed to update JSON file: {str(e)}")
#                     st.stop()
#
#                 # Add to vector store
#                 try:
#                     vector_store.add_documents(docs, ids=doc_ids)
#                     st.success(f"Добавлено {len(docs)} твитов, связанных с {project_name} (${project_symbol}).")
#                 except Exception as e:
#                     st.error(f"Failed to add documents to vector store: {str(e)}")
#             else:
#                 st.error("Не удалось получить твиты. Проверьте параметры запроса или Twitter API ключ.")
#     else:
#         st.error("Пожалуйста, введите URL CoinMarketCap.")
#
# # Chat with search
# st.subheader("Чат с поиском по векторной базе")
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
#     try:
#         response = agent.invoke(st.session_state["data"])
#         st.session_state['data'] = response
#     except Exception as e:
#         st.error(f"Failed to invoke agent: {str(e)}")
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

####

# import os
# import json
# import dotenv
# import streamlit as st
# from uuid import uuid4
# from datetime import datetime
# from urllib.parse import urlparse
# import requests
# from requests.adapters import HTTPAdapter
# from urllib3.util.retry import Retry
# import nest_asyncio
# from bs4 import BeautifulSoup
#
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_core.documents import Document
# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# from langchain_pinecone import PineconeVectorStore
# from pinecone import Pinecone, ServerlessSpec
# from langgraph.prebuilt import create_react_agent
#
# # Apply nest_asyncio to handle async issues in Streamlit
# nest_asyncio.apply()
#
# # Load environment variables
# dotenv.load_dotenv()
#
# # API keys
# api_key = os.getenv("GEMINI_API_KEY")
# pinecone_key = os.getenv("PINECONE_API_KEY")
# twitter_api_key = os.getenv("TWITTER_API_KEY") or "b45c33e1de7d49c2a761857d7ac9ec01"
#
# # Validate API key
# if not api_key:
#     st.error("GEMINI_API_KEY is not set. Please add it to the .env file or Streamlit Cloud Secrets.")
#     st.stop()
# if not pinecone_key:
#     st.error("PINECONE_API_KEY is not set. Please add it to the .env file or Streamlit Cloud Secrets.")
#     st.stop()
# if not twitter_api_key:
#     st.warning("TWITTER_API_KEY is not set, using default key.")
#
# # Debug print to verify API keys
# print(f"GEMINI_API_KEY: {api_key}")
# print(f"PINECONE_API_KEY: {pinecone_key}")
# print(f"TWITTER_API_KEY: {twitter_api_key}")
#
# # Initialize embeddings and Pinecone
# try:
#     embeddings = GoogleGenerativeAIEmbeddings(
#         model="models/text-embedding-004",
#         google_api_key=api_key
#     )
# except Exception as e:
#     st.error(f"Failed to initialize GoogleGenerativeAIEmbeddings: {str(e)}")
#     st.stop()
#
# try:
#     pc = Pinecone(api_key=pinecone_key)
# except Exception as e:
#     st.error(f"Failed to initialize Pinecone: {str(e)}")
#     st.stop()
#
# index_name = "task1"
#
# try:
#     if not pc.has_index(index_name):
#         pc.create_index(
#             name=index_name,
#             dimension=768,
#             metric="cosine",
#             spec=ServerlessSpec(cloud="aws", region="us-east-1")
#         )
# except Exception as e:
#     st.error(f"Failed to create Pinecone index: {str(e)}")
#     st.stop()
#
# index = pc.Index(index_name)
# vector_store = PineconeVectorStore(index=index, embedding=embeddings)
#
# # JSON for storing IDs
# json_path = "data_ai.json"
# if os.path.exists(json_path):
#     try:
#         with open(json_path, "r", encoding="utf-8") as f:
#             id_data = json.load(f)
#     except Exception as e:
#         st.error(f"Failed to read JSON file {json_path}: {str(e)}")
#         id_data = {}
# else:
#     id_data = {}
#
# # Function to parse CoinMarketCap project page
# def parse_coinmarketcap_project(url):
#     headers = {
#         "User-Agent": "Mozilla/5.0"
#     }
#
#     try:
#         response = requests.get(url, headers=headers)
#         response.raise_for_status()
#     except requests.RequestException as e:
#         print(f"Ошибка: {str(e)} при запросе {url}")
#         return None
#
#     soup = BeautifulSoup(response.text, "lxml")
#
#     # Name and symbol: parse from <h1>
#     name = "?"
#     symbol = "?"
#     h1 = soup.find("h1")
#     if h1:
#         lines = [line.strip() for line in h1.stripped_strings if line.strip()]
#         filtered = [line for line in lines if line.lower() != 'price' and line != '']
#         if len(filtered) >= 2:
#             name = filtered[0]
#             symbol = filtered[1]
#
#     # Twitter link
#     twitter_link = None
#     for a in soup.find_all("a", href=True):
#         href = a["href"]
#         if "twitter.com" in href:
#             twitter_link = href.strip()
#             break
#
#     if twitter_link and twitter_link.startswith("//"):
#         twitter_link = "https:" + twitter_link
#
#     return {
#         "name": name,
#         "symbol": symbol,
#         "twitter": twitter_link or "Not found",
#         "url": url
#     }
#
# # Function to search tweets
# def search_tweets_by_query(query: str, username: str, start_date: datetime, limit: int = 20, min_retweets: int = 0, min_replies: int = 0):
#     url = "https://api.twitterapi.io/twitter/tweet/advanced_search"
#     headers = {"x-api-key": twitter_api_key}
#     since_str = start_date.strftime("%Y-%m-%d")
#
#     session = requests.Session()
#     retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
#     session.mount("https://", HTTPAdapter(max_retries=retries))
#
#     all_tweets = []
#     remaining_limit = limit
#     max_per_request = 20
#
#     # Поиск твитов от официального аккаунта и связанных с проектом
#     queries = [f"from:{username}", f"({query}) OR ${project_symbol}"]
#
#     for q in queries:
#         current_limit = min(max_per_request, remaining_limit)
#         while current_limit > 0:
#             params = {
#                 "query": f"{q} since:{since_str} min_retweets:{min_retweets} min_replies:{min_replies}",
#                 "queryType": "Latest",
#                 "limit": current_limit
#             }
#
#             try:
#                 response = session.get(url, headers=headers, params=params)
#                 response.raise_for_status()
#                 data = response.json()
#                 tweets = data.get("tweets") or data.get("data") or []
#                 all_tweets.extend(tweets)
#
#                 if len(tweets) < current_limit:
#                     break
#
#                 remaining_limit -= current_limit
#             except requests.RequestException as e:
#                 print(f"❌ Ошибка при запросе к Twitter API для запроса '{q}': {str(e)}")
#                 break
#
#     # Удаляем дубликаты по tweet_id
#     unique_tweets = {tweet["id_str"]: tweet for tweet in all_tweets if "id_str" in tweet}.values()
#     return list(unique_tweets)[:limit]
#
# # LLM and agent
# try:
#     llm = ChatGoogleGenerativeAI(
#         model='gemini-2.0-flash',
#         google_api_key=api_key
#     )
# except Exception as e:
#     st.error(f"Failed to initialize ChatGoogleGenerativeAI: {str(e)}")
#     st.stop()
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
#     try:
#         docs = vector_store.similarity_search(user_text, k=3)
#         return docs
#     except Exception as e:
#         st.error(f"Error during vector store search: {str(e)}")
#         return []
#
# try:
#     agent = create_react_agent(
#         model=llm,
#         tools=[doc_ser]
#     )
# except Exception as e:
#     st.error(f"Failed to create agent: {str(e)}")
#     st.stop()
#
# # Streamlit interface
# st.title("Администрация Векторной Базы Данных")
#
# # Clear database
# st.subheader("Очистка базы данных")
# if st.button("Очистить векторную базу и JSON"):
#     try:
#         index = pc.Index(index_name)
#         index.delete(delete_all=True)
#         with open(json_path, "w", encoding="utf-8") as jf:
#             json.dump({}, jf, ensure_ascii=False)
#         st.success("Векторная база и JSON-файл очищены.")
#     except Exception as e:
#         st.error(f"Failed to clear database: {str(e)}")
#
# # Add tweets
# st.subheader("Добавить твиты в базу")
#
# coinmarketcap_url = st.text_input(
#     "Введите URL CoinMarketCap (например, https://coinmarketcap.com/currencies/legends-of-elumia/):")
# start_date = st.date_input("Выберите начальную дату:", value=datetime(2025, 7, 1))
# limit = st.number_input("Количество твитов:", min_value=1, max_value=100, value=20)
# min_retweets = st.number_input("Минимальное количество ретвитов:", min_value=0, value=0)
# min_replies = st.number_input("Минимальное количество ответов:", min_value=0, value=0)
#
# if st.button("Загрузить твиты"):
#     if coinmarketcap_url:
#         project_info = parse_coinmarketcap_project(coinmarketcap_url)
#         if not project_info or project_info["twitter"] == "Not found":
#             st.error("Не удалось найти Twitter URL на странице CoinMarketCap.")
#         else:
#             twitter_url = project_info["twitter"]
#             project_name = project_info["name"]
#             project_symbol = project_info["symbol"]
#             official_username = urlparse(twitter_url).path.strip("/").split("/")[-1]
#             st.info(f"Найден Twitter: {twitter_url} (Проект: {project_name}, Символ: {project_symbol})")
#
#             # Очистка существующих данных для проекта
#             try:
#                 vector_store.delete(filter={"coinmarketcap_url": coinmarketcap_url})
#                 st.info("Существующие данные для этого проекта удалены из векторной базы.")
#             except Exception as e:
#                 st.error(f"Ошибка при очистке базы данных: {str(e)}")
#
#             # Поиск твитов
#             tweets = search_tweets_by_query(project_name, official_username, start_date, limit, min_retweets, min_replies)
#
#             if tweets:
#                 docs = []
#                 doc_ids = []
#                 new_id_data = {}
#                 existing_ids = set(id_data.values())
#
#                 for tweet in tweets:
#                     text = tweet.get("text", "")
#                     if not text:
#                         continue
#
#                     tweet_id = tweet.get("id_str", str(uuid4()))
#                     if tweet_id in existing_ids:
#                         print(f"Твит {tweet_id} уже существует, пропускаем.")
#                         continue
#
#                     author_username = tweet.get("user", {}).get("screen_name", "unknown")
#                     is_official = author_username.lower() == official_username.lower()
#                     author_type = "official" if is_official else "external"
#
#                     public_metrics = tweet.get("public_metrics", {})
#                     retweet_count = public_metrics.get("retweet_count", tweet.get("retweet_count", 0))
#                     reply_count = public_metrics.get("reply_count", tweet.get("reply_count", 0))
#                     view_count = public_metrics.get("view_count", tweet.get("view_count", 0))
#
#                     print(f"Tweet ID: {tweet_id}, Retweets: {retweet_count}, Replies: {reply_count}, Views: {view_count}")
#
#                     doc = Document(
#                         page_content=text,
#                         metadata={
#                             "source": f"twitter_{official_username}",
#                             "tweet_id": tweet_id,
#                             "created_at": tweet.get("created_at", ""),
#                             "retweet_count": retweet_count,
#                             "reply_count": reply_count,
#                             "view_count": view_count,
#                             "project_name": project_name,
#                             "project_symbol": project_symbol,
#                             "coinmarketcap_url": coinmarketcap_url,
#                             "author_username": author_username,
#                             "author_type": author_type
#                         }
#                     )
#                     docs.append(doc)
#
#                     new_id = str(uuid4())
#                     doc_ids.append(new_id)
#                     key_name = f"twitter_{official_username}_{tweet_id}"
#                     new_id_data[key_name] = new_id
#
#                 # Обновление JSON
#                 try:
#                     existing_data = id_data if id_data else {}
#                     existing_data.update(new_id_data)
#                     with open(json_path, "w", encoding="utf-8") as jf:
#                         json.dump(existing_data, jf, ensure_ascii=False)
#                 except Exception as e:
#                     st.error(f"Failed to update JSON file: {str(e)}")
#                     st.stop()
#
#                 # Добавление в векторную базу
#                 try:
#                     vector_store.add_documents(docs, ids=doc_ids)
#                     st.success(f"Добавлено {len(docs)} твитов, связанных с {project_name} (${project_symbol}).")
#                 except Exception as e:
#                     st.error(f"Failed to add documents to vector store: {str(e)}")
#             else:
#                 st.error("Не удалось получить твиты. Проверьте параметры запроса или Twitter API ключ.")
#     else:
#         st.error("Пожалуйста, введите URL CoinMarketCap.")
#
# # Chat with search
# st.subheader("Чат с поиском по векторной базе")
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
#     try:
#         response = agent.invoke(st.session_state["data"])
#         st.session_state['data'] = response
#     except Exception as e:
#         st.error(f"Failed to invoke agent: {str(e)}")
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
# import os
# import json
# import dotenv
# import streamlit as st
# from uuid import uuid4
# from datetime import datetime, timedelta
# from urllib.parse import urlparse
# import requests
# from requests.adapters import HTTPAdapter
# from urllib3.util.retry import Retry
# import nest_asyncio
# from bs4 import BeautifulSoup
# import matplotlib.pyplot as plt
# import pandas as pd
#
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_core.documents import Document
# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# from langchain_pinecone import PineconeVectorStore
# from pinecone import Pinecone, ServerlessSpec
# from langgraph.prebuilt import create_react_agent
#
# # Apply nest_asyncio to handle async issues in Streamlit
# nest_asyncio.apply()
#
# # Load environment variables
# dotenv.load_dotenv()
#
# # API keys
# api_key = os.getenv("GEMINI_API_KEY")
# pinecone_key = os.getenv("PINECONE_API_KEY")
# twitter_api_key = os.getenv("TWITTER_API_KEY") or "c769bb7d2f3742828d13bad15b7262d1"
#
# # Validate API keys
# if not api_key:
#     st.error("GEMINI_API_KEY is not set. Please add it to the .env file or Streamlit Cloud Secrets.")
#     st.stop()
# if not pinecone_key:
#     st.error("PINECONE_API_KEY is not set. Please add it to the .env file or Streamlit Cloud Secrets.")
#     st.stop()
# if not twitter_api_key:
#     st.warning("TWITTER_API_KEY is not set, using default key.")
#
# # Debug print to verify API keys
# st.write(f"GEMINI_API_KEY: {api_key[:5]}... (hidden for security)")
# st.write(f"PINECONE_API_KEY: {pinecone_key[:5]}... (hidden for security)")
# st.write(f"TWITTER_API_KEY: {twitter_api_key[:5]}... (hidden for security)")
#
# # Initialize embeddings and Pinecone
# try:
#     embeddings = GoogleGenerativeAIEmbeddings(
#         model="models/text-embedding-004",
#         google_api_key=api_key
#     )
# except Exception as e:
#     st.error(f"Failed to initialize GoogleGenerativeAIEmbeddings: {str(e)}")
#     st.stop()
#
# try:
#     pc = Pinecone(api_key=pinecone_key)
# except Exception as e:
#     st.error(f"Failed to initialize Pinecone: {str(e)}")
#     st.stop()
#
# index_name = "task1"
#
# try:
#     if not pc.has_index(index_name):
#         pc.create_index(
#             name=index_name,
#             dimension=768,
#             metric="cosine",
#             spec=ServerlessSpec(cloud="aws", region="us-east-1")
#         )
# except Exception as e:
#     st.error(f"Failed to create Pinecone index: {str(e)}")
#     st.stop()
#
# index = pc.Index(index_name)
# vector_store = PineconeVectorStore(index=index, embedding=embeddings)
#
# # JSON for storing IDs
# json_path = "data_ai.json"
# if os.path.exists(json_path):
#     try:
#         with open(json_path, "r", encoding="utf-8") as f:
#             id_data = json.load(f)
#     except Exception as e:
#         st.error(f"Failed to read JSON file {json_path}: {str(e)}")
#         id_data = {}
# else:
#     id_data = {}
#
#
# # Function to parse CoinMarketCap project page
# def parse_coinmarketcap_project(url):
#     headers = {
#         "User-Agent": "Mozilla/5.0"
#     }
#
#     try:
#         response = requests.get(url, headers=headers, timeout=10)
#         response.raise_for_status()
#     except requests.RequestException as e:
#         st.error(f"Ошибка при запросе CoinMarketCap {url}: {str(e)}")
#         return None
#
#     soup = BeautifulSoup(response.text, "lxml")
#
#     # Name and symbol: parse from <h1>
#     name = "?"
#     symbol = "?"
#     h1 = soup.find("h1")
#     if h1:
#         lines = [line.strip() for line in h1.stripped_strings if line.strip()]
#         filtered = [line for line in lines if line.lower() != 'price' and line != '']
#         if len(filtered) >= 2:
#             name = filtered[0]
#             symbol = filtered[1]
#
#     # Twitter link
#     twitter_link = None
#     for a in soup.find_all("a", href=True):
#         href = a["href"]
#         if "twitter.com" in href or "x.com" in href:
#             twitter_link = href.strip()
#             break
#
#     if twitter_link and twitter_link.startswith("//"):
#         twitter_link = "https:" + twitter_link
#     elif twitter_link and "x.com" in twitter_link:
#         twitter_link = twitter_link.replace("x.com", "twitter.com")
#
#     st.write(f"Parsed CoinMarketCap: name={name}, symbol={symbol}, twitter={twitter_link}")
#     return {
#         "name": name,
#         "symbol": symbol,
#         "twitter": twitter_link or "Not found",
#         "url": url
#     }
#
#
# # Function to search tweets
# def search_tweets_by_query(query: str, username: str, project_name: str, project_symbol: str, start_date: datetime,
#                            limit: int = 20, min_retweets: int = 0, min_replies: int = 0):
#     url = "https://api.twitterapi.io/twitter/tweet/advanced_search"
#     headers = {"x-api-key": twitter_api_key}
#     since_str = start_date.strftime("%Y-%m-%d")
#
#     session = requests.Session()
#     retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
#     session.mount("https://", HTTPAdapter(max_retries=retries))
#
#     def fetch(query_string):
#         try:
#             params = {
#                 "query": f"{query_string} since:{since_str} min_retweets:{min_retweets} min_replies:{min_replies}",
#                 "queryType": "Latest",
#                 "limit": limit
#             }
#             st.write(f"Поиск по запросу: {params['query']}")
#             response = session.get(url, headers=headers, params=params, timeout=10)
#             response.raise_for_status()
#             data = response.json()
#
#             tweets = []
#             for key in ["tweets", "data", "results"]:
#                 if key in data:
#                     tweets = [t for t in data[key] if t.get("type") == "tweet"]
#                     break
#
#             result = []
#             for tweet in tweets:
#                 result.append({
#                     "id_str": str(tweet.get("id", tweet.get("id_str", str(uuid4())))),
#                     "text": tweet.get("text", ""),
#                     "created_at": tweet.get("created_at", ""),
#                     "user": {"screen_name": tweet.get("user", {}).get("screen_name", "unknown")},
#                     "public_metrics": {
#                         "retweet_count": tweet.get("retweetCount", 0),
#                         "reply_count": tweet.get("replyCount", 0),
#                         "view_count": tweet.get("viewCount", 0)
#                     }
#                 })
#             return result
#         except Exception as e:
#             st.error(f"Ошибка при получении твитов для запроса '{query_string}': {e}")
#             return []
#
#     # Запрос от официального аккаунта
#     from_query = f"from:{username}"
#     official_tweets = fetch(from_query)
#
#     # Запрос по ключевым словам
#     project_name_query = f'"{project_name}"'
#     project_symbol_query = f"${project_symbol}"
#     keyword_query = f"{project_name_query} OR {project_symbol_query}"
#     keyword_tweets = fetch(keyword_query)
#
#     # Объединение с удалением дубликатов по id_str
#     all_tweets = {tweet["id_str"]: tweet for tweet in official_tweets + keyword_tweets}
#     st.write(f"Объединено {len(all_tweets)} уникальных твитов.")
#     return list(all_tweets.values())
#
#
# # Function to search documents
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
#     try:
#         docs = vector_store.similarity_search(user_text, k=3)
#         return docs
#     except Exception as e:
#         st.error(f"Error during vector store search: {str(e)}")
#         return []
#
#
# # LLM and agent
# try:
#     llm = ChatGoogleGenerativeAI(
#         model='gemini-2.0-flash',
#         google_api_key=api_key
#     )
# except Exception as e:
#     st.error(f"Failed to initialize ChatGoogleGenerativeAI: {str(e)}")
#     st.stop()
#
# try:
#     agent = create_react_agent(
#         model=llm,
#         tools=[doc_ser]
#     )
# except Exception as e:
#     st.error(f"Failed to create agent: {str(e)}")
#     st.stop()
#
# # Streamlit interface
# st.title("Администрация Векторной Базы Данных")
#
# # Clear database
# st.subheader("Очистка базы данных")
# if st.button("Очистить векторную базу и JSON"):
#     try:
#         index = pc.Index(index_name)
#         index.delete(delete_all=True)
#         with open(json_path, "w", encoding="utf-8") as jf:
#             json.dump({}, jf, ensure_ascii=False)
#         st.success("Векторная база и JSON-файл очищены.")
#     except Exception as e:
#         st.error(f"Failed to clear database: {str(e)}")
#
# # Add tweets
# st.subheader("Добавить твиты в базу")
#
# coinmarketcap_url = st.text_input(
#     "Введите URL CoinMarketCap (например, https://coinmarketcap.com/currencies/legends-of-elumia/):")
# start_date = st.date_input("Выберите начальную дату:", value=datetime.now().date() - timedelta(days=7))
# limit = st.number_input("Количество твитов:", min_value=1, max_value=100, value=20)
# min_retweets = st.number_input("Минимальное количество ретвитов:", min_value=0, value=0)
# min_replies = st.number_input("Минимальное количество ответов:", min_value=0, value=0)
#
# if "tweets" not in st.session_state:
#     st.session_state["tweets"] = []
#
# if st.button("Загрузить твиты"):
#     if coinmarketcap_url:
#         project_info = parse_coinmarketcap_project(coinmarketcap_url)
#         if not project_info or project_info["twitter"] == "Not found":
#             st.error("Не удалось найти Twitter URL на странице CoinMarketCap.")
#         else:
#             twitter_url = project_info["twitter"]
#             project_name = project_info["name"]
#             project_symbol = project_info["symbol"]
#             official_username = urlparse(twitter_url).path.strip("/").split("/")[-1]
#             if not official_username:
#                 st.error("Не удалось извлечь username из Twitter URL.")
#                 st.stop()
#             st.info(
#                 f"Найден Twitter: {twitter_url} (Проект: {project_name}, Символ: {project_symbol}, Username: {official_username})")
#
#             # Очистка существующих данных для проекта
#             try:
#                 vector_store.delete(filter={"coinmarketcap_url": coinmarketcap_url})
#                 st.info("Существующие данные для этого проекта удалены из векторной базы.")
#             except Exception as e:
#                 st.error(f"Ошибка при очистке базы данных: {str(e)}")
#
#             # Поиск твитов
#             tweets = search_tweets_by_query(project_name, official_username, project_name, project_symbol, start_date,
#                                             limit, min_retweets, min_replies)
#             st.session_state["tweets"] = tweets  # Сохраняем твиты в сессии
#
#             if tweets:
#                 docs = []
#                 doc_ids = []
#                 new_id_data = {}
#                 existing_ids = set(id_data.values())
#
#                 for tweet in tweets:
#                     text = tweet.get("text", "")
#                     if not text:
#                         continue
#
#                     tweet_id = tweet.get("id_str", str(uuid4()))
#                     if tweet_id in existing_ids:
#                         st.write(f"Твит {tweet_id} уже существует, пропускаем.")
#                         continue
#
#                     author_username = tweet.get("user", {}).get("screen_name", "unknown")
#                     is_official = author_username.lower() == official_username.lower()
#                     author_type = "official" if is_official else "external"
#
#                     public_metrics = tweet.get("public_metrics", {})
#                     retweet_count = public_metrics.get("retweet_count", tweet.get("retweet_count", 0))
#                     reply_count = public_metrics.get("reply_count", tweet.get("reply_count", 0))
#                     view_count = public_metrics.get("view_count", tweet.get("view_count", 0))
#
#                     st.write(
#                         f"Tweet ID: {tweet_id}, Retweets: {retweet_count}, Replies: {reply_count}, Views: {view_count}")
#
#                     doc = Document(
#                         page_content=text,
#                         metadata={
#                             "source": f"twitter_{official_username}",
#                             "tweet_id": tweet_id,
#                             "created_at": tweet.get("created_at", ""),
#                             "retweet_count": retweet_count,
#                             "reply_count": reply_count,
#                             "view_count": view_count,
#                             "project_name": project_name,
#                             "project_symbol": project_symbol,
#                             "coinmarketcap_url": coinmarketcap_url,
#                             "author_username": author_username,
#                             "author_type": author_type
#                         }
#                     )
#                     docs.append(doc)
#
#                     new_id = str(uuid4())
#                     doc_ids.append(new_id)
#                     key_name = f"twitter_{official_username}_{tweet_id}"
#                     new_id_data[key_name] = new_id
#
#                 # Обновление JSON
#                 try:
#                     existing_data = id_data if id_data else {}
#                     existing_data.update(new_id_data)
#                     with open(json_path, "w", encoding="utf-8") as jf:
#                         json.dump(existing_data, jf, ensure_ascii=False)
#                 except Exception as e:
#                     st.error(f"Failed to update JSON file: {str(e)}")
#                     st.stop()
#
#                 # Добавление в векторную базу
#                 try:
#                     vector_store.add_documents(docs, ids=doc_ids)
#                     st.success(f"Добавлено {len(docs)} твитов, связанных с {project_name} (${project_symbol}).")
#                 except Exception as e:
#                     st.error(f"Failed to add documents to vector store: {str(e)}")
#             else:
#                 st.error("Не удалось получить твиты. Проверьте параметры запроса или Twitter API ключ.")
#                 st.write("Проверьте отладочную информацию выше для деталей.")
#     else:
#         st.error("Пожалуйста, введите URL CoinMarketCap.")
#
# # Analytics section
# # Analytics section
# st.subheader("Аналитика твитов")
# if st.button("Показать аналитику"):
#     if not st.session_state.get("tweets"):
#         st.error("Сначала загрузите твиты, нажав 'Загрузить твиты'.")
#     else:
#         # Fetch followers and following count
#         twitter_url = st.session_state["tweets"][0]["user"]["screen_name"] if st.session_state["tweets"] else None
#         if twitter_url:
#             try:
#                 url = f"https://api.twitterapi.io/twitter/user/lookup?usernames={twitter_url}"
#                 headers = {"x-api-key": twitter_api_key}
#                 response = requests.get(url, headers=headers, timeout=10)
#                 response.raise_for_status()
#                 data = response.json()
#                 user_data = data.get("data", [{}])[0]
#                 followers_count = user_data.get("public_metrics", {}).get("followers_count", 0)
#                 following_count = user_data.get("public_metrics", {}).get("following_count", 0)
#                 st.write(f"Followers: {followers_count}")
#                 st.write(f"Following: {following_count}")
#             except Exception as e:
#                 st.error(f"Ошибка при получении данных о пользователе: {e}")
#
#         # Prepare data for plotting
#         tweets_df = pd.DataFrame(st.session_state["tweets"])
#         # Ensure the 'user.screen_name' column exists
#         if "user.screen_name" not in tweets_df.columns:
#             tweets_df = tweets_df.join(pd.json_normalize(tweets_df["user"]).add_prefix("user."))
#
#         tweets_df["created_at"] = pd.to_datetime(tweets_df["created_at"], errors="coerce")
#         tweets_df["date"] = tweets_df["created_at"].dt.date
#
#         # Official tweets
#         official_tweets_df = tweets_df[tweets_df["user.screen_name"].str.lower() == twitter_url.lower()]
#         official_metrics = official_tweets_df.groupby("date").agg({
#             "view_count": "sum",
#             "retweet_count": "sum",
#             "reply_count": "sum"
#         }).reset_index()
#
#         # Other tweets
#         other_tweets_df = tweets_df[tweets_df["user.screen_name"].str.lower() != twitter_url.lower()]
#         other_metrics = other_tweets_df.groupby("date").agg({
#             "view_count": "sum",
#             "retweet_count": "sum",
#             "reply_count": "sum"
#         }).reset_index()
#
#         # Plotting
#         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
#
#         # Official tweets plot
#         ax1.plot(official_metrics["date"], official_metrics["view_count"], label="Views", marker="o")
#         ax1.plot(official_metrics["date"], official_metrics["retweet_count"], label="Retweets", marker="o")
#         ax1.plot(official_metrics["date"], official_metrics["reply_count"], label="Replies", marker="o")
#         ax1.set_title("Аналитика твитов официального аккаунта")
#         ax1.set_ylabel("Количество")
#         ax1.legend()
#         ax1.grid(True)
#
#         # Other tweets plot
#         ax2.plot(other_metrics["date"], other_metrics["view_count"], label="Views", marker="o")
#         ax2.plot(other_metrics["date"], other_metrics["retweet_count"], label="Retweets", marker="o")
#         ax2.plot(other_metrics["date"], other_metrics["reply_count"], label="Replies", marker="o")
#         ax2.set_title("Аналитика остальных твитов")
#         ax2.set_xlabel("Дата")
#         ax2.set_ylabel("Количество")
#         ax2.legend()
#         ax2.grid(True)
#
#         plt.xticks(rotation=45)
#         st.pyplot(fig)
# # Chat with search
# st.subheader("Чат с поиском по векторной базе")
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
#     try:
#         response = agent.invoke(st.session_state["data"])
#         st.session_state['data'] = response
#     except Exception as e:
#         st.error(f"Failed to invoke agent: {str(e)}")
#
# for message in st.session_state['data']['messages']:
#     if isinstance(message, HumanMessage):
#         role = "user"
#     elif isinstance(message, AIMessage):
#         role = "bot"
#     else:
#         continue
#
#     content = message.content.strip() if isinstance(message.content, str) else ""
#     if not content:
#         continue  # пропускаем пустые сообщения
#
#     with st.chat_message(role):
#         st.markdown(content)

###
import os
import json
import dotenv
import streamlit as st
from uuid import uuid4
from datetime import datetime, timedelta, timezone  # Add timezone import

from urllib.parse import urlparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import nest_asyncio
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import pandas as pd

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langgraph.prebuilt import create_react_agent

# Apply nest_asyncio to handle async issues in Streamlit
nest_asyncio.apply()

# Load environment variables
dotenv.load_dotenv()

# API keys
api_key = os.getenv("GEMINI_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")
twitter_api_key = os.getenv("TWITTER_API_KEY") or "c769bb7d2f3742828d13bad15b7262d1"

# Validate API keys
if not api_key:
    st.error("GEMINI_API_KEY is not set. Please add it to the .env file or Streamlit Cloud Secrets.")
    st.stop()
if not pinecone_key:
    st.error("PINECONE_API_KEY is not set. Please add it to the .env file or Streamlit Cloud Secrets.")
    st.stop()
if not twitter_api_key:
    st.warning("TWITTER_API_KEY is not set, using default key.")

# Debug print to verify API keys
st.write(f"GEMINI_API_KEY: {api_key[:5]}... (hidden for security)")
st.write(f"PINECONE_API_KEY: {pinecone_key[:5]}... (hidden for security)")
st.write(f"TWITTER_API_KEY: {twitter_api_key[:5]}... (hidden for security)")

# Initialize embeddings and Pinecone
try:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )
except Exception as e:
    st.error(f"Failed to initialize GoogleGenerativeAIEmbeddings: {str(e)}")
    st.stop()

try:
    pc = Pinecone(api_key=pinecone_key)
except Exception as e:
    st.error(f"Failed to initialize Pinecone: {str(e)}")
    st.stop()

index_name = "task1"

try:
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
except Exception as e:
    st.error(f"Failed to create Pinecone index: {str(e)}")
    st.stop()

index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# JSON for storing IDs
json_path = "data_ai.json"
if os.path.exists(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            id_data = json.load(f)
    except Exception as e:
        st.error(f"Failed to read JSON file {json_path}: {str(e)}")
        id_data = {}
else:
    id_data = {}

# Function to parse CoinMarketCap project page
def parse_coinmarketcap_project(url):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        st.error(f"Ошибка при запросе CoinMarketCap {url}: {str(e)}")
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
        if "twitter.com" in href or "x.com" in href:
            twitter_link = href.strip()
            break

    if twitter_link and twitter_link.startswith("//"):
        twitter_link = "https:" + twitter_link
    elif twitter_link and "x.com" in twitter_link:
        twitter_link = twitter_link.replace("x.com", "twitter.com")

    st.write(f"Parsed CoinMarketCap: name={name}, symbol={symbol}, twitter={twitter_link}")
    return {
        "name": name,
        "symbol": symbol,
        "twitter": twitter_link or "Not found",
        "url": url
    }

# Function to search tweets

from datetime import datetime, timedelta, timezone
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import streamlit as st
from uuid import uuid4
import pandas as pd


def search_tweets_by_query(query: str, username: str, project_name: str, project_symbol: str, start_date: datetime,
                           limit: int = 20, min_retweets: int = 0, min_replies: int = 0):
    url = "https://api.twitterapi.io/twitter/tweet/advanced_search"
    headers = {"x-api-key": twitter_api_key}
    since_str = start_date.strftime("%Y-%m-%d")

    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))

    def fetch(query_string):
        try:
            params = {
                "query": f"{query_string} since:{since_str} min_retweets:{min_retweets} min_replies:{min_replies}",
                "queryType": "Latest",
                "limit": limit
            }
            st.write(f"Поиск по запросу: {params['query']}")
            response = session.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Debug: Log the full API response
            st.write(f"Raw API response: {data}")

            tweets = []
            for key in ["tweets", "data", "results"]:
                if key in data:
                    tweets = [t for t in data[key] if t.get("type") == "tweet"]
                    break

            result = []
            processed_keys = set()
            for tweet in tweets:
                tweet_id = str(tweet.get("id", tweet.get("id_str", str(uuid4()))))

                # Try to get username from author.userName (twitterapi.io), user.username (Twitter API v2), or user.screen_name (v1.1)
                author = tweet.get("author", {})
                user = tweet.get("user", {})
                screen_name = author.get("userName", user.get("username", user.get("screen_name", None)))
                if not screen_name:
                    st.warning(f"Твит {tweet_id} не содержит userName, username, или screen_name, пропускаем.")
                    st.write(f"Author object for tweet {tweet_id}: {author}")
                    st.write(f"User object for tweet {tweet_id}: {user}")
                    continue

                unique_key = f"{tweet_id}_{screen_name}"
                if unique_key in processed_keys:
                    st.write(f"Твит {tweet_id} от {screen_name} уже обработан, пропускаем.")
                    continue
                processed_keys.add(unique_key)

                # Use createdAt (twitterapi.io) or created_at (Twitter API)
                created_at = tweet.get("createdAt", tweet.get("created_at", None))
                try:
                    created_at = pd.to_datetime(created_at, utc=True, errors="raise").isoformat()
                except (ValueError, TypeError) as e:
                    st.warning(f"Некорректный формат created_at для твита {tweet_id}: {created_at}. Пропускаем.")
                    continue

                # Extract public metrics
                public_metrics = tweet.get("public_metrics", {})
                retweet_count = public_metrics.get("retweet_count", tweet.get("retweetCount", 0))
                reply_count = public_metrics.get("reply_count", tweet.get("replyCount", 0))
                view_count = public_metrics.get("view_count", tweet.get("viewCount", 0))

                result.append({
                    "id_str": tweet_id,
                    "text": tweet.get("text", ""),
                    "created_at": created_at,
                    "user": {"screen_name": screen_name},  # Store as screen_name for backward compatibility
                    "public_metrics": {
                        "retweet_count": retweet_count,
                        "reply_count": reply_count,
                        "view_count": view_count
                    }
                })
            st.write(
                f"Fetched {len(result)} tweets with sample created_at: {result[0]['created_at'] if result else 'None'}")
            return result
        except Exception as e:
            st.error(f"Ошибка при получении твитов для запроса '{query_string}': {e}")
            st.write(f"Request params: {params}")
            return []

    # Запрос от официального аккаунта
    from_query = f"from:{username}"
    official_tweets = fetch(from_query)

    # Запрос по ключевым словам
    project_name_query = f'"{project_name}"'
    project_symbol_query = f"${project_symbol}"
    keyword_query = f"{project_name_query} OR {project_symbol_query}"
    keyword_tweets = fetch(keyword_query)

    # Объединение с удалением дубликатов по id_str и screen_name
    all_tweets = {}
    for tweet in official_tweets + keyword_tweets:
        tweet_id = tweet["id_str"]
        screen_name = tweet["user"]["screen_name"]
        unique_key = f"{tweet_id}_{screen_name}"
        all_tweets[unique_key] = tweet
    st.write(f"Объединено {len(all_tweets)} уникальных твитов.")
    return list(all_tweets.values())

# Function to search documents
def doc_ser(user_text: str):
    """
    Search for documents in the vector database based on user query.

    Args:
        user_text (str): The user's query text to search for similar documents.

    Returns:
        List[Document]: A list of the top 3 most similar documents from the vector database.
    """
    try:
        docs = vector_store.similarity_search(user_text, k=3)
        return docs
    except Exception as e:
        st.error(f"Error during vector store search: {str(e)}")
        return []

# LLM and agent
try:
    llm = ChatGoogleGenerativeAI(
        model='gemini-2.0-flash',
        google_api_key=api_key
    )
except Exception as e:
    st.error(f"Failed to initialize ChatGoogleGenerativeAI: {str(e)}")
    st.stop()

try:
    agent = create_react_agent(
        model=llm,
        tools=[doc_ser]
    )
except Exception as e:
    st.error(f"Failed to create agent: {str(e)}")
    st.stop()

# Streamlit interface
st.title("Администрация Векторной Базы Данных")

# Clear database
st.subheader("Очистка базы данных")
if st.button("Очистить векторную базу и JSON"):
    try:
        index = pc.Index(index_name)
        index.delete(delete_all=True)
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump({}, jf, ensure_ascii=False)
        st.success("Векторная база и JSON-файл очищены.")
    except Exception as e:
        st.error(f"Failed to clear database: {str(e)}")

# Add tweets
st.subheader("Добавить твиты в базу")

coinmarketcap_url = st.text_input(
    "Введите URL CoinMarketCap (например, https://coinmarketcap.com/currencies/legends-of-elumia/):")
start_date = st.date_input("Выберите начальную дату:", value=datetime.now().date() - timedelta(days=7))
limit = st.number_input("Количество твитов:", min_value=1, max_value=100, value=20)
min_retweets = st.number_input("Минимальное количество ретвитов:", min_value=0, value=0)
min_replies = st.number_input("Минимальное количество ответов:", min_value=0, value=0)

if "tweets" not in st.session_state:
    st.session_state["tweets"] = []
#################33

def normalize_url(url: str) -> str:
    """Убирает завершающий слэш"""
    return url.strip().rstrip("/")

def already_exists(tweet_id: str) -> bool:
    """Проверяет, есть ли вектор с таким tweet_id в Pinecone"""
    try:
        response = index.fetch(ids=[tweet_id])
        exists = tweet_id in (response.vectors if hasattr(response, 'vectors') else response.get("vectors", {}))
        if exists:
            st.write(f"Pinecone: tweet_id {tweet_id} уже существует")
        return exists
    except Exception as e:
        st.error(f"Ошибка при проверке tweet_id {tweet_id} в Pinecone: {str(e)}")
        return False


# ... (предыдущий код остается без изменений до секции загрузки твитов)

# Функция для проверки существующих пространств имен
def check_namespaces(index):
    try:
        stats = index.describe_index_stats()
        namespaces = stats.get("namespaces", {})
        st.write(f"Доступные пространства имен: {list(namespaces.keys())}")
        return namespaces
    except Exception as e:
        st.error(f"Ошибка при получении статистики индекса: {str(e)}")
        return {}


if st.button("Загрузить твиты"):
    if coinmarketcap_url:
        # Normalize coinmarketcap_url
        normalized_coinmarketcap_url = normalize_url(coinmarketcap_url)
        project_info = parse_coinmarketcap_project(coinmarketcap_url)
        if not project_info or project_info["twitter"] == "Not found":
            st.error("Не удалось найти Twitter URL на странице CoinMarketCap.")
        else:
            twitter_url = project_info["twitter"]
            project_name = project_info["name"]
            project_symbol = project_info["symbol"]
            official_username = urlparse(twitter_url).path.strip("/").split("/")[-1]
            if not official_username or official_username == "":
                st.error("Не удалось извлечь валидный username из Twitter URL.")
                st.stop()
            st.info(
                f"Найден Twitter: {twitter_url} (Проект: {project_name}, Символ: {project_symbol}, Username: {official_username})")

            # Clear session state
            st.session_state["tweets"] = []
            st.write("Очищена сессия st.session_state['tweets'].")

            # Check namespaces before deletion
            namespaces = check_namespaces(index)
            namespace = ""  # Используем дефолтное пространство имен, если не указано иное

            # Clear existing data in vector database
            try:
                if namespace in namespaces and namespaces[namespace].get("vector_count", 0) > 0:
                    vector_store.delete(
                        filter={
                            "$or": [
                                {"coinmarketcap_url": normalized_coinmarketcap_url},
                                {"coinmarketcap_url": normalized_coinmarketcap_url + "/"}
                            ]
                        },
                        namespace=namespace
                    )
                    st.info(f"Существующие данные для проекта удалены из векторной базы (namespace: {namespace}).")
                else:
                    st.info(f"Пространство имен '{namespace}' пусто или не существует, пропускаем удаление.")

                # Verify deletion
                verify_docs = vector_store.similarity_search(
                    "", k=1000,
                    filter={
                        "$or": [
                            {"coinmarketcap_url": normalized_coinmarketcap_url},
                            {"coinmarketcap_url": normalized_coinmarketcap_url + "/"}
                        ]
                    },
                    namespace=namespace
                )
                if verify_docs:
                    st.warning(
                        f"Внимание: найдено {len(verify_docs)} документов после очистки в namespace '{namespace}'.")
                    for doc in verify_docs:
                        st.write(
                            f"Остаток: tweet_id={doc.metadata['tweet_id']}, doc_id={doc.metadata.get('doc_id', 'N/A')}")
                else:
                    st.info(f"Пространство имен '{namespace}' успешно очищено или было пустым.")
            except Exception as e:
                st.error(f"Ошибка при очистке векторной базы: {str(e)}")
                if "Namespace not found" in str(e):
                    st.warning(f"Пространство имен '{namespace}' не найдено, продолжаем без удаления.")
                else:
                    st.stop()

            # Clear JSON entries for this project
            try:
                filtered_id_data = {
                    k: v for k, v in id_data.items()
                    if not k.startswith(f"twitter_{official_username}_")
                }
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(filtered_id_data, jf, ensure_ascii=False)
                id_data.clear()
                id_data.update(filtered_id_data)
                st.info("JSON данные для этого проекта очищены.")
            except Exception as e:
                st.error(f"Ошибка при очистке JSON: {str(e)}")
                st.stop()

            # Search tweets
            tweets = search_tweets_by_query(project_name, official_username, project_name, project_symbol, start_date,
                                            limit, min_retweets, min_replies)
            st.session_state["tweets"] = tweets
            st.write(f"Найдено {len(tweets)} уникальных твитов.")
            st.write(f"Содержимое st.session_state['tweets']: {tweets}")

            if tweets:
                uploaded = 0
                skipped = 0
                new_id_data = {}
                existing_ids = set()

                for tweet in tweets:
                    text = tweet.get("text", "")
                    if not text:
                        st.warning(f"Твит {tweet.get('id_str', 'unknown')} пропущен: пустой текст.")
                        continue

                    tweet_id = tweet.get("id_str", str(uuid4()))
                    if already_exists(tweet_id):
                        st.write(f"⏩ Пропущено (дубликат в Pinecone): tweet_id={tweet_id}")
                        skipped += 1
                        continue
                    if tweet_id in existing_ids:
                        st.write(f"⏩ Пропущено (дубликат в текущей сессии): tweet_id={tweet_id}")
                        skipped += 1
                        continue
                    existing_ids.add(tweet_id)

                    author_username = tweet.get("user", {}).get("screen_name", None)
                    if not author_username:
                        st.warning(f"Твит {tweet_id} не содержит screen_name, пропускаем.")
                        st.write(f"Проблемный твит: {tweet}")
                        continue

                    created_at = tweet.get("created_at")
                    if not created_at:
                        st.warning(f"Твит {tweet_id} не содержит created_at, пропускаем.")
                        st.write(f"Проблемный твит: {tweet}")
                        continue
                    try:
                        created_at = pd.to_datetime(created_at, utc=True, errors="raise").isoformat()
                    except (ValueError, TypeError) as e:
                        st.warning(f"Некорректный формат created_at для твита {tweet_id}: {created_at}. Пропускаем.")
                        continue

                    is_official = author_username.lower() == official_username.lower()
                    author_type = "official" if is_official else "external"

                    public_metrics = tweet.get("public_metrics", {})
                    retweet_count = public_metrics.get("retweet_count", tweet.get("retweetCount", 0))
                    reply_count = public_metrics.get("reply_count", tweet.get("replyCount", 0))
                    view_count = public_metrics.get("view_count", tweet.get("viewCount", 0))

                    st.write(
                        f"Добавляется твит: ID={tweet_id}, Author={author_username}, Type={author_type}, Created At={created_at}")

                    # Prepare document
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": f"twitter_{official_username}",
                            "tweet_id": tweet_id,
                            "created_at": created_at,
                            "retweet_count": retweet_count,
                            "reply_count": reply_count,
                            "view_count": view_count,
                            "project_name": project_name,
                            "project_symbol": project_symbol,
                            "coinmarketcap_url": normalized_coinmarketcap_url,
                            "author_username": author_username,
                            "author_type": author_type,
                            "doc_id": tweet_id
                        }
                    )

                    # Add to vector store
                    try:
                        vector_store.add_documents([doc], ids=[tweet_id], namespace=namespace)
                        st.success(f"✅ Загружено: tweet_id={tweet_id}")
                        uploaded += 1
                    except Exception as e:
                        st.error(f"Ошибка при добавлении твита {tweet_id}: {str(e)}")
                        continue

                    # Update JSON
                    key_name = f"twitter_{official_username}_{tweet_id}_{author_username}"
                    new_id_data[key_name] = tweet_id

                # Save JSON
                try:
                    existing_data = id_data if id_data else {}
                    existing_data.update(new_id_data)
                    with open(json_path, "w", encoding="utf-8") as jf:
                        json.dump(existing_data, jf, ensure_ascii=False)
                    st.write("JSON файл успешно обновлен.")
                except Exception as e:
                    st.error(f"Failed to update JSON file: {str(e)}")
                    st.stop()

                st.write(f"📥 Загружено: {uploaded}")
                st.write(f"🚫 Пропущено (дубликаты): {skipped}")

                # Verify vector database
                try:
                    verify_docs = vector_store.similarity_search(
                        "", k=1000,
                        filter={
                            "$or": [
                                {"coinmarketcap_url": normalized_coinmarketcap_url},
                                {"coinmarketcap_url": normalized_coinmarketcap_url + "/"}
                            ]
                        },
                        namespace=namespace
                    )
                    tweet_ids_in_db = [doc.metadata["tweet_id"] for doc in verify_docs]
                    duplicate_ids = [tid for tid in set(tweet_ids_in_db) if tweet_ids_in_db.count(tid) > 1]
                    if duplicate_ids:
                        st.warning(f"Обнаружены дубликаты в векторной базе: {duplicate_ids}")
                        for doc in verify_docs:
                            if doc.metadata["tweet_id"] in duplicate_ids:
                                st.write(
                                    f"Дубликат: tweet_id={doc.metadata['tweet_id']}, author_username={doc.metadata['author_username']}, doc_id={doc.metadata.get('doc_id', 'N/A')}")
                    else:
                        st.info("Дубликаты в векторной базе не найдены.")
                except Exception as e:
                    st.error(f"Ошибка при проверке векторной базы: {str(e)}")
            else:
                st.error("Не удалось получить твиты. Проверьте параметры запроса или Twitter API ключ.")
                st.write("Проверьте отладочную информацию выше для деталей.")
    else:
        st.error("Пожалуйста, введите URL CoinMarketCap.")

# if st.button("Загрузить твиты"):
#     if coinmarketcap_url:
#         # Normalize coinmarketcap_url
#         normalized_coinmarketcap_url = normalize_url(coinmarketcap_url)
#         project_info = parse_coinmarketcap_project(coinmarketcap_url)
#         if not project_info or project_info["twitter"] == "Not found":
#             st.error("Не удалось найти Twitter URL на странице CoinMarketCap.")
#         else:
#             twitter_url = project_info["twitter"]
#             project_name = project_info["name"]
#             project_symbol = project_info["symbol"]
#             official_username = urlparse(twitter_url).path.strip("/").split("/")[-1]
#             if not official_username or official_username == "":
#                 st.error("Не удалось извлечь валидный username из Twitter URL.")
#                 st.stop()
#             st.info(f"Найден Twitter: {twitter_url} (Проект: {project_name}, Символ: {project_symbol}, Username: {official_username})")
#
#             # Clear session state
#             st.session_state["tweets"] = []
#             st.write("Очищена сессия st.session_state['tweets'].")
#
#             # Clear existing data in vector database
#             try:
#                 vector_store.delete(filter={
#                     "$or": [
#                         {"coinmarketcap_url": normalized_coinmarketcap_url},
#                         {"coinmarketcap_url": normalized_coinmarketcap_url + "/"}
#                     ]
#                 })
#                 st.info("Существующие данные для этого проекта удалены из векторной базы.")
#                 # Verify deletion
#                 verify_docs = vector_store.similarity_search("", k=1000, filter={
#                     "$or": [
#                         {"coinmarketcap_url": normalized_coinmarketcap_url},
#                         {"coinmarketcap_url": normalized_coinmarketcap_url + "/"}
#                     ]
#                 })
#                 if verify_docs:
#                     st.warning(f"Внимание: найдено {len(verify_docs)} документов после очистки.")
#                     for doc in verify_docs:
#                         st.write(f"Остаток: tweet_id={doc.metadata['tweet_id']}, doc_id={doc.metadata.get('doc_id', 'N/A')}")
#             except Exception as e:
#                 st.error(f"Ошибка при очистке векторной базы: {str(e)}")
#                 st.stop()
#
#             # Clear JSON entries for this project
#             try:
#                 filtered_id_data = {
#                     k: v for k, v in id_data.items()
#                     if not k.startswith(f"twitter_{official_username}_")
#                 }
#                 with open(json_path, "w", encoding="utf-8") as jf:
#                     json.dump(filtered_id_data, jf, ensure_ascii=False)
#                 id_data.clear()
#                 id_data.update(filtered_id_data)
#                 st.info("JSON данные для этого проекта очищены.")
#             except Exception as e:
#                 st.error(f"Ошибка при очистке JSON: {str(e)}")
#                 st.stop()
#
#             # Search tweets
#             tweets = search_tweets_by_query(project_name, official_username, project_name, project_symbol, start_date, limit, min_retweets, min_replies)
#             st.session_state["tweets"] = tweets
#             st.write(f"Найдено {len(tweets)} уникальных твитов.")
#             st.write(f"Содержимое st.session_state['tweets']: {tweets}")
#
#             if tweets:
#                 uploaded = 0
#                 skipped = 0
#                 new_id_data = {}
#                 existing_ids = set()
#
#                 for tweet in tweets:
#                     text = tweet.get("text", "")
#                     if not text:
#                         st.warning(f"Твит {tweet.get('id_str', 'unknown')} пропущен: пустой текст.")
#                         continue
#
#                     tweet_id = tweet.get("id_str", str(uuid4()))
#                     if already_exists(tweet_id):
#                         st.write(f"⏩ Пропущено (дубликат в Pinecone): tweet_id={tweet_id}")
#                         skipped += 1
#                         continue
#                     if tweet_id in existing_ids:
#                         st.write(f"⏩ Пропущено (дубликат в текущей сессии): tweet_id={tweet_id}")
#                         skipped += 1
#                         continue
#                     existing_ids.add(tweet_id)
#
#                     author_username = tweet.get("user", {}).get("screen_name", None)
#                     if not author_username:
#                         st.warning(f"Твит {tweet_id} не содержит screen_name, пропускаем.")
#                         st.write(f"Проблемный твит: {tweet}")
#                         continue
#
#                     created_at = tweet.get("created_at")
#                     if not created_at:
#                         st.warning(f"Твит {tweet_id} не содержит created_at, пропускаем.")
#                         st.write(f"Проблемный твит: {tweet}")
#                         continue
#                     try:
#                         created_at = pd.to_datetime(created_at, utc=True, errors="raise").isoformat()
#                     except (ValueError, TypeError) as e:
#                         st.warning(f"Некорректный формат created_at для твита {tweet_id}: {created_at}. Пропускаем.")
#                         continue
#
#                     is_official = author_username.lower() == official_username.lower()
#                     author_type = "official" if is_official else "external"
#
#                     public_metrics = tweet.get("public_metrics", {})
#                     retweet_count = public_metrics.get("retweet_count", tweet.get("retweetCount", 0))
#                     reply_count = public_metrics.get("reply_count", tweet.get("replyCount", 0))
#                     view_count = public_metrics.get("view_count", tweet.get("viewCount", 0))
#
#                     st.write(f"Добавляется твит: ID={tweet_id}, Author={author_username}, Type={author_type}, Created At={created_at}")
#
#                     # Prepare document
#                     doc = Document(
#                         page_content=text,
#                         metadata={
#                             "source": f"twitter_{official_username}",
#                             "tweet_id": tweet_id,
#                             "created_at": created_at,
#                             "retweet_count": retweet_count,
#                             "reply_count": reply_count,
#                             "view_count": view_count,
#                             "project_name": project_name,
#                             "project_symbol": project_symbol,
#                             "coinmarketcap_url": normalized_coinmarketcap_url,
#                             "author_username": author_username,
#                             "author_type": author_type,
#                             "doc_id": tweet_id
#                         }
#                     )
#
#                     # Add to vector store
#                     try:
#                         vector_store.add_documents([doc], ids=[tweet_id])
#                         st.success(f"✅ Загружено: tweet_id={tweet_id}")
#                         uploaded += 1
#                     except Exception as e:
#                         st.error(f"Ошибка при добавлении твита {tweet_id}: {str(e)}")
#                         continue
#
#                     # Update JSON
#                     key_name = f"twitter_{official_username}_{tweet_id}_{author_username}"
#                     new_id_data[key_name] = tweet_id
#
#                 # Save JSON
#                 try:
#                     existing_data = id_data if id_data else {}
#                     existing_data.update(new_id_data)
#                     with open(json_path, "w", encoding="utf-8") as jf:
#                         json.dump(existing_data, jf, ensure_ascii=False)
#                     st.write("JSON файл успешно обновлен.")
#                 except Exception as e:
#                     st.error(f"Failed to update JSON file: {str(e)}")
#                     st.stop()
#
#                 st.write(f"📥 Загружено: {uploaded}")
#                 st.write(f"🚫 Пропущено (дубликаты): {skipped}")
#
#                 # Verify vector database
#                 try:
#                     verify_docs = vector_store.similarity_search("", k=1000, filter={
#                         "$or": [
#                             {"coinmarketcap_url": normalized_coinmarketcap_url},
#                             {"coinmarketcap_url": normalized_coinmarketcap_url + "/"}
#                         ]
#                     })
#                     tweet_ids_in_db = [doc.metadata["tweet_id"] for doc in verify_docs]
#                     duplicate_ids = [tid for tid in set(tweet_ids_in_db) if tweet_ids_in_db.count(tid) > 1]
#                     if duplicate_ids:
#                         st.warning(f"Обнаружены дубликаты в векторной базе: {duplicate_ids}")
#                         for doc in verify_docs:
#                             if doc.metadata["tweet_id"] in duplicate_ids:
#                                 st.write(f"Дубликат: tweet_id={doc.metadata['tweet_id']}, author_username={doc.metadata['author_username']}, doc_id={doc.metadata.get('doc_id', 'N/A')}")
#                     else:
#                         st.info("Дубликаты в векторной базе не найдены.")
#                 except Exception as e:
#                     st.error(f"Ошибка при проверке векторной базы: {str(e)}")
#             else:
#                 st.error("Не удалось получить твиты. Проверьте параметры запроса или Twitter API ключ.")
#                 st.write("Проверьте отладочную информацию выше для деталей.")
#     else:
#         st.error("Пожалуйста, введите URL CoinMarketCap.")

##################
def normalize_url(url):
    """Normalize URL by removing trailing slash."""
    return url.rstrip("/")


if st.button("Проверить и удалить дубликаты"):
    try:
        st.info("Поиск дубликатов...")

        # Normalize URL
        normalized_url = normalize_url(coinmarketcap_url)
        filter_url_variants = [
            normalized_url,
            normalized_url + "/"
        ]

        # Initialize
        dimension = index.describe_index_stats()['dimension']
        dummy_vector = [0.0] * dimension
        top_k = 1000
        all_docs = []

        # Fetch documents for both URL variants
        for url in filter_url_variants:
            query_filter = {"coinmarketcap_url": url}
            last_id = None
            while True:
                q_filter = query_filter.copy()
                if last_id:
                    q_filter["$and"] = [{"doc_id": {"$gt": last_id}}]
                query_result = index.query(
                    vector=dummy_vector,
                    top_k=top_k,
                    filter=q_filter,
                    include_metadata=True
                )
                matches = query_result.get("matches", [])
                st.write(f"Найдено {len(matches)} документов для URL: {url}")
                for match in matches:
                    metadata = match.get("metadata", {})
                    doc_id = match.get("id")
                    tweet_id = str(metadata.get("tweet_id")) if metadata.get("tweet_id") else None
                    if tweet_id:
                        all_docs.append(Document(
                            page_content="",
                            metadata={**metadata, "doc_id": doc_id, "tweet_id": tweet_id}
                        ))
                if len(matches) < top_k:
                    break
                last_id = matches[-1]["id"]

        st.write(f"Всего найдено документов: {len(all_docs)}")

        # Group by tweet_id
        tweet_id_to_docs = {}
        for doc in all_docs:
            tweet_id = doc.metadata["tweet_id"]
            tweet_id_to_docs.setdefault(tweet_id, []).append(doc)

        duplicates_found = False
        ids_to_delete = []

        for tweet_id, docs in tweet_id_to_docs.items():
            if len(docs) > 1:
                duplicates_found = True
                st.warning(f"Найден дубликат: tweet_id={tweet_id}, всего версий: {len(docs)}")
                # Keep the "best" document
                valid_doc = next((
                    d for d in docs
                    if d.metadata.get("author_username") != "unknown"
                    and d.metadata.get("created_at")
                    and d.metadata.get("author_type") == "official"
                ), docs[0])

                for doc in docs:
                    if doc != valid_doc:
                        doc_id = doc.metadata.get("doc_id")
                        ids_to_delete.append(doc_id)
                        st.write(f"Будет удален: doc_id={doc_id}, author={doc.metadata.get('author_username')}, created_at={doc.metadata.get('created_at')}")

        if ids_to_delete:
            try:
                index.delete(ids=ids_to_delete)
                st.success(f"Удалено {len(ids_to_delete)} дубликатов.")
            except Exception as e:
                st.error(f"Ошибка при удалении: {str(e)}")
        elif duplicates_found:
            st.warning("Обнаружены дубликаты, но не удалось определить ID для удаления.")
        else:
            st.success("Дубликаты не найдены.")

    except Exception as e:
        st.error(f"Ошибка при проверке векторной базы: {str(e)}")

if st.button("Удалить конкретный документ"):
    doc_id_to_delete = st.text_input("Введите ID документа для удаления (например, dc82727d-c885-4ca8-a4d2-789aa7037841):")
    if doc_id_to_delete:
        try:
            index.delete(ids=[doc_id_to_delete])
            st.success(f"Документ с ID {doc_id_to_delete} удален.")
        except Exception as e:
            st.error(f"Ошибка при удалении документа {doc_id_to_delete}: {str(e)}")


def get_user_info(username):
    st.write(f"Проверка профиля пользователя {username}...")
    url = f"https://api.twitterapi.io/twitter/user/{username}"
    headers = {"x-api-key": twitter_api_key}
    response = requests.get(url, headers=headers, timeout=10)
    st.write(f"Код ответа: {response.status_code}")
    st.write(f"Ответ: {response.text}")
    if response.status_code == 200:
        data = response.json()
        user = data.get("user") or data.get("data") or {}
        followers = user.get("followers") or user.get("public_metrics", {}).get("followers_count", 0)
        following = user.get("following") or user.get("public_metrics", {}).get("following_count", 0)
        return followers, following
    else:
        st.warning(f"Ошибка при получении профиля: {response.status_code} {response.text}")
        return 0, 0


st.subheader("Аналитика твитов")

try:
    # Get all projects from vector database
    all_docs = vector_store.similarity_search("", k=1000)
    unique_projects = set(
        doc.metadata.get("coinmarketcap_url") for doc in all_docs if doc.metadata.get("coinmarketcap_url")
    )
    if not unique_projects:
        st.error("Нет данных в векторной базе для анализа. Загрузите твиты сначала.")
    else:
        selected_project = st.selectbox("Выберите проект для анализа:", options=["Все проекты"] + list(unique_projects))

        # Filter by project
        filter_dict = {}
        if selected_project != "Все проекты":
            normalized_url = normalize_url(selected_project)
            filter_dict = {
                "$or": [
                    {"coinmarketcap_url": normalized_url},
                    {"coinmarketcap_url": normalized_url + "/"}
                ]
            }

        # Get relevant documents
        docs = vector_store.similarity_search("", k=1000, filter=filter_dict)
        if not docs:
            st.error("Нет данных для выбранного проекта в векторной базе.")
        else:
            # Convert to DataFrame
            data = []
            for doc in docs:
                metadata = doc.metadata
                data.append({
                    "text": doc.page_content,
                    "created_at": metadata.get("created_at", ""),
                    "view_count": metadata.get("view_count", 0),
                    "retweet_count": metadata.get("retweet_count", 0),
                    "reply_count": metadata.get("reply_count", 0),
                    "author_username": metadata.get("author_username", "unknown"),
                    "author_type": metadata.get("author_type", "external"),
                    "project_name": metadata.get("project_name", ""),
                    "project_symbol": metadata.get("project_symbol", ""),
                    "tweet_id": metadata.get("tweet_id", ""),
                    "doc_id": metadata.get("doc_id", "")
                })
            tweets_df = pd.DataFrame(data)

            # Debug: Check tweet data
            st.write("Все твиты в базе:",
                     tweets_df[["tweet_id", "author_username", "author_type", "created_at"]].to_dict(orient="records"))

            # Convert created_at to datetime
            tweets_df["created_at"] = pd.to_datetime(tweets_df["created_at"], errors="coerce", utc=True)
            tweets_df = tweets_df.dropna(subset=["created_at"])
            if tweets_df.empty:
                st.error("Нет данных с валидными датами для анализа. Проверьте формат created_at в векторной базе.")
            else:
                tweets_df["date"] = tweets_df["created_at"].dt.date

                # Split tweets by author_type
                official_tweets_df = tweets_df[tweets_df["author_type"] == "official"]
                other_tweets_df = tweets_df[tweets_df["author_type"] != "official"]

                # Fetch followers if official_username is available
                official_username = None
                if selected_project != "Все проекты":
                    first_official_doc = next((doc for doc in docs if doc.metadata.get("author_type") == "official"),
                                              None)
                    if first_official_doc:
                        official_username = first_official_doc.metadata.get("author_username")

                if official_username:
                    followers, following = get_user_info(official_username)
                    if followers != 0 or following != 0:
                        st.write(f"Подписчики: {followers}")
                        st.write(f"Подписки: {following}")
                    else:
                        st.warning(
                            f"Не удалось получить данные о подписчиках для {official_username}. Аналитика продолжится без этих данных.")
                else:
                    st.warning("Официальный аккаунт не найден в данных. Аналитика продолжится по author_type.")

                # Aggregate metrics
                official_metrics = pd.DataFrame(columns=["date", "view_count", "retweet_count", "reply_count"])
                other_metrics = pd.DataFrame(columns=["date", "view_count", "retweet_count", "reply_count"])

                if not official_tweets_df.empty:
                    official_metrics = official_tweets_df.groupby("date").agg({
                        "view_count": "sum",
                        "retweet_count": "sum",
                        "reply_count": "sum"
                    }).reset_index()
                    st.write(f"Найдено {len(official_tweets_df)} официальных твитов.")
                else:
                    st.warning("Нет твитов от официального аккаунта для анализа.")

                if not other_tweets_df.empty:
                    other_metrics = other_tweets_df.groupby("date").agg({
                        "view_count": "sum",
                        "retweet_count": "sum",
                        "reply_count": "sum"
                    }).reset_index()
                    st.write(f"Найдено {len(other_tweets_df)} других твитов.")
                else:
                    st.warning("Нет других твитов для анализа.")

                # # Plotting
                # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
                #
                # # Official tweets plot
                # if not official_metrics.empty:
                #     ax1.plot(official_metrics["date"], official_metrics["view_count"], label="Просмотры", marker="o")
                #     ax1.plot(official_metrics["date"], official_metrics["retweet_count"], label="Ретвиты", marker="o")
                #     ax1.plot(official_metrics["date"], official_metrics["reply_count"], label="Ответы", marker="o")
                # ax1.set_title("Аналитика твитов официального аккаунта")
                # ax1.set_ylabel("Количество")
                # ax1.legend()
                # ax1.grid(True)
                #
                # # Other tweets plot
                # if not other_metrics.empty:
                #     ax2.plot(other_metrics["date"], other_metrics["view_count"], label="Просмотры", marker="o")
                #     ax2.plot(other_metrics["date"], other_metrics["retweet_count"], label="Ретвиты", marker="o")
                #     ax2.plot(other_metrics["date"], other_metrics["reply_count"], label="Ответы", marker="o")
                # ax2.set_title("Аналитика остальных твитов")
                # ax2.set_xlabel("Дата")
                # ax2.set_ylabel("Количество")
                # ax2.legend()
                # ax2.grid(True)
                #
                # plt.xticks(rotation=45)
                # plt.tight_layout()
                # st.pyplot(fig)

                # Plotting
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

                # Official tweets plot
                if not official_metrics.empty:
                    # Left axis for view_count
                    ax1.plot(official_metrics["date"], official_metrics["view_count"], label="Просмотры", marker="o",
                             color="blue")
                    ax1.set_ylabel("Просмотры", color="blue")
                    ax1.tick_params(axis="y", labelcolor="blue")

                    # Right axis for retweet_count and reply_count
                    ax1_twin = ax1.twinx()
                    ax1_twin.plot(official_metrics["date"], official_metrics["retweet_count"], label="Ретвиты",
                                  marker="o", color="green")
                    ax1_twin.plot(official_metrics["date"], official_metrics["reply_count"], label="Ответы", marker="o",
                                  color="red")
                    ax1_twin.set_ylabel("Ретвиты / Ответы", color="black")
                    ax1_twin.tick_params(axis="y", labelcolor="black")

                    # Combine legends
                    lines1, labels1 = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax1_twin.get_legend_handles_labels()
                    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
                else:
                    ax1.text(0.5, 0.5, "Нет данных для официальных твитов", horizontalalignment="center",
                             verticalalignment="center")
                ax1.set_title("Аналитика твитов официального аккаунта")
                ax1.set_xlabel("Дата")
                ax1.grid(True)
                ax1.tick_params(axis="x", rotation=45)

                # Other tweets plot
                if not other_metrics.empty:
                    # Left axis for view_count
                    ax2.plot(other_metrics["date"], other_metrics["view_count"], label="Просмотры", marker="o",
                             color="blue")
                    ax2.set_ylabel("Просмотры", color="blue")
                    ax2.tick_params(axis="y", labelcolor="blue")

                    # Right axis for retweet_count and reply_count
                    ax2_twin = ax2.twinx()
                    ax2_twin.plot(other_metrics["date"], other_metrics["retweet_count"], label="Ретвиты", marker="o",
                                  color="green")
                    ax2_twin.plot(other_metrics["date"], other_metrics["reply_count"], label="Ответы", marker="o",
                                  color="red")
                    ax2_twin.set_ylabel("Ретвиты / Ответы", color="black")
                    ax2_twin.tick_params(axis="y", labelcolor="black")

                    # Combine legends
                    lines1, labels1 = ax2.get_legend_handles_labels()
                    lines2, labels2 = ax2_twin.get_legend_handles_labels()
                    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
                else:
                    ax2.text(0.5, 0.5, "Нет данных для других твитов", horizontalalignment="center",
                             verticalalignment="center")
                ax2.set_title("Аналитика остальных твитов")
                ax2.set_xlabel("Дата")
                ax2.grid(True)
                ax2.tick_params(axis="x", rotation=45)

                plt.tight_layout()
                st.pyplot(fig)

                # Display raw data
                st.write("Официальные твиты:",
                         official_tweets_df[["tweet_id", "text", "created_at", "author_username", "author_type"]])
                st.write("Остальные твиты:",
                         other_tweets_df[["tweet_id", "text", "created_at", "author_username", "author_type"]])
except Exception as e:
    st.error(f"Ошибка при обработке аналитики: {str(e)}")

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
    try:
        response = agent.invoke(st.session_state["data"])
        st.session_state['data'] = response
    except Exception as e:
        st.error(f"Failed to invoke agent: {str(e)}")

for message in st.session_state['data']['messages']:
    if isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, AIMessage):
        role = "bot"
    else:
        continue

    content = message.content.strip() if isinstance(message.content, str) else ""
    if not content:
        continue  # пропускаем пустые сообщения

    with st.chat_message(role):
        st.markdown(content)