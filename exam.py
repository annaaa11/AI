
import os
import json
import dotenv
import streamlit as st
from uuid import uuid4
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import nest_asyncio
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langgraph.prebuilt import create_react_agent
import matplotlib.dates as mdates

from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from uuid import uuid4
import streamlit as st
import pandas as pd

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

# def search_tweets_by_query(query: str, username: str, project_name: str, project_symbol: str, start_date: datetime,
#                           limit: int = 60, min_retweets: int = 0, min_replies: int = 0):
#     url = "https://api.twitterapi.io/twitter/tweet/advanced_search"
#     headers = {"x-api-key": twitter_api_key}
#     since_str = start_date.strftime("%Y-%m-%d")
#
#     session = requests.Session()
#     retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
#     session.mount("https://", HTTPAdapter(max_retries=retries))
#
#     def fetch_paginated(query_string, is_official=False, max_tweets=limit):
#         all_tweets = []
#         processed_keys = set()
#         next_token = None
#         tweets_fetched = 0
#
#         while tweets_fetched < max_tweets:
#             try:
#                 # Exclude official account from keyword query
#                 if not is_official:
#                     query_string = f"{query_string} -from:{username}"
#                 params = {
#                     "query": f"{query_string} since:{since_str} min_retweets:{min_retweets} min_replies:{min_replies}",
#                     "queryType": "Latest",
#                     "limit": min(20, max_tweets - tweets_fetched)  # Request up to 20 tweets per call
#                 }
#                 if next_token:
#                     params["next_token"] = next_token
#
#                 st.write(f"DEBUG: Выполняется запрос: {params['query']}, next_token={next_token}")
#                 response = session.get(url, headers=headers, params=params, timeout=10)
#                 response.raise_for_status()
#                 data = response.json()
#
#                 tweets = []
#                 for key in ["tweets", "data", "results"]:
#                     if key in data:
#                         tweets = [t for t in data[key] if t.get("type") == "tweet"]
#                         break
#                 st.write(f"DEBUG: Получено {len(tweets)} твитов для запроса '{query_string}'")
#
#                 for tweet in tweets:
#                     tweet_id = str(tweet.get("id", tweet.get("id_str", str(uuid4()))))
#                     author = tweet.get("author", {})
#                     user = tweet.get("user", {})
#                     screen_name = author.get("userName", user.get("username", user.get("screen_name", None)))
#                     if not screen_name:
#                         st.write(f"DEBUG: Пропущен твит {tweet_id}: отсутствует имя пользователя")
#                         continue
#
#                     unique_key = f"{tweet_id}_{screen_name}"
#                     if unique_key in processed_keys:
#                         st.write(f"DEBUG: Пропущен твит {tweet_id}: дубликат по ключу {unique_key}")
#                         continue
#                     processed_keys.add(unique_key)
#
#                     created_at_raw = tweet.get("createdAt") or tweet.get("created_at")
#                     if not created_at_raw:
#                         st.write(f"DEBUG: Пропущен твит {tweet_id}: отсутствует дата создания")
#                         continue
#
#                     try:
#                         parsed_date = pd.to_datetime(created_at_raw, utc=True, errors="raise")
#                         created_at = parsed_date.isoformat()
#                     except (ValueError, TypeError) as e:
#                         st.write(f"DEBUG: Пропущен твит {tweet_id}: некорректная дата создания ({created_at_raw}), ошибка: {str(e)}")
#                         continue
#
#                     public_metrics = tweet.get("public_metrics", {})
#                     retweet_count = int(public_metrics.get("retweet_count", tweet.get("retweetCount", 0)))
#                     reply_count = int(public_metrics.get("reply_count", tweet.get("replyCount", 0)))
#                     view_count = int(public_metrics.get("view_count", tweet.get("viewCount", 0)))
#
#                     # Client-side filtering for non-official tweets
#                     if not is_official and (retweet_count < min_retweets or reply_count < min_replies):
#                         st.write(f"DEBUG: Пропущен твит {tweet_id}: retweet_count={retweet_count}, reply_count={reply_count} не соответствуют min_retweets={min_retweets}, min_replies={min_replies}")
#                         continue
#
#                     author_type = "official" if screen_name.lower() == username.lower() else "external"
#                     all_tweets.append({
#                         "id_str": tweet_id,
#                         "text": tweet.get("text", ""),
#                         "created_at": created_at,
#                         "user": {"screen_name": screen_name},
#                         "public_metrics": {
#                             "retweet_count": retweet_count,
#                             "reply_count": reply_count,
#                             "view_count": view_count
#                         },
#                         "author_type": author_type,
#                         "project_name": project_name,
#                         "project_symbol": project_symbol
#                     })
#                     tweets_fetched += 1
#
#                 next_token = data.get("next_token")
#                 if not next_token or len(tweets) == 0:
#                     st.write(f"DEBUG: Нет следующей страницы для запроса '{query_string}'")
#                     break
#
#             except Exception as e:
#                 st.error(f"Ошибка при получении твитов для запроса '{query_string}': {e}")
#                 break
#
#         return all_tweets[:max_tweets]
#
#     # Fetch official tweets
#     from_query = f"from:{username}"
#     official_tweets = fetch_paginated(from_query, is_official=True, max_tweets=limit)
#     st.write(f"DEBUG: Официальные твиты ({len(official_tweets)}): {[t['id_str'] for t in official_tweets]}")
#
#     # Fetch non-official tweets
#     remaining_limit = limit - len(official_tweets)
#     project_name_query = f'"{project_name}"'
#     project_symbol_query = f"${project_symbol}"
#     keyword_query = f"{project_name_query} OR {project_symbol_query}"
#     keyword_tweets = fetch_paginated(keyword_query, is_official=False, max_tweets=remaining_limit)
#     st.write(f"DEBUG: Неофициальные твиты ({len(keyword_tweets)}): {[t['id_str'] for t in keyword_tweets]}")
#
#     # Combine tweets
#     all_tweets = official_tweets + keyword_tweets
#     st.write(f"DEBUG: Всего твитов после объединения: {len(all_tweets)}")
#
#     # Deduplicate by tweet_id
#     unique_tweets = {}
#     for tweet in all_tweets:
#         tweet_id = tweet["id_str"]
#         if tweet_id not in unique_tweets:
#             unique_tweets[tweet_id] = tweet
#         else:
#             st.write(f"DEBUG: Пропущен дубликат твита {tweet_id}")
#     all_tweets = list(unique_tweets.values())
#     st.write(f"DEBUG: Всего уникальных твитов: {len(all_tweets)}")
#
#     return all_tweets


def search_tweets_by_query(query: str, username: str, project_name: str, project_symbol: str, start_date: datetime,
                          limit: int = 60, min_retweets: int = 0, min_replies: int = 0):
    url = "https://api.twitterapi.io/twitter/tweet/advanced_search"
    headers = {"x-api-key": twitter_api_key}
    since_str = start_date.strftime("%Y-%m-%d")

    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))

    def fetch_paginated(query_string, is_official=False, max_tweets=limit):
        all_tweets = []
        processed_keys = set()
        next_token = None
        tweets_fetched = 0

        while tweets_fetched < max_tweets:
            try:
                params = {
                    "query": f"{query_string} since:{since_str} min_retweets:{min_retweets} min_replies:{min_replies}",
                    "queryType": "Latest",
                    "limit": min(20, max_tweets - tweets_fetched)
                }
                if next_token:
                    params["next_token"] = next_token

                st.write(f"DEBUG: Выполняется запрос: {params['query']}, next_token={next_token}")
                response = session.get(url, headers=headers, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                # Debug full API response structure
                if tweets_fetched == 0:
                    st.write(f"DEBUG: Структура ответа API: {list(data.keys())}")
                    if "tweets" in data and data["tweets"]:
                        st.write(f"DEBUG: Пример первого твита: {data['tweets'][0]}")

                tweets = []
                for key in ["tweets", "data", "results"]:
                    if key in data:
                        tweets = [t for t in data[key] if t.get("type") == "tweet"]
                        break
                st.write(f"DEBUG: Получено {len(tweets)} твитов для запроса '{query_string}'")

                for tweet in tweets:
                    tweet_id = str(tweet.get("id", tweet.get("id_str", str(uuid4()))))
                    author = tweet.get("author", {})
                    user = tweet.get("user", {})
                    screen_name = author.get("userName", user.get("username", user.get("screen_name", None)))
                    if not screen_name:
                        st.write(f"DEBUG: Пропущен твит {tweet_id}: отсутствует имя пользователя")
                        continue

                    unique_key = f"{tweet_id}_{screen_name}"
                    if unique_key in processed_keys:
                        st.write(f"DEBUG: Пропущен твит {tweet_id}: дубликат по ключу {unique_key}")
                        continue
                    processed_keys.add(unique_key)

                    created_at_raw = tweet.get("createdAt") or tweet.get("created_at")
                    if not created_at_raw:
                        st.write(f"DEBUG: Пропущен твит {tweet_id}: отсутствует дата создания")
                        continue

                    try:
                        parsed_date = pd.to_datetime(created_at_raw, utc=True, errors="raise")
                        created_at = parsed_date.isoformat()
                    except (ValueError, TypeError) as e:
                        st.write(f"DEBUG: Пропущен твит {tweet_id}: некорректная дата создания ({created_at_raw}), ошибка: {str(e)}")
                        continue

                    public_metrics = tweet.get("public_metrics", {})
                    retweet_count = int(public_metrics.get("retweet_count", tweet.get("retweetCount", 0)))
                    reply_count = int(public_metrics.get("reply_count", tweet.get("replyCount", 0)))
                    view_count = int(public_metrics.get("view_count", tweet.get("viewCount", 0)))

                    # Client-side filtering for non-official tweets
                    if not is_official and (retweet_count < min_retweets or reply_count < min_replies):
                        st.write(f"DEBUG: Пропущен твит {tweet_id}: retweet_count={retweet_count}, reply_count={reply_count} не соответствуют min_retweets={min_retweets}, min_replies={min_replies}")
                        continue

                    author_type = "official" if screen_name.lower() == username.lower() else "external"
                    all_tweets.append({
                        "id_str": tweet_id,
                        "text": tweet.get("text", ""),
                        "created_at": created_at,
                        "user": {"screen_name": screen_name},
                        "public_metrics": {
                            "retweet_count": retweet_count,
                            "reply_count": reply_count,
                            "view_count": view_count
                        },
                        "author_type": author_type,
                        "project_name": project_name,
                        "project_symbol": project_symbol
                    })
                    tweets_fetched += 1

                    # Debug raw data for first few tweets
                    if tweets_fetched <= 3:
                        st.write(f"DEBUG: Твит {tweet_id}: created_at_raw={created_at_raw}, retweet_count={retweet_count}, reply_count={reply_count}")

                next_token = data.get("next_token")
                if not next_token or len(tweets) == 0:
                    st.write(f"DEBUG: Нет следующей страницы для запроса '{query_string}'")
                    break

            except Exception as e:
                st.error(f"Ошибка при получении твитов для запроса '{query_string}': {e}")
                break

        return all_tweets[:max_tweets]

    # Fetch official tweets
    from_query = f"from:{username}"
    official_tweets = fetch_paginated(from_query, is_official=True, max_tweets=limit)
    st.write(f"DEBUG: Официальные твиты ({len(official_tweets)}): {[t['id_str'] for t in official_tweets]}")

    # Fetch non-official tweets
    remaining_limit = limit - len(official_tweets)
    project_name_query = f"RWA OR Inc."  # Relaxed query to match individual words
    project_symbol_query = f"${project_symbol}"
    keyword_query = f"{project_name_query} OR {project_symbol_query}"
    keyword_tweets = fetch_paginated(keyword_query, is_official=False, max_tweets=remaining_limit)
    st.write(f"DEBUG: Неофициальные твиты ({len(keyword_tweets)}): {[t['id_str'] for t in keyword_tweets]}")

    # Combine and deduplicate tweets
    all_tweets = official_tweets + keyword_tweets
    st.write(f"DEBUG: Всего твитов после объединения: {len(all_tweets)}")

    unique_tweets = {}
    for tweet in all_tweets:
        tweet_id = tweet["id_str"]
        if tweet_id not in unique_tweets:
            unique_tweets[tweet_id] = tweet
        else:
            st.write(f"DEBUG: Пропущен дубликат твита {tweet_id}")
    all_tweets = list(unique_tweets.values())
    st.write(f"DEBUG: Всего уникальных твитов: {len(all_tweets)}")

    return all_tweets


# Function to search documents (unchanged as requested)
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

# Function to get user info
def get_user_info(username):
    url = f"https://api.twitterapi.io/twitter/user/{username}"
    headers = {"x-api-key": twitter_api_key}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            user = data.get("user") or data.get("data") or {}
            followers = user.get("followers") or user.get("public_metrics", {}).get("followers_count", 0)
            following = user.get("following") or user.get("public_metrics", {}).get("following_count", 0)
            return followers, following
        else:
            return 0, 0
    except Exception:
        return 0, 0

# Function to check if tweet exists in Pinecone
def already_exists(tweet_id: str, namespace: str = "") -> bool:
    try:
        response = index.fetch(ids=[tweet_id], namespace=namespace)
        exists = tweet_id in (response.vectors if hasattr(response, 'vectors') else response.get("vectors", {}))
        return exists
    except Exception as e:
        st.error(f"Ошибка при проверке tweet_id {tweet_id} в Pinecone: {str(e)}")
        return False

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
st.title("Crypto Twitter Search")


# Add tweets
st.subheader("Добавить твиты в базу")

# Button to clear Pinecone and JSON
if st.button("Очистить базу"):
    namespace = ""
    try:
        vector_store.delete(delete_all=True, namespace=namespace)
        st.success(f"Все записи в Pinecone (неймспейс '{namespace}') удалены.")
    except Exception as e:
        if "Namespace not found" not in str(e):
            st.error(f"Ошибка при очистке Pinecone: {str(e)}")
        else:
            st.info(f"Неймспейс '{namespace}' уже пуст.")

    try:
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump({}, jf, ensure_ascii=False)
        id_data.clear()
        st.success("JSON-файл очищен.")
    except Exception as e:
        st.error(f"Ошибка при очистке JSON: {str(e)}")

coinmarketcap_url = st.text_input(
    "Введите URL CoinMarketCap (например, https://coinmarketcap.com/currencies/legends-of-elumia/):")
start_date = st.date_input("Выберите начальную дату:", value=datetime.now().date() - timedelta(days=7))
limit = st.number_input("Количество твитов:", min_value=1, max_value=100, value=20)
min_retweets = st.number_input("Минимальное количество ретвитов:", min_value=0, value=0)
min_replies = st.number_input("Минимальное количество ответов:", min_value=0, value=0)


def normalize_url(url: str) -> str:
    return url.strip().rstrip("/")


if st.button("Загрузить твиты"):
    if coinmarketcap_url:
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

            st.session_state["tweets"] = []
            namespace = ""

            # Check Pinecone contents before deletion
            try:
                filter_dict = {
                    "$or": [
                        {"coinmarketcap_url": normalized_coinmarketcap_url},
                        {"coinmarketcap_url": normalized_coinmarketcap_url + "/"}
                    ]
                }
                existing_docs = vector_store.similarity_search("", k=1000, namespace=namespace, filter=filter_dict)
                existing_tweet_ids = [doc.metadata["tweet_id"] for doc in existing_docs if "tweet_id" in doc.metadata]
                st.write(
                    f"DEBUG: Найдено {len(existing_docs)} записей для проекта '{normalized_coinmarketcap_url}' перед очисткой.")
                if existing_tweet_ids:
                    st.write(f"DEBUG: Пример tweet_id в базе: {existing_tweet_ids[:5]}")
                else:
                    st.write("DEBUG: Записи для проекта отсутствуют в базе.")
            except Exception as e:
                st.warning(f"DEBUG: Ошибка при проверке содержимого Pinecone: {str(e)}. Продолжаем выполнение.")

            # Delete records from Pinecone
            try:
                vector_store.delete(filter=filter_dict, namespace=namespace)
                # Verify deletion
                post_delete_docs = vector_store.similarity_search("", k=1000, namespace=namespace, filter=filter_dict)
                st.write(
                    f"DEBUG: Найдено {len(post_delete_docs)} записей для проекта '{normalized_coinmarketcap_url}' после очистки.")
                if post_delete_docs:
                    st.warning(
                        "DEBUG: Очистка не удалила все записи. Остались tweet_id: {[doc.metadata['tweet_id'] for doc in post_delete_docs[:5]]}")
            except Exception as e:
                if "Namespace not found" not in str(e):
                    st.warning(f"Ошибка при очистке векторной базы: {str(e)}. Продолжаем выполнение.")
                else:
                    st.write("DEBUG: Неймспейс не найден, очистка не требуется.")

            # Clean JSON
            try:
                # Check JSON contents before cleaning
                st.write(f"DEBUG: Всего записей в JSON до очистки: {len(id_data)}")
                keys_to_delete = [k for k in id_data.keys() if k.startswith(f"twitter_{official_username}_")]
                st.write(f"DEBUG: Найдено {len(keys_to_delete)} ключей для удаления: {keys_to_delete[:5]}")

                # Filter out project-related keys
                filtered_id_data = {
                    k: v for k, v in id_data.items()
                    if not k.startswith(f"twitter_{official_username}_")
                }
                # Write to JSON
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(filtered_id_data, jf, ensure_ascii=False, indent=2)
                id_data.clear()
                id_data.update(filtered_id_data)
                st.write(f"DEBUG: JSON очищен. Текущее количество записей: {len(id_data)}")
            except Exception as e:
                st.warning(f"Ошибка при очистке JSON: {str(e)}. Продолжаем выполнение.")

            # # Check Pinecone contents before deletion
            # try:
            #     existing_docs = vector_store.similarity_search("", k=1000, namespace=namespace)
            #     st.write(f"DEBUG: Найдено {len(existing_docs)} записей в неймспейсе '{namespace}' перед очисткой.")
            #     existing_tweet_ids = [doc.metadata["tweet_id"] for doc in existing_docs if "tweet_id" in doc.metadata]
            #     if existing_tweet_ids:
            #         st.write(f"DEBUG: Пример tweet_id в базе: {existing_tweet_ids[:5]}")
            # except Exception as e:
            #     st.error(f"DEBUG: Ошибка при проверке содержимого Pinecone: {str(e)}")
            #
            # # Delete records from Pinecone
            # try:
            #     vector_store.delete(
            #         filter={
            #             "$or": [
            #                 {"coinmarketcap_url": normalized_coinmarketcap_url},
            #                 {"coinmarketcap_url": normalized_coinmarketcap_url + "/"}
            #             ]
            #         },
            #         namespace=namespace
            #     )
            #     # Verify deletion
            #     post_delete_docs = vector_store.similarity_search("", k=1000, namespace=namespace)
            #     st.write(f"DEBUG: Найдено {len(post_delete_docs)} записей в неймспейсе '{namespace}' после очистки.")
            # except Exception as e:
            #     if "Namespace not found" not in str(e):
            #         st.error(f"Ошибка при очистке векторной базы: {str(e)}")
            #         st.stop()
            #
            # # Clean JSON
            # try:
            #     filtered_id_data = {
            #         k: v for k, v in id_data.items()
            #         if not k.startswith(f"twitter_{official_username}_")
            #     }
            #     with open(json_path, "w", encoding="utf-8") as jf:
            #         json.dump(filtered_id_data, jf, ensure_ascii=False)
            #     id_data.clear()
            #     id_data.update(filtered_id_data)
            #     st.write(f"DEBUG: JSON очищен. Текущее количество записей: {len(id_data)}")
            # except Exception as e:
            #     st.error(f"Ошибка при очистке JSON: {str(e)}")
            #     st.stop()

            tweets = search_tweets_by_query(project_name, official_username, project_name, project_symbol, start_date,
                                            limit, min_retweets, min_replies)
            st.session_state["tweets"] = tweets
            st.write(f"Найдено {len(tweets)} уникальных твитов.")

            if tweets:
                uploaded = 0
                skipped = 0
                new_id_data = {}
                existing_ids = set()

                for tweet in tweets:
                    text = tweet.get("text", "")
                    if not text:
                        st.write(f"DEBUG: Пропущен твит: отсутствует текст")
                        continue

                    tweet_id = tweet.get("id_str", str(uuid4()))
                    st.write(f"DEBUG: Обработка твита с id: {tweet_id}, текст: {text[:30]}...")

                    if already_exists(tweet_id, namespace=namespace):
                        st.write(f"DEBUG: Твит {tweet_id} уже существует в Pinecone, пропущен")
                        skipped += 1
                        continue
                    if tweet_id in existing_ids:
                        st.write(f"DEBUG: Твит {tweet_id} уже обработан в текущей сессии, пропущен")
                        skipped += 1
                        continue
                    existing_ids.add(tweet_id)

                    author_username = tweet.get("user", {}).get("screen_name", None)
                    if not author_username:
                        st.write(f"DEBUG: Пропущен твит {tweet_id}: отсутствует имя пользователя")
                        continue

                    created_at = tweet.get("created_at")
                    if not created_at:
                        st.write(f"DEBUG: Пропущен твит {tweet_id}: отсутствует дата создания")
                        continue
                    st.write(f"DEBUG: Твит {tweet_id} ({author_username}): created_at={created_at}")

                    is_official = author_username.lower() == official_username.lower()
                    author_type = "official" if is_official else "external"

                    public_metrics = tweet.get("public_metrics", {})
                    retweet_count = public_metrics.get("retweet_count", tweet.get("retweetCount", 0))
                    reply_count = public_metrics.get("reply_count", tweet.get("replyCount", 0))
                    view_count = public_metrics.get("view_count", tweet.get("viewCount", 0))

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

                    try:
                        vector_store.add_documents([doc], ids=[tweet_id], namespace=namespace)
                        uploaded += 1
                        st.write(f"DEBUG: Загружен твит: {tweet_id}")
                    except Exception as e:
                        st.error(f"Ошибка при добавлении твита {tweet_id}: {str(e)}")
                        continue

                    key_name = f"twitter_{official_username}_{tweet_id}_{author_username}"
                    new_id_data[key_name] = tweet_id

                try:
                    existing_data = id_data if id_data else {}
                    existing_data.update(new_id_data)
                    with open(json_path, "w", encoding="utf-8") as jf:
                        json.dump(existing_data, jf, ensure_ascii=False)
                    st.write(f"DEBUG: JSON обновлен. Новых записей: {len(new_id_data)}")
                except Exception as e:
                    st.error(f"Failed to update JSON file: {str(e)}")
                    st.stop()

                st.write(f"📥 Загружено: {uploaded}")
                st.write(f"🚫 Пропущено (дубликаты): {skipped}")

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
                    else:
                        st.info("Дубликаты в векторной базе не найдены.")
                except Exception as e:
                    st.error(f"Ошибка при проверке векторной базы: {str(e)}")
            else:
                st.error("Не удалось получить твиты. Проверьте параметры запроса или Twitter API ключ.")
    else:
        st.error("Пожалуйста, введите URL CoinMarketCap.")

# Analytics
st.subheader("Аналитика твитов")

try:
    all_docs = vector_store.similarity_search("", k=1000)
    unique_projects = sorted(set(
        doc.metadata.get("coinmarketcap_url") for doc in all_docs if doc.metadata.get("coinmarketcap_url")
    ))
    if not unique_projects:
        st.error("Нет данных в векторной базе для анализа. Загрузите твиты сначала.")
    else:
        selected_project = st.selectbox("Выберите проект для анализа:", options=unique_projects, index=0)

        if st.button("Вывести аналитику"):
            filter_dict = {
                "$or": [
                    {"coinmarketcap_url": normalize_url(selected_project)},
                    {"coinmarketcap_url": normalize_url(selected_project) + "/"}
                ]
            }

            docs = vector_store.similarity_search("", k=1000, filter=filter_dict)
            if not docs:
                st.error("Нет данных для выбранного проекта в векторной базе.")
            else:
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

                tweets_df["created_at"] = pd.to_datetime(tweets_df["created_at"], errors="coerce", utc=True)
                tweets_df = tweets_df.dropna(subset=["created_at"])
                if tweets_df.empty:
                    st.error("Нет данных с валидными датами для анализа. Проверьте формат created_at в векторной базе.")
                else:
                    tweets_df["date"] = tweets_df["created_at"].dt.date

                    # Get project info for display
                    project_info = parse_coinmarketcap_project(selected_project)
                    if project_info and project_info["twitter"] != "Not found":
                        twitter_url = project_info["twitter"]
                        project_name = project_info["name"]
                        project_symbol = project_info["symbol"]
                        st.write(f"Twitter: {twitter_url} (Проект: {project_name}, Символ: {project_symbol})")
                    else:
                        st.warning("Не удалось получить информацию о проекте из CoinMarketCap.")

                    official_tweets_df = tweets_df[tweets_df["author_type"] == "official"]
                    other_tweets_df = tweets_df[tweets_df["author_type"] != "official"]



                    first_official_doc = next((doc for doc in docs if doc.metadata.get("author_type") == "official"), None)
                    if first_official_doc:
                        official_username = first_official_doc.metadata.get("author_username")
                        followers, following = get_user_info(official_username)
                        st.write(f"Подписчики: {followers}")
                        st.write(f"Подписки: {following}")
                        # Calculate engagement ratio
                        if not official_tweets_df.empty and followers > following:
                            avg_engagement = (
                                        official_tweets_df["retweet_count"] + official_tweets_df["reply_count"]).mean()
                            engagement_ratio = avg_engagement / (followers - following)
                            st.write(f"Коэффициент вовлеченности фолловеров: {engagement_ratio:.4f}")
                        else:
                            st.warning(
                                "Коэффициент вовлеченности не может быть рассчитан: недостаточно данных или нулевая/отрицательная разница фолловеров и подписок.")



                    # Aggregate metrics by datetime (not just date) to preserve hours
                    official_metrics = pd.DataFrame(
                        columns=["created_at", "view_count", "retweet_count", "reply_count"])
                    other_metrics = pd.DataFrame(columns=["created_at", "view_count", "retweet_count", "reply_count"])

                    if not official_tweets_df.empty:
                        official_metrics = official_tweets_df.groupby("created_at").agg({
                            "view_count": "sum",
                            "retweet_count": "sum",
                            "reply_count": "sum"
                        }).reset_index()
                        st.write(f"Найдено {len(official_tweets_df)} официальных твитов.")
                    else:
                        st.warning("Нет твитов от официального аккаунта для анализа.")

                    if not other_tweets_df.empty:
                        other_metrics = other_tweets_df.groupby("created_at").agg({
                            "view_count": "sum",
                            "retweet_count": "sum",
                            "reply_count": "sum"
                        }).reset_index()
                        st.write(f"Найдено {len(other_tweets_df)} других твитов.")
                    else:
                        st.warning("Нет других твитов для анализа.")

                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

                    # Plot official tweets
                    if not official_metrics.empty:
                        ax1.plot(official_metrics["created_at"], official_metrics["view_count"], label="Просмотры",
                                 marker="o", color="blue")
                        ax1.set_ylabel("Просмотры", color="blue")
                        ax1.tick_params(axis="y", labelcolor="blue")

                        ax1_twin = ax1.twinx()
                        ax1_twin.plot(official_metrics["created_at"], official_metrics["retweet_count"],
                                      label="Ретвиты", marker="o", color="green")
                        ax1_twin.plot(official_metrics["created_at"], official_metrics["reply_count"], label="Ответы",
                                      marker="o", color="red")
                        ax1_twin.set_ylabel("Ретвиты / Ответы", color="black")
                        ax1_twin.tick_params(axis="y", labelcolor="black")

                        # Set x-axis to show date and time with 3-hour intervals
                        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=3))  # Ticks every 3 hours
                        ax1.xaxis.set_major_formatter(
                            mdates.DateFormatter("%Y-%m-%d %H:%M"))  # Format: YYYY-MM-DD HH:MM
                        ax1.tick_params(axis="x", rotation=45)

                        lines1, labels1 = ax1.get_legend_handles_labels()
                        lines2, labels2 = ax1_twin.get_legend_handles_labels()
                        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
                    else:
                        ax1.text(0.5, 0.5, "Нет данных для официальных твитов", horizontalalignment="center",
                                 verticalalignment="center")
                    ax1.set_title("Аналитика твитов официального аккаунта")
                    ax1.set_xlabel("Дата и время")
                    ax1.grid(True)

                    # Plot other tweets
                    if not other_metrics.empty:
                        ax2.plot(other_metrics["created_at"], other_metrics["view_count"], label="Просмотры",
                                 marker="o", color="blue")
                        ax2.set_ylabel("Просмотры", color="blue")
                        ax2.tick_params(axis="y", labelcolor="blue")

                        ax2_twin = ax2.twinx()
                        ax2_twin.plot(other_metrics["created_at"], other_metrics["retweet_count"], label="Ретвиты",
                                      marker="o", color="green")
                        ax2_twin.plot(other_metrics["created_at"], other_metrics["reply_count"], label="Ответы",
                                      marker="o", color="red")
                        ax2_twin.set_ylabel("Ретвиты / Ответы", color="black")
                        ax2_twin.tick_params(axis="y", labelcolor="black")

                        # Set x-axis to show date and time with 3-hour intervals
                        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=3))  # Ticks every 3 hours
                        ax2.xaxis.set_major_formatter(
                            mdates.DateFormatter("%Y-%m-%d %H:%M"))  # Format: YYYY-MM-DD HH:MM
                        ax2.tick_params(axis="x", rotation=45)

                        lines1, labels1 = ax2.get_legend_handles_labels()
                        lines2, labels2 = ax2_twin.get_legend_handles_labels()
                        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
                    else:
                        ax2.text(0.5, 0.5, "Нет данных для других твитов", horizontalalignment="center",
                                 verticalalignment="center")
                    ax2.set_title("Аналитика остальных твитов")
                    ax2.set_xlabel("Дата и время")
                    ax2.grid(True)

                    plt.tight_layout()
                    st.pyplot(fig)
                    # official_metrics = pd.DataFrame(columns=["date", "view_count", "retweet_count", "reply_count"])
                    # other_metrics = pd.DataFrame(columns=["date", "view_count", "retweet_count", "reply_count"])
                    #
                    # if not official_tweets_df.empty:
                    #     official_metrics = official_tweets_df.groupby("date").agg({
                    #         "view_count": "sum",
                    #         "retweet_count": "sum",
                    #         "reply_count": "sum"
                    #     }).reset_index()
                    #     st.write(f"Найдено {len(official_tweets_df)} официальных твитов.")
                    # else:
                    #     st.warning("Нет твитов от официального аккаунта для анализа.")
                    #
                    # if not other_tweets_df.empty:
                    #     other_metrics = other_tweets_df.groupby("date").agg({
                    #         "view_count": "sum",
                    #         "retweet_count": "sum",
                    #         "reply_count": "sum"
                    #     }).reset_index()
                    #     st.write(f"Найдено {len(other_tweets_df)} других твитов.")
                    # else:
                    #     st.warning("Нет других твитов для анализа.")
                    #
                    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
                    #
                    # if not official_metrics.empty:
                    #     ax1.plot(official_metrics["date"], official_metrics["view_count"], label="Просмотры", marker="o",
                    #              color="blue")
                    #     ax1.set_ylabel("Просмотры", color="blue")
                    #     ax1.tick_params(axis="y", labelcolor="blue")
                    #
                    #     ax1_twin = ax1.twinx()
                    #     ax1_twin.plot(official_metrics["date"], official_metrics["retweet_count"], label="Ретвиты",
                    #                   marker="o", color="green")
                    #     ax1_twin.plot(official_metrics["date"], official_metrics["reply_count"], label="Ответы",
                    #                   marker="o", color="red")
                    #     ax1_twin.set_ylabel("Ретвиты / Ответы", color="black")
                    #     ax1_twin.tick_params(axis="y", labelcolor="black")
                    #
                    #     lines1, labels1 = ax1.get_legend_handles_labels()
                    #     lines2, labels2 = ax1_twin.get_legend_handles_labels()
                    #     ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
                    # else:
                    #     ax1.text(0.5, 0.5, "Нет данных для официальных твитов", horizontalalignment="center",
                    #              verticalalignment="center")
                    # ax1.set_title("Аналитика твитов официального аккаунта")
                    # ax1.set_xlabel("Дата")
                    # ax1.grid(True)
                    # ax1.tick_params(axis="x", rotation=45)
                    #
                    # if not other_metrics.empty:
                    #     ax2.plot(other_metrics["date"], other_metrics["view_count"], label="Просмотры", marker="o",
                    #              color="blue")
                    #     ax2.set_ylabel("Просмотры", color="blue")
                    #     ax2.tick_params(axis="y", labelcolor="blue")
                    #
                    #     ax2_twin = ax2.twinx()
                    #     ax2_twin.plot(other_metrics["date"], other_metrics["retweet_count"], label="Ретвиты",
                    #                   marker="o", color="green")
                    #     ax2_twin.plot(other_metrics["date"], other_metrics["reply_count"], label="Ответы", marker="o",
                    #                   color="red")
                    #     ax2_twin.set_ylabel("Ретвиты / Ответы", color="black")
                    #     ax2_twin.tick_params(axis="y", labelcolor="black")
                    #
                    #     lines1, labels1 = ax2.get_legend_handles_labels()
                    #     lines2, labels2 = ax2_twin.get_legend_handles_labels()
                    #     ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
                    # else:
                    #     ax2.text(0.5, 0.5, "Нет данных для других твитов", horizontalalignment="center",
                    #              verticalalignment="center")
                    # ax2.set_title("Аналитика остальных твитов")
                    # ax2.set_xlabel("Дата")
                    # ax2.grid(True)
                    # ax2.tick_params(axis="x", rotation=45)
                    #
                    # plt.tight_layout()
                    # st.pyplot(fig)

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
        continue

    with st.chat_message(role):
        st.markdown(content)
