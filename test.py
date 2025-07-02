import os
import json
import dotenv
import streamlit as st
from uuid import uuid4

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langgraph.prebuilt import create_react_agent


dotenv.load_dotenv()


api_key = os.getenv("GEMINI_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")

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

json_path = "data_ai.json"
if os.path.exists(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        id_data = json.load(f)
else:
    id_data = {}

# ====== LLM и агент ======

secrets = st.secrets.get('GEMINI_API_KEY')

llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',  # назва моделі
    google_api_key=secrets,  # ваша API
)

def doc_ser(user_text: str):
    """
    Функция поиска документов по запросу пользователя через векторную базу.

    :param user_text: Текст запроса от пользователя
    :return: Список документов (Document), наиболее похожих на запрос
    """
    docs = vector_store.similarity_search(user_text, k=3)
    return docs

agent = create_react_agent(
    model=llm,
    tools=[doc_ser]
)

# ====== Streamlit ======

#st.set_page_config(page_title="Адмінка Векторної Бази Даних", layout="wide")
st.title("Адміністрація Векторної Бази Даних")

# ==== Додавання нового файлу  ====

st.subheader("Додати новий документ")

uploaded_file = st.file_uploader("Завантажте файл (формат .txt)", type=["txt"])

if uploaded_file is not None:
    # Читаємо текст з файлу
    text = uploaded_file.read().decode("utf-8")

    # Розбиваємо на "сирі" блоки по трьох порожніх рядках
    raw_blocks = text.split("\n\n\n")

    blocks = []
    for block in raw_blocks:
        block = block.strip()
        if block != "":
            blocks.append(block)

    docs = []
    doc_ids = []
    new_id_data = {}

    for block in blocks:
        lines = block.splitlines()
        if not lines:
            continue
        block_title = lines[0].strip()
        content = block.strip()
        doc = Document(
            page_content=content,
            metadata={
                "file": uploaded_file.name,
                "block": block_title
            }
        )
        docs.append(doc)

        new_id = str(uuid4())
        doc_ids.append(new_id)

        key_name = f"{uploaded_file.name} | {block_title}"
        new_id_data[key_name] = new_id

    # Завантажуємо існуючий JSON або створюємо пустий словник
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as jf:
            existing_data = json.load(jf)
    else:
        existing_data = {}

    # Оновлюємо словник
    existing_data.update(new_id_data)

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(existing_data, jf)

    # Додаємо документи у векторну базу
    vector_store.add_documents(docs, ids=doc_ids)

    st.success(f"Додано {len(docs)} блок(ів) з файлу {uploaded_file.name} до бази.")

# ==== Чат з агентом і пошуком ====

st.subheader(" Чат з пошуком по векторній базі")



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


for massage in st.session_state['data']['messages']:
    if isinstance(massage, HumanMessage):
        role = "user"
    elif isinstance(massage, AIMessage):
        role = "bot"
    else:
        continue

    with st.chat_message(role):
        st.markdown(massage.content)