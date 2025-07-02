# Добавте в створену базу даних файл
# data/lesson_rag/huge_file.txt про умови користування гуглом
# Оскільки файл надто великий, то його треба добавляти
# частинами. Для цього:
#  прочитайте вміст файлу
#  розділіть його на окремі блоки(між блоками два
# порожніх рядка, дивись файл)
#  отримайте перший рядок кожного блоку – це його
# назва
#  створіть документи для кожного блоку. В метаданих:
# o назва файлу
# o назва блоку
#  створіть ID та добавте все в існуючу базу даних
#  добавте ID у json файл
#  перевірте агента

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_community.utilities import GoogleSerperAPIWrapper
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

import os
import dotenv
import json
from uuid import uuid4

# Завантаження ключів
dotenv.load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')

# Створення моделей
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=api_key
)

llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    google_api_key=api_key
)

# Підключення до Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index_name = "task1"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Додати великий файл частинами
huge_file_path = "data/lesson_rag/huge_file.txt"

if os.path.exists(huge_file_path):
    with open(huge_file_path, "r", encoding="utf-8") as f:
        huge_text = f.read()

    raw_blocks = huge_text.split("\n\n\n")
    blocks = []
    for block in raw_blocks:
        block = block.strip()
        if block != "":
            blocks.append(block)

    docs = []
    doc_ids = []
    id_data = {}

    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue
        block_title = lines[0].strip()
        content = "\n".join(lines).strip()
        doc = Document(
            page_content=content,
            metadata={
                "file_path": huge_file_path,
                "block_title": block_title
            }
        )
        docs.append(doc)
        new_id = str(uuid4())
        doc_ids.append(new_id)
        file_name = os.path.basename(huge_file_path)  # Получаем только имя файла без пути
        key_name = file_name + " | " + block_title  # Склеиваем имя файла и заголовок блока
        id_data[key_name] = new_id  # Добавляем в словарь ID по ключу

    # Додати до існуючого JSON
    json_path = "data_ai.json"
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as jf:
            existing_data = json.load(jf)
    else:
        existing_data = {}

    existing_data.update(id_data)

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(existing_data, jf, indent=2, ensure_ascii=False)

    # Додати у векторну базу
    vector_store.add_documents(docs, ids=doc_ids)
    print(f"Додано {len(docs)} блоків з huge_file.txt до векторної бази.")
else:
    print("Файл huge_file.txt не знайдено. Пропускаємо додавання.")

# Функція для пошуку документів
def doc_ser(user_text: str):
    """
    Шукає документи, які підходять під запит користувача.
    Це документи з бази даних про штучний інтелект.
    Повертає список документів, знайдених за допомогою пошуку
    """
    print("doc_ser")
    docs = vector_store.similarity_search(user_text, k=2)
    return docs

# Створення агента
agent = create_react_agent(
    model=llm,
    tools=[doc_ser]
)

# Початкове повідомлення системи
messages = {"messages": [
    SystemMessage("""
    Ти чат-бот, який відповідає на питання про штучний інтелект.
    Спочатку пробуй знайти відповіді у документах через "doc_ser".
    Якщо не знайдеш — дай відповідь сам.
    """)
]}

# Чат-цикл
while True:
    user_input = input("Ви: ")
    if user_input.strip() == '':
        print("👋 Завершення чату.")
        break

    user_message = HumanMessage(user_input)
    messages["messages"].append(user_message)

    messages = agent.invoke(messages)

    ai_message = messages["messages"][-1]
    print("Бот:", ai_message.content)
