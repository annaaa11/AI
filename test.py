# # пошук потрібного документа
# # RAG -- (пошук - відповідь - генерація)
#
# # документ1 -- Суп корисний при застуді
# # документ2 -- Суп придумали в Китаї
# # документ3 -- Бігати більше 10 км шкідливо для здоров'я
#
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from pinecone import Pinecone, ServerlessSpec
# from langchain_core.documents import Document
# from langchain_pinecone import PineconeVectorStore
#
# import dotenv
# import os
# from uuid import uuid4
#
# # завантажити api ключі з папки .env
# dotenv.load_dotenv()
#
# # отримати сам ключ
# api_key = os.getenv('GEMINI_API_KEY')
# pinecone_api_key = os.getenv('PINECONE_API_KEY')
#
# # створення моделі для кодування текстів
# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/text-embedding-004",  # назва моделі
#     google_api_key=api_key
# )
#
# # # кодування тексту
# # vec1 = embeddings.embed_query('Фільм чудовий')
# # vec2 = embeddings.embed_query('Цей фільм чудовий')
# # vec3 = embeddings.embed_query('Дуже хороший фільм')
# #
# # # закодовані числа(вектори)
# # print(vec1)
# # print(vec2)
# # print(vec3)
# #
# # # кількість чисел у векторі
# # print(len(vec1))  # 768
#
# # створення векторної бази даних
# pc = Pinecone(api_key=pinecone_api_key)
#
# # назва таблиці з документами
# index_name = "itstep"
#
# if not pc.has_index(index_name):
#     pc.create_index(
#         name=index_name,   # назва таблиці
#         dimension=768,     # кількість чисел у векторі
#         metric="cosine",   # формула для обрахунку схожості
#         spec=ServerlessSpec(
#             cloud="aws",         # хмарна платформа(Амазон)
#             region="us-east-1"   # регіон де знаходиться сервер(впливає на оплату)
#         )
#     )
#
# index = pc.Index(index_name)
#
# vector_store = PineconeVectorStore(
#     index=index,
#     embedding=embeddings
# )
#
# # створення документів
# # документ1 -- Суп корисний при застуді
# doc1 = Document(
#     page_content="Суп корисний при застуді",  # вміст документа
#     # мета дані(додаткова інформація)
#     metadata={
#         'type': "здоров'я",
#         'author': "Anton Halysh"
#     }
# )
#
# # документ2 -- Суп придумали в Китаї
# doc2 = Document(
#     page_content="Суп придумали в Китаї",  # вміст документа
#     # мета дані(додаткова інформація)
#     metadata={
#         'type': "історія",
#         'author': "Anton Halysh",
#         'date': "2025"
#     }
# )
#
# # документ3 -- Бігати більше 10 км шкідливо для здоров'я
# doc3 = Document(
#     page_content="Бігати більше 10 км шкідливо для здоров'я",  # вміст документа
#     # мета дані(додаткова інформація)
#     metadata={
#         'type': "здоров'я",
#         'author': "Anton Halysh"
#     }
# )
#
# # добавляння документів
# # створити ID для документів
#
# docs = [doc1, doc2, doc3]
# ids = [str(uuid4()) for _ in range(len(docs))]
#
# #vector_store.add_documents(docs, ids=ids)
#
# # як дістати потрібний документ
# user_text = 'Розкажи щось цікаве про суп'
#
# # пошук схожого документа
# docs = vector_store.similarity_search(
#     user_text,   # запит від користувача
#     k=2,          # кількість документів
#     # фільтр по метаданих
#     filter={'type': "здоров'я"}  # документи про здоров'я
# )
#
# for doc in docs:
#     print(doc)

# Завдання 1
# Створіть векторну базу даних, де кожен документ – це
# вміст файлу з папки data/lesson_rag/files
#  добавте в метадані шлях до файлу
#  створіть для кожного документу ID
#  збережіть створені ID та назви відповідних файлів в
# окремий json файл
# Перевірте чи працює правильно пошук
#
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from pinecone import Pinecone, ServerlessSpec
# from langchain_core.documents import Document
# from langchain_pinecone import PineconeVectorStore
# import json
# import dotenv
# import os
# from uuid import uuid4
#

#
# # завантажити api ключі з папки .env
# dotenv.load_dotenv()
#
# # отримати сам ключ
# api_key = os.getenv('GEMINI_API_KEY')
# pinecone_api_key = os.getenv('PINECONE_API_KEY')
#
# # створення моделі для кодування текстів
# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/text-embedding-004",  # назва моделі
#     google_api_key=api_key
# )
#
#
#
# # створення векторної бази даних
# pc = Pinecone(api_key=pinecone_api_key)
#
# # назва таблиці з документами
# index_name = "task1"
#
# if not pc.has_index(index_name):
#     pc.create_index(
#         name=index_name,   # назва таблиці
#         dimension=768,     # кількість чисел у векторі
#         metric="cosine",   # формула для обрахунку схожості
#         spec=ServerlessSpec(
#             cloud="aws",         # хмарна платформа(Амазон)
#             region="us-east-1"   # регіон де знаходиться сервер(впливає на оплату)
#         )
#     )
#
# index = pc.Index(index_name)
#
# vector_store = PineconeVectorStore(
#     index=index,
#     embedding=embeddings
# )
#

# file_names = os.listdir("data/lesson_rag/files")
# print(file_names)
# user_docs = []
# for file_name in file_names:
#     file_name = f"data/lesson_rag/files/{file_name}"
#     with open(file_name, "r", encoding="UTF-8") as file:
#         data = file.read()
#         doc = Document(
#             page_content=data,
#             metadata= {"file_path": file_name}
#                     )
#         user_docs.append(doc)
#
# #print(user_docs)
# ids = [str(uuid4()) for _ in range(len(file_names))]
# id_data = {}
# for i in range(len(ids)):
#     id = ids[i]
#     name_file = file_names[i]
#     id_data[name_file] = id
#
# # print(id_data)
# with open('data_ai.json', 'w' ) as file:
#     json.dump(id_data, file)
#
# vector_store.add_documents(user_docs, ids=ids)




# Завдання 2
# На основі створеної бази даних створіть агента та
# реалізуйте його у вигляді чат бота

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.utilities import GoogleSerperAPIWrapper
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)

import json
import dotenv
import os
from uuid import uuid4



# завантажити api ключі з папки .env
dotenv.load_dotenv()

# отримати сам ключ
api_key = os.getenv('GEMINI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')

# створення моделі для кодування текстів
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",  # назва моделі
    google_api_key=api_key
)



# створення векторної бази даних
pc = Pinecone(api_key=pinecone_api_key)

# назва таблиці з документами
index_name = "task1"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,   # назва таблиці
        dimension=768,     # кількість чисел у векторі
        metric="cosine",   # формула для обрахунку схожості
        spec=ServerlessSpec(
            cloud="aws",         # хмарна платформа(Амазон)
            region="us-east-1"   # регіон де знаходиться сервер(впливає на оплату)
        )
    )

index = pc.Index(index_name)

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)




# завантаження API ключів з .env
dotenv.load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')
api_key_serper = os.getenv('SERPER_API_KEY')

# створення LLM моделі
llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    google_api_key=api_key,
)

def doc_ser(user_text: str ):
    '''
    Ищет документы, которые подходят под вопрос пользователя. Это база данных документов
    про искусственный интеллект.

    :param user_text: str, вопрос пользователя
    :return: list[Document], cписок найденных документов
    '''
    print("doc_ser")
    docs = vector_store.similarity_search(user_text,k=2)
    return docs



# створення агента
agent = create_react_agent(
    model=llm,
    tools=[doc_ser]
)

# початкове повідомлення системи
messages = {"messages": [
    SystemMessage("""
    Ты чат-бот, который дает ответы на вопросы пользователя про искусственный интеллект.
    Давай приоритет данным, полученным с помощью "doc_ser". Если ты не можешь найти ответ 
    с помощью "doc_ser", то давай ответ самостоятельно.
    
    """)
]}


# основний цикл чату
while True:
    user_input = input("Ви: ")

    if user_input.strip() == '':
        break

    user_message = HumanMessage(user_input)
    messages["messages"].append(user_message)

    messages = agent.invoke(messages)

    # відповідь бота
    ai_message = messages["messages"][-1]
    print(ai_message.content)