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

with open('data_ai.json', 'r' ) as file:
    id_data = json.load(file)

file_name = 'ai_ethics.txt'

id = id_data[file_name]

vector_store.delete([id])

with open(f"data/lesson_rag/files/{file_name}", 'r' , encoding="UTF-8") as file:
    data = file.read()
    doc = Document(
            page_content=data,
            metadata= {"file_path": file_name}
                    )
vector_store.add_documents([doc], ids=[id])


# for file_name in file_names:
#     file_name = f"data/lesson_rag/files/{file_name}"
#     with open(file_name, "r", encoding="UTF-8") as file:
#         data = file.read()
#         doc = Document(
#             page_content=data,
#             metadata= {"file_path": file_name}
#                     )
#         user_docs.append(doc)