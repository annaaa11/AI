
# Напишіть чат бота, з інструментом по рекомендації
# ресторанів.
# Для цього скористайтесь
# GoogleSerperAPIWrapper(type="places")
# Інструмент повинен отримувати запит для пошуку та
# повертати таку інформацію про ресторани:
#  назва
#  посилання на сайт(якщо є)
#  рейтинг

################

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.utilities import GoogleSerperAPIWrapper
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)

import dotenv
import os

# завантаження API ключів з .env
dotenv.load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')
api_key_serper = os.getenv('SERPER_API_KEY')

# створення LLM моделі
llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    google_api_key=api_key,
)

# створення інструмента для пошуку ресторанів
searcher = GoogleSerperAPIWrapper(api_key_serper=api_key_serper, type="places")

def restaurant_recommendation(query: str):
    """
    Пошук ресторанів за запитом користувача.

    :param query: str, запит користувача (типу "ресторан Київ піца")
    :return: list, інформація про знайдені ресторани
    """
    result = searcher.results(query)
    #print(result)

    restaurants = []
    if 'places' in result:
        for place in result['places']:
            new_data = {'title': place['title'], 'website': place['website'], 'rating': place['rating']}
            #print(new_data)
            restaurants.append(new_data)

    return restaurants

# створення агента
agent = create_react_agent(
    model=llm,
    tools=[restaurant_recommendation]
)

# початкове повідомлення системи
messages = {"messages": [
    SystemMessage("""
    Ти чат-бот, який допомагає рекомендувати ресторани.
    Твоя задача — отримати від користувача запит для пошуку ресторанів
    і видати перелік ресторанів з назвою, посиланням на сайт (якщо є)
    та рейтингом. Якщо запит не про ресторан або місце — виведи повідомлення:
    «немає відповідної інформації».
    Кожен ресторан виводь з нового рядка у такому форматі:
    Назва: ...
    Сайт: ...
    Рейтинг: ...
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
