
# Напишіть чат бота, який імітує замовлення піци. Чат бот
# має бути ввічливим. Реалізуйте функціонал:
# 1. Показати меню
# 2. Задати питання
# 3. Зробити замовлення
# 4. Змінити замовлення
# 5. Підтвердити замовлення(бот має показати суму
# замовлення)



from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    trim_messages
)

import json
import dotenv
import os
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
# завантажити api ключі з папки .env
dotenv.load_dotenv()

# отримати сам ключ
api_key = os.getenv('GEMINI_API_KEY')

# створення чат моделі
# Велика мовна модель(llm)

llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',  # назва моделі
    google_api_key=api_key,    # ваша API
)


# Меню піци з складом і розмірами
MENU = {
    "Маргарита": {
        "маленька": {"ціна": 120, "см": 25},
        "велика": {"ціна": 180, "см": 35},
        "склад": "Томатний соус, моцарела, базилік"
    },
    "Пепероні": {
        "маленька": {"ціна": 140, "см": 25},
        "велика": {"ціна": 200, "см": 35},
        "склад": "Томатний соус, моцарела, пепероні"
    },
    "4 сири": {
        "маленька": {"ціна": 160, "см": 25},
        "велика": {"ціна": 220, "см": 35},
        "склад": "Моцарела, дорблю, пармезан, чеддер"
    },
    "Гавайська": {
        "маленька": {"ціна": 150, "см": 25},
        "велика": {"ціна": 210, "см": 35},
        "склад": "Томатний соус, моцарела, шинка, ананас"
    },
    "Вегетаріанська": {
        "маленька": {"ціна": 130, "см": 25},
        "велика": {"ціна": 190, "см": 35},
        "склад": "Томатний соус, моцарела, болгарський перець, гриби, оливки, кукурудза"
    }
}

# Ініціалізація порожнього замовлення
order = {}

# Схема парсингу
schemas = [
    ResponseSchema(name='pizza', description='назва піци'),
    ResponseSchema(name='size', description='розмір піци (маленька / велика)')
]

parser = StructuredOutputParser.from_response_schemas(schemas)
instructions = parser.get_format_instructions()

prompt_extract = PromptTemplate.from_template(
    """
    Твоя задача — дістати назву піци і розмір із тексту замовлення.

    Текст: {text}

    Формат відповіді:
    {instructions}
    """,
    partial_variables={"instructions": instructions}
)

chain_extract = prompt_extract | llm | parser

# Формуємо меню текст
menu_text = """
1. Маргарита
Томатний соус, моцарела, базилік.
Маленька 25 см — 120 грн, велика 35 см — 180 грн

2. Пепероні
Томатний соус, моцарела, пепероні.
Маленька 25 см — 140 грн, велика 35 см — 200 грн

3. 4 сири
Моцарела, дорблю, пармезан, чеддер.
Маленька 25 см — 160 грн, велика 35 см — 220 грн

4. Гавайська
Томатний соус, моцарела, шинка, ананас.
Маленька 25 см — 150 грн, велика 35 см — 210 грн

5. Вегетаріанська
Томатний соус, моцарела, болгарський перець, гриби, оливки, кукурудза.
Маленька 25 см — 130 грн, велика 35 см — 190 грн
"""

# Системне повідомлення
system_prompt = """
Ти — ввічливий чат-бот для замовлення піци. 

Твої функції:
- показати меню
- прийняти замовлення
- змінити замовлення
- підтвердити замовлення і показати загальну суму

Меню:
{menu}

Формат відповіді:
- Будь чемним.
- Використовуй дружній тон.
- Якщо треба показати меню — виведи його у зручному вигляді.
- Якщо замовлення прийнято — підтверди його.
- Якщо підтверджено — виведи повне замовлення і суму.
"""


messages = [SystemMessage(system_prompt.format(menu=menu_text))]

# Тріммер
trimmer = trim_messages(
    strategy='last',  # залишати останні повідомлення
    token_counter=len,  # умовно рахуємо кількість повідомлень як довжину списку
    max_tokens=5,  # максимум 5 повідомлень
    start_on='human',  # історія починається з HumanMessage
    end_on='human',  # історія закінчується на HumanMessage
    include_system=True  # SystemMessage залишати
)

while True:
    user_input = input("Ви: ")
    if user_input == '':
        break

    messages.append(HumanMessage(user_input))
    messages = trimmer.invoke(messages)

    response = llm.invoke(messages)
    print(f"Бот: {response.content}")

    try:
        parsed = chain_extract.invoke({"text": response.content})
        pizza = parsed['pizza']
        size = parsed['size']
        if pizza in MENU and size in MENU[pizza]:
            order[pizza] = size
            print(f"Додано до замовлення: {pizza} ({size})")
    except:
        pass

    messages[0] = SystemMessage(system_prompt.format(menu=menu_text))
    messages.append(response)

    if "підтвердити" in user_input.lower():
        total = 0
        print("\nВаше замовлення:")
        for pizza_name, pizza_size in order.items():
            price = MENU[pizza_name][pizza_size]["ціна"]
            size_cm = MENU[pizza_name][pizza_size]["см"]
            print(f"- {pizza_name} ({pizza_size}, {size_cm} см) — {price} грн")
            total += price
        print(f"Загальна сума замовлення: {total} грн")
