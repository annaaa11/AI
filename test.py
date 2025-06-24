
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


# Напишіть чат бота, який допомагає у вивченні
# англійської мови з наступним функціоналом:
#  якщо користувач просить перекласти слово або фразу
# то дається переклад слова та приклад використання в
# реченні
#  якщо користувач просить перекласти речення, то
# дається переклад самого речення, а також пояснення
# граматики, наприклад структура there is\are, питання в
# різних часових формах, тощо.

# Модифікуйте попереднє завдання таким чином, щоб в
# SystemMessage
# користувачем.
# передавався
# список
# вивчених
# слів
# Для цього напишіть окрему модель яка буде діставати з
# відповіді(AIMessage) усі англійські слова(вважаємо що
# користувач знає лише ті слова, про які йому сказала модель).
# Список вивчених слів треба зберігати в json файлі та
# відвантажувати при запуску програми.
# Змініть функціонал таким чином:
#  якщо користувач просить перекласти слово або фразу
# то дається переклад слова та приклад використання в
# реченні з вивченими словами
#  якщо користувач просить перекласти речення, то
# додатково пояснюється значення невідомих слів


# схема для відповідей
schemas = [
    ResponseSchema(name='words', description='list of English words')

]

# створення парсер
parser = StructuredOutputParser.from_response_schemas(schemas)

# отримати інструкція для llm
instructions = parser.get_format_instructions()

prompt = PromptTemplate.from_template(
    """
    Твоя задача достать все английские слова из тексту.

    Текст: {text}

    Формат відповіді:
    {instructions}
    """,
    partial_variables={"instructions": instructions}
)

chain_eng_words = prompt | llm | parser

# response = chain_book_selector.invoke({
#     "text": recommendation
# })


trimmer = trim_messages(
    strategy='last',  # залишати останні повідомлення

    token_counter=len,  # рахуємо кількість повідомлень
    max_tokens=5,  # залишати максимум 5 повідомлення(System, AI, Human)

    start_on='human',  # історія завжди починатиметься з HumanMessage
    end_on='human',  # історія завжди закінчуватиметься з HumanMessage
    include_system=True  # SystemMessage не чіпати
)

prompt = SystemMessagePromptTemplate.from_template("""
    Ти — помічник з вивчення англійської мови. Твоя задача — перекладати слова або речення, а також надавати прості, зрозумілі пояснення граматики, якщо йдеться про речення.

    # Якщо користувач просить перекласти слово, ти:
    - даєш переклад
    - надаєш приклад речення з цим словом
    - даешь пример, используя знакомые слова

    # Якщо користувач просить перекласти речення, ти:
    - даєш переклад
    - пояснюєш граматичну структуру, яку вжито

    Використовуй просту мову. 

    Знакомые слова:{learn_words}

    Ось кілька прикладів:

    ---

    Користувач: Що означає слово genuine
    Асистент: Переклад: genuine — справжній, щирий.  
    Приклад: She gave me a **genuine** smile.  
    (Вона подарувала мені щиру усмішку.)

    ---

    Користувач: Переклади слово achievement і дай приклад у реченні.
    Асистент: Переклад: achievement — досягнення.  
    Приклад: Climbing Mount Everest was a great **achievement**.  
    (Сходження на Еверест було великим досягненням.)

    ---

    Користувач: Як буде відповідальність англійською
    Асистент: Переклад: responsibility.  
    Приклад: Taking care of a pet is a big **responsibility**.  
    (Піклування про домашню тварину — велика відповідальність.)

    ---

    Користувач: Переклади речення: Я щойно закінчив домашнє завдання.
    Асистент: Переклад: I have just finished my homework.  
    Граматика: Це **Present Perfect** — час, який ми використовуємо для дій, що щойно завершились. Структура: *have/has + дієслово в 3-й формі*.

    ---

    Користувач: На столі є дві книжки.
    Асистент: Переклад: There are two books on the table.  
    Граматика: Структура **there is/are** використовується, щоб сказати, що щось десь є. *There are* — для множини.

    ---

    Користувач: Як сказати: Якби я знав, я б тобі допоміг.
    Асистент: Переклад: If I knew, I would help you.  
    Граматика: Це умовне речення другого типу (Second Conditional), що описує уявну ситуацію. Структура: *if + past simple*, *would + інфінітив*.

    ---

    Користувач: Скажи англійською: Мене звати Оля, і я працюю вчителем.
    Асистент: Переклад: My name is Olya, and I work as a teacher.  
    Граматика: Це **Present Simple** — використовується для фактів і звичних дій. Структура: підмет + дієслово.

    ---

    Продовжуй у такому ж стилі.
    """)
learn_words = set()

prompt1 = """..."""
messages = [SystemMessage(prompt1.format(learn_words=learn_words))]

while True:
    message = input("Ви: ")

    if message == '':
        break

    human_mes = HumanMessage(message)

    messages.append(human_mes)

    messages = trimmer.invoke(messages)

    response = llm.invoke(messages)

    eng_words = chain_eng_words.invoke({
        "text": response.content
    })

    for word in eng_words['words']:
        learn_words.add(word.lower())

    # messages[0] = prompt.invoke({'learn_words': learn_words})

    messages[0] = SystemMessage(prompt1.format(learn_words=learn_words))

    messages.append(response)

    print(response.content)
    print(eng_words)