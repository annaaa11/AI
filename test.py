# # # Напишіть промпт для створення плану навчального
# # # курсу з певної теми для цільової айдиторії(початківці,
# # # професіонали, діти, тощо).
# # # Вхідні параметри: тема, опис цільової аудиторії
# # # Реалізуйте двома способами:
# # #  Zero-shot
# # #  Few-shot
# #
# #
# # from langchain_google_genai import GoogleGenerativeAI
# # from langchain.prompts import PromptTemplate
# #
# # import dotenv
# # import os
# #
# # # завантажити api ключі з папки .env
# # dotenv.load_dotenv()
# #
# # # отримати сам ключ
# # api_key = os.getenv('GEMINI_API_KEY')
# #
# # # створення моделі
# # # Велика мовна модель(llm)
# #
# # llm = GoogleGenerativeAI(
# #     model='gemini-2.0-flash',  # назва моделі
# #     google_api_key=api_key,  # ваша API
# # )
# #
# #
# # #  Zero-shot
# # prompt1 = PromptTemplate.from_template("""
# # Створи детальний план навчального курсу.
# #
# # Тема: {тема}
# #
# # Цільова аудиторія: {аудиторія}
# #
# # Вимоги:
# # - Курс має відповідати рівню підготовки аудиторії.
# # - Поділи курс на модулі або тижні.
# # - Для кожного модуля додай короткий опис і ключові підтемі.
# # - За потреби запропонуй формати занять (лекція, практика, проєкт, тест).
# # - Завершуй курс фінальним проєктом або тестом знань.
# #
# # Почни з назви курсу.
# # """)
# #
# #
# # #  Few-shot
# # prompt2 = PromptTemplate.from_template("""
# # Створи план навчального курсу за заданою темою і цільовою аудиторією. Нижче приклад структури:
# #
# # ---
# # Тема: Основи програмування
# # Аудиторія: Діти 10-13 років
# #
# # Назва курсу: "Програмування — це цікаво!"
# # 1. Вступ до програмування
# #    - Що таке комп’ютери та як вони працюють
# #    - Перша програма в Scratch
# # 2. Логіка і алгоритми
# #    - Що таке алгоритм
# #    - Ігри для тренування логічного мислення
# # 3. Змінні та цикли
# #    - Введення у змінні
# #    - Створення циклів у Scratch
# # ...
# # Фінальний проєкт: Створити власну гру в Scratch
# #
# # ---
# #
# # Тепер створи подібний курс за наступними параметрами:
# #
# # Тема: {тема}
# #
# # Аудиторія: {аудиторія}
# #
# # Дотримуйся схожої структури: назва курсу, модулі, підтемі, фінальний проєкт.
# # """)
# #
# #
# # chain1 = prompt1 | llm
# # chain2 = prompt2 | llm
# #
# # response1 = chain1.invoke({
# #     'тема': 'Основи кібербезпеки',
# #     'аудиторія': 'Студенти першого курсу ІТ-спеціальностей'
# # })
# # response2 = chain1.invoke({
# #     'тема': 'Основи кібербезпеки',
# #     'аудиторія': 'Студенти першого курсу ІТ-спеціальностей'
# # })
# #
# # print(response1)
# # print(response2)
# #
# # from langchain_google_genai import GoogleGenerativeAI
# # from langchain.prompts import PromptTemplate
# # from langchain.output_parsers import ResponseSchema, StructuredOutputParser
# #
# # import json
# # import dotenv
# # import os
# #
# # # завантажити api ключі з папки .env
# # dotenv.load_dotenv()
# #
# # # отримати сам ключ
# # api_key = os.getenv('GEMINI_API_KEY')
# #
# # # створення моделі
# # # Велика мовна модель(llm)
# #
# # llm = GoogleGenerativeAI(
# #     model='gemini-2.0-flash',  # назва моделі
# #     google_api_key=api_key,  # ваша API
# # )
# #
# # # Користувач задає питання по книзі
# # # Ваша задача:
# # # 1. Дати відповідь на питання
# # # 2. Порекомендувати схожі книги(на ту саму тему, того ж автора, жанру, ...)
# #
# # # має назву книги і питання -- хочемо отримати всю інформацію про книгу
# # # і відповідь на питання
# #
# # # маючи інформацію про книгу порекомендувати щось схоже
# #
# #
# # # ------------------------------
# #
# # # схема для відповідей
# # schemas = [
# #     ResponseSchema(name='answer', description='відповідь на питання користувача'),
# #     ResponseSchema(name='theme', description='головна тема книги'),
# #     ResponseSchema(name='author', description='автор книги'),
# #     ResponseSchema(name='jaunre', description='жанр книги')
# # ]
# #
# # # створення парсер
# # parser = StructuredOutputParser.from_response_schemas(schemas)
# #
# # # отримати інструкція для llm
# # instructions = parser.get_format_instructions()
# #
# # # print(instructions)
# #
# # # створення промпта
# # prompt = PromptTemplate.from_template(
# #     """
# #     Ти асистент онлайн книгарні. Твоя задача давати відповіді
# #     на питання користувачів. Відповіді мають бути чіткі та інформативні.
# #     Загальний стиль спілкування ввічливий, іноді можеш використовувати
# #     неформальний стиль.
# #     Також ти повинен визначити параметри книжки про яку питає
# #     користувач(наприклад жанр, автор, тема, ...)
# #
# #     Питання: {question}
# #
# #     Формат відповіді:
# #     {instructions}
# #     """,
# #     partial_variables={"instructions": instructions}
# # )
# #
# # # створення ланцюга
# # chain = prompt | llm | parser
# #
# # response = chain.invoke({
# #     "question": "Коли була написана книга 1984",
# # })
# #
# # print(response)
# # print(response['theme'])
# # print(type(response))
# #
# # # # збереження у файл
# # # with open('response.json', 'w', encoding='UTF-8') as file:
# # #     json.dump(response, file)
# # #
# # # # завантаження з файлу
# # # with open('response.json', 'r', encoding='UTF-8') as file:
# # #     new_response = json.load(file)
# # #
# # # print(new_response)
# #
# #
# # # рекомендація схожих книг
# # prompt = PromptTemplate.from_template(
# #     """
# #     Ти асистент онлайн книгарні. Твоя задача давати рекомендації книг
# #     користувачам певно жанру, теми та автора. Запропонуй по 3-5 книг по
# #     кожному пункту.
# #     Загальний стиль спілкування ввічливий, іноді можеш використовувати
# #     неформальний стиль.
# #
# #     Жанр: {jaunre}
# #     Автор: {author}
# #     Тема: {theme}
# #
# #     Відповідь дай у вигляді списку, познач які книги до якого
# #     пункту відносяться
# #     * Книги на схожу тему
# #     * Книги того ж автора
# #     * Книги того ж жанру
# #     """
# # )
# #
# # chain_recommendation = prompt | llm
# #
# # recommendation = chain_recommendation.invoke({
# #     "jaunre": response['jaunre'],
# #     "author": response['author'],
# #     "theme": response['theme'],
# # })
# #
# # # print(recommendation)
# #
# # # дістати всі назви книг з рекомендації
# #
# # schemas = [
# #     ResponseSchema(name='books', description='список з назвами книг')
# # ]
# #
# # parser = StructuredOutputParser.from_response_schemas(schemas)
# # instructions = parser.get_format_instructions()
# #
# # prompt = PromptTemplate.from_template(
# #     """
# #     Твоя задача дістати назви усіх книг з тексту.
# #
# #     Текст: {text}
# #
# #     Формат відповіді:
# #     {instructions}
# #     """,
# #     partial_variables={"instructions": instructions}
# # )
# #
# # chain_book_selector = prompt | llm | parser
# #
# # response = chain_book_selector.invoke({
# #     "text": recommendation
# # })
# #
# # print(response)
# #
# # for book in response['books']:
# #     print(book)
#
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import PromptTemplate
# from langchain_core.messages import (
#     HumanMessage,
#     AIMessage,
#     SystemMessage,
#     trim_messages
# )
#
#
# import json
# import dotenv
# import os
#
# # завантажити api ключі з папки .env
# dotenv.load_dotenv()
#
# # отримати сам ключ
# api_key = os.getenv('GEMINI_API_KEY')
#
# # створення чат моделі
# # Велика мовна модель(llm)
#
# llm = ChatGoogleGenerativeAI(
#     model='gemini-2.0-flash',  # назва моделі
#     google_api_key=api_key,    # ваша API
# )
#
# # response = llm.invoke("Привіт")
# #
# # print(type(response))
# # print(response)
# # print(repr(response))
#
#
# # # історія спілкування(чат)
# # messages = [
# #     SystemMessage("""
# #     Ти ввічливий чат бот. Твоя заача давати відповіді на питання
# #     користувача.
# #
# #     Закінчуй усі відповіді фразою "Чи залишились іще питання?"
# #     """),
# #     HumanMessage('Привіт'),
# #     AIMessage('Привіт, чим можу допомогти? Чи залишились іще питання?'),
# #     HumanMessage('Яка столиця Франції?'),
# #     AIMessage('Париж. Чи залишились іще питання?'),
# #     HumanMessage("Розкажи пару фактів про це місто")
# # ]
# #
# # response = llm.invoke(messages)
# #
# # print(response.content)
#
# # простий чат бот
#
# # messages = [
# #     SystemMessage("""
# #     Ти ввічливий чат бот. Твоя заача давати відповіді на питання
# #     користувача.
# #
# #     Закінчуй усі відповіді фразою "Чи залишились іще питання?"
# #     """)
# # ]
# #
# # # основний цикл
# # while True:
# #     # отримати повідомлення від користувача
# #     user_text = input('Ви: ')
# #
# #     # умова зупинки
# #     if user_text == '':
# #         break
# #
# #     # змінити тип даних на HumanMessage
# #     human_message = HumanMessage(user_text)
# #
# #     # змінити історію
# #     messages.append(human_message)
# #
# #     # застосувати модель
# #     response = llm.invoke(messages)
# #     # response -- AIMessage
# #
# #     # змінити історію
# #     messages.append(response)
# #
# #     # вивід результату
# #     print(f'AI: {response.content}')
#
# # очищення історії повідомлень
#
# # створення трімер
# trimmer = trim_messages(
#     strategy='last',  # залишати останні повідомлення
#
#     token_counter=len,  # рахуємо кількість повідомлень
#     max_tokens=5,  # залишати максимум 5 повідомлення(System, AI, Human)
#
#     start_on='human',  # історія завжди починатиметься з HumanMessage
#     end_on='human',  # історія завжди закінчуватиметься з HumanMessage
#     include_system=True  # SystemMessage не чіпати
# )
#
# messages = [
#     SystemMessage("""
#     Ти ввічливий чат бот. Твоя заача давати відповіді на питання
#     користувача.
#
#     Закінчуй усі відповіді фразою "Чи залишились іще питання?"
#     """)
# ]
#
# # # основний цикл
# # while True:
# #     # отримати повідомлення від користувача
# #     user_text = input('Ви: ')
# #
# #     # умова зупинки
# #     if user_text == '':
# #         break
# #
# #     # змінити тип даних на HumanMessage
# #     human_message = HumanMessage(user_text)
# #
# #     # змінити історію
# #     messages.append(human_message)
# #
# #     # почистити історію
# #     messages = trimmer.invoke(messages)
# #
# #     # застосувати модель
# #     response = llm.invoke(messages)
# #     # response -- AIMessage
# #
# #     # змінити історію
# #     messages.append(response)
# #
# #     # вивід результату
# #     print(f'AI: {response.content}')
# #
# #     # вивід історії
# #     print('\nІсторія')
# #     for message in messages:
# #         print(repr(message))
# #     print()
#
#
# # теж саме через ланцюг
# chain = trimmer | llm
#
# while True:
#     # отримати повідомлення від користувача
#     user_text = input('Ви: ')
#
#     # умова зупинки
#     if user_text == '':
#         break
#
#     # змінити тип даних на HumanMessage
#     human_message = HumanMessage(user_text)
#
#     # змінити історію
#     messages.append(human_message)
#
#     # застосувати модель
#     response = chain.invoke(messages)
#     # response -- AIMessage
#
#     # змінити історію
#     messages.append(response)
#
#     # вивід результату
#     print(f'AI: {response.content}')
#
#     # вивід історії
#     print('\nІсторія')
#     for message in messages:
#         print(repr(message))
#     print()


# Напишіть чат бота, який допомагає у вивченні
# англійської мови з наступним функціоналом:
#  якщо користувач просить перекласти слово або фразу
# то дається переклад слова та приклад використання в
# реченні
#  якщо користувач просить перекласти речення, то
# дається переклад самого речення, а також пояснення
# граматики, наприклад структура there is\are, питання в
# різних часових формах, тощо.
# Приклади реалізуйте як HumanMessage та AIMessage

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

messages = [prompt.invoke({'learn_words': learn_words})]

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

    messages[0] = prompt.invoke({'learn_words': learn_words})


    messages.append(response)

    print(response.content)
    print(eng_words)