
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.prompts import PromptTemplate
from langchain.output_parsers import JsonOutputToolsParser
import os
import dotenv

dotenv.load_dotenv()
api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
repo_id = 'microsoft/Phi-3-mini-4k-instruct'
#repo_id = 'meta-llama/Llama-3.2-1B-Instruct'

prompt = PromptTemplate.from_template(
    '<s>[INST]Ти помічник в написанні коду. Твоя задача генерувати лише код за запитом. Пояснення не потрібні[/INST]</s>'
    ""

    '[INST]Створи клас на {language} про {animal}'
    "```YOUR ANSWEAR ```[/INST]"
)

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.7,
    max_new_tokens=200,
    frequency_penalty=1.2,
)
#
#
# model = prompt | llm
#
# result = model.invoke({
#     'language': 'с++',
#     'animal': 'cat'
# })
#
# print(result)



# Завдання 1

def generate_code_zero_shot(language: str, description: str) -> str:
    return f"""
    [INST]
    Ти – досвідчений програміст. Напиши функцію мовою {language}, яка вирішує наступну задачу: {description}.
    [/INST]
    """


def generate_code_few_shot(language: str, description: str) -> str:
    return """
    [INST]
    Ти – досвідчений програміст. Напиши функцію мовою {language}, яка вирішує наступну задачу: {description}.

    Приклад 1:
    Вхідні дані: Python, "Знайти суму всіх елементів списку"
    Вихідний код:
    ```python
def sum_list(lst):
    return sum(lst)
    ```

    Приклад 2:
    Вхідні дані: JavaScript, "Піднести число до степеня"
    Вихідний код:
    ```javascript
function power(base, exponent) {
    return Math.pow(base, exponent);
}
```

Використай це для створення коду.[/INST]
""" + f"Вхідні дані: {language}, {description}"

# Завдання 2

def rephrase_text_zero_shot(text: str) -> str:

    return f"""
    [INST]
    Ти – редактор текстів. Переормулюй цей текст у формальному стилі:
    {text}
    [/INST]
    """

def rephrase_text_few_shot(text: str) -> str:

    return f"""
    [INST]
    Ти – реактор текстів. Переформулюй цей текст у формальному стилі.
    
    Приклад 1:
    Неформальний: "Привіт, ти можеш допомогти мені з цим питанням?"
    Формальний: "Добрий день, чи могли б Ви допомогт
     мені з цим питанням?"
    
    Приклад 2:
    Неформальний: "Я не встигаю доробити звіт, бо був зайнятий."
    Формальний: "Я не встиг завершити звіт, оскільки ма
     інші важлив
     за
    дання."
    
    Неформальний: {text}
    Формальний:[/INST]
    """

# Завдання 3

def compare_phones_zero_shot(product_descriptions: str, user_request):
    return f"""
    [INST]
    Ти – консультант із вибору смартфонів. Порівняй два телефони за їх характеристиками, використовуючи такі дані:
    {product_descriptions}

    Запит користувача:
    {user_request}
    [/INST]
    """


def compare_phones_chain_of_thoughts(services: str, user_query: str) -> str:
    return f"""
    [INST]
    Вам надано список SPA-послуг:
    {services}
    
    Проаналізуйте запит користувача та виконайте наступні кроки:
    1. Визначте, які потреби чи побажання має користувач.
    2. Перевірте, які послуги найбільше відповідають цим потребам.
    3. Виберіть найбільш підходящу послугу та обґрунтуйте свій вибір.
    
    Запит користувача: {user_query}
    
    Формат відповіді:
    - Назва послуги
    - Пояснення вибору, враховуючи запит користувача
    [/INST]
    """

# Завдання 4

def summarize_resume(resume: str) -> str:
    return """
    [INST]
    Ти – HR-аналітик. Підсумуй резюме у вигляді словника. Використай структуру:
    ```json
    {
        "Ім'я": "...",
        "Досвід роботи": "...",
        "Навички": "...",
        "Освіта": "...",
        "Контакти": "..."
    }
    ```

    Резюме:""" + f"""{resume}
    [/INST]
    """


with open('data/lesson10/products.txt', encoding='utf-8') as file:
    doc = file.read()


prompt = compare_phones_chain_of_thoughts(doc, "Масаж спини але недовго")

template = PromptTemplate.from_template("""
[INST]Підсумуйте резюме у вигляді словника. У словнику мають бути наступні ключі:
- 'Освіта': інформація про освіту.
- 'Досвід': професійний досвід.
- 'Навички': ключові навички.
- 'Мови': мови, якими володіє кандидат.
- 'Зв'язок': контактна інформація.

Резюме: {resume}[/INST]
""")

llm = ChatHuggingFace(llm=llm, verbose=False)
chain = template | llm

# Вхідне резюме для аналізу
resume_text = """
Ім'я: Антон
Контакт: телефон: +380123456789, email: anton@example.com
Навички: Python, машинне навчання, комп'ютерний зір
Досвід роботи:
1. Компанія: XYZ, посада: Data Scientist, тривалість: 3 роки, обов'язки: аналіз даних, розробка моделей.
2. Компанія: ABC, посада: Junior Developer, тривалість: 1 рік, обов'язки: розробка веб-додатків.
Освіта: Київський університет, ступінь: магістр, тривалість: 2015-2020.
Мови: Українська, англійська
Сертифікати: Сертифікат з машинного навчання, сертифікат з Python
Досягнення: переможець хакатону по машинному навчанню
"""

# Отримання результату
#result = chain.invoke({'resume': resume_text})
result = llm.invoke("What is the capital of France? Only one word")
# Виведення результату
print(result)

