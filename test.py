#
# Напишіть модель для генерації персонального плану
# тренувань з двох ланцюгів:
#  Перший ланцюг отримує мету тренування(схуднення,
# набір м’язів, тощо) та повертає список вправ
#  Другий ланцюг отримує список вправ, рівень
# підготовки користувача(низький, середній,
# професіонал) та кількість часу на тиждень(в годинах)
# і повертає план тренувань


from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

import dotenv
import os
import json

# завантажити api ключі з .env
dotenv.load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

# створення моделі
llm = GoogleGenerativeAI(
    model='gemini-2.0-flash',
    google_api_key=api_key,
)

# ---------- Перший ланцюг: генерація списку вправ ----------

# схема для першого ланцюга
schemas_exercises = [
    ResponseSchema(name='goal', description='мета тренування'),
    ResponseSchema(name='exercises', description='список рекомендованих вправ')
]

parser_exercises = StructuredOutputParser.from_response_schemas(schemas_exercises)
instructions_exercises = parser_exercises.get_format_instructions()

prompt_exercises = PromptTemplate.from_template(
    """
    Ти персональний тренер. Твоя задача — створити список вправ
    для користувача відповідно до його мети тренування.

    Мета тренування: {goal}

    Формат відповіді:
    {instructions}
    """,
    partial_variables={"instructions": instructions_exercises}
)

chain_exercises = prompt_exercises | llm | parser_exercises

# ---------- Другий ланцюг: генерація плану тренувань ----------

schemas_plan = [
    ResponseSchema(name='plan', description='детальний план тренувань на тиждень')
]

parser_plan = StructuredOutputParser.from_response_schemas(schemas_plan)
instructions_plan = parser_plan.get_format_instructions()

prompt_plan = PromptTemplate.from_template(
    """
    Ти персональний тренер. Твоя задача — створити детальний план
    тренувань на тиждень відповідно до рівня підготовки користувача,
    доступного часу та списку вправ.

    Рівень підготовки: {level}
    Час на тиждень (годин): {hours}
    Список вправ: {exercises}

    Формат відповіді:
    {instructions}
    """,
    partial_variables={"instructions": instructions_plan}
)

chain_plan = prompt_plan | llm | parser_plan

# ---------- Виклик ----------

# приклад вхідних даних
user_goal = "схуднення"
user_level = "середній"
user_hours = 4

# перший ланцюг — отримати список вправ
response_exercises = chain_exercises.invoke({
    "goal": user_goal
})

print("Список вправ:")
print(response_exercises['exercises'])

# другий ланцюг — отримати план тренувань
response_plan = chain_plan.invoke({
    "level": user_level,
    "hours": user_hours,
    "exercises": response_exercises['exercises']
})

print("\nПлан тренувань:")
print(response_plan['plan'])

