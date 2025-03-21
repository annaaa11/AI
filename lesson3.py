from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.output_parsers import StrOutputParser
import dotenv


dotenv.load_dotenv()
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.7,
    max_new_tokens=200,
    #frequency_penalty=1.2,
)

# 2. Ланцюг для визначення теми питання
response_schema = [
    ResponseSchema(name='question', description='Питання задане користувачем'),
    ResponseSchema(name='topic', description="Категорія до якої належить питання")
]

parser = StructuredOutputParser.from_response_schemas(response_schema)
format_instructions = parser.get_format_instructions()

topic_prompt = PromptTemplate.from_template(
    template="Визнач, до якої категорії належить це питання: '{question}'. "
    "Вибери одну з: Наука, Історія, Технології."
    "Відповідай у наступному форматі:\n{format_instructions}",
    partial_variables={"format_instructions": format_instructions}
)

topic_chain = topic_prompt | llm | parser


answer_prompt = PromptTemplate.from_template(
    "Дай коротку відповідь на питання: {question}\n"
    "Порекомендуй інші цікаві теми з {topic} які пов'язані з питанням {question}. Наведи список з 3-5 речей, лише назви"
)

answer_chain = answer_prompt | llm | StrOutputParser()

chain = topic_chain | answer_chain

#print(chain.invoke({"question": "Коли була висадка на місяць?"}))



# -----------------------------------
# schemas = [
#     ResponseSchema(name="topic", description="Категорія питання (Наука, Історія, Технології)"),
#     ResponseSchema(name="short_answer", description="Коротка відповідь на питання"),
#     ResponseSchema(name="long_answer", description="Розгорнута відповідь на питання"),
# ]
#
# # 🔹 Ініціалізуємо StructuredOutputParser
# output_parser = StructuredOutputParser.from_response_schemas(schemas)
# format_instructions = output_parser.get_format_instructions()
#
# # 🔹 Створюємо промпт із форматуванням
# prompt = PromptTemplate(
#     template="Відповідай у наступному форматі:\n{format_instructions}\n\nПитання: {question}",
#     input_variables=["question"],
#     partial_variables={"format_instructions": format_instructions},
# )
#
# # 🔹 Ланцюг для отримання структурованих даних
# chain = prompt | llm | output_parser
#
# # 🔹 Тестуємо
# question = "Що таке квантова механіка?"
# result = chain.invoke({"question": question})
#
# # 🔹 Виводимо результат
# print(result)

# skills_schema = [
#     ResponseSchema(name="job_description", description="Опис вакансії"),
#     ResponseSchema(name="skills", description="Ключові навички, необхідні для вакансії")
# ]
#
# skills_parser = StructuredOutputParser.from_response_schemas(skills_schema)
# format_instructions = skills_parser.get_format_instructions()
#
# skills_prompt = PromptTemplate.from_template(
#     "Витягни ключові навички з вакансії: '{job_description}'.\n"
#     "Відповідай у наступному форматі:\n{format_instructions}",
#     partial_variables={"format_instructions": format_instructions}
# )
#
# skills_chain = skills_prompt | llm | skills_parser
#
# # 🔹 Генерація резюме
# resume_prompt = PromptTemplate.from_template(
#     "Склади резюме для кандидата з такими навичками: {skills}.\n"
#     "Опис кандидата: {candidate_description}."
# )
#
# resume_chain = resume_prompt | llm | StrOutputParser()
#
# # 🔹 Об'єднаний ланцюг
# resume_generation_chain = skills_chain | resume_chain
#
# # 🔹 Тест
# result = resume_generation_chain.invoke({
#     "job_description": "Python-розробник, знання Flask, SQL, Docker.",
#     "candidate_description": "3 роки досвіду в бекенді, розробка REST API."
# })
# print(result)
