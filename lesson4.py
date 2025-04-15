from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.tools.google_search import GoogleSearchRun
import dotenv


#search = GoogleSearchRun()
#tools = [Tool(name="Search", func=search.run, description="Шукає інформацію в Google")]


dotenv.load_dotenv()
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.7,
    max_new_tokens=200,
    #frequency_penalty=1.2,
)


# agent = initialize_agent(
#     tools=tools,
#     llm=llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # або інший тип агента
#     verbose=True
# )


#response = agent.invoke("Яка столиця Франції?")
#print(response)

from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

# agent = initialize_agent(
#     tools=[search],
#     llm=llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # або інший тип агента
#     #verbose=True
# )

# response = agent.invoke("Whose is Crimea?")
#
# print(response)
# print(llm.invoke(f'[INST]Переклади на українську: {response}[/INST]'))
#

from langchain.tools import Tool
from langchain_core.tools import tool
from langchain.agents import initialize_agent
import re

# @tool
# def check_password_strength(password: str) -> str:
#     """
#     Check password strength
#     :param password: user password
#     :return: response with detailed description
#     """
#     issues = []
#
#     if len(password) < 8:
#         issues.append("❌ Пароль занадто короткий (менше 8 символів).")
#     else:
#         issues.append("✅ Довжина пароля достатня.")
#
#     if not any(c.islower() for c in password) or not any(c.isupper() for c in password):
#         issues.append("❌ Пароль повинен містити літери у різних регістрах.")
#     else:
#         issues.append("✅ Є літери у верхньому та нижньому регістрах.")
#
#     if not any(c.isdigit() for c in password):
#         issues.append("❌ Пароль повинен містити хоча б одну цифру.")
#     else:
#         issues.append("✅ Є цифри.")
#
#     if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
#         issues.append("❌ Пароль повинен містити хоча б один спеціальний символ.")
#     else:
#         issues.append("✅ Є спеціальні символи.")
#
#     return "\n".join(issues)
#
#
# # Створення інструменту для агента
# password_tool = Tool(
#     name="Password Strength Checker",
#     func=check_password_strength,
#     description="Перевіряє складність паролю за кількома критеріями"
# )
#
# # Створення агента
# agent = initialize_agent(
#     tools=[check_password_strength],
#     llm=llm,
#     agent="zero-shot-react-description",
#     verbose=False
# )
#
# # Тест агента
# response = agent.run("[INST]Перевір пароль: MySecurePss[/INST]")
# print(response)


from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

chat = ChatHuggingFace(llm=llm)

messages = [
    ("system", "You are a helpful translator. Translate the user sentence to French."),
    ("human", "I love programming."),
]

print(chat.invoke(messages))