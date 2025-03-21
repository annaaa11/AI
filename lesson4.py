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

agent = initialize_agent(
    tools=[search],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # або інший тип агента
    #verbose=True
)

print(agent.invoke("[INST]Чий Крим?[\INST]"))

