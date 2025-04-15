from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3",
    temperature=0.7,
)

chat = ChatHuggingFace(llm=llm)

print('Start')
# print(chat.invoke([
#     {
#         "role": "user",
#         "content": "Hello, how are you?",
#     },
#     {
#         "role": "assistant",
#         "content": "I'm doing well, thank you for asking.",
#     },
#     {
#         "role": "user",
#         "content": "Can you tell me a joke?",
#     }
# ]))

# print(chat.invoke([
#     SystemMessage(content='You are chatbot that answer like captain Jack Sparrow'),
#     HumanMessage(content='Hello'),
# ]))

from  langchain_core.messages import trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState


def call_chat(state: MessagesState):
    messages = state['messages']
    response = chat.invoke(messages)
    return {'messages': response}


graph = StateGraph(state_schema=MessagesState)

graph.add_node('model', call_chat)

graph.add_edge(START, 'model')
graph.add_edge(START, 'model')

memory = MemorySaver()
app = graph.compile(checkpointer=memory)


state = {'messages': [
    SystemMessage('You are polite chat bot'),
    HumanMessage('Hello, my name is Anton')
]}

config = {"configurable": {"thread_id": '1'}}

print(app.invoke(state, config))

state2 = {'messages':[
    HumanMessage('What is my name?')
]}

trim_messages(state['messages'],
              )


print(app.invoke(state2, config))