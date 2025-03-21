from langchain_huggingface import HuggingFaceEndpoint
import os
import dotenv


dotenv.load_dotenv()
api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"


llm = HuggingFaceEndpoint(
    repo_id=repo_id,  # ID моделі на Hugging Face (наприклад, "mistralai/Mistral-7B-Instruct-v0.1")
    temperature=0.1,  # Впливає на випадковість генерації (нижчі значення → більш передбачувані відповіді)
    max_new_tokens=50,  # Максимальна кількість нових токенів, які модель може згенерувати
    top_p=0.8,  # Nucleus sampling: модель вибирає токени, сукупна ймовірність яких становить 80%
    top_k=10,  # Модель розглядає лише 10 найімовірніших токенів на кожному кроці
    model_kwargs={
        'frequency_penalty': 1.2,  # Штраф за повторюваність слів (вищі значення → менше повторень)
        'presence_penalty': 0.5,  # Стимулює введення нових слів у відповідь (вищі значення → більша різноманітність)
    }
)

# model = prompt | llm
#
# text = model.invoke({'animal': 'програміст'})
# print(text)
#
#
# # response = llm.invoke("Придумай коротку історію про кота(4 речення)")
# # response = response.replace('. ', '.\n')
# # print(response)
#
# print(llm.invoke("[INST]Craft a Python function to convert Celsius to Fahrenheit. If water boils at 100°C, what's that in Fahrenheit?[/INST]"))


with open('data/lesson9/rules.txt', encoding='utf-8') as f:
    text = f.read()



prompt = f"""
[INST]Ти асисент по допомозі клієнтам атракціону. Твоє завдання давати відповіді на питання базуючись на документі.[/INST]

<s>Документ: {text}</s>

<s>Питання: З якого віку можна користуватись атракціоном?</s>
"""

# prompt = f"""
# Ти асисент по допомозі клієнтам атракціону. Твоє завдання давати відповіді на питання базуючись на документі. Відповіді мають бути короткими.
#
# Документ: {text}
#
# Питання: З якого віку можна користуватись атракціоном?
# Дай відповідь одним реченням
# """

#print(llm.invoke("[INST]Яка столиця Франції? Відповідь в одне слово[/INST]"))
print(llm.invoke("Яка столиця Франції?"))
