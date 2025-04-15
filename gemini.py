from langchain_google_genai import GoogleGenerativeAI
import dotenv
import os

dotenv.load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

llm = GoogleGenerativeAI(model="gemini-2.0-flash",
                         google_api_key=api_key)

print(llm.invoke('hello'))