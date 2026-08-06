import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Use the latest model which is fully supported by the new SDK
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

result = llm.invoke("What is the Capital of India?")
print(result.content)