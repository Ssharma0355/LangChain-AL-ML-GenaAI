import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

script_dir = Path(__file__).resolve().parent
load_dotenv(script_dir / ".env") or load_dotenv(script_dir.parent / ".env")

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

if not api_key:
    raise ValueError("OPENAI_API_KEY is missing from .env")

model = ChatOpenAI(
    model="openai/gpt-4o",
    base_url=base_url,
    api_key=api_key.strip("'\" "),
    max_completion_tokens=200,  # Caps potential credit consumption
)

result = model.invoke("What is the Captian of Indian Cricket Team?")

print("--- Output ---")
print(result.content)