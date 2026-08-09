from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Initialize embeddings with the specified model and reduced dimensions
embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

documents = [
    "Delhi is Capital of India",
    "Kolkata is Capital of West Bengal",
    "Paris is Capital of France"
]

# Generate vector embedding for the text
result = embedding.embed_documents(documents)

print(f"Vector length: {len(result)}")
print(result)