from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)

documents = [
    # Document 01: Sachin Tendulkar
    "Sachin Tendulkar is widely regarded as the 'Little Master' and holds the record for scoring the most international runs in cricket history.",
    
    # Document 02: MS Dhoni
    "MS Dhoni is the only captain in cricket history to win all three major ICC trophies: the 2007 T20 World Cup, 2011 ODI World Cup, and 2013 Champions Trophy.",
    
    # Document 03: Rohit Sharma
    "Rohit Sharma is the only player to score three double-centuries in One Day International (ODI) cricket.",
    
    # Document 04: Jasprit Bumrah
    "Jasprit Bumrah is considered one of the most lethal fast bowlers of the modern era, famous for his unique slinging action and pinpoint yorkers.",
    
    # Document 05: AB de Villiers
    "AB de Villiers is known as 'Mr. 360' for his extraordinary ability to hit the ball to any part of the ground.",
    
    # Document 06: Steve Smith
    "Steve Smith is known for his unconventional batting stance and exceptional consistency in Test cricket.",
    
    # Document 07: Sir Don Bradman
    "Sir Don Bradman is statistically the greatest Test batsman of all time, finishing his career with an incredible batting average of 99.94.",
    
    # Document 08: Ben Stokes
    "Ben Stokes is one of the world's leading all-rounders, known for producing match-winning performances under pressure in major finals.",
    
    # Document 09: Kane Williamson
    "Kane Williamson is widely respected for his calm leadership and technical mastery across all formats of the game.",
    
    # Document 10: Rashid Khan
    "Rashid Khan is one of the most successful leg-spin bowlers in modern T20 cricket, known for his quick wrist action and deceptive googly."
]
query = "Tell me about about Sachin Tendulkar"

doc_embedding = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

# print(cosine_similarity([query_embedding], doc_embedding))

scores =  cosine_similarity([query_embedding], doc_embedding,)[0]

# print(list(enumerate(scores)))
index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print("Your Question was!", query)
print(documents[index])
print("Similaraties score is: ",score)
