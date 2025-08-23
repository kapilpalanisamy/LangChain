from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding=OpenAIEmbeddings(model='text_embedding-3-large',dimensions=300)

document=[
    "Virat Kohli is a celebrated Indian cricketer known for his aggressive batting style and leadership.",
    "Sachin Tendulkar, often regarded as the 'God of Cricket,' holds numerous records and is an icon in the sport.",
    "Rohit Sharma is an accomplished Indian cricketer, admired for his elegant stroke play and record-breaking performances in limited-overs cricket.",
    "Jasprit Bumrah is a prominent Indian fast bowler recognized for his distinctive bowling action and ability to deliver crucial wickets."
]

query="tell me about Virat Kohli"

doc_embedding=embedding.embed_documents(document)
query_embedding=embedding.embed_query(query)

similarity_scores=cosine_similarity([query_embedding], doc_embedding)[0]
index,score=sorted(list(enumerate(similarity_scores)), key=lambda x: x[1])[-1]
print(query)
print(document[index])
print("Similarity Score:", score)