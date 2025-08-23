from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv

embedding=OpenAIEmbeddings(model='text_embedding-3-large',dimensions=32)

documents=[
    "Delhi is the capital of India",
    "Kolkata is the capital of westbengal",
    "Paris is the capital of france"
]
result=embedding.embed_query(documents)

print(str(result))