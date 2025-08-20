from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
#temparature is the amount of the creativity that we want 
model=ChatOpenAI(model='gpt-4.1',temperature=2,max_completion_tokens=10)

result=model.invoke("What is the capital of India")

print(result.content)