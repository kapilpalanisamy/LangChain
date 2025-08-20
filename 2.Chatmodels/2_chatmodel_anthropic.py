from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model=ChatAnthropic(model="claude-opus-4-1-20250805")

result=model.invoke("What is the capital of India")

print(result.content)