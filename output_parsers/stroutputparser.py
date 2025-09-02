from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from huggingface_hub import login
import os

load_dotenv()

# Login to Hugging Face
hf_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
if not hf_token:
    raise ValueError("Please set HUGGINGFACEHUB_API_TOKEN in your .env file")
login(hf_token)

llm=HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"  # Fixed model ID and task name
)

model=ChatHuggingFace(llm=llm)

#1st prompt
template1=PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=['topic']
)
#2nd prompt summary
template2=PromptTemplate(
    template="write a 5 line summary on the following text. /n  {text}",
    input_variables=['text']
)

prompt1=template1.invoke({'topic':'black hole'})

result=model.invoke(prompt1)

prompt2=template2.invoke({'text':result.content})

result1=model.invoke(prompt2)

print(result1.content)
