from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"  # Fixed model ID and task name
)

model=ChatHuggingFace(llm=llm)

prompt=PromptTemplate(
    input_variables=['topic'],
    template='generate 5 interesting facts about {topic}'
)
#define the input
topic=input("Enter a topic: ")

#format the prompt manually using prompttempalte
formatted_prompt=prompt.format(topic=topic)

#call the llm directly
blog_title=model.invoke(formatted_prompt)

#print the output
print(blog_title)