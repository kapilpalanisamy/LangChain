from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"  # Fixed model ID and task name
)

model=ChatHuggingFace(llm=llm)

class Person(BaseModel):

    name: str = Field(description="The name of the person")
    age: int = Field(gt=18, description="The age of the person")
    city: str = Field(description="The city where the person lives")

parser=PydanticOutputParser(pydantic_object=Person)

template=PromptTemplate(
    template="""Generate details of a fictional {place} person.
Please provide the information in the following JSON format:
{format_instruction}

The output should be a valid JSON object with exactly these fields:
- name: a string with the person's full name
- age: an integer greater than 18
- city: a string with the person's city

Ensure the output is a valid JSON object.""",
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

'''prompt=template.invoke({'place':'Indian'})

result=model.invoke(prompt)

final=parser.parse(result.content)

print(final)'''

chain=template | model | parser
result=chain.invoke({'place':'American'})
print(result)