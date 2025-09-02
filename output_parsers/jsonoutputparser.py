from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"  # Fixed model ID and task name
)

model=ChatHuggingFace(llm=llm)
parser=JsonOutputParser()
template=PromptTemplate(
    template='give the name ,age and city of a fictional person \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}

)


'''prompt=template.format()

result=model.invoke(prompt)

final=parser.parse(result.content)'''
#instead of this we can write

chain=template | model | parser
final=chain.invoke({})
print(type(final))  # This will print the type of the parsed object
print(final)  # This will print the parsed JSON object