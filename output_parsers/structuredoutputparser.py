from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import StructuredOutputParser,ResponseSchema

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"  # Fixed model ID and task name
)

model=ChatHuggingFace(llm=llm)

schema=[
    ResponseSchema(name='fact_1',description='fact 1 about the topic'),
    ResponseSchema(name='fact_2',description='fact 2 about the topic'),
    ResponseSchema(name='fact_3',description='fact 3 about the topic'),
]

parser=StructuredOutputParser.from_response_schemas(schema)

template=PromptTemplate(
    template="give 3 facts about the topic {topic} \n {format_instructions}",
    input_variables=['topic'], 
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

'''
prompt=template.invoke({'topic':'black hole'})
result=model.invoke(prompt)
final_result=parser.parse(result.content)
'''
chain=template | model | parser
result=chain.invoke({'topic':'black hole'})
print(result)