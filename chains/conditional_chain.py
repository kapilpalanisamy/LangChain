from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel,RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = HuggingFaceEndpoint(
    repo_id="google/gemma-2b-it",  # Using the base model which is better for general tasks
    task="text-generation"
)
llm = ChatHuggingFace(llm=model)

parser=StrOutputParser()

class Feedback(BaseModel):

    sentiment: Literal['positive','negative'] = Field(description='sentiment of the feedback')

parser2=PydanticOutputParser(pydantic_object=Feedback)

prompt1=PromptTemplate(
    template="""Analyze the sentiment of the following feedback and classify it as either 'positive' or 'negative'.

Feedback: {feedback}

Instructions: 
1. Return ONLY a JSON object
2. The response must contain a 'sentiment' field with EXACTLY either 'positive' or 'negative' as the value
3. Follow this exact format:
{{"sentiment": "positive"}} or {{"sentiment": "negative"}}

{format_instruction}""",
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

classifier_chain= prompt1 | llm | parser2

prompt2=PromptTemplate(
    template='write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3=PromptTemplate(
    template='write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain=RunnableBranch(
    (lambda x:x.sentiment=='positive',prompt2 | model | parser),
    (lambda x:x.sentiment == 'negative',prompt3 | model | parser),
    RunnableLambda(lambda x: "No sentiment detected")
    )

chain=classifier_chain | branch_chain
result=chain.invoke({'feedback':'The product quality is excellent '})

print(result)