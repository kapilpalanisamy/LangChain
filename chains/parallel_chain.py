from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model1=HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"  # Fixed model ID and task name
)

model2=HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"  # Fixed model ID and task name
)

model1=ChatHuggingFace(llm=model1)
model2=ChatHuggingFace(llm=model2)

prompt1=PromptTemplate(
    template='generate a short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2=PromptTemplate(
    template='generate 5 short question answers from the following  \n {text}',
    input_variables=['text']
)

prompt3=PromptTemplate(
    template='Merge the provided notes and question answers into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes','quiz']
)

parser=StrOutputParser()

parellel_chain=RunnableParallel({
    'notes':prompt1 | model1 | parser,
    'quiz':prompt2 | model2 | parser
}
)

merge_chain=prompt3 | model1 | parser

chain= parellel_chain | merge_chain

text="""scikit-learn (formerly scikits.learn and also known as sklearn) is a free and open-source machine learning library for the Python programming language.[3] It features various classification, regression and clustering algorithms including support-vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy. Scikit-learn is a NumFOCUS fiscally sponsored project.[4]The scikit-learn project started as scikits.learn, a Google Summer of Code project by French data scientist David Cournapeau. The name of the project derives from its role as a "scientific toolkit for machine learning", originally developed and distributed as a third-party extension to SciPy.[5] The original codebase was later rewritten by other developers.[who?] In 2010, contributors Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort and Vincent Michel, from the French Institute for Research in Computer Science and Automation in Saclay, France, took leadership of the project and released the first public version of the library on February 1, 2010.[6] In November 2012, scikit-learn as well as scikit-image were described as two of the "well-maintained and popular" scikits libraries.[7] In 2019, it was noted that scikit-learn is one of the most popular machine learning libraries on GitHub.[8]
"""

result = chain.invoke({'text':text})

print(result)

chain.get_graph().print_ascii()  # to visualize the chain