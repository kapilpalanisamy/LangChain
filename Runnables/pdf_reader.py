from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.document_loaders import TextLoader
from langchain_core.vectorstores import FAISS

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"  # Fixed model ID and task name
)

model=ChatHuggingFace(llm=llm)

#Load the document
loader = TextLoader("docs.txt") # Ensure docs.txt exists

documents = loader.load()

# Split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

docs = text_splitter.split_documents(documents)

# Convert text into embeddings & store in FAISS
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())

# Create a retriever (fetches relevant documents)
retriever = vectorstore.as_retriever()

# Manually Retrieve Relevant Documents
query = "What are the key takeaways from the document?"

retrieved_docs = retriever.get_relevant_documents(query)

# Combine Retrieved Text into a Single Prompt
retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])

# Manually Pass Retrieved Text to LLM
prompt = f"Based on the following text, answer the question: {query}\n\n{retrieved_text}"

answer = llm.predict(prompt)

# Print the Answer
print("Answer:", answer)