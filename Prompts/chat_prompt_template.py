from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import SystemMessage,HumanMessage,AIMessage

chat_template = ChatPromptTemplate(
    [   
        ('system','You are a helpful {domain} expert.'),
        ('human','Tell in simple terms,what is {topic}?'),
    ]
)

prompt=chat_template.invoke({"domain": "AI", "topic": "LangChain"})

print(prompt)