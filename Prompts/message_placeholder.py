from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

#chat template
chat_template = ChatPromptTemplate(
    [   
        ('system','You are a helpful customer support agent.'),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human','{query}')
    ]
)
chat_history=[]
#load chat history
with open('chathistory.txt','r') as file:
    chat_history.extend(file.readlines())
print(chat_history)

#create prompt
prompt = chat_template.invoke({
    "chat_history": chat_history,
    "query":"Where is my refund?"
})

print(prompt)
