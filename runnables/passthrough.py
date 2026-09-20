from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

load_dotenv()

prompt1 = ChatPromptTemplate.from_messages(
    [
        ("system",
        """You are code generator based on the topic user will give you"""),
        ("human",
         """{topic}""")
    ]
)

prompt2 = PromptTemplate.from_template(
    "Explain the {code} in detail"
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

chain1 = prompt1 | llm | StrOutputParser()

parallel_runnable = RunnableParallel(
    {
        'code': RunnablePassthrough(),
        'explanation': prompt2 | llm | StrOutputParser()
    }
)

final_chain = chain1 | parallel_runnable

response = final_chain.invoke({'topic': 'Write a code to reverse linkedlist in python'})

print("Code: ")
print(response['code'])
print("\n")
print("\n")
print("\n")
print("Code explanation: ")
print(response['explanation'])

