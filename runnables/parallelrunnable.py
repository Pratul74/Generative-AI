from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from rich import print
import time

load_dotenv()

detailed_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
        """You are a helpful teacher"""
         ),
         ("human", """Give me detailed explanation of {topic}""")
    ]

)

short_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
You are a helpful teacher
"""),
("human", """Give me summary explanation of {topic}""")
    ]
)

llm = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash-lite"
)

parallel_runnable = RunnableParallel(
    {
        'short': short_prompt | llm | StrOutputParser(),
        'detailed': detailed_prompt | llm | StrOutputParser()
    }
)

response = parallel_runnable.invoke({'topic': "Biggest Blackhole in the universe"})

print(response['detailed'])
print("\n")
print("\n")
print(response['short'])

