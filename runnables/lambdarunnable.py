from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

#Load all the environment variables stored in .env file
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash-lite"
)

prompt_template1= ChatPromptTemplate.from_messages(
    [
        ("system",
        """You are helpful teacher
        Gives answer in friendly way"""
         ),
        ("human", """Give me a deatiled information on topic1: {topic1} and what is the relation of topic1: {topic1} with topic2: {topic2}""")
    ]
)

prompt_template2= ChatPromptTemplate.from_messages(
    [
        ("system",
        """You are helpful teacher
        Gives answer in friendly way"""
         ),
        ("human", """topic3: {topic3}""")
    ]
)

parrellel_chains = RunnableParallel(
    {
        'first': RunnableLambda(lambda x: x['first_message']) | prompt_template1 | llm | StrOutputParser(),
        'second': RunnableLambda(lambda x: x['second_message']) | prompt_template2 | llm | StrOutputParser()
    }
)

messages = {
    'first_message': {'topic1': 'what is the sagitarrius A black hole', 'topic2': 'Milkyway galaxy'},
    'second_message': {'topic3': 'Elon musk'}
}

response = parrellel_chains.invoke(messages)

print(response)