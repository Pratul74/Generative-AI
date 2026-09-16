from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system",
    """
You are an intelligent document assistant.

Your task is to answer questions using ONLY the retrieved document context.

Guidelines:
- Use only facts present in the context.
- Do not rely on prior knowledge.
- Do not invent information.
- If the answer is partially available, answer only the available part and mention that additional information is not present in the document.
- If the answer is completely unavailable, respond exactly:
  "I couldn't find the answer in the provided document."
- If the context contains conflicting information, mention the conflict instead of choosing one.
- Use bullet points when appropriate.
- Keep answers factual and concise.
"""
     ), ("human", """
    context: 
    {context}

    question:
    {question}
""")
])

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

embeddings = MistralAIEmbeddings()

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="chroma_db"
)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
    
)

print("---------------------------Welcome to your ChatBot. Enter 0 for exit-------------------------------")
while True:
    question = input("You: ")
    if question == "0":
        break
    documents=retriever.invoke(question)
    context = "".join([doc.page_content for doc in documents])
    messages= prompt.invoke({
        "context": context,
        "question": question
    })
    response = llm.invoke(messages)
    print(response.content[0]['text'])