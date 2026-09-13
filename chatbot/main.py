from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model= "gemini-3.5-flash-lite"
)



messages =[
    SystemMessage("You are a professional chatbot")
]

print("---------------------Welcome to My ChatBot-------------------------")
while True:
    human_prompt = input("You: ")
    if human_prompt == "0":
        break
    messages.append(HumanMessage(human_prompt))
    response = model.invoke(messages)
    messages.append(AIMessage(response.content[0]['text']))
    print("Bot: ", response.content[0]['text'])

