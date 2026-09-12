from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-3.5-flash-lite", temperature=1)

response = model.invoke("Tell me about Donald Trump")

print(response.content)