from langchain_mistralai import ChatMistralAI
import os
from dotenv import load_dotenv

load_dotenv()

model = ChatMistralAI(
    model= "mistral-small-2603",
    max_tokens=100
)

response = model.invoke("Hello")

print(response.content)