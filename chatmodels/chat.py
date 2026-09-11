from dotenv import load_dotenv
from 

load_dotenv()

model = init_chat_model("ollama:mistral-small-2603")

response = model.invoke("Tell me about Donald Trump")

print(response.content)