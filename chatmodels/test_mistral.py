from mistralai import Mistral
from dotenv import load_dotenv
import os

load_dotenv(override=True)

client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

response = client.chat.complete(
    model="mistral-small-2603",
    messages=[
        {
            "role": "user",
            "content": "Hello"
        }
    ]
)

print(response.choices[0].message.content)