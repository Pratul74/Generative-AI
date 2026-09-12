from dotenv import load_dotenv
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="openai-community/gpt2",
    task = "text-generation",
    pipeline_kwargs={
        "max_new_tokens": 256,
        "temperature": 0.7,
    },
)


response = llm.invoke("Who is python ?")

print(response)

