from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

data = PyPDFLoader("Pratul-Resume.pdf")

docs = data.load()

print(docs[0].page_content)