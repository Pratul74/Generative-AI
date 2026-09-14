from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

load_dotenv()

data = TextLoader("Cold_emails.txt")

docs = data.load()

print(docs[0].page_content)