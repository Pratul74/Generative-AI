from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

data = PyPDFLoader("Pratul-Resume.pdf")

docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 20
)

chunks = splitter.split_documents(docs)

print(len(chunks))
print("\n")
print("\n")

for chunk in chunks:
    print(chunk.page_content)
    print("\n")