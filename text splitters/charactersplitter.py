from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

data = PyPDFLoader("Pratul-Resume.pdf")

docs = data.load()

print(docs[0].page_content)
print("\n\n")

splitter = CharacterTextSplitter(
    separator="",
    chunk_size=100,
    chunk_overlap=20,
)

chunks = splitter.split_documents(docs)

for i in chunks:
    print(i.page_content)
    print("\n")


