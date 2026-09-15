from dotenv import load_dotenv
from langchain_text_splitters import TokenTextSplitter
from langchain_community.document_loaders import TextLoader

load_dotenv()

data = TextLoader("Cold_emails.txt")

docs = data.load()

splitter = TokenTextSplitter(
    chunk_size = 5,
    chunk_overlap= 2
)

chunks = splitter.split_documents(docs)

for chunk in chunks:
    print(chunk.page_content)
    print("\n")

