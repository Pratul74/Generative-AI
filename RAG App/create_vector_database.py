from langchain_docling.loader import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


#load the pdf
loader = DoclingLoader("System design by Alex Xu(Volume 1).pdf")
docs = loader.load()

docs = filter_complex_metadata(docs)


#Split the pdf
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)
chunks = splitter.split_documents(docs)

#Create Vector Embeddings
embedding_model = MistralAIEmbeddings()


#Store in Vector database

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma_db"
)