from dotenv import load_dotenv
from langchain_arxiv import ArxivRetriever

retriver = ArxivRetriever(
                        load_max_documents=3,
                        load_all_available_meta=True
                )

docs= retriver.invoke("Large Language Model")

for doc in docs:
    print(doc.page_content)