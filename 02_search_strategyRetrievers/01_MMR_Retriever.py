

from langchain_community.embeddings import OpenAIEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

docs=[

    Document(page_content="LangChain helps developers build applications using large language models."),
    Document(page_content="LangChain is a framework designed to create LLM-powered applications."),
    Document(page_content="LangChain allows developers to connect LLMs with external data sources."),
    Document(page_content="Python is a programming language used for backend development."),
    Document(page_content="LangChain simplifies the development of AI-driven applications.")
]

#initialize the embedding model

embeddingModel=OpenAIEmbeddings()

vectorStore=FAISS.from_documents(documents=docs,embedding=embeddingModel)

#enable mmr in the retriver

retriver= vectorStore.as_retriever(
search_type="mmr",# this enables mmr
search_kwargs={"k":3,"lambda_mult":0.5} ##if lambda_mult= 1 then mmr will behave same as similariy search,if 0 we get diverse results
)


query="what is langchain"
result=retriver.invoke(query)

for i,doc in enumerate(result):
    print(f"\n--------Result {i+1}----")
    print(doc.page_content)

