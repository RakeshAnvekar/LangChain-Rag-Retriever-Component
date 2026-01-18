from langchain_community.retrievers import WikipediaRetriever

#initialize the retriver  (optonal:Set  language and top k results
wiki_retriever = WikipediaRetriever(language="en", top_k_results=2)

#define your query

query="the geopolitical history of india and pakistan from the perspective of major world powers."

#get relevant documents
relevant_docs = wiki_retriever.invoke(query)

#print the relevant documents

for doc in relevant_docs:
    print(doc.page_content)

