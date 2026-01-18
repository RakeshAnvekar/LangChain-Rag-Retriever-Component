from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from langchain.retrievers.multi_query import MultiQueryRetriever





documents = [
    Document(page_content="Protein is essential for muscle growth and repair in the human body."),
    Document(page_content="Adequate protein intake helps maintain muscle mass and overall health."),
    Document(page_content="Consuming enough protein supports tissue repair and body recovery."),
    Document(page_content="High-quality protein plays a key role in strengthening muscles."),
    Document(page_content="Protein helps the body recover after physical activity and exercise."),

    Document(page_content="Excessive protein consumption may put stress on the kidneys."),
    Document(page_content="Too much dietary protein can lead to dehydration if water intake is low."),
    Document(page_content="Overconsumption of protein may cause digestive discomfort."),
    Document(page_content="Very high protein diets can imbalance essential nutrients."),

    Document(page_content="Protein intake should be balanced with carbohydrates and fats."),
    Document(page_content="A healthy diet includes moderate protein along with other nutrients."),
    Document(page_content="Balanced protein consumption supports long-term health and wellness."),

    Document(page_content="Plant-based proteins can be a healthy alternative to animal protein."),
    Document(page_content="Lean protein sources are better for heart health."),
    Document(page_content="Protein requirements vary based on age, activity level, and health.")
]

#initialize open ai embeddings

embedding_model=OpenAIEmbeddings()

#create FAISS vector store
vectorStore=FAISS(
    documents=documents,
    embedding_model=embedding_model
)

#create similarity_retriver Retriver

similarity_retriver=vectorStore.as_retriever(search_type="similarity",search_kwargs={"k":5})

#create multi query retriver

multiQueryRetriever=MultiQueryRetriever.from_llm(
    vectorStore.as_retriever(search_kwargs={"k":5}), 
    llm=ChatOpenAI(model="gpt-3.5.turbo"))##internally we need to tell which llm you are using to create diffrent version of the queries


query="How to be fit"

similarity_retriver_result=similarity_retriver.invoke(query)
multiQueryRetriever_result=multiQueryRetriever.invoke(query)


for i,doc in enumerate(similarity_retriver_result):
    print(f"\n--------Result {i+1}----")
    print(doc.page_content)

print("--------------------------------------------------------------")
for i,doc in enumerate(multiQueryRetriever_result):
    print(f"\n--------Result {i+1}----")
    print(doc.page_content)
