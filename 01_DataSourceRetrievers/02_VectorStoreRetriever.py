from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import OpenAIEmbeddings

from langchain_core.documents import Document

from dotenv import load_dotenv
load_dotenv()

#Your source doeuments
documents=[

    Document(page_content="India and Pakistan have a complex geopolitical history marked by partition, wars, and ongoing conflicts primarily over the Kashmir region. Major world powers like the US, China, and Russia have played significant roles in shaping the dynamics between the two nations through diplomatic interventions, military aid, and strategic alliances."),
    Document(page_content="The partition of British India in 1947 led to the creation of India and Pakistan, resulting in massive population exchanges and violence. The geopolitical interests of major world powers during the Cold War further complicated relations, with the US and USSR vying for influence in South Asia."),
    Document(page_content="India's non-alignment policy during the Cold War allowed it to maintain a degree of independence from major world powers, while Pakistan aligned more closely with the US and China. This alignment influenced military and economic aid, impacting the regional balance of power."),
    Document(page_content="The Kargil War in 1999 and the 2001 Indian Parliament attack heightened tensions between India and Pakistan, drawing international attention. Major world powers called for restraint and dialogue, emphasizing the need for peaceful resolution of disputes."),
    Document(page_content="In recent years, the geopolitical landscape has evolved with China's Belt and Road Initiative and increased US-India cooperation. These developments have implications for Pakistan's strategic calculations and its relations with both India and major world powers."),

]

#initialize embedding model
embedding_model=OpenAIEmbeddings()

#create chroma vector store from documents
vector_store=Chroma.from_documents(
    documents,
    embedding_model,
    collection_name="MyCollection")


#convert vectror store to retriver


retriver= vector_store.as_retriever(search_kwargs={"k":1})

#define your query
query="about The Kargil War in 1999"
#get relevant documents
relevant_docs=retriver.invoke(query)## it will perform semilarity search,

for doc in relevant_docs:
    print(doc.page_content)



# insted of converting vector store to retriver you can directly use the vector store to get similar documents as below
#similar_docs=vector_store.similarity_search(query,k=2)
#the difference is it uses only one stargey to get the rlevent vectors.
#what if you want to  use diffrent stargey, we can achive this by retriever
# another benifit is if we convert to etriver its a runnable, so we can use it in chain


