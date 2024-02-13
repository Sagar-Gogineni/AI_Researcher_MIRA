import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import streamlit as st

chroma_client = chromadb.Client()

@st.cache_data
def generate_embedding(_chunks,_metadata,openAI_API_KEY):
    
    openai_embfunc = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openAI_API_KEY,
                model_name="text-embedding-ada-002"
            )

    metadata=[]    
    for idx, doc in enumerate(_chunks):
        #print("*********************************"+str(chroma_client.list_collections()))
        collection = chroma_client.get_or_create_collection(name="my_collection", embedding_function=openai_embfunc)
        metadata.append(_metadata[idx])
        for d in doc:
            collection.upsert(
                
                documents=d.page_content,
                #metadatas=metadata,
                ids=str(metadata)
            )

def fetch_similar_text(query,openAI_API_KEY):

    openai_embfunc = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openAI_API_KEY,
                model_name="text-embedding-ada-002"
            )

    collection = chroma_client.get_or_create_collection(name="my_collection", embedding_function=openai_embfunc)
    results = collection.query(
    query_texts=query,#["what are the different prompting techniques discussed in this paper"],
    n_results=10
    )
    #print(results['documents'][0])
    return results['documents'][0]

def clear_collections():
    chroma_client.delete_collection(name='my_collection')