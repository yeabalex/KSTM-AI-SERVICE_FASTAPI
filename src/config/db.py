from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
import os

embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
def create_vectordb(documents, url):
    
    # Metadata enrichment with URL
    for doc in documents:
        doc.metadata["source_url"] = url
    
    # Create vector database with optimized parameters
    db = Neo4jVector.from_documents(
        documents=documents,
        embedding=embedding,
        username=os.getenv("NEO4J_USER"),
        password=os.getenv("NEO4J_PASSWORD"),
        url=os.getenv("NEO4J_URI"),
        index_name="ecom_index",  # Consistent index name
        node_label="ECOM"  # Consistent node label
    )
    
    return db


def load_vectordb():
    return Neo4jVector.from_existing_index(
        embedding=embedding,
        username=os.getenv("NEO4J_USER"),
        password=os.getenv("NEO4J_PASSWORD"),
        url=os.getenv("NEO4J_URI"),
        index_name="ecom_index",
        node_label="ECOM"
    )