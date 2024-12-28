def create_conversational_chain(vectorstore):
    try:
        retriever = vectorstore.as_retriever()
        llm = OpenAI()
        conversation_chain = ConversationalRetrievalChain.from_llm(llm, retriever)
        return conversation_chain
    except Exception as e:
        logging.error(f"Error creating conversational chain: {e}")
        raise
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

# Initialize Elasticsearch client
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

def index_embeddings_to_es(index_name, embeddings_with_metadata):
    try:
        actions = []
        for embedding in embeddings_with_metadata:
            action = {
                "_index": index_name,
                "_id": embedding['metadata']['id'],
                "_source": {
                    "embedding": embedding["embedding"],
                    "metadata": embedding["metadata"]
                }
            }
            actions.append(action)
        
        bulk(es, actions)
        print("Embeddings indexed in ElasticSearch successfully.")
    except Exception as e:
        print(f"Error indexing embeddings: {e}")
        raise

# Example usage:
embeddings_with_metadata = [{"embedding": [0.1, 0.2, 0.3], "metadata": {"id": "1", "type": "pdf"}}]
index_embeddings_to_es("multimodal_embeddings", embeddings_with_metadata)
