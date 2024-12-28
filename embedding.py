def generate_embeddings(data_batch):
    try:
        embeddings = []
        for content, metadata in data_batch:
            if isinstance(content, str):
                embedding = embedding_model.encode(content)
            else:
                embedding = np.zeros(384)  # Placeholder for non-text content
            embeddings.append({"embedding": embedding, "metadata": metadata})
        return embeddings
    except Exception as e:
        logging.error(f"Error in generating embeddings: {e}")
        raise
def store_embeddings_in_redis(embeddings_with_metadata):
    try:
        for embedding in embeddings_with_metadata:
            chat_cache.set(f"embedding_{embedding['metadata']['id']}", json.dumps(embedding))
        logging.info("Embeddings stored in Redis cache.")
    except Exception as e:
        logging.error(f"Error storing embeddings in Redis cache: {e}")
        raise

def store_embeddings_in_vector_store(embeddings_with_metadata):
    try:
        vectors = [embedding["embedding"] for embedding in embeddings_with_metadata]
        metadata = [embedding["metadata"] for embedding in embeddings_with_metadata]
        
        vectorstore = FAISS.from_documents(vectors, metadata)
        return vectorstore
    except Exception as e:
        logging.error(f"Error storing embeddings in vector store: {e}")
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