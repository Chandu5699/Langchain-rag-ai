import logging
from elasticsearch import Elasticsearch
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import ElasticsearchVectorSearch
from langchain.embeddings import OpenAIEmbeddings
from py2neo import Graph

# Setup logging
logging.basicConfig(filename='search_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Elasticsearch setup
es = Elasticsearch("http://localhost:9200")

# Neo4j setup for knowledge graph
neo4j_url = "bolt://localhost:7687"
neo4j_user = "neo4j"
neo4j_password = "password"
graph = Graph(neo4j_url, auth=(neo4j_user, neo4j_password))

# Setup the model (T5 or BART)
model_name = "t5-base"  # You can change this to "facebook/bart-large" for BART
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Chat buffer memory to store chat message history
chat_buffer = []

# Define a prompt template to generate different queries from the original query
query_variation_template = PromptTemplate(
    input_variables=["original_query"],
    template="""
    You are a helpful assistant. Your task is to generate different versions of the original user query by paraphrasing it in multiple ways.
    The user query is:

    Original Query: {original_query}

    Please generate 5 variations of this query that would ask the same question but in different ways.
    1.
    2.
    3.
    4.
    5.
    """
)

def generate_query_variations(query, llm):
    try:
        # Generate query variations using the LLM
        prompt = query_variation_template.format(original_query=query)
        response = llm.generate(prompt)

        # Split the response into separate lines to get individual variations
        query_variations = response.strip().split("\n")

        # Retrieve documents for each query variation
        all_retrieved_docs = []
        for query_variant in query_variations:
            docs = retrieve_documents_from_query(query_variant)
            all_retrieved_docs.extend(docs)

        # Store the query and responses in chat buffer
        chat_buffer.append({"role": "user", "content": query})
        for variant in query_variations:
            chat_buffer.append({"role": "assistant", "content": variant})

        return all_retrieved_docs  # Return the aggregated documents from all variations
    except Exception as e:
        logging.error(f"Error generating query variations: {e}")
        return None

#vector serach# Initialize the hybrid search function using Elasticsearch and Neo4j
def hybrid_search(query, index="documents", num_results=5):
    try:
        # Vector Search: Use OpenAI Embeddings to create vector representation of the query
        embeddings = OpenAIEmbeddings()
        query_vector = embeddings.embed(query)

        # Elasticsearch query (Hybrid: Match query with both text search and vector search)
        body = {
            "query": {
                "bool": {
                    "should": [
                        {"match": {"text": query}},  # Traditional text search
                        {"knn": {
                            "field": "vector",  # Assuming 'vector' is the field storing document embeddings
                            "query_vector": query_vector,
                            "k": num_results
                        }}
                    ]
                }
            }
        }
        # Execute the hybrid search query
        response = es.search(index=index, body=body)
        # Parse the results
        parsed_results = parse_elasticsearch_results(response)

        # Knowledge Graph Query (using Neo4j)
        kg_results = knowledge_graph_query(query)
        parsed_results.extend(kg_results)

        # Store the query and search results in chat buffer
        chat_buffer.append({"role": "user", "content": query})
        chat_buffer.append({"role": "assistant", "content": str(parsed_results)})

        return parsed_results

    except Exception as e:
        logging.error(f"Error in hybrid search: {str(e)}")
        return []

def parse_elasticsearch_results(response):
    try:
        # Extract the top results
        results = [hit["_source"] for hit in response["hits"]["hits"]]
        parsed_results = []
        
        for result in results:
            parsed_result = {
                "title": result.get("title", "No title"),
                "text": result.get("text", "No text content"),
                "score": result.get("_score", 0)
            }
            parsed_results.append(parsed_result)
        
        return parsed_results
    except Exception as e:
        logging.error(f"Error parsing Elasticsearch results: {str(e)}")
        return []

def knowledge_graph_query(query):
    try:
        # Example Neo4j query to find related nodes and relationships
        kg_query = f"""
        MATCH (n)-[r]->(m)
        WHERE n.name CONTAINS '{query}' OR m.name CONTAINS '{query}'
        RETURN n.name AS node1, type(r) AS relationship, m.name AS node2
        LIMIT 10
        """
        results = graph.run(kg_query)
        parsed_results = [{"node1": record["node1"], "relationship": record["relationship"], "node2": record["node2"]} for record in results]
        return parsed_results
    except Exception as e:
        logging.error(f"Error executing knowledge graph query: {str(e)}")
        return []

def get_chain_of_thought_prompt(query, documents):
    # Custom Chain of Thought (CoT) prompt template to guide conversational reasoning
    thought_process = """
    We have a question and multiple documents. Let's reason through these documents step by step, 
    analyzing the relevance of each one to the query, and then combining the insights to form the best possible answer. 
    
    Query: {query}
    
    Here is the breakdown of the documents and their reasoning:

    Document 1: {doc_1_type} (Type of Document)
    Reasoning for Document 1:
    {reasoning_1}
    
    Document 2: {doc_2_type} (Type of Document)
    Reasoning for Document 2:
    {reasoning_2}
    
    Document 3: {doc_3_type} (Type of Document)
    Reasoning for Document 3:
    {reasoning_3}
    
    In conclusion, based on the reasoning from all the documents, the final answer to the query is:
    """
    
    # Create the prompt template with dynamic variables
    prompt = PromptTemplate(
        input_variables=["query", "doc_1_type", "reasoning_1", "doc_2_type", "reasoning_2", "doc_3_type", "reasoning_3"],
        template=thought_process
    )
    
    # Render the prompt with the provided query and documents
    return prompt.format(
        query=query,
        doc_1_type=documents[0]["type"],
        reasoning_1=documents[0]["reasoning"],
        doc_2_type=documents[1]["type"],
        reasoning_2=documents[1]["reasoning"],
        doc_3_type=documents[2]["type"],
        reasoning_3=documents[2]["reasoning"]
    )

def parse_model_outputs(outputs, tokenizer):
    try:
        decoded_outputs = []
        for output in outputs:
            decoded_output = tokenizer.decode(output, skip_special_tokens=True)
            decoded_outputs.append(decoded_output)
        return decoded_outputs
    except Exception as e:
        logging.error(f"Error parsing model outputs: {str(e)}")
        return []

# Ranking the documents using T5/BART
def rank_documents_with_model(documents, query, model, tokenizer):
    try:
        ranked_documents = []
        for doc in documents:
            input_text = f"Rank this document: {doc['text']} based on the query: {query}"
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
            outputs = model.generate(inputs['input_ids'], max_length=50)
            rank_score = parse_model_outputs(outputs, tokenizer)[0]
            ranked_documents.append((doc, rank_score))
        
        # Sort documents based on the generated rank score
        ranked_documents.sort(key=lambda x: x[1], reverse=True)
        return [doc[0] for doc in ranked_documents]

    except Exception as e:
        logging.error(f"Error in ranking documents: {str(e)}")
        return documents

def parse_cot_result(response):
    try:
        # Assuming the response is a string, you can format it or extract key parts
        formatted_response = response.strip()
        return formatted_response
    except Exception as e:
        logging.error(f"Error parsing CoT result: {str(e)}")
        return "Sorry, I couldn't generate an answer."

# Define the Conversational Chain LLM with Chain of Thought (CoT)
def generate_answer_with_cot(query, documents):
    try:
        # Step 1: Rank documents using the T5/BART model
        ranked_documents = rank_documents_with_model(documents, query, model, tokenizer)
        
        # Step 2: Create the Chain of Thought prompt template
        # Here we only show 3 documents, but it can be scaled for more
        prompt = get_chain_of_thought_prompt(
            query=query,
            documents=ranked_documents[:3]  # Limiting to top 3 documents for simplicity
        )

        # Load Conversational Chain LLM (e.g., OpenAI's GPT)
        llm = OpenAI(model="text-davinci-003", temperature=0.7)

        # Prepare the retrieval chain
        retriever = ElasticsearchVectorSearch.from_documents(ranked_documents, OpenAIEmbeddings())
        conversational_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)

        # Get the answer from the conversational chain using the CoT prompt
        response = conversational_chain.run(input=prompt)
        parsed_response = parse_cot_result(response)

        # Store the query and final response in chat buffer
        chat_buffer.append({"role": "user", "content": query})
        chat_buffer.append({"role": "assistant", "content": parsed_response})

        return parsed_response

    except Exception as e:
        logging.error(f"Error in generating answer with Chain of Thought: {str(e)}")
        return "Sorry, I couldn't generate an answer."

# Main execution
def main(query):
    try:
        # Step 1: Perform Hybrid Search to get the top documents
        documents = hybrid_search(query)

        if not documents:
            logging.info("No documents retrieved in the search.")
            return "Sorry, no relevant documents were found."

        # Step 2: Generate an answer using the Chain of Thought approach
        answer = generate_answer_with_cot(query, documents)

        logging.info("Successfully generated the answer.")
        return answer

    except Exception as e:
        logging.error(f"Error in the main execution: {str(e)}")
        return "An error occurred while processing the query."

# Example Query
if __name__ == "__main__":
    query = "What is the capital of France?"
    result = main(query)
    print(result)