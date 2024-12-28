import logging
from elasticsearch import Elasticsearch
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import ElasticsearchVectorSearch
from langchain.embeddings import OpenAIEmbeddings

# Setup logging
logging.basicConfig(filename='search_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Elasticsearch setup
es = Elasticsearch("http://localhost:9200")

# Setup the model (T5 or BART)
model_name = "t5-base"  # You can change this to "facebook/bart-large" for BART
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
from langchain.prompts import PromptTemplate

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
def process_query(query):
    try:
        # Step 1: Generate different variations of the query
        query_variations = query_variation_template.run(original_query=original_query)
        print("Generated Query Variations:", query_variations)
        
        # Step 2: Retrieve documents for each query variation
        all_retrieved_docs = []
        for query in query_variations.splitlines():
            docs = retrieve_documents_from_query(query)
            all_retrieved_docs.extend(docs)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
# Initialize the hybrid search function using Elasticsearch
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
        # Extract the top results
        results = [hit["_source"] for hit in response["hits"]["hits"]]
        return results

    except Exception as e:
        logging.error(f"Error in hybrid search: {str(e)}")
        return []


def get_chain_of_thought_prompt(query, documents):
    # Custom Chain of Thought (CoT) prompt template to guide conversational reasoning
    thought_process = """
    We have a question and multiple documents. Let's reason through these documents step by step, 
    analyzing the relevance of each one to the query, and then combining the insights to form the best possible documents. 
    
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

# Example usage:
documents = [
    {"type": "pdf", "reasoning": "This document explains the main topic in depth, providing detailed analysis relevant to the query."},
    {"type": "json", "reasoning": "The data extracted from this JSON document gives concrete figures that directly support the answer."},
    {"type": "audio", "reasoning": "The transcript of this audio interview provides conversational context that sheds light on the query."}
]

query = "What are the key insights provided in these documents?"

# Generate the Chain of Thought prompt
prompt = get_chain_of_thought_prompt(query, documents)
print(prompt)

# Ranking the documents using T5/BART
def rank_documents_with_model(documents, query):
    try:
        ranked_documents = []
        for doc in documents:
            input_text = f"Rank this document: {doc['text']} based on the query: {query}"
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
            outputs = model.generate(inputs['input_ids'], max_length=50)
            rank_score = tokenizer.decode(outputs[0], skip_special_tokens=True)
            ranked_documents.append((doc, rank_score))
        
        # Sort documents based on the generated rank score
        ranked_documents.sort(key=lambda x: x[1], reverse=True)
        return [doc[0] for doc in ranked_documents]

    except Exception as e:
        logging.error(f"Error in ranking documents: {str(e)}")
        return documents

# Define the Conversational Chain LLM with Chain of Thought (CoT)
def generate_answer_with_cot(query, documents):
    try:
        # Step 1: Rank documents using the T5/BART model
        ranked_documents = rank_documents_with_model(documents, query)
        
        # Step 2: Create the Chain of Thought prompt template
        # Here we only show 3 documents, but it can be scaled for more
        prompt = get_chain_of_thought_prompt(
            query=query,
            doc_1_text=ranked_documents[0]["text"],
            reasoning_1="Explain why this document is relevant.",
            doc_2_text=ranked_documents[1]["text"],
            reasoning_2="Explain why this document is relevant.",
            doc_3_text=ranked_documents[2]["text"],
            reasoning_3="Explain why this document is relevant."
        )

        # Load Conversational Chain LLM (e.g., OpenAI's GPT)
        llm = OpenAI(model="text-davinci-003", temperature=0.7)

        # Prepare the retrieval chain
        retriever = ElasticsearchVectorSearch.from_documents(ranked_documents, OpenAIEmbeddings())
        conversational_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)

        # Get the answer from the conversational chain using the CoT prompt
        response = conversational_chain.run(input=prompt.format(query=query))
        return response

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
query = "What is the capital of France?"
result = main(query)
print(result)