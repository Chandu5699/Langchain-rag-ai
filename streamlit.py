import streamlit as st
import logging
from data_processor import DataProcessor
from vector_store_manager import VectorStoreManager
from input_tracker import InputTracker
from response_generator import ResponseGenerator
from multimodal_model import MultimodalEmbeddingModel, MultimodalLLM

# Configure logging
logging.basicConfig(
    filename="rag_system.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def main():
    """Main Streamlit application"""
    st.set_page_config(page_title='Enhanced Multimodal RAG System')
    st.header('Intelligent Multimodal Document Analysis System 🤖')

    # Initialize managers and processors
    emb = MultimodalEmbeddingModel()
    llm = MultimodalLLM()
    data_processor = DataProcessor()
    vector_store_manager = VectorStoreManager(emb)

    # Initialize session state
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "last_processed_hash" not in st.session_state:
        st.session_state.last_processed_hash = None
    if "needs_update" not in st.session_state:
        st.session_state.needs_update = False

    # Input selection
    input_choice = st.radio(
        "Choose input method:",
        ["Enter Blog URL", "Upload PDF", "Upload Audio", "Upload Image"]
    )

    # Input handling
    input_source = None
    input_type = None

    if input_choice == "Enter Blog URL":
        input_source = st.text_input('Enter the URL:')
        input_type = "url"
    elif input_choice == "Upload PDF":
        input_source = st.file_uploader("Upload PDF", type="pdf")
        input_type = "pdf"
    elif input_choice == "Upload Audio":
        input_source = st.file_uploader("Upload Audio", type=["wav", "mp3"])
        input_type = "audio"
    elif input_choice == "Upload Image":
        input_source = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        input_type = "image"

    # Check for input changes
    if input_source:
        current_hash = InputTracker.get_input_hash(input_source)
        if InputTracker.has_input_changed(current_hash, st.session_state.last_processed_hash):
            st.session_state.needs_update = True
            st.warning("⚠️ Input has changed - Vector store needs to be updated before querying")

    # Vector store update section
    if st.session_state.needs_update and input_source:
        st.sidebar.markdown("### Vector Store Status")
        if st.sidebar.button('Update Vector Store'):
            try:
                with st.spinner('Processing input and updating vector store...'):
                    docs, current_hash = data_processor.process_input(input_source, input_type)
                    st.session_state.vector_store = vector_store_manager.create_or_update(docs)
                    st.session_state.last_processed_hash = current_hash
                    st.session_state.needs_update = False
                st.success('✅ Vector Store Updated Successfully!')
                logging.info("Vector store updated successfully.")
            except Exception as e:
                st.error(f"Error updating vector store: {str(e)}")
                logging.error(f"Error updating vector store: {str(e)}")
                return

    # Question handling
    user_question = st.text_input('Ask a question:')

    # Generate Response button
    if st.button('Generate Response', key='generate_response'):
        if not user_question:
            st.error("Please enter a question first.")
        elif st.session_state.needs_update:
            st.error("⚠️ Please update the vector store before asking questions about the new content")
        elif st.session_state.vector_store is None:
            st.error("⚠️ Please provide input and update the vector store first")
        else:
            try:
                with st.spinner('Generating response...'):
                    response_generator = ResponseGenerator(llm, st.session_state.vector_store)
                    response = response_generator.generate_response(user_question)
                    st.write(response)
                    logging.info("Response generated successfully.")
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                logging.error(f"Error generating response: {str(e)}")

if __name__ == '__main__':
    main()
