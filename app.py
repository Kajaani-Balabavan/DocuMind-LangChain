import streamlit as st
import os
from dotenv import load_dotenv
from src.document_loader import DocumentLoader
from src.vector_store import VectorStoreManager
from src.rag_chain import RAGChain
from src.utils import (
    display_chat_message, 
    create_confidence_chart,
    create_sources_chart,
    format_sources, 
    validate_file_upload,
    get_system_stats,
    export_conversation,
    create_document_summary,
    display_document_preview
)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="DocuMind - LangChain RAG Q&A System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stats-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
        color: #333;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .source-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #28a745;
        margin: 0.5rem 0;
        color: #333;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    .chat-container {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .feature-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        margin: 0.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'document_loader' not in st.session_state:
        st.session_state.document_loader = DocumentLoader()
    
    if 'vector_store_manager' not in st.session_state:
        st.session_state.vector_store_manager = VectorStoreManager(
            vector_store_type=st.session_state.get('vector_store_type', 'faiss')
        )
    
    if 'rag_chain' not in st.session_state:
        hf_token = os.getenv('HF_TOKEN') or st.secrets.get('HF_TOKEN', None)
        st.session_state.rag_chain = RAGChain(hf_token=hf_token)
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    
    if 'all_documents' not in st.session_state:
        st.session_state.all_documents = []

def main():
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 DocuMind - LangChain RAG System</h1>
        <p>Advanced document Q&A powered by LangChain, vector embeddings, and conversational AI</p>
        <div style="margin-top: 1rem;">
            <span class="feature-highlight">🔗 LangChain</span>
            <span class="feature-highlight">🧠 Conversational Memory</span>
            <span class="feature-highlight">🔍 Vector Search</span>
            <span class="feature-highlight">📊 Source Analytics</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Model Configuration
        with st.expander("🤖 Model Settings", expanded=False):
            vector_store_type = st.selectbox(
                "Vector Store",
                ["faiss", "chroma"],
                index=0 if st.session_state.get('vector_store_type', 'faiss') == 'faiss' else 1
            )
            
            chain_type = st.selectbox(
                "Chain Type",
                ["conversational", "simple"],
                help="Conversational chains remember chat history"
            )
            
            retrieval_k = st.slider(
                "Sources to Retrieve",
                min_value=1,
                max_value=10,
                value=4,
                help="Number of source documents to retrieve"
            )
            
            if st.button("🔄 Apply Settings"):
                # Update vector store if changed
                if vector_store_type != st.session_state.get('vector_store_type'):
                    st.session_state.vector_store_type = vector_store_type
                    st.session_state.vector_store_manager = VectorStoreManager(
                        vector_store_type=vector_store_type
                    )
                
                st.session_state.chain_type = chain_type
                st.session_state.retrieval_k = retrieval_k
                st.success("Settings applied!")
        
        st.header("📁 Document Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            help="Upload PDF, TXT, or DOCX files to create your knowledge base"
        )
        
        # Process uploaded files
        if uploaded_files:
            new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
            
            if new_files:
                with st.spinner(f"Processing {len(new_files)} files..."):
                    try:
                        # Process documents
                        documents = st.session_state.document_loader.process_uploaded_files(new_files)
                        
                        if documents:
                            # Add to vector store
                            st.session_state.vector_store_manager.add_documents(documents)
                            
                            # Setup/update RAG chain
                            retriever = st.session_state.vector_store_manager.get_retriever(
                                k=st.session_state.get('retrieval_k', 4)
                            )
                            st.session_state.rag_chain.setup_qa_chain(
                                retriever, 
                                chain_type=st.session_state.get('chain_type', 'conversational')
                            )
                            
                            # Update session state
                            st.session_state.all_documents.extend(documents)
                            st.session_state.processed_files.extend([f.name for f in new_files])
                            
                            st.success(f"✅ Successfully processed {len(new_files)} files!")
                            st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error processing files: {str(e)}")
        
        # Document summary
        if st.session_state.all_documents:
            st.header("📊 Document Summary")
            doc_summary = create_document_summary(st.session_state.all_documents)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Files", doc_summary["total_docs"])
                st.metric("Chunks", doc_summary["total_chunks"])
            with col2:
                st.metric("Characters", f"{doc_summary['total_chars']:,}")
                st.metric("Avg per Chunk", f"{doc_summary['total_chars'] // doc_summary['total_chunks']:,}")
            
            # File details
            with st.expander("📄 File Details"):
                for filename, stats in doc_summary["files"].items():
                    st.write(f"**{filename}**")
                    st.write(f"- Chunks: {stats['chunks']}")
                    st.write(f"- Characters: {stats['total_chars']:,}")
                    st.write(f"- Type: {stats['file_type']}")
                    st.write("---")
        
        # System statistics
        if st.session_state.processed_files:
            st.header("⚡ System Stats")
            stats = get_system_stats(st.session_state.vector_store_manager, st.session_state.rag_chain)
            
            for key, value in stats.items():
                st.markdown(f"""
                <div class="stats-card">
                    <strong>{key}:</strong> {value}
                </div>
                """, unsafe_allow_html=True)
        
        # Memory management
        if st.session_state.rag_chain.memory:
            st.header("🧠 Memory Management")
            memory_summary = st.session_state.rag_chain.get_memory_summary()
            st.info(memory_summary)
            
            if st.button("🗑️ Clear Memory"):
                st.session_state.rag_chain.clear_memory()
                st.rerun()
        
        # Export conversation
        if st.session_state.chat_history:
            st.header("💾 Export")
            if st.button("📥 Export Conversation"):
                export_text = export_conversation(st.session_state.chat_history)
                st.download_button(
                    label="Download Conversation",
                    data=export_text,
                    file_name="documind_conversation.md",
                    mime="text/markdown"
                )
        
        # Clear all data
        if st.button("🗑️ Clear All Data", type="secondary"):
            for key in ['vector_store_manager', 'rag_chain', 'chat_history', 'processed_files', 'all_documents']:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("All data cleared!")
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for message in st.session_state.chat_history:
                display_chat_message(message["role"], message["content"])
        
        # Query input
        if query := st.chat_input("Ask a question about your documents..."):
            if not st.session_state.processed_files:
                st.warning("Please upload some documents first!")
            else:
                # Add user message to chat
                st.session_state.chat_history.append({"role": "user", "content": query})
                
                with chat_container:
                    display_chat_message("user", query)
                
                # # Get RAG response
                # with st.spinner("🤔 Analyzing documents and generating response..."):
                #     response = st.session_state.rag_chain.query(query, include_sources=True)

                with chat_container:
                    thinking_placeholder = st.empty()
                    thinking_placeholder.markdown("🤖 _Analyzing documents..._")

                response = st.session_state.rag_chain.query(query, include_sources=True)
                assistant_message = response["answer"]

                # Replace placeholder with formatted assistant message
                thinking_placeholder.markdown(f"🧠 **Answer:** {assistant_message}")

                
                # Display assistant response
                assistant_message = response["answer"]
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_message})
                
                with chat_container:
                    display_chat_message("assistant", assistant_message)
                
                # Store response for sidebar display
                st.session_state.last_response = response
                st.rerun()
    
    with col2:
        st.header("📊 Response Analysis")
        
        if hasattr(st.session_state, 'last_response'):
            response = st.session_state.last_response
            
            # Confidence visualization
            if response.get("confidence", 0) > 0:
                st.subheader("🎯 Confidence Score")
                fig = create_confidence_chart(response["confidence"])
                st.plotly_chart(fig, use_container_width=True)
            
            # Response metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sources Used", response.get("sources_count", 0))
            with col2:
                st.metric("Confidence", f"{response.get('confidence', 0):.1%}")
            
            # Source documents analysis
            source_docs = response.get("source_documents", [])
            if source_docs:
                st.subheader("📚 Source Documents")
                
                # Source distribution chart
                sources_chart = create_sources_chart(source_docs)
                if sources_chart:
                    st.plotly_chart(sources_chart, use_container_width=True)
                
                # Detailed source information
                formatted_sources = format_sources(source_docs)
                
                for source in formatted_sources:
                    with st.expander(f"Source {source['source_num']}: {source['filename']}"):
                        st.write(f"**File:** {source['filename']}")
                        st.write(f"**Page:** {source['page']}")
                        st.write("**Content:**")
                        st.text_area(
                            "Content", 
                            source['content'], 
                            height=100,
                            key=f"source_{source['source_num']}"
                        )
                        
                        # Show metadata
                        if source['metadata']:
                            st.write("**Metadata:**")
                            st.json(source['metadata'])
            
            # # Query suggestions
            # st.subheader("💡 Suggested Questions")
            # suggestions = [
            #     "What are the main topics in these documents?",
            #     "Can you summarize the key points?",
            #     "What are the most important findings?",
            #     "Are there any specific recommendations?",
            #     "What conclusions can be drawn?"
            # ]
            
            # for suggestion in suggestions:
            #     if st.button(suggestion, key=f"suggestion_{suggestion[:20]}"):
            #         st.session_state.suggested_query = suggestion
            #         st.rerun()
        
        else:
            st.info("Upload documents and ask a question to see response analysis!")
            
            # Document preview
            if st.session_state.all_documents:
                display_document_preview(st.session_state.all_documents)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p><strong>DocuMind - LangChain RAG System</strong></p>
        <p>🔗 Built with LangChain • 🤗 Hugging Face • 🔍 FAISS/Chroma • ⚡ Streamlit</p>
        <p>📧 Contact: kajaani1705@gmail.com </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()