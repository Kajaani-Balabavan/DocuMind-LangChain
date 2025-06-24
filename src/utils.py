import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import List, Dict
from langchain.schema import Document

def display_chat_message(role: str, content: str):
    """Display chat message with proper styling"""
    with st.chat_message(role):
        st.write(content)

def create_confidence_chart(confidence: float):
    """Create a confidence score visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Score (%)"},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def create_sources_chart(sources: List[Document]):
    """Create a chart showing source document distribution"""
    if not sources:
        return None
    
    # Count sources by filename
    source_counts = {}
    for doc in sources:
        filename = doc.metadata.get('filename', 'Unknown')
        source_counts[filename] = source_counts.get(filename, 0) + 1
    
    fig = px.bar(
        x=list(source_counts.keys()),
        y=list(source_counts.values()),
        title="Source Document Distribution",
        labels={'x': 'Documents', 'y': 'Number of Chunks Used'}
    )
    fig.update_layout(height=300)
    return fig

def format_sources(sources: List[Document], max_length: int = 300):
    """Format source documents for display"""
    formatted_sources = []
    for i, doc in enumerate(sources, 1):
        content = doc.page_content
        metadata = doc.metadata
        
        truncated = content[:max_length] + "..." if len(content) > max_length else content
        filename = metadata.get('filename', 'Unknown')
        page = metadata.get('page', 'N/A')
        
        formatted_sources.append({
            "source_num": i,
            "content": truncated,
            "filename": filename,
            "page": page,
            "metadata": metadata
        })
    
    return formatted_sources

def validate_file_upload(uploaded_file):
    """Validate uploaded file"""
    if uploaded_file is None:
        return False, "Please upload a file."
    
    allowed_extensions = ['.pdf', '.txt', '.docx']
    file_extension = '.' + uploaded_file.name.lower().split('.')[-1]
    
    if file_extension not in allowed_extensions:
        return False, f"Unsupported file type. Please upload: {', '.join(allowed_extensions)}"
    
    if uploaded_file.size > 50 * 1024 * 1024:  # 50MB limit
        return False, "File size too large. Please upload files smaller than 50MB."
    
    return True, "File is valid."

def get_system_stats(vector_store_manager, rag_chain):
    """Get comprehensive system statistics"""
    stats = {}
    
    # Vector store stats
    if vector_store_manager.vector_store:
        vs_stats = vector_store_manager.get_stats()
        stats.update({
            "Vector Store Type": vs_stats.get("type", "Unknown"),
            "Documents in Store": vs_stats.get("documents", 0),
            "Embedding Model": vs_stats.get("embedding_model", "Unknown")
        })
    
    # Chain stats
    if rag_chain:
        stats.update({
            "LLM Model": rag_chain.model_name,
            "Memory Window": rag_chain.memory.k if rag_chain.memory else 0,
            "Chain Type": "Conversational" if hasattr(rag_chain.qa_chain, 'memory') else "Simple"
        })
    
    return stats

def export_conversation(chat_history: List[Dict]) -> str:
    """Export conversation history to text format"""
    if not chat_history:
        return "No conversation to export."
    
    export_text = "# DocuMind Conversation Export\n\n"
    for i, message in enumerate(chat_history, 1):
        role = message["role"].title()
        content = message["content"]
        export_text += f"## {role} {i}\n{content}\n\n"
    
    return export_text

def create_document_summary(documents: List[Document]) -> Dict:
    """Create a summary of loaded documents"""
    if not documents:
        return {"total_docs": 0, "total_chunks": 0, "files": []}
    
    file_stats = {}
    total_chars = 0
    
    for doc in documents:
        filename = doc.metadata.get('filename', 'Unknown')
        if filename not in file_stats:
            file_stats[filename] = {
                "chunks": 0,
                "total_chars": 0,
                "file_type": doc.metadata.get('file_type', 'Unknown')
            }
        
        file_stats[filename]["chunks"] += 1
        file_stats[filename]["total_chars"] += len(doc.page_content)
        total_chars += len(doc.page_content)
    
    return {
        "total_docs": len(file_stats),
        "total_chunks": len(documents),
        "total_chars": total_chars,
        "files": file_stats
    }

def display_document_preview(documents: List[Document], max_docs: int = 3):
    """Display a preview of loaded documents"""
    if not documents:
        st.info("No documents loaded yet.")
        return
    
    st.subheader("📄 Document Preview")
    
    for i, doc in enumerate(documents[:max_docs]):
        with st.expander(f"Document {i+1}: {doc.metadata.get('filename', 'Unknown')}"):
            st.write(f"**File:** {doc.metadata.get('filename', 'Unknown')}")
            st.write(f"**Type:** {doc.metadata.get('file_type', 'Unknown')}")
            st.write(f"**Characters:** {len(doc.page_content)}")
            st.write("**Content Preview:**")
            st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
    
    if len(documents) > max_docs:
        st.info(f"Showing {max_docs} of {len(documents)} document chunks.")