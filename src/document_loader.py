import os
from typing import List, Dict, Any
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredFileLoader
)
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)
from langchain.schema import Document
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader


class DocumentLoader:
    """Enhanced document loader using LangChain"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.supported_formats = ['.pdf', '.txt', '.docx']
    
    def load_document_from_bytes(self, file_content: bytes, filename: str) -> List[Document]:
        """Load document from bytes using appropriate LangChain loader"""
        file_ext = os.path.splitext(filename.lower())[1]
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        try:
            documents = self._load_with_langchain_loader(tmp_file_path, file_ext, filename)
            return documents
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    def _load_with_langchain_loader(self, file_path: str, file_ext: str, original_filename: str) -> List[Document]:
        """Load document using appropriate LangChain loader"""
        try:
            if file_ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_ext == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_ext == '.docx':
                loader = Docx2txtLoader(file_path)
            else:
                # Fallback to unstructured loader
                loader = UnstructuredFileLoader(file_path)
            
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    'filename': original_filename,
                    'file_type': file_ext,
                    'source': original_filename
                })
            
            return documents
            
        except Exception as e:
            st.error(f"Error loading {original_filename}: {str(e)}")
            # Fallback: create document from raw text
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                return [Document(
                    page_content=content,
                    metadata={
                        'filename': original_filename,
                        'file_type': file_ext,
                        'source': original_filename
                    }
                )]
            except Exception as fallback_error:
                raise Exception(f"Failed to load document: {str(fallback_error)}")
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks using LangChain text splitter"""
        return self.text_splitter.split_documents(documents)
    
    def process_uploaded_files(self, uploaded_files) -> List[Document]:
        """Process multiple uploaded files and return split documents"""
        all_documents = []
        
        for uploaded_file in uploaded_files:
            try:
                # Load document
                documents = self.load_document_from_bytes(
                    uploaded_file.getvalue(),
                    uploaded_file.name
                )
                
                # Split into chunks
                split_docs = self.split_documents(documents)
                all_documents.extend(split_docs)
                
                st.success(f"✅ Processed {uploaded_file.name}: {len(split_docs)} chunks")
                
            except Exception as e:
                st.error(f"❌ Failed to process {uploaded_file.name}: {str(e)}")
        
        return all_documents