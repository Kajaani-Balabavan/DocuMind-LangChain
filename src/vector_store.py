import os
from typing import List, Optional
from langchain.vectorstores import FAISS, Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


class VectorStoreManager:
    """Enhanced vector store manager using LangChain"""
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 vector_store_type: str = "faiss",
                 persist_directory: str = "./data/vector_db"):
        
        self.embedding_model_name = embedding_model
        self.vector_store_type = vector_store_type.lower()
        self.persist_directory = persist_directory
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vector_store = None
        
        # Ensure directory exists
        os.makedirs(persist_directory, exist_ok=True)
    
    def create_vector_store(self, documents: List[Document]) -> None:
        """Create a new vector store from documents"""
        if not documents:
            raise ValueError("No documents provided to create vector store")
        
        try:
            if self.vector_store_type == "faiss":
                self.vector_store = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
            elif self.vector_store_type == "chroma":
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory
                )
                self.vector_store.persist()
            else:
                raise ValueError(f"Unsupported vector store type: {self.vector_store_type}")
                
            st.success(f"✅ Created {self.vector_store_type.upper()} vector store with {len(documents)} documents")
            
        except Exception as e:
            st.error(f"❌ Error creating vector store: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to existing vector store"""
        if not documents:
            return
        
        try:
            if self.vector_store is None:
                self.create_vector_store(documents)
            else:
                if self.vector_store_type == "faiss":
                    # For FAISS, we need to merge stores
                    new_store = FAISS.from_documents(
                        documents=documents,
                        embedding=self.embeddings
                    )
                    self.vector_store.merge_from(new_store)
                elif self.vector_store_type == "chroma":
                    self.vector_store.add_documents(documents)
                    self.vector_store.persist()
                
                st.success(f"✅ Added {len(documents)} documents to vector store")
                
        except Exception as e:
            st.error(f"❌ Error adding documents: {str(e)}")
            raise
    
    def get_retriever(self, 
                     search_type: str = "similarity", 
                     k: int = 4,
                     search_kwargs: Optional[dict] = None):
        """Get a retriever from the vector store"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Add documents first.")
        
        if search_kwargs is None:
            search_kwargs = {"k": k}
        else:
            search_kwargs["k"] = k
        
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
    
    def similarity_search(self, 
                         query: str, 
                         k: int = 4,
                         filter_dict: Optional[dict] = None) -> List[Document]:
        """Perform similarity search"""
        if self.vector_store is None:
            return []
        
        try:
            if filter_dict:
                return self.vector_store.similarity_search(
                    query=query,
                    k=k,
                    filter=filter_dict
                )
            else:
                return self.vector_store.similarity_search(query=query, k=k)
        except Exception as e:
            st.error(f"Error in similarity search: {str(e)}")
            return []
    
    def save_local(self, folder_path: str) -> None:
        """Save vector store locally"""
        if self.vector_store is None:
            raise ValueError("No vector store to save")
        
        try:
            if self.vector_store_type == "faiss":
                self.vector_store.save_local(folder_path)
            elif self.vector_store_type == "chroma":
                # Chroma persists automatically
                pass
            
            st.success(f"✅ Vector store saved to {folder_path}")
            
        except Exception as e:
            st.error(f"❌ Error saving vector store: {str(e)}")
            raise
    
    def load_local(self, folder_path: str) -> None:
        """Load vector store from local storage"""
        try:
            if self.vector_store_type == "faiss" and os.path.exists(os.path.join(folder_path, "index.faiss")):
                self.vector_store = FAISS.load_local(
                    folder_path=folder_path,
                    embeddings=self.embeddings
                )
                st.success(f"✅ Loaded FAISS vector store from {folder_path}")
                
            elif self.vector_store_type == "chroma" and os.path.exists(self.persist_directory):
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                st.success(f"✅ Loaded Chroma vector store from {self.persist_directory}")
                
        except Exception as e:
            st.warning(f"Could not load existing vector store: {str(e)}")
    
    def get_stats(self) -> dict:
        """Get vector store statistics"""
        if self.vector_store is None:
            return {"documents": 0, "type": self.vector_store_type}
        
        try:
            if self.vector_store_type == "faiss":
                return {
                    "documents": self.vector_store.index.ntotal,
                    "type": "FAISS",
                    "embedding_model": self.embedding_model_name
                }
            elif self.vector_store_type == "chroma":
                collection = self.vector_store._collection
                return {
                    "documents": collection.count(),
                    "type": "Chroma",
                    "embedding_model": self.embedding_model_name
                }
        except:
            return {"documents": "Unknown", "type": self.vector_store_type}