from typing import List, Dict, Any, Optional
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseRetriever
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from .prompts import get_qa_prompt, get_condense_prompt
import streamlit as st


class RAGChain:
    """Modern RAG chain using langchain_huggingface"""

    def __init__(
        self,
        hf_token: Optional[str] = None,
        model_name: str = "deepseek-ai/DeepSeek-R1",
        # provider: str = "together",
        provider: str = "hyperbolic",
        temperature: float = 0.7,
        max_tokens: int = 512,
        memory_window: int = 5,
    ):
        self.hf_token = hf_token
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = provider

        # Initialize LLM
        self.llm = self._initialize_llm()

        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            k=memory_window,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )

        self.qa_chain = None
        self.retriever = None

    def _initialize_llm(self):
        """Initialize the language model using langchain_huggingface"""
        try:
            llm = HuggingFaceEndpoint(
                repo_id=self.model_name,
                provider=self.provider,
                temperature=self.temperature,
                max_new_tokens=self.max_tokens,
                huggingfacehub_api_token=self.hf_token,
            )
            model = ChatHuggingFace(llm=llm)
            st.success(f"✅ Using {self.model_name} via {self.provider}")
            return model
        except Exception as e:
            st.error(f"❌ Failed to initialize LLM: {str(e)}")
            raise

    def setup_qa_chain(self, retriever: BaseRetriever, chain_type: str = "conversational"):
        """Setup the QA chain"""
        self.retriever = retriever

        try:
            if chain_type == "conversational":
                self.qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=retriever,
                    memory=self.memory,
                    return_source_documents=True,
                    verbose=True,
                    combine_docs_chain_kwargs={
                        "prompt": get_qa_prompt()
                    },
                    condense_question_prompt=get_condense_prompt()
                )
            else:
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": get_qa_prompt()}
                )

            st.success(f"✅ QA chain setup complete ({chain_type})")

        except Exception as e:
            st.error(f"❌ Error setting up QA chain: {str(e)}")
            raise

    def query(self, question: str, include_sources: bool = True) -> Dict[str, Any]:
        """Ask a question"""
        if self.qa_chain is None:
            return {
                "answer": "Please upload documents and setup the QA chain first.",
                "source_documents": [],
                "confidence": 0.0
            }

        try:
            with st.spinner("🤔 Thinking..."):
                result = self.qa_chain.invoke({"question": question})

            answer = result.get("answer", "I don't have enough information to answer.")
            source_docs = result.get("source_documents", [])
            confidence = self._calculate_confidence(question, source_docs)

            return {
                "answer": answer,
                "source_documents": source_docs if include_sources else [],
                "confidence": confidence,
                "sources_count": len(source_docs)
            }

        except Exception as e:
            st.error(f"❌ Query error: {str(e)}")
            return {
                "answer": f"Error: {str(e)}",
                "source_documents": [],
                "confidence": 0.0
            }

    def _calculate_confidence(self, question: str, source_docs: List) -> float:
        """Estimate confidence"""
        if not source_docs:
            return 0.0
        try:
            base = min(len(source_docs) * 0.2, 0.8)
            avg_len = sum(len(doc.page_content) for doc in source_docs) / len(source_docs)
            bonus = min(avg_len / 1000, 0.2)
            return min(base + bonus, 1.0)
        except:
            return 0.5

    def clear_memory(self):
        if self.memory:
            self.memory.clear()
            st.success("✅ Memory cleared")

    def get_memory_summary(self) -> str:
        if not self.memory or not hasattr(self.memory, 'chat_memory'):
            return "No memory"
        messages = self.memory.chat_memory.messages
        return f"Conversation history: {len(messages)} messages" if messages else "No conversation history"

    def update_chain_config(self, **kwargs):
        """Update settings and reinitialize model"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        if any(k in kwargs for k in ['temperature', 'max_tokens', 'model_name', 'provider']):
            self.llm = self._initialize_llm()
            if self.retriever:
                self.setup_qa_chain(self.retriever)
