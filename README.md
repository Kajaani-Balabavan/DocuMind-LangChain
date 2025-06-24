# 🧠 [DocuMind - LangChain RAG System](https://kajaani-balabavan-documind-langchain-app-db5ifn.streamlit.app/)

A RAG (Retrieval-Augmented Generation) system built with **LangChain**, **Streamlit**, and **Hugging Face APIs**. Features conversational memory, advanced document processing, and multiple vector store options.

## ✨ Key Features

### 🔗 LangChain Integration
- **Conversational Memory**: Remembers context across questions
- **Advanced Chains**: ConversationalRetrievalChain and RetrievalQA

### 📄 Document Processing
- **Multi-format Support**: PDF, DOCX, TXT files
- **Smart Chunking**: Recursive text splitting with overlap
- **Metadata Preservation**: File source and page tracking

### 🔍 Vector Search
- **Multiple Backends**: FAISS and Chroma support
- **Optimized Embeddings**: HuggingFace sentence-transformers

### 🤖 AI-Powered Responses
- **Hugging Face Models**: API integration
- **Source Attribution**: Track answer origins

## 🛠️ Tech Stack

- **Framework**: LangChain + Streamlit
- **Vector Stores**: FAISS, Chroma
- **Embeddings**: HuggingFace sentence-transformers
- **LLM**: Hugging Face Inference API

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/documind-langchain-rag.git
cd DocuMind-LangChain
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Setup
```bash
HF_TOKEN= your huggingface token
```

### 4. Run Application
```bash
streamlit run app.py
```

### 5. Start Using!
1. **Upload Documents**: Use the sidebar to upload PDF, DOCX, or TXT files
2. **Configure Settings**: Choose vector store type and chain configuration
3. **Ask Questions**: Use the chat interface for intelligent Q&A
4. **Analyze Responses**: View confidence scores and source documents

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain**: For the powerful RAG framework
- **Hugging Face**: For model APIs and transformers
- **Streamlit**: For the intuitive web framework
- **FAISS/Chroma**: For efficient vector storage
