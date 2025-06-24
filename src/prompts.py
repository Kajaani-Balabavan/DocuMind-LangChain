from langchain.prompts import PromptTemplate

def get_qa_prompt():
    """Get the QA prompt template"""
    template = """You are a helpful AI assistant that answers questions based on the provided context. 
Use the following pieces of context to answer the question at the end. 

If you don't know the answer based on the context provided, just say that you don't know. 
Don't try to make up an answer.

Keep your answer concise but comprehensive, and always base it on the provided context.

Context:
{context}

Question: {question}

Helpful Answer:"""
    
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

def get_condense_prompt():
    """Get the prompt for condensing follow-up questions"""
    template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
    
    return PromptTemplate(
        template=template,
        input_variables=["chat_history", "question"]
    )

def get_custom_qa_prompt(system_message: str):
    """Get a custom QA prompt with system message"""
    template = f"""{system_message}

Use the following pieces of context to answer the question at the end.

Context:
{{context}}

Question: {{question}}

Answer:"""
    
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

# Predefined prompts for different use cases
PROMPTS = {
    "default": get_qa_prompt(),
    "concise": get_custom_qa_prompt(
        "You are a helpful AI assistant. Provide concise, direct answers based on the context."
    ),
    "detailed": get_custom_qa_prompt(
        "You are a thorough AI assistant. Provide detailed, comprehensive answers with explanations based on the context."
    ),
    "analytical": get_custom_qa_prompt(
        "You are an analytical AI assistant. Provide answers with analysis, insights, and relevant details from the context."
    )
}