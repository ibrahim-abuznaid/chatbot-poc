import os
from dotenv import load_dotenv

# Load environment variables from .env file early
load_dotenv()

# Set up LangSmith environment variables early.
os.environ["LANGSMITH_TRACING"] = "true"
# Ensure the tenant is set; if LANGSMITH_TENANT is not in your env, fallback to 'rag-building-requirements'
os.environ["LANGSMITH_TENANT"] = os.getenv("LANGSMITH_TENANT", "rag-building-requirements")

import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langsmith import traceable
from langsmith.wrappers import wrap_openai
import openai

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Create OpenAI client (with tracing via wrap_openai)
openai_client = wrap_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

# Setup Components
llm = ChatOpenAI(
    model="gpt-4o", 
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Query Translation Step
@traceable(name="query_translation")
def translate_query(query):
    """Translate or refine the query with LangSmith tracing"""
    translation_prompt = ChatPromptTemplate.from_template(
        "Rewrite the following query to make it more clear and specific for a search: {query}"
    )
    translation_chain = translation_prompt | llm
    translated_query = translation_chain.invoke({"query": query}).content
    return translated_query

# Document Preparation
@traceable(name="document_preparation")
def prepare_documents(documents):
    """Split documents into chunks with tracing"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20
    )
    splits = text_splitter.split_documents(documents)
    return splits

# Vector Store Indexing
@traceable(name="vector_store_creation")
def create_vector_store(document_splits):
    """Create a vector store from document splits with tracing"""
    vectorstore = Chroma.from_documents(
        documents=document_splits, 
        embedding=embeddings
    )
    return vectorstore

# Traceable Retrieval
@traceable(name="document_retrieval", run_type="retriever")
def retrieve_documents(vectorstore, query):
    """Retrieve most similar documents with tracing"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    retrieved_docs = retriever.invoke(query)
    
    formatted_docs = [
        {
            "page_content": doc.page_content,
            "metadata": dict(doc.metadata)
        } for doc in retrieved_docs
    ]
    
    return formatted_docs

# Retrieval and RAG Chain
@traceable(name="rag_chain_creation")
def create_rag_chain(vectorstore):
    """Create a retrieval-augmented generation chain with tracing"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    rag_prompt = ChatPromptTemplate.from_template(
        """You are an expert assistant. Answer the question based only on the following context: 
        {context}
        
        Question: {question}
        """
    )
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        | rag_prompt 
        | llm
    )
    
    return rag_chain

# Main RAG Pipeline
@traceable(name="rag_pipeline")
def process_query(user_query):
    """Complete RAG pipeline with comprehensive LangSmith tracing"""
    try:
        # Load documents from docs.txt (assumed to be in the same directory)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        docs_path = os.path.join(script_dir, "docs.txt")
        loader = TextLoader(docs_path, encoding="utf-8")
        documents = loader.load()

        # Query Translation
        refined_query = translate_query(user_query)
        
        # Document Preparation
        document_splits = prepare_documents(documents)
        
        # Vector Store Creation
        vectorstore = create_vector_store(document_splits)
        
        # Document Retrieval
        retrieved_docs = retrieve_documents(vectorstore, refined_query)
        
        # RAG Chain
        rag_chain = create_rag_chain(vectorstore)
        
        # Generate Response
        response = rag_chain.invoke(refined_query)
        
        return {
            "original_query": user_query,
            "refined_query": refined_query,
            "response": response.content,
            "retrieved_documents": retrieved_docs
        }
        
    except Exception as e:
        print(f"Pipeline error: {str(e)}")
        raise Exception(f"Failed to process query: {str(e)}")

@app.route("/api/send-message", methods=["POST"])
def send_message():
    try:
        data = request.json
        message = data.get("message")
        
        if not message:
            return jsonify({"error": "No message provided"}), 400
            
        result = process_query(message)
        
        return jsonify({
            "reply": result["response"],
            "context": {
                "original_query": result["original_query"],
                "refined_query": result["refined_query"],
                "retrieved_documents": result["retrieved_documents"]
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    port = int(os.getenv("PORT", 3001))
    app.run(host="0.0.0.0", port=port, debug=True) 