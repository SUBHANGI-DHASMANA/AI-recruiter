import json
import io
import fitz 
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama

# Function to process PDF resume from bytes and extract relevant data
from langchain_core.documents import Document

def process_pdf_resume_from_bytes(pdf_bytes: bytes) -> dict:
    # Use fitz to open PDF directly from bytes
    doc = fitz.open(stream=pdf_bytes)
    
    # Extract text from each page
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    print("Document text extracted successfully.")
    
    # Create a proper Document object
    data = [Document(page_content=text)]
    
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(data)
    print("Document split into chunks.")
    
    # Create vector database
    ollama.pull("nomic-embed-text")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        collection_name="simple-rag",
    )
    print("Done adding to vector database.")
    
    # Set up retriever with limited results
    model = "llama3.2"
    llm = ChatOllama(model=model)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})  # Limit to 3 results
    
    # Create prompt template
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """
    prompt = PromptTemplate.from_template(template)
    
    # Create chain with limited retrieval
    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )
    
    # Extract resume details
    resume_details = {}
    resume_details["name"] = chain.invoke("Extract full name from the document.")
    resume_details["email"] = chain.invoke("Find email address in the document.")
    resume_details["experience"] = chain.invoke("Summarize professional experience.")
    resume_details["skills"] = chain.invoke("List key skills mentioned.")
    
    return resume_details