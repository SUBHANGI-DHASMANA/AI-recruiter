import streamlit as st
import asyncio
import fitz  # PyMuPDF for PDF processing
from agents.orchestrator import OrchestratorAgent  # Ensure this is correctly imported
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Streamlit app setup
st.set_page_config(page_title="Job Application Processor", layout="wide")
st.title("Job Application Processor")

# Sidebar information
st.sidebar.title("About")
st.sidebar.write("This application uses AI agents to process resumes and provide job recommendations.")

# File upload section
uploaded_file = st.file_uploader("Upload Resume (PDF format)", type=["pdf"])

# Initialize OrchestratorAgent
orchestrator = OrchestratorAgent()

def process_pdf_resume_from_bytes(pdf_bytes: bytes) -> dict:
    # Use fitz to open PDF directly from bytes
    doc = fitz.open(stream=pdf_bytes)
    
    # Extract text from each page and create Document objects for each chunk of text.
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    
    # Create Document object with extracted text.
    data = [Document(page_content=text)]
    
    # Split the document into chunks.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(data)
    
    # Create vector database.
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        collection_name="simple-rag",
    )

    # Prepare resume data for processing.
    resume_data = {
        'full_text': text,
        'chunks': [chunk.page_content for chunk in chunks]  # Use chunk contents as needed.
    }

    return resume_data

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()
    
    try:
        resume_data = process_pdf_resume_from_bytes(pdf_bytes)
        
        # Use asyncio to run the process_application method.
        result = asyncio.run(orchestrator.process_application(resume_data))
        
        st.write(result)  # Display the processed result in Streamlit app
        
    except Exception as e:
        st.error(f"An error occurred: {e}")  # Display error messages if any
