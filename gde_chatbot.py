import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import GoogleDriveLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import re
from datetime import datetime

# Configure the app
st.set_page_config(
    page_title="GDE Program Assistant", 
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Authentication ---
def authenticate_user(email):
    """Check if email is authorized"""
    if not email:
        return False
    return email.endswith("@google.com") or "advocu.com" in email.lower()

# --- Document Processing ---
@st.cache_resource(ttl=86400)  # Refresh daily
def load_and_process_documents():
    """Load and process all GDE documents"""
    docs = []
    
    # Load from Google Drive (main program guide)
    try:
        loader = GoogleDriveLoader(
            document_ids=["1MDmMSWhjtq1i9w1wOwyJhG89YOZS9LVBB-WaKpeSfWw"],
            credentials_path="service_account.json"
        )
        docs.extend(loader.load())
    except Exception as e:
        st.error(f"Error loading Google Doc: {e}")
    
    # Load additional PDFs from local 'documents' folder
    try:
        pdf_loader = PyPDFLoader("documents/")
        docs.extend(pdf_loader.load())
    except Exception as e:
        st.error(f"Error loading PDFs: {e}")
    
    # Process documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " (?=\\d+\\. )", " ", ""]  # Preserve numbered sections
    )
    
    processed_docs = []
    for doc in docs:
        # Add metadata for source tracking
        doc.metadata["loaded_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not doc.metadata.get("source"):
            doc.metadata["source"] = "Unknown GDE Document"
        processed_docs.extend(text_splitter.split_documents([doc]))
    
    return processed_docs

# --- Location Awareness ---
def get_regional_contact(country_code):
    """Get regional contact based on country code"""
    contacts = {
        "NA": {"name": "North America Team", "email": "na-gde@google.com", "countries": ["US", "CA", "MX"]},
        "EMEA": {"name": "EMEA Team", "email": "emea-gde@google.com", "countries": ["GB", "DE", "FR", "IT", "ES"]},
        "APAC": {"name": "APAC Team", "email": "apac-gde@google.com", "countries": ["IN", "JP", "KR", "SG", "AU"]},
        "LATAM": {"name": "LATAM Team", "email": "latam-gde@google.com", "countries": ["BR", "AR", "CL"]}
    }
    
    for region, data in contacts.items():
        if country_code.upper() in data["countries"]:
            return data
    
    return {"name": "Global GDE Team", "email": "global-gde@google.com"}

# --- Chatbot Initialization ---
@st.cache_resource
def initialize_chatbot():
    """Initialize the retrieval-augmented chatbot"""
    docs = load_and_process_documents()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # Custom prompt for GDE-specific answers
    prompt = ChatPromptTemplate.from_template("""
    You are an expert assistant for the Google Developer Experts (GDE) program.
    Answer questions STRICTLY based on the provided context from official GDE documents.
    
    Important rules:
    1. If the answer isn't in the documents, say "I couldn't find this information in the GDE documentation."
    2. For policy questions, always cite the specific document and section.
    3. Be concise but thorough.
    4. Format responses using markdown for readability.
    
    Context:
    {context}
    
    Question: {input}
    
    Answer:
    """)
    
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    return create_retrieval_chain(retriever, document_chain)

# --- UI Components ---
def show_authentication():
    """Authentication sidebar"""
    with st.sidebar:
        st.title("🔐 GDE Authentication")
        email = st.text_input("Enter your Google or Advocu email:")
        
        if email and not authenticate_user(email):
            st.error("Access restricted to GDEs and Googlers")
            st.stop()
        
        country = st.selectbox(
            "Select your country:",
            ["US", "CA", "MX", "BR", "GB", "DE", "FR", "IN", "JP", "Other"]
        )
        
        st.markdown("---")
        st.markdown("**Documentation Status**")
        docs = load_and_process_documents()
        unique_sources = list(set(d.metadata["source"] for d in docs))
        for source in sorted(unique_sources):
            st.caption(f"✓ {source}")
        
        return email, country

def show_chat_interface():
    """Main chat interface"""
    st.title("🤖 GDE Program Assistant")
    st.caption("Ask me anything about the GDE program - policies, applications, reporting, and more")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Your GDE question..."):
        if not st.session_state.get("authenticated"):
            st.warning("Please authenticate in the sidebar")
            st.stop()
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get chatbot response
        with st.chat_message("assistant"):
            with st.spinner("Consulting GDE docs..."):
                try:
                    chatbot = initialize_chatbot()
                    response = chatbot.invoke({"input": prompt})
                    answer = response["answer"]
                    
                    # Add regional contact if location-relevant
                    location_keywords = ["country", "region", "contact", "local", "move", "relocate"]
                    if any(keyword in prompt.lower() for keyword in location_keywords):
                        contact = get_regional_contact(st.session_state.country)
                        answer += f"\n\n**Regional Contact**: {contact['name']} ({contact['email']})"
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# --- Main App Flow ---
def main():
    # Authentication
    email, country = show_authentication()
    st.session_state.authenticated = authenticate_user(email)
    st.session_state.country = country
    
    # Chat interface
    show_chat_interface()

if __name__ == "__main__":
    if not os.path.exists("service_account.json"):
        st.error("Missing service_account.json - needed for Google Drive access")
    else:
        main()
