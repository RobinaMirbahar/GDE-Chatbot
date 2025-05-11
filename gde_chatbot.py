import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import GoogleDriveLoader, PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import re
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Configure the app
st.set_page_config(
    page_title="GDE Program Assistant", 
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Constants ---
DOCUMENT_SOURCES = {
    "welcome_guide": "http://bit.ly/welcome-gde",
    "program_overview": "https://docs.google.com/document/d/1CYq8965IO7flA8UW1gbDKSwezO7EvcsIPKxU4iL2whk/edit",
    "travel_policy": "http://goo.gle/travel-program-policy"
}

# Initialize Google Drive credentials
def get_drive_credentials():
    try:
        credentials = service_account.Credentials.from_service_account_file(
            "service_account.json",
            scopes=["https://www.googleapis.com/auth/drive.readonly"]
        )
        return credentials
    except Exception as e:
        st.error(f"Failed to load Google Drive credentials: {str(e)}")
        return None

# --- Authentication ---
def authenticate_user(email):
    """Check if email is authorized"""
    if not email:
        return False
    return email.endswith("@google.com") or "advocu.com" in email.lower()

# --- Document Processing ---
@st.cache_resource(ttl=86400)  # Refresh daily
def load_and_process_documents():
    """Load and process all specified GDE documents"""
    docs = []
    credentials = get_drive_credentials()
    
    if not credentials:
        return []
    
    # Load Welcome Guide (PDF from URL)
    try:
        welcome_loader = WebBaseLoader(DOCUMENT_SOURCES["welcome_guide"])
        welcome_docs = welcome_loader.load()
        for doc in welcome_docs:
            doc.metadata["source"] = "GDE Welcome Guide"
            doc.metadata["document_type"] = "welcome"
        docs.extend(welcome_docs)
    except Exception as e:
        st.error(f"Error loading Welcome Guide: {e}")
    
    # Load Program Overview (Google Doc)
    try:
        program_loader = GoogleDriveLoader(
            document_ids=["1CYq8965IO7flA8UW1gbDKSwezO7EvcsIPKxU4iL2whk"],
            credentials=credentials
        )
        program_docs = program_loader.load()
        for doc in program_docs:
            doc.metadata["source"] = "GDE Program Overview"
            doc.metadata["document_type"] = "overview"
        docs.extend(program_docs)
    except Exception as e:
        st.error(f"Error loading Program Overview: {e}")
    
    # Load Travel Policy (PDF from URL)
    try:
        travel_loader = WebBaseLoader(DOCUMENT_SOURCES["travel_policy"])
        travel_docs = travel_loader.load()
        for doc in travel_docs:
            doc.metadata["source"] = "GDE Travel Policy"
            doc.metadata["document_type"] = "travel"
        docs.extend(travel_docs)
    except Exception as e:
        st.error(f"Error loading Travel Policy: {e}")
    
    # Process documents with special handling for different formats
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " (?=\\d+\\. )", " ", ""]
    )
    
    processed_docs = []
    for doc in docs:
        # Clean and standardize document content
        doc.page_content = clean_document_content(doc.page_content, doc.metadata["document_type"])
        doc.metadata["loaded_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        processed_docs.extend(text_splitter.split_documents([doc]))
    
    return processed_docs

def clean_document_content(content, doc_type):
    """Clean document content based on its type"""
    if doc_type == "welcome":
        # Remove headers/footers from PDF
        content = re.sub(r'Page \d+ of \d+', '', content)
    elif doc_type == "travel":
        # Clean policy document formatting
        content = re.sub(r'\s{3,}', '  ', content)
    return content.strip()

# --- Location Awareness ---
def get_regional_contact(country_code):
    """Enhanced regional contact with timezone awareness"""
    contacts = {
        "NA": {
            "name": "North America Team", 
            "email": "na-gde@google.com",
            "countries": ["US", "CA", "MX"],
            "office_hours": "9AM-5PM PST"
        },
        "EMEA": {
            "name": "EMEA Team", 
            "email": "emea-gde@google.com",
            "countries": ["GB", "DE", "FR", "IT", "ES", "NL"],
            "office_hours": "9AM-5PM CET"
        },
        "APAC": {
            "name": "APAC Team", 
            "email": "apac-gde@google.com",
            "countries": ["IN", "JP", "KR", "SG", "AU", "NZ"],
            "office_hours": "9AM-5PM SGT"
        },
        "LATAM": {
            "name": "LATAM Team", 
            "email": "latam-gde@google.com",
            "countries": ["BR", "AR", "CL", "CO"],
            "office_hours": "9AM-5PM BRT"
        }
    }
    
    for region, data in contacts.items():
        if country_code.upper() in data["countries"]:
            return data
    
    return {
        "name": "Global GDE Team", 
        "email": "global-gde@google.com",
        "office_hours": "24/7"
    }

# --- Chatbot Initialization ---
@st.cache_resource
def initialize_chatbot():
    """Initialize the retrieval-augmented chatbot with document-specific handling"""
    docs = load_and_process_documents()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # Document-specific prompt template
    prompt = ChatPromptTemplate.from_template("""
    You are an expert assistant for the Google Developer Experts (GDE) program.
    Strictly answer questions ONLY using information from the following documents:
    - GDE Welcome Guide (welcome_guide)
    - GDE Program Overview (program_overview)
    - GDE Travel Policy (travel_policy)
    
    Important rules:
    1. If the answer isn't in these documents, say: "This information isn't available in the current GDE documentation."
    2. Always cite the specific document name when answering.
    3. For policy questions, include the relevant section if possible.
    4. Format responses clearly using markdown.
    
    Context from documents:
    {context}
    
    Question: {input}
    
    Answer:
    """)
    
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 5,
            "score_threshold": 0.7
        }
    )
    
    return create_retrieval_chain(retriever, document_chain)

# --- UI Components ---
def show_authentication():
    """Enhanced authentication sidebar with document info"""
    with st.sidebar:
        st.title("🔐 GDE Authentication")
        email = st.text_input("Enter your Google or Advocu email:")
        
        if email and not authenticate_user(email):
            st.error("Access restricted to GDEs and Googlers")
            st.stop()
        
        country = st.selectbox(
            "Select your country:",
            ["US", "CA", "MX", "BR", "GB", "DE", "FR", "IN", "JP", "SG", "AU", "Other"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("**Documentation Loaded**")
        st.caption("✓ GDE Welcome Guide")
        st.caption("✓ GDE Program Overview")
        st.caption("✓ GDE Travel Policy")
        
        st.markdown("---")
        st.markdown("**Need Help?**")
        contact = get_regional_contact(country)
        st.caption(f"Regional contact: {contact['email']}")
        st.caption(f"Office hours: {contact['office_hours']}")
        
        return email, country

def show_chat_interface():
    """Enhanced chat interface with document awareness"""
    st.title("🤖 GDE Program Assistant")
    st.caption("Ask me about the Welcome Guide, Program Overview, or Travel Policy")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I can answer questions about:\n- GDE Welcome Guide\n- Program Overview\n- Travel Policy\n\nWhat would you like to know?"}
        ]
    
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
            with st.spinner("Searching GDE documents..."):
                try:
                    chatbot = initialize_chatbot()
                    response = chatbot.invoke({"input": prompt})
                    answer = response["answer"]
                    
                    # Enhance with document source info
                    sources = list(set(doc.metadata["source"] for doc in response["context"]))
                    if sources:
                        answer += f"\n\n*Sources: {', '.join(sources)}*"
                    
                    # Add regional contact for location-specific questions
                    if any(keyword in prompt.lower() for keyword in ["country", "region", "contact", "timezone"]):
                        contact = get_regional_contact(st.session_state.country)
                        answer += f"\n\n**Your Regional Contact**: {contact['name']} ({contact['email']})"
                        answer += f"\n**Office Hours**: {contact['office_hours']}"
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error processing your question: {str(e)}")

# --- Main App Flow ---
def main():
    # Check for service account file first
    if not os.path.exists("service_account.json"):
        st.error("Missing service_account.json - needed for Google Drive access")
        st.info("Please ensure you've:")
        st.markdown("""
        1. Created a Google Cloud service account
        2. Enabled Google Drive API
        3. Downloaded the JSON key file
        4. Named it `service_account.json` in this folder
        5. Shared your documents with the service account email
        """)
        return
    
    # Authentication
    email, country = show_authentication()
    st.session_state.authenticated = authenticate_user(email)
    st.session_state.country = country
    
    # Chat interface
    show_chat_interface()

if __name__ == "__main__":
    main()
