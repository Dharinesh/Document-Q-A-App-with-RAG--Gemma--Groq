import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Load GROQ API KEY and Google API KEY
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("Document Q&A")

# Initialize Groq language model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma-7b-it") #change to the needed model

model_name = "gemma-7b-it"

# Define prompt template
prompt = ChatPromptTemplate.from_template(
    """ Answer the questions based on the provided context only. Please provide the most accurate response based on the question <context> {context} <context> Questions:{input} """
)

# Function to perform document embedding and create vector store
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./pdf")

        # Data ingestion
        st.session_state.docs = st.session_state.loader.load()

        # Document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        # Chunk creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])

        # Create vector store from embedded documents
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Documents Embedding
if st.button("Embed Documents", help="Click to load and embed PDF files"):
    vector_embedding()
    st.success("Vector Store DB is Ready")

# Main content area
prompt1 = st.text_input("Enter the Question")
if st.button("Answer"):
    import time
    if prompt1:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        print("Response time :", time.process_time() - start)
        st.write(response['answer'])

        # With a streamlit expander for document similarity search
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
                
st.markdown(f"**Model used: {model_name}**")