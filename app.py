import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import tempfile

# Page config
st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📚", layout="centered")

st.title("📚 PDF RAG Q&A Chatbot")
st.markdown("Upload your PDFs and ask any questions — answers come directly from your documents!")

# Sidebar for API key
with st.sidebar:
    st.header("Settings")
    openai_api_key = st.text_input("OpenAI API Key", type="password", value="")
    if not openai_api_key:
        st.info("Enter your OpenAI API key to start")
        st.stop()

os.environ["OPENAI_API_KEY"] = openai_api_key

# Session state for vectorstore
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# File uploader
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    if st.button("Process PDFs"):
        with st.spinner("Processing PDFs... This may take a minute"):
            documents = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load()
                documents.extend(docs)

            # Split text
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)

            # Create embeddings and vectorstore
            embeddings = OpenAIEmbeddings()
            st.session_state.vectorstore = Chroma.from_documents(texts, embeddings)

            # QA chain
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            prompt_template = """Use only the following context to answer the question. If you don't know, say "I don't know".

Context: {context}

Question: {question}
Answer:"""
            PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4}),
                chain_type_kwargs={"prompt": PROMPT}
            )

            st.success(f"Processed {len(uploaded_files)} PDF(s)! You can now ask questions.")

# Chat interface
if st.session_state.qa_chain:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your PDFs"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.qa_chain.run(prompt)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
