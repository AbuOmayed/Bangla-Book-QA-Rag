import os
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Streamlit UI title
st.title("📘 Bangla PDF Book Q&A App\n*Created by Abu Omayed*")

# Prompt Template for LLM
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}
Context: {context}
Answer:
"""

# Path for saving uploaded PDFs
pdf_path = "./pdfstore/"
os.makedirs(pdf_path, exist_ok=True)

# Create Embedding LLM and VectorStore
embeddings = OllamaEmbeddings(model="deepseek-r1:14b")
vector_store = InMemoryVectorStore(embeddings)
model = OllamaLLM(model="deepseek-r1:14b")

# Upload PDF file
def upload_pdf(file):
    with open(pdf_path + file.name, "wb") as f:
        f.write(file.getbuffer())

# Load PDF content
def load_pdf(path):
    loader = PDFPlumberLoader(path)
    documents = loader.load()
    return documents

# Split text into chunks
def split_text(document):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return splitter.split_documents(document)

# Index chunks into vector store
def index_docs(chunks):
    vector_store.add_documents(chunks)

# Retrieve similar documents from vector store
def retrieve_text(query):
    return vector_store.similarity_search(query=query)

# Answer questions
def answer_question(question, docs):
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

# File upload UI
uploaded_file = st.file_uploader("Upload a Book (Bangla)", type="pdf", accept_multiple_files=False)

if uploaded_file:
    upload_pdf(uploaded_file)
    document = load_pdf(pdf_path + uploaded_file.name)
    chunked_document = split_text(document)
    index_docs(chunked_document)

    question = st.chat_input("Ask a question about the uploaded book:")
    if question:
        st.chat_message("user").write(question)
        related_docs = retrieve_text(question)
        answer = answer_question(question, related_docs)
        st.chat_message("assistant").write(answer)
