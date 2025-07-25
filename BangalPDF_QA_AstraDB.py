# Import important packages
import cassio
from langchain_community.vectorstores import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_community.document_loaders import PDFPlumberLoader

# SET ASTRADB databased token and key
ASTRA_DB_APPLICATION_TOKEN = "AstraCS:hnOInRytyQwvyDgILOZFQXcs:cdcc8356c7d8edb438dd79ee36cb9d545463a7919c86ce03647a67dc75c41f24"
ASTRA_DB_ID = "2ed9341f-f627-4b80-8a9a-a7908feb6ed6"

# Read the PDF
pdf = PDFPlumberLoader("E:\\RAG+Langchain\\AI Engineer (Level-1) Assessment.pdf")
documents = pdf.load()

# Connect to the database
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# Create langchain Embedding and LLM
embeddings = OllamaEmbeddings(model="llama2") # Use OpenAIEmbeddings(model="text-embedding-ada-002") for better Performance
model = OllamaLLM(model="deepseek-r1:14b")

# Create Langchain Vector Store
astra_vector_store = Cassandra(
    embedding=embeddings,
    table_name="Bangla_1st_paper",
    session=None,
    keyspace=None,
)

# Split the documents using Character
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=300,
    add_start_index=True,
)

# Split the documents
split_documents = text_splitter.split_documents(documents)

# Extract the page content from each document
texts = [doc.page_content for doc in split_documents]

# Load the dataset into Vector Store
astra_vector_store.add_texts(texts[:100])
astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

# Run Question and Answer
question = True
while True:
    if question:
        question_text = input("\nWhat is your question? ").strip()
    else:
        question_text = input("\nWhat is your next question? ").strip()

    if question_text.lower() == "exit":
        break
    if question_text == "":
        continue
    question = False
    print("\nQuestion: \"%s\"" % question_text)
    answer = astra_vector_index.query(question_text, llm=model)
    print("Answer: \"%s\"" % answer)