from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}
Context: {context}
Answer:
"""

# Read the PDF
pdf = PDFPlumberLoader("E:\RAG+Langchain\HSC26-Bangla1st-Paper.pdf")
documents = pdf.load()


# Create langchain Embedding and LLM
embeddings = OllamaEmbeddings(model="deepseek-r1:14b")
vector_store = InMemoryVectorStore(embeddings)
model = OllamaLLM(model="deepseek-r1:14b")


#Split the text into chunk
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 300,
    add_start_index=True,
)
texts = text_splitter.split_documents(documents)


# Index chunks into vector store
vector_store.add_documents(texts)

# Answer questions
def answer_question(question, docs):
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})


# Run Question and Answer
question = True

while True:
    if question:
        question_text = input("\nWhat is your question?").strip()
    else:
        question_text = input("\nWhat is your next question?").strip()

    if question_text.lower() == "exit":
        break
    if question_text == "":
        continue
    question = False
    print("\n Question: \"%s\"" % question_text)
    related_documents = vector_store.similarity_search(query=question_text)
    answer = answer_question(question_text, related_documents)
    print(f"Answer: \"{answer}\"\n")
