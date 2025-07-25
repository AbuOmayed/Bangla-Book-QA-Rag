# 📘 Bangla Book QA RAG

**Bangla Book QA RAG** is a bilingual Retrieval-Augmented Generation (RAG) system for answering questions from the *HSC26 Bangla 1st Paper textbook* in **Bangla** and **English**. This tool supports both CLI-based and web-based (Streamlit) interfaces, using **Ollama LLMs** for response generation and **LangChain** for pipeline orchestration.

---
## 🚀 Features

- ✅ Accepts questions in **Bangla** and **English**
- 📄 Parses and indexes Bangla textbook PDF
- 🧠 Retrieves relevant document chunks from vector store
- 💬 Answers questions using **LLMs via Ollama** (deepseek-r1:14b)
- 🖥️ Offers both a **CLI** version and a **Streamlit web app**
- 💾 Supports **in-memory** and **AstraDB** vector storage
- 🔍 Efficient document retrieval using **LangChain**
- 🧬 Embedding using **DeepSeek and llama2**
---

## 🏗️ Architecture Overview

<p align="center">
  <img src="https://github.com/user-attachments/assets/2a50bd7b-3e07-4171-9ee6-63fd7373215d" width="600">
</p>

---

## 🖥️ App Interfaces

<p align="center">
  <img src="" width="600">
</p>

---

## 🧪 Sample Queries

| Question | Answer |
|---------|--------|
| বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল? | ১৫ বছর |
| আনুপমের ভাষায় সুপুরুষ কাকে বলে? | শম্ভুনাথ |
| কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে? | মামা |

---

### 📷 Screenshots

<p align="center">
  <img src="https://github.com/user-attachments/assets/9136efa6-87a2-4c8f-b7b7-7b2aedf219c5" width="600">
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/97806550-71f5-486a-8408-080076ef46e9" width="600">
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/a7ecb26a-a6e1-4a6c-841c-4d029bec4277" width="600">
</p>

---
## 🛠️ Tools, Libraries & Packages

- **LangChain** – Retrieval & QA orchestration
- **Ollama LLMs** – (`deepseek-r1:14b`, `llama2`)
- **Streamlit** – Web-based UI
- **PDFPlumber** – Bangla-friendly PDF text extraction
- **AstraDB (Cassandra)** – Vector DB for persistent storage
- **RecursiveCharacterTextSplitter** – Smart chunking for context-aware retrieval

---

## 📡 API Documentation (Not Implemented)

- Currently CLI and GUI only.
- API (FastAPI or Flask) can be integrated for production use.

---

## 📊 Evaluation Matrix (Planned/Future Work)

| Metric | Value | Notes |
|--------|-------|-------|
| Top-1 Accuracy | -- | Manual testing with known questions |
| Response Time | -- | On local CPU (LLM performance varies) |
| BLEU/ROUGE | -- | Planned for future implementation |

---

## ❓Question Answer Section

### 📄 What method or library did you use to extract the text, and why?

- **Library**: `PDFPlumberLoader` (via LangChain)
- **Reason**: PDFPlumber preserves **Bangla Unicode** and handles line breaks and ligatures better than PyPDF2 or pdfminer.
- **Challenge**: Some PDF pages had irregular formatting (broken words, missing punctuation), requiring chunking with tolerance for broken structure.

### ✂️ What chunking strategy did you choose?

- **Method**: `RecursiveCharacterTextSplitter`
- **Config**: `chunk_size=1000`, `chunk_overlap=300`
- **Why**: This strategy balances coherence and retrieval relevance. It avoids splitting in mid-sentence/word, helping embeddings maintain semantic continuity.

### 🧬 What embedding model did you use?

- **Model**: `OllamaEmbeddings(model="deepseek-r1:14b")`
- **Why**: It's multilingual and effective in capturing contextual meaning in Bangla and English.
- **Advantage**: Fine-grained semantic understanding that outperforms basic word vectors (e.g., TF-IDF).

### 🧠 How are you comparing the query with your stored chunks?

- **Method**: `vector_store.similarity_search(query=question)`
- **Similarity**: Cosine distance (via FAISS or AstraDB backend)
- **Why**: Vector similarity ensures semantically close chunks, even if they don’t share surface-level words.

### 🧐 How do you ensure meaningful comparison between question and document chunks?

- Chunks are semantically rich (1000 chars) and overlapping (300 chars), ensuring broader context.
- Queries are passed to embedding model to generate query vectors before comparison.
- If the query is vague or missing context, fallback is:
  - Either generic answers (e.g., “I don't know”)
  - Or retrieving the most relevant-looking chunk regardless of full match.

### ✅ Do the results seem relevant?

- **Yes**, for clear, factual, and Bangla-language textbook queries.
- **Improvements possible**:
  - Add metadata-based filtering (e.g., chapter tags)
  - Use sentence-level chunking for finer granularity
  - Explore OpenAI or Cohere embeddings for richer semantic capture
  - Add query rewriting for vague or incomplete questions

---

## 📁 Repository Structure

```
📦 bangla-book-qa-rag
 ┣ 📄 README.md
 ┣ 📄 requirements.txt
 ┣ 📄 BanglaPDF_QA_Apps.py         # Streamlit Web App
 ┣ 📄 BanglaPDF_QA (main).py       # CLI App with local store
 ┣ 📄 BangalPDF_QA_AstraDB.py      # CLI App with AstraDB
 ┣ 📁 pdfstore/                    # Uploaded PDFs for app
```
## ⚙️ Setup Instructions

```bash
# Clone the repository
git clone https://github.com/yourusername/bangla-book-qa-rag.git
cd bangla-book-qa-rag

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Run the Streamlit app
streamlit run BanglaPDF_QA_Apps.py

# OR run CLI version with local vector store
python BanglaPDF_QA\ \(main\).py

# OR run CLI version with AstraDB
python BangalPDF_QA_AstraDB.py
```

## 🤝 Credits

Created by **Abu Omayed**
