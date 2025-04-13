## GenAI Powered Document Q&A System (RAG with FastAPI)

This project implements an end-to-end **Retrieval-Augmented Generation (RAG)** system that lets you:

- 📤 Upload a `.pdf` or `.txt` document
- ❓ Ask natural language questions about its content
- 💡 Get AI-generated answers using a HuggingFace LLM

Powered by **LangChain**, **HuggingFace Transformers**, **Chroma Vector DB** and served with **FastAPI**.

---

## 🚀 Features

- ✅ Accepts PDF or Text documents via API
- ✅ Splits documents into semantic chunks
- ✅ Converts chunks to vector embeddings using `MiniLM`
- ✅ Stores vectors in ChromaDB for fast semantic search
- ✅ Answers user queries with relevant chunks + LLM (`Flan-T5`)

---

## 📁 Project Structure

├── main.py # FastAPI endpoints ├── rag_utils.py # Core RAG pipeline (load, split, embed, retrieve, answer) ├── chroma_store/ # Directory to persist Chroma vector store ├── requirements.txt # Python dependencies

| Component        | Purpose                         |
|------------------|----------------------------------|
| **FastAPI**       | Backend API framework            |
| **LangChain**     | Document processing & RAG chain  |
| **HuggingFace**   | LLM and Embedding models         |
| **ChromaDB**      | Vector store for semantic search |
| **Sentence-Transformers** | For efficient text embeddings |

## How It Works (RAG Pipeline) 🤔

    Upload: A document is uploaded and temporarily saved.

    Chunk: It is split into small, overlapping text chunks.

    Embed: Each chunk is converted into a vector using MiniLM.

    Store: ChromaDB stores the vectors on disk.

    Query: When you ask a question, similar chunks are retrieved.

    Answer: Retrieved chunks + question are fed to Flan-T5 to generate a final answer.


# API-1 (/upload): To upload the document

![API-1 upload](https://github.com/user-attachments/assets/e7e178b9-8e36-4a20-9be0-d9c7372b5154)

# API-2 (/ask): To ask the question

![API-2 ask](https://github.com/user-attachments/assets/aded88b0-cc07-4a79-8b21-b44d2e048e7b)
