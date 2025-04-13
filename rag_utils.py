import os
import tempfile
import shutil
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

PERSIST_DIRECTORY = "chroma_store"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-base"

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def process_document(content: bytes, filename: str):
  
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
        tmp.write(content)
        filepath = tmp.name

    # Load document
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(filepath)
    else:
        loader = TextLoader(filepath)

    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

  
    vectordb = Chroma.from_documents(chunks, embedding_model, persist_directory=PERSIST_DIRECTORY)
    vectordb.persist()
    os.remove(filepath)

def ask_question(query: str) -> str:
    vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model)
    retriever = vectordb.as_retriever()

    docs = retriever.get_relevant_documents(query)

    print("=== Retrieved Documents ===")
    for doc in docs:
        print(doc.page_content)

    # Use LLM pipeline
    llm_pipeline = pipeline("text2text-generation", model=LLM_MODEL_NAME)
    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa.invoke(query)

