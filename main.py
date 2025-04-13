from fastapi import FastAPI, UploadFile, File
from rag_utils import process_document, ask_question

app = FastAPI()

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    content = await file.read()
    filename = file.filename
    process_document(content, filename)
    return {"message": "Document uploaded and processed successfully."}

@app.get("/ask")
async def get_answer(query: str):
    answer = ask_question(query)
    return {"answer": answer}
