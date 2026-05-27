from fastapi import APIRouter, UploadFile, File
from app.services.extractor import extract_text
from app.services.vectorstore import add_to_vectorstore

router = APIRouter()


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()

    text = extract_text(file.filename, content)

    add_to_vectorstore(text)

    return {"message": "File uploaded and processed successfully"}
