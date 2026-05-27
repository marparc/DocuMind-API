import fitz
from PIL import Image
import pytesseract
import io


def extract_text(filename, file_bytes):
    if filename.endswith(".pdf"):
        return extract_pdf(file_bytes)
    elif filename.endswith((".png", ".jpg", ".jpeg")):
        return extract_image(file_bytes)
    else:
        return file_bytes.decode("utf-8")


def extract_pdf(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def extract_image(file_bytes):
    image = Image.open(io.BytesIO(file_bytes))
    return pytesseract.image_to_string(image)
