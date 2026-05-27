import os
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def generate_answer(question, context):
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a STRICT document QA system.\n"
                    "\n"
                    "========================\n"
                    "CORE PRINCIPLE\n"
                    "========================\n"
                    "You are NOT allowed to reason using the document.\n"
                    "You are ONLY allowed to extract text.\n"
                    "\n"
                    "========================\n"
                    "PROCESS (MANDATORY)\n"
                    "========================\n"
                    "STEP 1: TOC MATCHING\n"
                    "- Use Table of Contents to select EXACTLY ONE section.\n"
                    "\n"
                    "STEP 2: SECTION EXTRACTION\n"
                    "- Find the full text of ONLY that section in the document body.\n"
                    "\n"
                    "========================\n"
                    "HARD RULES\n"
                    "========================\n"
                    "1. You MUST NOT summarize using knowledge or reasoning.\n"
                    "2. You MUST ONLY use sentences that appear inside the selected section.\n"
                    "3. You MUST NOT combine information from other sections.\n"
                    "4. You MUST NOT rephrase using external inference.\n"
                    "5. If information is not explicitly written, omit it.\n"
                    "\n"
                    "========================\n"
                    "FAIL CONDITION\n"
                    "========================\n"
                    "- If the selected section contains no relevant text, output:\n"
                    "  'The information is not available in the provided document.'\n"
                    "\n"
                    "========================\n"
                    "OUTPUT FORMAT\n"
                    "========================\n"
                    "- Return ONLY extracted sentences.\n"
                    "- No added explanation.\n"
                    "- No interpretation.\n"
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}",
            },
        ],
        "temperature": 0.0,
    }
    response = requests.post(url, json=payload, headers=headers)

    # DEBUG: show real response if something breaks
    try:
        data = response.json()
    except Exception:
        return f"Invalid JSON from Groq: {response.text}"

    # HANDLE ERROR FROM GROQ
    if "error" in data:
        return f"Groq API Error: {data['error']['message']}"

    # SAFE ACCESS
    if "choices" not in data:
        return f"Unexpected response: {data}"

    return data["choices"][0]["message"]["content"]
