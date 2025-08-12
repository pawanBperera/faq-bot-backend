from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from model.retrieval import FAQRetrievalModel

app = FastAPI()

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend URL if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FAQ_PATH = "faq.json"
model = FAQRetrievalModel(FAQ_PATH, threshold=0.35)  # tweak threshold if needed

# Fallback: we still load raw JSON in case we want exact lookup (optional)
with open(FAQ_PATH, "r", encoding="utf-8") as f:
    faq_data = json.load(f)

class Question(BaseModel):
    query: str

@app.get("/")
def home():
    return {"message": "FAQ Bot API is running!"}

@app.post("/ask")
def ask_question(query: dict):
    user_input = query.get("query", "")
    # 1) Try ML retrieval
    result = model.answer(user_input)

    if result and result.get("answer"):
        return {"answer": result["answer"]}

    # 2) If low confidence, try your old exact match as a backup
    ui_lower = user_input.strip().lower()
    for item in faq_data:
        if ui_lower == item["question"].strip().lower():
            return {"answer": item["answer"]}

    # 3) Final fallback
    return {"answer": "Sorry, I don't understand the question. Please try again or contact support."}

@app.post("/reload")
def reload_faq():
    """Call this after editing faq.json to refresh vectors without restarting."""
    model.reload()
    return {"status": "reloaded"}
