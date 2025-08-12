# model/retrieval.py
import json
import joblib
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class FAQRetrievalModel:
    def __init__(self, faq_path: str, model_dir: str = "model", threshold: float = 0.35):
        self.faq_path = Path(faq_path)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.threshold = threshold
        self.vectorizer = None
        self.matrix = None
        self.questions = []
        self.answers = []

        self._load_data()
        self._fit()

    def _load_data(self):
        data = json.loads(self.faq_path.read_text(encoding="utf-8"))
        self.questions = [d["question"].strip() for d in data]
        self.answers   = [d["answer"] for d in data]

    def _fit(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True
        )
        self.matrix = self.vectorizer.fit_transform(self.questions)

        # cache for faster cold starts (optional)
        joblib.dump(self.vectorizer, self.model_dir / "vectorizer.joblib")
        joblib.dump(self.matrix,    self.model_dir / "matrix.joblib")

    def reload(self):
        self._load_data()
        self._fit()

    def answer(self, query: str, top_k: int = 3):
        if not query or not query.strip():
            return None

        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix).flatten()
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        # Build alternatives (excluding best)
        alt_indices = np.argsort(-sims).tolist()
        alt_indices = [i for i in alt_indices if i != best_idx][:top_k-1]
        alternatives = [{"question": self.questions[i], "score": float(sims[i])} for i in alt_indices]

        result = {
            "matched_question": self.questions[best_idx],
            "answer": self.answers[best_idx],
            "score": best_score,
            "alternatives": alternatives
        }

        if best_score < self.threshold:
            return {
                "matched_question": None,
                "answer": None,
                "score": best_score,
                "alternatives": alternatives
            }
        return result
