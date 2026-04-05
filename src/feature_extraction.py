from transformers import pipeline
from sentence_transformers import SentenceTransformer
import math
from typing import List, Dict, Tuple, Optional
class FeatureExtractor:
    def __init__(self):
        # Models
        self.toxicity_model = pipeline(
            "text-classification",
            model="facebook/roberta-hate-speech-dynabench-r4-target",
            device="cpu"
        )

        self.threat_model = pipeline(
            "text-classification",
            model="tomh/toxigen_roberta",
           device="cpu"
        )

        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
           device="cpu"
        )

        self.prev_embedding: Optional[List[float]] = None
        self.prev_user_msg: str = ""
        self.prev_assistant_msg: str = ""

    def _embed(self, text: str) -> List[float]:
        return self.embedding_model.encode(text).tolist()

    def _cosine_distance(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x ** 2 for x in a))
        norm_b = math.sqrt(sum(x ** 2 for x in b))
        similarity = dot / (norm_a * norm_b + 1e-9)
        return (1.0 - similarity) / 2.0  # normalized [0,1]


    def get_toxicity_score(self, text: str) -> float:
        result = self.toxicity_model(text, truncation=True, max_length=512)[0]
        return round(
            result["score"] if result["label"] == "hate"
            else 1.0 - result["score"],
            4
        )

    def get_threat_score(self, text: str) -> float:
        result = self.threat_model(text, truncation=True, max_length=512)[0]
        return round(
            result["score"] if result["label"] == "LABEL_1"
            else 1.0 - result["score"],
            4
        )

    def get_obfuscation_score(self, text: str) -> float:
        leet_chars = set("014358@$!")
        hits = sum(c in leet_chars for c in text)
        return round(min(hits / max(len(text), 1) * 3, 1.0), 4)


    def extract_features(
        self,
        user_msg: str,
        assistant_msg: str = ""
    ) -> Dict[str, float]:
        """
        Extract features using:
        - current user message
        - previous user + assistant context
        - internal conversation memory
        """

        # -------- Topic shift --------
        current_embedding = self._embed(user_msg)

        if self.prev_embedding is None:
            topic_shift = 0.0
        else:
            topic_shift = round(
                self._cosine_distance(current_embedding, self.prev_embedding),
                4
            )

        # -------- Context-aware text --------
        context_text = f"{self.prev_user_msg} {self.prev_assistant_msg}".strip()
        combined_text = f"{context_text} {user_msg}".strip()

        features = {
            # Current message signals
            "toxicity_score": self.get_toxicity_score(user_msg),
            "obfuscation_score": self.get_obfuscation_score(user_msg),

            # Context-aware signal
            "threat_score": self.get_threat_score(combined_text),

            # Conversation behavior
            "topic_shift_score": topic_shift,
        }

        self.prev_embedding = current_embedding
        self.prev_user_msg = user_msg
        self.prev_assistant_msg = assistant_msg

        return features

  
    def reset(self):
        self.prev_embedding = None
        self.prev_user_msg = ""
        self.prev_assistant_msg = ""
