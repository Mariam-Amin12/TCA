import math
from typing import List


class FeatureExtractor:
    def __init__(self, toxicity_model, threat_model, embedding_model, refusal_model):
        self.toxicity_model= toxicity_model
        self.threat_model = threat_model
        self.embedding_model = embedding_model
        self.refusal_model= refusal_model
        self.baseline_embedding = None
        self.turn_embeddings = []
        self.drift_history = []

    def reset(self):
        self.baseline_embedding= None
        self.turn_embeddings= []
        self.drift_history= []

    def extract_features(
        self,
        user_msg: str,
        assistant_msg: str = "",
        user_msg2: str = "",
        assistant_msg2: str = "",
    ) -> dict:
        combined_text = f"{user_msg2} {assistant_msg2}".strip()

        current_embed = self._embed(combined_text) if combined_text else []

        topic_shift_score= self._get_topic_shift(current_embed)
        cumulative_drift = self._get_cumulative_drift(current_embed)
        drift_acceleration = self._get_drift_acceleration(current_embed)

        if current_embed:
            self.turn_embeddings.append(current_embed)
            if self.baseline_embedding is None:
                self.baseline_embedding = current_embed

        features = {
            "toxicity_score":  self._get_toxicity_score(combined_text),
            "threat_score":self._get_threat_score(combined_text),
            "topic__score":topic_shift_score,
            "cumulative_drift":cumulative_drift,
            "drift_acceleration": drift_acceleration,
            "post_refusal":self._get_post_refusal(assistant_msg2),
        }
        return features

    def _get_topic_shift(self, current_embed: List[float]) -> float:
        if not current_embed or not self.turn_embeddings:
            return 0.0
        return round(self._cosine_distance(current_embed, self.turn_embeddings[-1]), 4)

    def _get_cumulative_drift(self, current_embed: List[float]) -> float:
        if not current_embed:
            return 0.0
        if self.baseline_embedding is None:
            self.drift_history.append(0.0)
            return 0.0

        drift = round(self._cosine_distance(current_embed, self.baseline_embedding), 4)
        self.drift_history.append(drift)
        return drift

    def _get_drift_acceleration(self, current_embed: List[float]) -> float:
       
        if not current_embed or len(self.turn_embeddings) < 2:
            return 0.0

        recent_drift= self._cosine_distance(current_embed, self.turn_embeddings[-1])
        earlier_drift= self._cosine_distance(self.turn_embeddings[-1], self.turn_embeddings[-2])
        acceleration= recent_drift - earlier_drift        

        return round( acceleration, 4)

    def _get_post_refusal(self, assistant_msg2: str) -> float:
        if not assistant_msg2:
            return 0.0
        assistant_emb= self.embedding_model.encode([assistant_msg2])
        assistant_pred = self.refusal_model.predict(assistant_emb)[0]
        return float(assistant_pred) if assistant_pred != 0 else 0.0

    def _get_toxicity_score(self, text: str) -> float:
        if not text:
            return 0.0
        result = self.toxicity_model(text, truncation=True, max_length=512)[0]
        return result["score"] if result["label"] == "hate" else 1.0 - result["score"]

    def _get_threat_score(self, text: str) -> float:
        if not text:
            return 0.0
        result = self.threat_model(text, truncation=True, max_length=512)[0]
        return result["score"] if result["label"] == "LABEL_1" else 1.0 - result["score"]

    def _embed(self, text: str) -> List[float]:
        return self.embedding_model.encode(text).tolist()

    @staticmethod
    def _cosine_distance(a: List[float], b: List[float]) -> float:
        dot= sum(x * y for x, y in zip(a, b))
        norm_a= math.sqrt(sum(x ** 2 for x in a))
        norm_b = math.sqrt(sum(x ** 2 for x in b))
        similarity = dot / (norm_a * norm_b + 1e-9)
        return 1.0 - similarity