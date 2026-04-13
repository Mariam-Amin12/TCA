import math
from typing import List


class FeatureExtractor:
    def __init__(self, toxicity_model, threat_model, embedding_model, refusal_model):
        self.toxicity_model = toxicity_model
        self.threat_model = threat_model
        self.embedding_model = embedding_model
        self.refusal_model = refusal_model

        self.prev_embedding       = None
        self.baseline_embedding   = None   
        self.turn_embeddings      = []    
      
    def extract_features(
        self,
        user_msg: str,
        assistant_msg: str = "",
        user_msg2: str = "",
        assistant_msg2: str = "",
    ) -> dict:
        combined_text = f"{user_msg2} {assistant_msg2}".strip()

        features = {
            "toxicity_score":      self._get_toxicity_score(combined_text),
            "threat_score":        self._get_threat_score(combined_text),
            "topic_shift_score":   self._get_topic_shift(user_msg, assistant_msg, user_msg2, assistant_msg2),
            "cumulative_drift":    self._get_cumulative_drift(),
            "drift_acceleration":  self._get_drift_acceleration(),
            "post_refusal":        self._get_post_refusal(assistant_msg2),
        }
        return features

    def _get_post_refusal(self, assistant_msg2):
        if assistant_msg2 == "":
            return 0.0

        assistant_emb = self.embedding_model.encode([assistant_msg2])
        assistant_pred = self.refusal_model.predict(assistant_emb)[0]

        if assistant_pred == 0:
            return 0.0

        return assistant_pred


    def reset(self):
        self.prev_embedding     = None
        self.baseline_embedding = None
        self.turn_embeddings    = []


        

    def _get_cumulative_drift(self) -> float:
        current_embed = self.prev_embedding
        self.turn_embeddings.append(current_embed)

        if self.baseline_embedding is None:
            self.baseline_embedding = current_embed
            return 0.0

        drift = self._cosine_distance(current_embed, self.baseline_embedding)
        return round(drift, 4)

    def _get_drift_acceleration(self) -> float:
        if len(self.turn_embeddings) < 3:
            return 0.0

        recent_drift = self._cosine_distance(
            self.turn_embeddings[-1], self.turn_embeddings[-2]
        )
        earlier_drift = self._cosine_distance(
            self.turn_embeddings[-2], self.turn_embeddings[-3]
        )
        acceleration = recent_drift - earlier_drift
        return round(max(acceleration, 0.0), 4)  
        
   

    def _get_toxicity_score(self, text: str) -> float:
        result = self.toxicity_model(text, truncation=True, max_length=512)[0]
        raw = result["score"] if result["label"] == "hate" else 1.0 - result["score"]
        return raw 

    def _get_threat_score(self, text: str) -> float:
        result = self.threat_model(text, truncation=True, max_length=512)[0]
        raw = result["score"] if result["label"] == "LABEL_1" else 1.0 - result["score"]
        return raw

    def _get_topic_shift(self, user_msg: str, assistant_msg: str, user_msg2: str, assistant_msg2: str) -> float:
        embed_prev_context = self.prev_embedding if self.prev_embedding is not None else self._embed(user_msg + assistant_msg)
        embed_current      = self._embed(assistant_msg2 + user_msg2)
        if self.prev_embedding is None:
            self.prev_embedding = embed_current
        topic_shift = round(self._cosine_distance(embed_current, embed_prev_context), 4)

        self.prev_embedding = embed_current
       
        return topic_shift

    def _embed(self, text: str) -> List[float]:
        return self.embedding_model.encode(text).tolist()

    @staticmethod
    def _cosine_distance(a: List[float], b: List[float]) -> float:
        dot    = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x ** 2 for x in a))
        norm_b = math.sqrt(sum(x ** 2 for x in b))
        similarity = dot / (norm_a * norm_b + 1e-9)
        return 1.0 - similarity  