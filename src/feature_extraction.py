

import math
from typing import List, Optional, Dict
from transformers import pipeline
from sentence_transformers import SentenceTransformer


class FeatureExtractor:
    def __init__(self):
        self.toxicity_model = pipeline(
            "text-classification",
            model="facebook/roberta-hate-speech-dynabench-r4-target",
            device="cpu",
        )
        self.threat_model = pipeline(
            "text-classification",
            model="tomh/toxigen_roberta",
            device="cpu",
        )
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )

        self.prev_embedding: Optional[List[float]] = None

    import math

    def _boost(self, raw: float, floor: float = 0.08, steepness: float = 12) -> float:
        """
        Dead zone below `floor` (noise suppression).
        Sigmoid amplification above it.
        
        floor=0.08 means scores < 0.08 stay near 0 (benign text ignored)
        steepness=12 controls how sharply it rises after the floor
        
        Examples (floor=0.08, steepness=12):
        raw=0.02 → 0.01   (benign, suppressed)
        raw=0.05 → 0.04   (still suppressed)
        raw=0.10 → 0.18   (mild signal, small bump)
        raw=0.20 → 0.55   (clear signal, meaningful)
        raw=0.35 → 0.84   (strong signal)
        raw=0.50 → 0.96   (very strong)
        """
        shifted = raw - floor
        return round(max(1 / (1 + math.exp(-steepness * shifted)) - 0.5, 0) * 2, 4)
    
    def extract_features(
        self,
        user_msg: str,
        assistant_msg: str = "",
        user_msg2: str = "",
    ) -> Dict[str, float]:
    
      
        combined_text = f"{user_msg} {assistant_msg} {user_msg2}"

        features = {
            "toxicity_score":    self._get_toxicity_score(combined_text),
            "threat_score":      self._get_threat_score(combined_text),
            "obfuscation_score": self._get_obfuscation_score(user_msg2),
            "topic_shift_score": self._get_topic_shift(user_msg, assistant_msg, user_msg2),
        }
        return features

    def reset(self):
        """Call between conversations so topic-shift starts fresh."""
        self.prev_embedding = None
    def _get_toxicity_score(self, text: str) -> float:
        result = self.toxicity_model(text, truncation=True, max_length=512)[0]
        raw = result["score"] if result["label"] == "hate" else 1.0 - result["score"]
        return self._boost(raw, floor=0.05, steepness=9)  # was floor=0.10

    def _get_threat_score(self, text: str) -> float:
        result = self.threat_model(text, truncation=True, max_length=512)[0]
        raw = result["score"] if result["label"] == "LABEL_1" else 1.0 - result["score"]
        return self._boost(raw, floor=0.05, steepness=9)  # was floor=0.10
    def _get_obfuscation_score(self, text: str) -> float:
        leet_chars = set("014358@$!")
        hits = sum(c in leet_chars for c in text)
        # Increased multiplier: 3 → 6  (leet text like "h0w" now scores meaningfully)
        return round(min(hits / max(len(text), 1) * 6, 1.0), 4)



    def _get_topic_shift(self, user_msg: str, assistant_msg: str, user_msg2: str) -> float:
    
        embed_context  = self._embed(user_msg + assistant_msg)
        embed_reply    = self._embed(assistant_msg)
        embed_current  = self._embed(user_msg2)

        if self.prev_embedding is None:
            topic_shift = 0.0
        else:
            topic_shift = round(
                max(
                    self._cosine_distance(embed_context, self.prev_embedding),
                    self._cosine_distance(embed_reply,   self.prev_embedding),
                    self._cosine_distance(embed_current, self.prev_embedding),
                ),
                4,
            )

        self.prev_embedding = embed_context
        return topic_shift

    def _embed(self, text: str) -> List[float]:
        return self.embedding_model.encode(text).tolist()

    @staticmethod
    def _cosine_distance(a: List[float], b: List[float]) -> float:
        dot    = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x ** 2 for x in a))
        norm_b = math.sqrt(sum(x ** 2 for x in b))
        similarity = dot / (norm_a * norm_b + 1e-9)
        return (1.0 - similarity)