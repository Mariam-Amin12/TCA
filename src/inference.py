
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from feature_extraction import FeatureExtractor
from risk_calculator import RiskCalculator
DEFAULT_FEATURE_INFO   = os.getenv("FEATURE_INFO_PATH","config/feature_info.json")
DEFAULT_RISK_PARAMS    = os.getenv("RISK_PARAMS_PATH","config/optimized_params_risk.json")
DEFAULT_SCALER         = os.getenv("SCALER_PATH", "models/scaler.pkl")
DEFAULT_MODEL          = os.getenv("MODEL_PATH", "models/best_model_latest.pkl")
DEFAULT_REFUSAL_MODEL  = os.getenv("REFUSAL_MODEL_PATH", "models/refusal_classifier.pkl")
DEFAULT_THRESHOLD      = float(os.getenv("DETECTION_THRESHOLD", "0.5"))


class JailbreakMultiTurnDetector:
  

    def __init__(
        self,
        feature_info_path: str = DEFAULT_FEATURE_INFO,
        risk_params_path: str = DEFAULT_RISK_PARAMS,
        scaler_path:Optional[str] = DEFAULT_SCALER,
        model_path:str = DEFAULT_MODEL,
        refusal_model_path: str = DEFAULT_REFUSAL_MODEL,
        threshold: float = DEFAULT_THRESHOLD,
        device: str = "cpu",
    ):
        self.threshold = threshold

        with open(feature_info_path, "r") as f:
            feature_info = json.load(f)

        self.selected_features: List[str] = feature_info["selected_features"]
        resolved_scaler = scaler_path or feature_info.get("scaler", DEFAULT_SCALER)

        if os.path.exists(risk_params_path):
            with open(risk_params_path, "r") as f:
                risk_params = json.load(f)
        else:

            risk_params = {}
        self._risk_calc = RiskCalculator(**risk_params)

        self._scaler = joblib.load(resolved_scaler)

        self._model = joblib.load(model_path)
        self._has_proba = hasattr(self._model, "predict_proba")

        self._toxicity_model = pipeline(
            "text-classification",
            model="facebook/roberta-hate-speech-dynabench-r4-target",
            device=device,
        )
        self._threat_model = pipeline(
            "text-classification",
            model="tomh/toxigen_roberta",
            device=device,
        )
        self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self._refusal_model   = joblib.load(refusal_model_path)


        self._feature_extractor: Optional[FeatureExtractor] = None
        self._prev_user:      str   = ""
        self._prev_assistant: str   = ""
        self._prev_prog:      float = 0.0
        self._turn_id:        int   = 0

        self.new_conversation()


    def new_conversation(self) -> None:
      
        self._feature_extractor = FeatureExtractor(
            toxicity_model = self._toxicity_model,
            threat_model = self._threat_model,
            embedding_model = self._embedding_model,
            refusal_model= self._refusal_model,
        )
        self._prev_user = ""
        self._prev_assistant = ""
        self._prev_prog  = 0.0
        self._turn_id = 0

    def observe_turn(
        self,
        user_msg:str,
        assistant_msg: str = "",
    ) -> Dict:
        
        turn_id = self._turn_id

        if turn_id == 0:
            raw = self._feature_extractor.extract_features(
                user_msg="",assistant_msg="",
                user_msg2=user_msg, assistant_msg2=assistant_msg,
            )
        else:
            raw = self._feature_extractor.extract_features(
                user_msg=self._prev_user, assistant_msg=self._prev_assistant,
                user_msg2=user_msg, assistant_msg2=assistant_msg,
            )

        interaction_risk = self._risk_calc.compute_interaction_risk(raw)
        pattern_risk     = self._risk_calc.compute_pattern_risk(raw)
        progressive_risk = self._risk_calc.calculate_progressive_risk(raw, self._prev_prog)

        row = {
            **raw,
            "interaction_risk": interaction_risk,
            "pattern_risk":pattern_risk,
            "progressive_risk": progressive_risk,
            "prev_progressive":self._prev_prog,
        }

        row_df = pd.DataFrame([row])    
        print (self.selected_features)
        row_df = row_df[self.selected_features]
        print (row_df.columns)
        X_scaled = pd.DataFrame(
            self._scaler.transform(row_df),
            columns=row_df.columns,
        )

        prediction  = int(self._model.predict(X_scaled)[0])
        if self._has_proba:
            probability = float(self._model.predict_proba(X_scaled)[0, 1])
        else:
            probability = float(prediction)

        self._prev_user = user_msg
        self._prev_assistant = assistant_msg
        self._prev_prog = progressive_risk
        self._turn_id += 1

        return {
            "turn_id": turn_id,
            "prediction": prediction,
            "is_attack": bool(probability >= self.threshold),
            "probability":  round(probability, 4),
            "progressive_risk":round(progressive_risk, 4),
            "interaction_risk": round(interaction_risk, 4),
            "pattern_risk": round(pattern_risk, 4),
            "raw_features": {k: round(float(v), 4) for k, v in raw.items()},
        }

    def score_conversation(
        self,
        turns: List[Dict[str, str]],
    ) -> List[Dict]:
        
        self.new_conversation()
        results = []
        for turn in turns:
            result = self.observe_turn(
                user_msg      = turn.get("user", ""),
                assistant_msg = turn.get("assistant", ""),
            )
            results.append(result)
        return results

    @property
    def current_turn(self) -> int:
        return self._turn_id

    @property
    def current_risk(self) -> float:
        return round(self._prev_prog, 4)
