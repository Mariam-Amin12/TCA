from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Deque
from collections import deque
from pydantic import BaseModel
from feature_extraction import FeatureExtractor  # assume your existing class
from risk_calculator import RiskCalculator      # assume your existing class

# ── Pydantic models for structured response ─────────────────────

class ProgressionSummary(BaseModel):
    trend: str | None = None
    dominant_flag: str | None = None
    summary: str | None = None

class TCAResult(BaseModel):
    risk_level: int | None = None
    threat_score: float | None = None
    obfuscation_score: float | None = None
    toxicity_score: float | None = None
    topic_shift_score: float | None = None
    historical_risk: float | None = None
    progression_summary: ProgressionSummary | None = None


# ── Decision maker ─────────────────────────────────────────────
class DecisionMaker:
    """
    Optimized DecisionMaker for multi-turn jailbreak detection.
    Thresholds tuned based on T0-T20 conversation logs.
    """
    def __init__(
        self,
        t_warn: float = 0.08,           # Lowered from 0.1 to catch early probing
        t_block: float = 0.16,          # Lowered from 0.2 to stop slow-burn attacks
        window_size: int = 3,
        spike_threshold: float = 0.15,  # More sensitive to sudden jumps
        trend_threshold: float = 0.03,  # Catches shallow rising slopes
        sustained_threshold: float = 0.08, # Matches the 'ALLOW [SUSTAINED]' floor in logs
        obfuscation_threshold: float = 0.12, # Catches the 0.18 obfuscation seen in T17
    ):
        self.t_warn = t_warn
        self.t_block = t_block
        self.window_size = window_size
        self.spike_threshold = spike_threshold
        self.trend_threshold = trend_threshold
        self.sustained_threshold = sustained_threshold
        self.obfuscation_threshold = obfuscation_threshold

        self.risk_window: Deque[float] = deque(maxlen=window_size)
        self.prev_progressive = 0.0

    def decide(self, features: Dict[str, float], risk_scores: Dict[str, float]) -> Dict:
        prog = risk_scores["progressive_risk"]
        self.risk_window.append(prog)

        flags = self._detect_flags(features, prog)
        decision = self._make_decision(prog, flags)
        level = self._risk_to_level(prog)

        self.prev_progressive = prog

        return {
            "decision": decision,
            "flags": flags,
            "risk_level": level,
        }

    def _detect_flags(self, features: Dict[str, float], prog: float) -> List[str]:
        flags = []
        window = list(self.risk_window)

        # 1. SPIKE: Rapid escalation between turns
        if prog - self.prev_progressive > self.spike_threshold:
            flags.append("SPIKE")

        # 2. OBFUSCATION: High effort to hide intent
        if features["obfuscation_score"] > self.obfuscation_threshold:
            flags.append("OBFUSCATION")

        # 3. TREND: Consistency in rising risk
        if len(window) >= 3 and self._slope(window) > self.trend_threshold:
            flags.append("TREND")

        # 4. SUSTAINED: High average risk over the window
        if len(window) >= 3 and (sum(window) / len(window)) > self.sustained_threshold:
            flags.append("SUSTAINED")

        return flags

    def _make_decision(self, prog: float, flags: List[str]) -> str:
        """
        Logic:
        - If progressive risk hits the ceiling (t_block), block immediately.
        - If we are in the warning zone (t_warn) AND any suspicious pattern 
          (TREND, SUSTAINED, or SPIKE) is detected, upgrade to BLOCK.
        """
        # Absolute Block
        if prog >= self.t_block:
            return "BLOCK"

        # Escalation Logic: WARN -> BLOCK
        if prog >= self.t_warn:
            # If we see any bad patterns while in WARN zone, we kill the conv
            if any(f in flags for f in ["TREND", "SUSTAINED", "SPIKE", "OBFUSCATION"]):
                return "BLOCK"
            return "WARN"

        # Obfuscation Check (even if prog is low)
        if "OBFUSCATION" in flags and prog > (self.t_warn / 2):
            return "BLOCK"

        return "ALLOW"

    @staticmethod
    def _slope(values: List[float]) -> float:
        n = len(values)
        if n < 2: return 0.0
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        den = sum((i - x_mean) ** 2 for i in range(n))
        return num / den if den else 0.0

    @staticmethod
    def _risk_to_level(prog: float) -> int:
        return min(int(prog * 10), 10)

    def reset(self):
        self.risk_window.clear()
        self.prev_progressive = 0.0

# ── Turn record for CSV/logging ───────────────────────────────

@dataclass
class TurnRecord:
    turn_id: int
    timestamp: str
    text: str
    toxicity_score: float = 0.0
    threat_score: float = 0.0
    obfuscation_score: float = 0.0
    topic_shift_score: float = 0.0
    interaction_risk: float = 0.0
    pattern_risk: float = 0.0
    historical_risk: float = 0.0
    progressive_risk: float = 0.0
    risk_level: int = 0
    flags: str = ""
    decision: str = "ALLOW"


# ── Main analyzer ────────────────────────────────────────────

class TCAAnalyzer:
    """Coordinates feature extraction, risk calculation, and decision-making."""
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        self.extractor = FeatureExtractor()
        self.calculator = RiskCalculator(alpha, beta, gamma)
        self.decider = DecisionMaker()
        self.history: List[TurnRecord] = []

    def process_turn(self, user_msg: str, assistant_msg: str = "") -> TCAResult:
        """Process one turn and return structured TCAResult."""
        turn_id = len(self.history)
        timestamp = datetime.now().isoformat(timespec="seconds")

        # 1. Extract features
        features = self.extractor.extract_features(user_msg, assistant_msg)

        # 2. Calculate risks
        risk_scores = {
            "interaction_risk": self.calculator.compute_interaction_risk(features),
            "pattern_risk": self.calculator.compute_pattern_risk(features),
            "progressive_risk": self.calculator.calculate_progressive_risk(features),
            "historical_risk": self.calculator.historical_risk,
        }

        # 3. Make decision
        decision_output = self.decider.decide(features, risk_scores)

        # 4. Build TCAResult
        result = TCAResult(
            **features,
            **risk_scores,
            **decision_output,
            progression_summary=ProgressionSummary(
                trend=self._trend_label(),
                dominant_flag=decision_output["flags"][0] if decision_output["flags"] else None,
                summary=f"Turn {turn_id}: {decision_output['decision']}",
            ),
        )

        # 5. Store record for logging
        record = TurnRecord(
            turn_id=turn_id,
            timestamp=timestamp,
            text=user_msg,
            toxicity_score=features["toxicity_score"],
            threat_score=features["threat_score"],
            obfuscation_score=features["obfuscation_score"],
            topic_shift_score=features["topic_shift_score"],
            interaction_risk=risk_scores["interaction_risk"],
            pattern_risk=risk_scores["pattern_risk"],
            historical_risk=risk_scores["historical_risk"],
            progressive_risk=risk_scores["progressive_risk"],
            risk_level=decision_output["risk_level"],
            flags=",".join(decision_output["flags"]),
            decision=decision_output["decision"],
        )
        self.history.append(record)
        self._print_turn(record)

        return result

    def reset(self):
        self.extractor.reset()
        self.calculator.reset()
        self.decider.reset()
        self.history.clear()

    def _trend_label(self) -> str:
        """Determine trend from risk window."""
        window = list(self.decider.risk_window)
        if len(window) < 3:
            return "stable"
        slope = DecisionMaker._slope(window)
        if slope > 0.02:
            return "escalating"
        if slope < -0.02:
            return "declining"
        return "stable"

    def _print_turn(self, r: TurnRecord):
        bar = "█" * int(r.progressive_risk * 20) + "░" * (20 - int(r.progressive_risk * 20))
        flags = f"  [{r.flags}]" if r.flags else ""
        print(f"T{r.turn_id}  [{bar}]  prog={r.progressive_risk:.3f}  → {r.decision}{flags}")
        print(f"    {r.text[:70]!r}")
        print(f"    tox={r.toxicity_score:.2f}  thr={r.threat_score:.2f}  "
              f"emo={r.emotion_score:.2f}  obf={r.obfuscation_score:.2f}  "
              f"shift={r.topic_shift_score:.2f}\n")

    def save_to_csv(self, path="tca_output.csv"):
        """Save all processed turns to CSV."""
        import csv
        from pathlib import Path
        if not self.history:
            print("No turns recorded.")
            return
        rows = [asdict(r) for r in self.history]
        with open(Path(path), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows)} turns → {Path(path).resolve()}")

