from collections import deque
from typing import List

class DecisionMaker:
    """
    Thresholds
    ----------
    t_warn               : progressive_risk ≥ this → WARN
    t_block              : progressive_risk ≥ this → BLOCK
    spike_threshold      : jump from last turn that triggers SPIKE flag
    trend_threshold      : slope over window that triggers TREND flag
    sustained_threshold  : window average that triggers SUSTAINED flag
    obfuscation_threshold: obfuscation_score that triggers OBFUSCATION flag
    window_size          : how many recent turns to consider for TREND/SUSTAINED
    """
    def __init__(
    self,
    t_warn=0.15,               # was 0.22 (too high), original was 0.10 (too low)
    t_block=0.28,              # was 0.42 (too high), original was 0.20 (too low)
    window_size=3,
    spike_threshold=0.15,
    trend_threshold=0.05,
    sustained_threshold=0.14,  # was 0.18
    obfuscation_threshold=0.10,
    ):
        self.t_warn = t_warn
        self.t_block = t_block
        self.spike_threshold = spike_threshold
        self.trend_threshold = trend_threshold
        self.sustained_threshold = sustained_threshold
        self.obfuscation_threshold = obfuscation_threshold
 
        self.risk_window = deque(maxlen=window_size)
 
        self.prev_progressive = 0.0
 
    def decide(self, features: dict, risk_scores: dict) -> dict:
        prog = risk_scores["progressive_risk"]
        self.risk_window.append(prog)
 
        flags    = self._detect_flags(features, prog)
        decision = self._make_decision(prog, flags)
        level    = min(int(prog * 10), 10)   # map 0-1 float → 0-10 int
 
        self.prev_progressive = prog
 
        return {"decision": decision, "flags": flags, "risk_level": level}
 
    def reset(self):
        self.risk_window.clear()
        self.prev_progressive = 0.0   # ← reset to 0 for a fresh conversation
 
    def _detect_flags(self, features: dict, prog: float) -> List[str]:
        flags  = []
        window = list(self.risk_window)
 
        if prog - self.prev_progressive > self.spike_threshold:
            flags.append("SPIKE")
 
        if features["obfuscation_score"] > self.obfuscation_threshold:
            flags.append("OBFUSCATION")
 
        if len(window) >= 3 and self._slope(window) > self.trend_threshold:
            flags.append("TREND")
 
        if len(window) >= 3 and sum(window) / len(window) > self.sustained_threshold:
            flags.append("SUSTAINED")
 
        return flags
 
    def _make_decision(self, prog: float, flags: List[str]) -> str:
        if "OBFUSCATION" in flags and "SPIKE" in flags:
            return "BLOCK"
 
        if prog >= self.t_block:
            base = "BLOCK"
        elif prog >= self.t_warn:
            base = "WARN"
        else:
            base = "ALLOW"
 
        soft_flags = [f for f in flags if f in ("TREND", "SUSTAINED", "SPIKE")]
        if base == "WARN" and len(soft_flags) >= 2:
            return "BLOCK"
 
        return base
 
    @staticmethod
    def _slope(values: List[float]) -> float:
        """Linear regression slope over a short list of values."""
        n      = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        den = sum((i - x_mean) ** 2 for i in range(n))
        return num / den if den else 0.0
 