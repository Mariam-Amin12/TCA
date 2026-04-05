class RiskCalculator:
    def __init__(self, alpha=0.45, beta=0.35, gamma=0.20):
        # alpha+beta+gamma = 1.0 → progressive stays bounded in [0,1]
        # Shifted weight FROM memory TOWARD current signals
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

    def compute_interaction_risk(self, features: dict) -> float:
        # More balanced: toxicity now contributes 40% instead of 20%
        return (
            0.60 * features["threat_score"] +
            0.40 * features["toxicity_score"]
        )

    def compute_pattern_risk(self, features: dict) -> float:
        # More balanced: obfuscation weight raised from 20% → 35%
        return (
            0.65 * features["topic_shift_score"] +
            0.35 * features["obfuscation_score"]
        )

    def calculate_progressive_risk(self, features: dict, prev_progressive: float) -> float:
        interaction_risk = self.compute_interaction_risk(features)
        pattern_risk     = self.compute_pattern_risk(features)
        progressive = (
            self.alpha * prev_progressive +
            self.beta  * interaction_risk +
            self.gamma * pattern_risk
        )
        return round(min(progressive, 1.0), 4)   # explicit clamp for safety