class RiskCalculator:
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        self.historical_risk = 0.0
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def compute_interaction_risk(self, f: dict) -> float:
        return (
            0.6 * f["threat_score"] +
            0.2 * f["obfuscation_score"] +
            0.2 * f["toxicity_score"] 
            
        )


    def compute_pattern_risk(self, f: dict) -> float:
        return (
            0.8 * f["topic_shift_score"] +
            0.2 * f["obfuscation_score"]
        )

    def calculate_progressive_risk(self, f: dict) -> float:
        interaction_risk = self.compute_interaction_risk(f)
        pattern_risk = self.compute_pattern_risk(f)

        progressive = (
            self.alpha * self.historical_risk +
            self.beta * interaction_risk +
            self.gamma * pattern_risk
        )

        # update memory
        self.historical_risk = progressive

        return round(progressive, 4)

    def reset(self):
        self.historical_risk = 0.0


