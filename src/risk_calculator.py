class RiskCalculator:
    def __init__(self, alpha=0.45, beta=0.25, gamma=0.30,
                 inter_alpha=0.5, inter_beta=0.3, inter_gamma=0.2,
                 pattern_alpha=0.4, pattern_beta=0.3, pattern_gamma=0.3):
        self.alpha = alpha   
        self.beta  = beta    
        self.gamma = gamma 
        self.inter_alpha = inter_alpha
        self.inter_beta  = inter_beta   
        self.inter_gamma = inter_gamma
        self.pattern_alpha = pattern_alpha
        self.pattern_beta  = pattern_beta
        self.pattern_gamma = pattern_gamma

    def compute_interaction_risk(self, features: dict) -> float:
        return (
            self.inter_alpha * features["threat_score"] +
            self.inter_beta * features["toxicity_score"] +
            self.inter_gamma * features["post_refusal"]     
        )

    def compute_pattern_risk(self, features: dict) -> float:
        return (
            self.pattern_alpha * features["topic_shift_score"] +
            self.pattern_beta * features["cumulative_drift"] +  
            self.pattern_gamma * features["drift_acceleration"] 
        )

    def calculate_progressive_risk(self, features: dict, prev_progressive: float) -> float:
        interaction_risk = self.compute_interaction_risk(features)
        pattern_risk     = self.compute_pattern_risk(features)
        progressive = (
            self.alpha * prev_progressive +
            self.beta  * interaction_risk +
            self.gamma * pattern_risk
        )
        return round(min(progressive, 1.0), 4)