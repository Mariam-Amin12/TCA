class TurnRecord:
    turn_id:          int
    timestamp:        str
    text:             str
    toxicity_score:   float
    threat_score:     float
    obfuscation_score:float
    topic_shift_score:float
    interaction_risk: float
    pattern_risk:     float
    historical_risk:  float   # = prev_progressive (what we knew *before* this turn)
    progressive_risk: float   # updated score *after* this turn
    risk_level:       int     # 0-10 integer version of progressive_risk
    flags:            str     # comma-separated flag names, e.g. "SPIKE,TREND"
    decision:         str     # ALLOW / WARN / BLOCK
 