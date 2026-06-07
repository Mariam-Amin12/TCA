import json
from pyexpat import model

from inference import JailbreakMultiTurnDetector
from visualize import plot_conversation, plot_conversations


detector = JailbreakMultiTurnDetector()

with open("data/raw/conversation_multiTurn_scaleAI.json", "r", encoding="utf-8") as f:
    dataset_scaleai = json.load(f)

multi_turn_for_viz = [item for item in dataset_scaleai if len(item["turns"]) > 4]

for conv_id, convo in enumerate(multi_turn_for_viz[5:10]):
    result = detector.score_conversation(convo["turns"])
    plot_conversation(result, conv_id)


