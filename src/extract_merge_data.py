import json
from feature_extraction import FeatureExtractor
import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline

toxicity_model = pipeline(
    "text-classification",
    model="facebook/roberta-hate-speech-dynabench-r4-target",
    device="cpu",
)
threat_model = pipeline(
    "text-classification",
    model="tomh/toxigen_roberta",
    device="cpu",
)

sentence_model= SentenceTransformer('all-MiniLM-L6-v2')
embedding_model = sentence_model

refusal_model= joblib.load("models/refusal_classifier.pkl")
threshold = 0.5



def process_scale_ai(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        dataset_scaleai = json.load(f)

    multi_turn_ScaleAI = [
        item for item in dataset_scaleai
        if len(item["turns"])
    ]

    print(f"Found {len(multi_turn_ScaleAI)} Scale AI conversations")

    all_rows = []

    for conv_id, convo in enumerate(multi_turn_ScaleAI):
        turns = convo["turns"]
        prev_user = ""
        prev_assistant= ""

        feature_extractor = FeatureExtractor(
            toxicity_model=toxicity_model,
            threat_model=threat_model,
            embedding_model=embedding_model,
            refusal_model=refusal_model,
        )

        for turn_id, turn in enumerate(turns):
            user_msg = turn["attack_message"]
            assistant_msg = turn["target_response"]
            label = turn["judge_result"]

            if turn_id == 0:
                features = feature_extractor.extract_features(
                    user_msg="",
                    assistant_msg="",
                    user_msg2=user_msg,
                    assistant_msg2=assistant_msg,
                )
            else:
                features = feature_extractor.extract_features(
                    user_msg=prev_user,
                    assistant_msg=prev_assistant,
                    user_msg2=user_msg,
                    assistant_msg2=assistant_msg,
                )

            all_rows.append({
                "conv_id": conv_id,
                "turn_id": turn_id,
                "label": label,
                **features
            })
            prev_user = user_msg
            prev_assistant = assistant_msg
    return pd.DataFrame(all_rows)



def process_crescendomation(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    multi_turn = [
        item for item in dataset
        if len(item["conversation"])
    ]
    print(f"Found {len(multi_turn)} Crescendomation conversations")

    all_rows = []

    for conv_id, convo in enumerate(multi_turn):
        turns = convo["conversation"]
        prev_user = ""
        prev_assistant = ""

        feature_extractor = FeatureExtractor(
            toxicity_model=toxicity_model,
            threat_model=threat_model,
            embedding_model=embedding_model,
            refusal_model=refusal_model,
        )

        for turn_id, turn in enumerate(turns):
            user_msg = turn["user"]
            assistant_msg = turn["assistant"]
            label = turn["label"]

            if turn_id == 0:
                features = feature_extractor.extract_features(
                    user_msg="",
                    assistant_msg="",
                    user_msg2=user_msg,
                    assistant_msg2=assistant_msg,
                )
            else:
                features = feature_extractor.extract_features(
                    user_msg=prev_user,
                    assistant_msg=prev_assistant,
                    user_msg2=user_msg,
                    assistant_msg2=assistant_msg,
                )

            all_rows.append({
                "conv_id": conv_id,
                "turn_id": turn_id,
                "label": label,
                **features
            })

            prev_user = user_msg
            prev_assistant = assistant_msg

    return pd.DataFrame(all_rows)



def process_opposite_day(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    multi_turn = [
        item for item in dataset
        if len(item["conversation"])
    ]
    print(f"Found {len(multi_turn)} Opposite Day conversations")
    all_rows = []

    for conv_id, convo in enumerate(multi_turn):
        turns = convo["conversation"]
        prev_user = ""
        prev_assistant = ""

        feature_extractor = FeatureExtractor(
            toxicity_model=toxicity_model,
            threat_model=threat_model,
            embedding_model=embedding_model,
            refusal_model=refusal_model,
        )

        for turn_id, turn in enumerate(turns):
            user_msg = turn["user"]
            assistant_msg = turn["assistant"]
            label = turn["label"]

            if turn_id == 0:
                features = feature_extractor.extract_features(
                    user_msg="",
                    assistant_msg="",
                    user_msg2=user_msg,
                    assistant_msg2=assistant_msg,
                )
            else:
                features = feature_extractor.extract_features(
                    user_msg=prev_user,
                    assistant_msg=prev_assistant,
                    user_msg2=user_msg,
                    assistant_msg2=assistant_msg,
                )

            all_rows.append({
                "conv_id": conv_id,
                "turn_id": turn_id,
                "label": label,
                **features
            })
            prev_user = user_msg
            prev_assistant = assistant_msg

    return pd.DataFrame(all_rows)


def process_benign_dataset(path: str) -> pd.DataFrame:
    all_rows = []
    conv_id = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            convs = item["conversations"]

            feature_extractor = FeatureExtractor(
                toxicity_model=toxicity_model,
                threat_model=threat_model,
                embedding_model=embedding_model,
                refusal_model=refusal_model,
            )

            prev_user = ""
            prev_assistant = ""
            turn_id = 0
            i = 0

            while i < len(convs):
                user_text = ""
                assist_text = ""

                if i < len(convs) and convs[i]["from"] == "human":
                    user_text = convs[i]["value"]
                    i += 1

                if i < len(convs) and convs[i]["from"] == "gpt":
                    assist_text = convs[i]["value"]
                    i += 1

                
                if not assist_text:
                    continue

                features = feature_extractor.extract_features(
                    user_msg=prev_user,
                    assistant_msg=prev_assistant,
                    user_msg2=user_text,
                    assistant_msg2=assist_text,
                )

                all_rows.append({
                    "conv_id": conv_id,
                    "turn_id": turn_id,
                    "label": 0,
                    **features
                })

                prev_user = user_text
                prev_assistant = assist_text
                turn_id += 1

            conv_id += 1  

    return pd.DataFrame(all_rows)

def main():


    # df_scale_ai = process_scale_ai(
    #     "data/raw/conversation_multiTurn_scaleAI.json"
    # )
    # df_crescendomation = process_crescendomation(
    #     "data/raw/multi_turn_conversations_from_crescendomation_08.json"
    # )
    # df_opposite_day = process_opposite_day(
    #     "data/raw/multi_turn_conversations_from_opposite_day_08.json"
    # )



    # df_new_benign = process_benign_dataset(
    #     "data/raw/benign_data.jsonl",

    # )

    # df_scale_ai.to_csv("data/cleaned/cleaned_train_multi_turn_ScaleAI.csv", index=False)
    # df_crescendomation.to_csv("data/cleaned/cleaned_train_multi_turn_Crescendomation.csv", index=False)
    # df_opposite_day.to_csv("data/cleaned/cleaned_train_multi_turn_OppositeDay.csv", index=False)
    # df_new_benign.to_csv("data/cleaned/cleaned_train_multi_turn_Benign.csv")

    df_new_benign = pd.read_csv("data/cleaned/cleaned_train_multi_turn_Benign.csv")
    df_scale_ai = pd.read_csv("data/cleaned/cleaned_train_multi_turn_ScaleAI.csv")
    df_crescendomation = pd.read_csv("data/cleaned/cleaned_train_multi_turn_Crescendomation.csv")
    df_opposite_day = pd.read_csv("data/cleaned/cleaned_train_multi_turn_OppositeDay.csv")

    offset = 0

    for dataset_df in [
        df_crescendomation,
        df_opposite_day,
        df_new_benign,
        df_scale_ai,
    ]:
        dataset_df["conv_id"] += offset
        offset = dataset_df["conv_id"].max() + 1
    df = pd.concat(
        [df_crescendomation, df_opposite_day, df_new_benign, df_scale_ai],
        ignore_index=True
    )

    print(df.shape)

    df.to_csv("data/merged/multi_turn_data.csv", index=False)

    return df


if __name__ == "__main__":
    main()