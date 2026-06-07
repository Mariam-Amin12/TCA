import json
import os

import numpy as np
import optuna
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from risk_calculator import RiskCalculator

df = pd.read_csv("data/merged/multi_turn_data.csv")

conv_ids = df["conv_id"].unique()

train_conv_ids, valid_conv_ids = train_test_split(
    conv_ids,
    test_size=0.2,
    random_state=42,
    shuffle=True,
)

train_df = df[df["conv_id"].isin(train_conv_ids)].copy()
valid_df = df[df["conv_id"].isin(valid_conv_ids)].copy()


def recompute_risks(df: pd.DataFrame, calc: RiskCalculator) -> pd.DataFrame:

    df = df.sort_values(["conv_id", "turn_id"]).reset_index(drop=True)

    interaction_list = []
    pattern_list = []
    progressive_list = []
    prev_prog_list = []

    for _, group in df.groupby("conv_id", sort=False):

        prev = 0.0

        for _, row in group.iterrows():

            interaction = calc.compute_interaction_risk(row)
            pattern = calc.compute_pattern_risk(row)
            prog = calc.calculate_progressive_risk(row, prev)

            interaction_list.append(interaction)
            pattern_list.append(pattern)
            progressive_list.append(prog)
            prev_prog_list.append(prev)

            prev = prog

    df["interaction_risk"] = interaction_list
    df["pattern_risk"] = pattern_list
    df["progressive_risk"] = progressive_list
    df["prev_progressive"] = prev_prog_list

    return df

def suggest_triplet(trial: optuna.Trial, prefix: str):

    a = trial.suggest_float(f"{prefix}_a", 0.0, 1.0)
    b = trial.suggest_float(f"{prefix}_b", 0.0, 1.0 - a)
    c = 1.0 - a - b

    return a, b, c

def best_f1_score(y_true, scores):

    best_f1 = 0.0
    best_threshold = 0.5

    for threshold in np.arange(0.01, 1.00, 0.01):

        preds = (scores >= threshold).astype(int)

        score = f1_score(
            y_true,
            preds,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_f1, best_threshold

def objective(trial: optuna.Trial):

    alpha, beta, gamma = suggest_triplet(trial, "main")

    ia, ib, ig = suggest_triplet(trial, "inter")

    pa, pb, pg = suggest_triplet(trial, "pattern")

    params = dict(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        inter_alpha=ia,
        inter_beta=ib,
        inter_gamma=ig,
        pattern_alpha=pa,
        pattern_beta=pb,
        pattern_gamma=pg,
    )

    risk_calc = RiskCalculator(**params)

    valid_scored = recompute_risks(
        valid_df.copy(),
        risk_calc,
    )

    best_f1, _ = best_f1_score(
        valid_scored["label"],
        valid_scored["progressive_risk"],
    )

    return best_f1

if __name__ == "__main__":

    study = optuna.create_study(direction="maximize")

    study.optimize(
        objective,
        n_trials=500,
        show_progress_bar=True,
    )

    print("\nBest F1:", study.best_value)
    print("Best Params:", study.best_params)

    bp = study.best_params

    alpha = bp["main_a"]
    beta = bp["main_b"]
    gamma = 1.0 - alpha - beta

    inter_alpha = bp["inter_a"]
    inter_beta = bp["inter_b"]
    inter_gamma = 1.0 - inter_alpha - inter_beta

    pattern_alpha = bp["pattern_a"]
    pattern_beta = bp["pattern_b"]
    pattern_gamma = 1.0 - pattern_alpha - pattern_beta

    final_params = {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "inter_alpha": inter_alpha,
        "inter_beta": inter_beta,
        "inter_gamma": inter_gamma,
        "pattern_alpha": pattern_alpha,
        "pattern_beta": pattern_beta,
        "pattern_gamma": pattern_gamma,
    }

    final_calc = RiskCalculator(**final_params)

    valid_scored = recompute_risks(
        valid_df.copy(),
        final_calc,
    )

    final_f1, final_threshold = best_f1_score(
        valid_scored["label"],
        valid_scored["progressive_risk"],
    )

    

    os.makedirs("config", exist_ok=True)

    with open(
        "config/optimized_params_risk.json",
        "w",
    ) as f:
        json.dump(
            final_params,
            f,
            indent=4,
        )

    print("\nBest Validation F1:", final_f1)
    print("Best Threshold:", final_threshold)
    print("Saved to config/optimized_params_risk.json")