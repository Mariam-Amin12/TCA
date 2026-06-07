

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from risk_calculator import RiskCalculator

MERGED_DATA_PATH  ="data/merged/multi_turn_data.csv"
PROCESSED_DIR  ="data/processed"
FIGURES_DIR    ="reports/figures"
SCALER_SAVE_PATH  ="models/scaler.pkl"         

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR,exist_ok=True)
os.makedirs(os.path.dirname(SCALER_SAVE_PATH), exist_ok=True)

def recompute_risks(df:pd.DataFrame, calc:RiskCalculator) -> pd.DataFrame:
    df =df.sort_values(["conv_id", "turn_id"]).reset_index(drop=True)

    interaction_list =[]
    pattern_list  =[]
    progressive_list =[]
    prev_prog_list=[]

    for _cid, group in df.groupby("conv_id", sort=False):
        prev =0.0
        for _, row in group.iterrows():
            interaction =calc.compute_interaction_risk(row)
            pattern  =calc.compute_pattern_risk(row)
            prog     =calc.calculate_progressive_risk(row, prev)

            interaction_list.append(interaction)
            pattern_list.append(pattern)
            progressive_list.append(prog)
            prev_prog_list.append(prev)   # record BEFORE updating

            prev =prog

    df["interaction_risk"]  =interaction_list
    df["pattern_risk"]   =pattern_list
    df["progressive_risk"]  =progressive_list
    df["prev_progressive"]  =prev_prog_list
    return df


def find_correlated_features(corr_matrix:pd.DataFrame, threshold:float =0.9):
    upper =corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    pairs =[]
    for col in upper.columns:
        if col in ("target", "label"):
            continue
        hi_corr =upper.index[abs(upper[col]) > threshold].tolist()
        for fc in hi_corr:
            if fc not in ("target", "label"):
                pairs.append({
                    "feature_A":col,
                    "feature_B":fc,
                    "correlation":round(corr_matrix.loc[col, fc], 3),
                })
    return pd.DataFrame(pairs)


def main():

    df =pd.read_csv(MERGED_DATA_PATH)

    # params_path ="config/optimized_params_risk.json"
    # if os.path.exists(params_path):
    #     with open(params_path) as f:
    #         Params =json.load(f)
    # else:
    #     params ={}

    
    Params ={
        'alpha':np.float64(0.5555555555555556),
        'beta': np.float64(0.22222222222222227),
        'gamma':np.float64(0.22222222222222227),
        'inter_alpha':np.float64(0.22222222222222227),
        'inter_beta':np.float64(0.5555555555555556),
        'inter_gamma':np.float64(0.22222222222222227),
        'pattern_alpha':np.float64(0.16666666666666669),
        'pattern_beta': np.float64(0.6666666666666667),
        'pattern_gamma':np.float64(0.16666666666666669),
    }


    calc =RiskCalculator(**Params)
    df=recompute_risks(df, calc)

    # feature_cols =[
    #     "toxicity_score", "threat_score",
    #     "topic_drift_score", "cumulative_drift", "drift_acceleration",
    #     "post_refusal",
    #     "interaction_risk", "pattern_risk", "progressive_risk", "prev_progressive",
    # ]

    feature_cols = [fo for fo in df.columns if fo not in ["conv_id", "turn_id", "label"]]
    missing =[c for c in feature_cols  if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns:{missing}")

    X =df[feature_cols].copy()
    y =df["label"].copy()

    X_train, X_temp, y_train, y_temp =train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test =train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    X_train =X_train.reset_index(drop=True)
    X_val=X_val.reset_index(drop=True)
    X_test  =X_test.reset_index(drop=True)
    y_train =y_train.reset_index(drop=True)
    y_val=y_val.reset_index(drop=True)
    y_test  =y_test.reset_index(drop=True)

    meta_train =df.loc[X_train.index, ["conv_id", "turn_id"]].reset_index(drop=True) \
        if False else None  # handled below via index tracking
    df_indexed =df[["conv_id", "turn_id"] + feature_cols + ["label"]].copy()
    df_indexed =df_indexed.reset_index(drop=True)

    idx_all=df_indexed.index.tolist()
    idx_train, idx_temp =train_test_split(idx_all, test_size=0.40, random_state=42,
                                           stratify=df_indexed["label"])
    idx_val,idx_test =train_test_split(idx_temp, test_size=0.50, random_state=42,
                                           stratify=df_indexed.loc[idx_temp, "label"])

    X_train =df_indexed.loc[idx_train, feature_cols].reset_index(drop=True)
    X_val=df_indexed.loc[idx_val,feature_cols].reset_index(drop=True)
    X_test  =df_indexed.loc[idx_test,feature_cols].reset_index(drop=True)
    y_train =df_indexed.loc[idx_train, "label"].reset_index(drop=True)
    y_val=df_indexed.loc[idx_val,"label"].reset_index(drop=True)
    y_test  =df_indexed.loc[idx_test,"label"].reset_index(drop=True)

    meta_train =df_indexed.loc[idx_train, ["conv_id", "turn_id"]].reset_index(drop=True)
    meta_val=df_indexed.loc[idx_val,["conv_id", "turn_id"]].reset_index(drop=True)
    meta_test  =df_indexed.loc[idx_test,["conv_id", "turn_id"]].reset_index(drop=True)

    print(X_train.describe().loc[["min", "max", "mean", "std"]].T.round(5).to_string())

    var_selector =VarianceThreshold(threshold=0.002)
    var_selector.fit(X_train)
    selected_var =X_train.columns[var_selector.get_support()].tolist()

    X_train =pd.DataFrame(var_selector.transform(X_train), columns=selected_var)
    X_val=pd.DataFrame(var_selector.transform(X_val),columns=selected_var)
    X_test  =pd.DataFrame(var_selector.transform(X_test),columns=selected_var)

    corr_with_target =X_train.corrwith(y_train).abs().sort_values(ascending=False)

    fig, ax =plt.subplots(figsize=(8, 5))
    corr_with_target.plot(kind="barh", ax=ax)
    ax.set_title("Feature correlation with target (train set)")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/feature_correlation.png")
    plt.close()

    selected_corr =corr_with_target[corr_with_target > 0.005].index.tolist()
    X_train =X_train[selected_corr]
    X_val=X_val[selected_corr]
    X_test  =X_test[selected_corr]
    print(f"   Kept {len(selected_corr)} features:{selected_corr}")

    df_corr     =X_train.copy()
    df_corr["target"] =y_train.values
    corr_matrix =df_corr.corr()


    correlated_pairs =find_correlated_features(corr_matrix, threshold=0.9)
    if not correlated_pairs.empty:
        print(f"\n   Highly correlated pairs:\n{correlated_pairs.to_string(index=False)}")

    features_to_remove =set()
    for _, row in correlated_pairs.iterrows():
        f1, f2 =row["feature_A"], row["feature_B"]
        # keep the one more correlated with the target
        c1 =abs(corr_matrix.loc[f1, "target"])
        c2 =abs(corr_matrix.loc[f2, "target"])
        features_to_remove.add(f2 if c1 >=c2 else f1)

    features_to_keep =[c for c in X_train.columns if c not in features_to_remove]
    X_train =X_train[features_to_keep]
    X_val=X_val[features_to_keep]
    X_test  =X_test[features_to_keep]
    print(f"\n   Removed :{features_to_remove}")
    print(f"   Kept    :{features_to_keep}")

    scaler =RobustScaler()
    X_train_scaled =pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=features_to_keep
    )
    X_val_scaled =pd.DataFrame(
        scaler.transform(X_val),
        columns=features_to_keep
    )
    X_test_scaled =pd.DataFrame(
        scaler.transform(X_test),
        columns=features_to_keep
    )


    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"   Scaler saved → {SCALER_SAVE_PATH}")
    rfecv =RFECV(
        estimator =RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        step   =1,
        cv     =5,
        scoring="f1_macro",
        n_jobs =-1,
    )
    rfecv.fit(X_train_scaled, y_train)

    selected_rfecv =X_train_scaled.columns[rfecv.support_].tolist()
    print(f"Optimal features ({len(selected_rfecv)}):{selected_rfecv}")

    # plot RFECV curve
    fig, ax =plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(rfecv.cv_results_["mean_test_score"]) + 1),
            rfecv.cv_results_["mean_test_score"])
    ax.set_xlabel("Number of features")
    ax.set_ylabel("CV F1 macro")
    ax.set_title("RFECV — optimal number of features")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/rfecv_curve.png")
    plt.close()

    X_train_final =X_train_scaled[selected_rfecv]
    X_val_final=X_val_scaled[selected_rfecv]
    X_test_final  =X_test_scaled[selected_rfecv]

    print("\n Saving processed CSVs…")

    def save_split(X_feat, meta, y, path):
        out =X_feat.copy()
        out["conv_id"] =meta["conv_id"].values
        out["turn_id"] =meta["turn_id"].values
        out["label"]=y.values
        out.to_csv(path, index=False)
        print(f"   Saved {path}  shape={out.shape}")

    save_split(X_train_final, meta_train, y_train, f"{PROCESSED_DIR}/train.csv")
    save_split(X_val_final,meta_val,y_val,f"{PROCESSED_DIR}/validation.csv")
    save_split(X_test_final,meta_test,y_test,f"{PROCESSED_DIR}/test.csv")
    feature_info ={
        "selected_features":selected_rfecv,
        "scaler":SCALER_SAVE_PATH,
    }
    with open(f"config/feature_info.json", "w") as f:
        json.dump(feature_info, f, indent=4)

    print(f"\nFinal features:{selected_rfecv}")
    return selected_rfecv


if __name__ =="__main__":
    main()