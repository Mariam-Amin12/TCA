
import argparse
import os
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from evaluate import MultiTurnJailbreakEvaluator

TRAIN_PATH= "data/processed/train.csv"
VALIDATION_PATH= "data/processed/validation.csv"
TEST_PATH= "data/processed/test.csv"

FIGURES_DIR= "reports/figures"
MODELS_DIR= "models"
MLFLOW_DB= "models/mlflow.db"
EVAL_REPORT= "reports/evaluation_report.txt"

EXPERIMENT_NAME= "multi_turn_jailbreak_experiment+"

MODEL_PARAMS= {
    "logistic_regression":{
        "penalty":["l2"],
        "C":[0.1,1.0,10.0],
        "solver":["lbfgs"],
        "max_iter":[10_000],
        "random_state":[42],
    },
    "decision_tree":{
        "max_depth":[5,10,20,None],
        "min_samples_split":[2,5,10],
        "min_samples_leaf":[5,10],
        "random_state":[42],
    },
    "random_forest":{
        "n_estimators":[100,200,300],
        "max_depth":[5,7,10,20,None],
        "random_state":[42],
    },
    "xgboost":{
        "n_estimators":[100,200,300],
        "max_depth":[3,6,7,10],
        "learning_rate":[0.05,0.1,0.01],
        "random_state":[42],
    },
    "svc":{
        "C":[0.1,1.0,10.0,5],
        "kernel":["rbf","linear"],
    },
}

CANDIDATE_MODELS= {
    "logistic_regression":LogisticRegression(),
    "decision_tree":DecisionTreeClassifier(),
    "random_forest":RandomForestClassifier(n_jobs=-1),
    "xgboost":XGBClassifier(n_jobs=-1,eval_metric="logloss"),
    "svc":SVC(probability=True),
}


def save_confusion_matrix(cm,path:str,title:str) -> str:
    os.makedirs(os.path.dirname(path),exist_ok=True)
    plt.figure()
    sns.heatmap(cm,annot=True,fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def evaluate_model(y_true,y_pred):
    """Standard sklearn metrics."""
    acc= accuracy_score(y_true,y_pred)
    f1= f1_score(y_true,y_pred,average="macro")
    report= classification_report(y_true,y_pred)
    cm= confusion_matrix(y_true,y_pred)
    return acc,f1,report,cm


def group_by_conv(X:pd.DataFrame,y_true,y_pred):
 
    df= X[["conv_id","turn_id"]].copy()
    df["_label"]= list(y_true)
    df["_pred"]= list(y_pred)
    df= df.sort_values(["conv_id","turn_id"]).reset_index(drop=True)

    conv_ids,y_true_g,y_pred_g= [],[],[]
    for cid,grp in df.groupby("conv_id",sort=False):
        conv_ids.append(cid)
        y_true_g.append(grp["_label"].tolist())
        y_pred_g.append(grp["_pred"].tolist())

    return conv_ids,y_true_g,y_pred_g


def append_eval_report(path:str,text:str):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    with open(path,"a",encoding="utf-8") as f:
        f.write(text + "\n")


class ModelTrainer:
    def __init__(self,train_path,val_path,test_path):
        train_df= pd.read_csv(train_path)
        val_df= pd.read_csv(val_path)
        test_df= pd.read_csv(test_path)

        self.X_train= train_df.drop(columns=["label"])
        self.y_train= train_df["label"]
        self.X_val= val_df.drop(columns=["label"])
        self.y_val= val_df["label"]
        self.X_test= test_df.drop(columns=["label"])
        self.y_test= test_df["label"]

        print(f"Train :{self.X_train.shape}  |  Val :{self.X_val.shape}  |  Test :{self.X_test.shape}")

      
        meta_cols= [c for c in ["conv_id","turn_id"] if c in self.X_train.columns]
        self.feature_cols= [c for c in self.X_train.columns if c not in meta_cols]
        print(f"Features:{len(self.feature_cols)}  (meta columns:{meta_cols})")

        # write the feature columns in a txt file
        append_eval_report(EVAL_REPORT,
            f"\n{'='*55}\n" 
            f"FEATURES USED FOR MODELING\n"
            f"{'-'*55}\n"
            f"{','.join(self.feature_cols)}\n"
            f"{'='*55}\n"
        )

        self.jb_evaluator= MultiTurnJailbreakEvaluator()
        self.best_model= None
        self.best_name= None
        self.best_val_f1= -1.0
        self.best_test_f1= -1.0    
        mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
        mlflow.set_experiment(EXPERIMENT_NAME)
        experiment= mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        self.experiment_id= experiment.experiment_id

    def _train_one(self,name:str,model,param_grid:dict):

        run_name= f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name,experiment_id=self.experiment_id):

        
            grid= GridSearchCV(
                estimator= model,
                param_grid= param_grid,
                cv= 3,
                scoring= "f1_macro",
                n_jobs= -1,
            )
            grid.fit(self.X_train[self.feature_cols],self.y_train)
            best= grid.best_estimator_
           
            y_train_pred= best.predict(self.X_train[self.feature_cols])
            y_val_pred= best.predict(self.X_val[self.feature_cols])
            y_test_pred= best.predict(self.X_test[self.feature_cols])
            if hasattr(best,"predict_proba"):
                y_train_prob= best.predict_proba(self.X_train[self.feature_cols])[:,1]
                y_val_prob= best.predict_proba(self.X_val[self.feature_cols])[:,1]
                y_test_prob= best.predict_proba(self.X_test[self.feature_cols])[:,1]
            else:
                
                y_train_prob= y_train_pred.astype(float)
                y_val_prob= y_val_pred.astype(float)
                y_test_prob= y_test_pred.astype(float)

            train_acc,train_f1,train_rep,train_cm= evaluate_model(self.y_train,y_train_pred)
            val_acc,val_f1,val_rep,val_cm= evaluate_model(self.y_val,y_val_pred)
            test_acc,test_f1,test_rep,test_cm= evaluate_model(self.y_test,y_test_pred)

            base= f"{FIGURES_DIR}/{name}"
            train_cm_path= save_confusion_matrix(train_cm,f"{base}_train_cm.png",f"{name} — Train")
            val_cm_path= save_confusion_matrix(val_cm,f"{base}_val_cm.png",f"{name} — Val")
            test_cm_path= save_confusion_matrix(test_cm,f"{base}_test_cm.png",f"{name} — Test")
            conv_ids_tr,yt_tr,yp_tr= group_by_conv(self.X_train,self.y_train,y_train_prob)
            append_eval_report(EVAL_REPORT,
            f"\n{'='*55}\n" 
            f"\ntrain classification report"  
            )
            jb_train= self.jb_evaluator.evaluate(yt_tr,yp_tr)

            conv_ids_va,yt_va,yp_va= group_by_conv(self.X_val,self.y_val,y_val_prob)
            append_eval_report(EVAL_REPORT,
            f"\n{'='*55}\n" 
            f"\nval classification report"  
            )
            jb_val= self.jb_evaluator.evaluate(yt_va,yp_va)

            conv_ids_te,yt_te,yp_te= group_by_conv(self.X_test,self.y_test,y_test_prob)
            append_eval_report(EVAL_REPORT,
            f"\n{'='*55}\n" 
            f"\ntest classification report"
            )
            jb_test= self.jb_evaluator.evaluate(yt_te,yp_te)

            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics({
            "cv_f1_macro": grid.best_score_,

            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "test_acc": test_acc,
            "test_f1": test_f1,

           
            "train_true_asr": jb_train["dataset_asr"],
            "train_conv_recall": jb_train["conversation_detection"]["recall"],
            "train_conv_precision": jb_train["conversation_detection"]["precision"],
            "train_conv_f1": jb_train["conversation_detection"]["f1"],
            "train_turn_f1": jb_train["turn_f1"],
    

            "val_true_asr": jb_val["dataset_asr"],
            "val_conv_recall": jb_val["conversation_detection"]["recall"],
            "val_conv_precision": jb_val["conversation_detection"]["precision"],
            "val_conv_f1": jb_val["conversation_detection"]["f1"],
            "val_turn_f1": jb_val["turn_f1"],
            

           
            "test_true_asr": jb_test["dataset_asr"],
            "test_conv_recall": jb_test["conversation_detection"]["recall"],
            "test_conv_precision": jb_test["conversation_detection"]["precision"],
            "test_conv_f1": jb_test["conversation_detection"]["f1"],
            "test_turn_f1": jb_test["turn_f1"],
            
        })

            mlflow.log_text(train_rep,"train_classification_report.txt")
            mlflow.log_text(val_rep,"val_classification_report.txt")
            mlflow.log_text(test_rep,"test_classification_report.txt")

            mlflow.log_artifact(train_cm_path,"confusion_matrices")
            mlflow.log_artifact(val_cm_path,"confusion_matrices")
            mlflow.log_artifact(test_cm_path,"confusion_matrices")

            mlflow.sklearn.log_model(best,name)
            
            
            append_eval_report(EVAL_REPORT,
                f"\n{'='*55}\n"
                f"Model :{name}\n"
                f"Params:{grid.best_params_}\n"
                f"{'='*55}\n"
                f" Train  acc={train_acc:.4f}  f1={train_f1:.4f}\n"
                f"Val    acc={val_acc:.4f}  f1={val_f1:.4f}\n"
                f"Test   acc={test_acc:.4f}  f1={test_f1:.4f}\n"
                f"conv-f1={jb_test['conversation_detection']['f1']:.4f}"
            )

        if val_f1 > self.best_val_f1:
            self.best_val_f1= val_f1
            self.best_model= best
            self.best_test_f1= test_f1
            self.best_name= name
            print(f"\n  New best model:{name}  (val F1={val_f1:.4f})")

        return test_f1

    def train_all(self):
        for name,model in CANDIDATE_MODELS.items():
            self._train_one(name,model,MODEL_PARAMS[name])

        print(f"\n{'='*55}")
        print(f"  Best model  :{self.best_name}")
        print(f"  Test F1     :{self.best_test_f1:.4f}")
        print(f"{'='*55}")
        return self.best_model

    def save_best(self):
        if self.best_model is None:
            raise RuntimeError("No model trained yet. Call train_all() first.")

        os.makedirs(MODELS_DIR,exist_ok=True)

        # versioned save
        existing= [
            f for f in os.listdir(MODELS_DIR)
            if f.startswith("best_model_") and f.endswith(".pkl")
            and "latest" not in f
        ]
        version= len(existing)
        version_path= os.path.join(MODELS_DIR,f"best_model_v{version}_{self.best_name}.pkl")
        latest_path= os.path.join(MODELS_DIR,"best_model_latest.pkl")

        joblib.dump(self.best_model,version_path)
        joblib.dump(self.best_model,latest_path)

        print(f"\n  Saved → {version_path}")
        print(f"  Saved → {latest_path}")

        append_eval_report(EVAL_REPORT,
            f"\n{'='*55}\n"
            f"BEST MODEL SAVED\n"
            f"  Name    :{self.best_name}\n"
            f"  Test F1 :{self.best_test_f1:.4f}\n"
            f"  Path    :{version_path}\n"
            f"{'='*55}\n"
        )

        return latest_path


def parse_args():
    p= argparse.ArgumentParser(description="Train jailbreak classifiers")
    p.add_argument("--train",default=TRAIN_PATH)
    p.add_argument("--val",default=VALIDATION_PATH)
    p.add_argument("--test",default=TEST_PATH)
    return p.parse_args()


if __name__== "__main__":
    args= parse_args()

    os.makedirs(FIGURES_DIR,exist_ok=True)
    os.makedirs(MODELS_DIR,exist_ok=True)
    os.makedirs("reports",exist_ok=True)

    trainer= ModelTrainer(
        train_path= args.train,
        val_path= args.val,
        test_path= args.test,
    )

    trainer.train_all()
    trainer.save_best()