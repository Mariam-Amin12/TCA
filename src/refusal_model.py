import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import mlflow
import mlflow.xgboost
import optuna
 
import xgboost as xgb
import joblib

def run_refusal_pipeline(
    data_path="data/cleaned_train.csv",
    model_path="refusal_classifier.pkl",
    embed_model=None,
    experiment_name="refusal_detection_xgb",
    n_trials=50
):
    df = pd.read_csv(data_path)
    texts = df["response"].tolist()
    labels = df["label"].tolist()

    embeddings = embed_model.encode(texts)

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42
    )

    mlflow.set_experiment(experiment_name)
    mlflow.log_param("embedding_model", str(embed_model))
    mlflow.log_param("dataset", data_path)

    def objective(trial):
        with mlflow.start_run(nested=True):
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 5)
            }

            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)

            y_prob = model.predict_proba(X_val)[:, 1]
            threshold = trial.suggest_float("threshold", 0.2, 0.7)
            y_pred = (y_prob > threshold).astype(int)
            f1 = f1_score(y_val, y_pred)
            
            mlflow.xgboost.log_model(final_model, "model")

            mlflow.log_params(params)
            mlflow.log_param("threshold", threshold)
            mlflow.log_metric("f1_val", f1)
            return f1

    with mlflow.start_run():
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best = study.best_params
        threshold = best.pop("threshold")

        final_model = xgb.XGBClassifier(**best)
        final_model.fit(X_train, y_train)

        y_prob = final_model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > threshold).astype(int)
        test_f1 = f1_score(y_test, y_pred)

        mlflow.log_params(best)
        mlflow.log_param("best_threshold", threshold)
        mlflow.log_metric("f1_test", test_f1)
        mlflow.xgboost.log_model(final_model, "model")

        joblib.dump(final_model, model_path)

    def predict(texts):
        embs = embed_model.encode(texts)
        pred = final_model.predict(embs)
        probs = final_model.predict_proba(embs)[:, 1]
        return [{"text": t, "label": int(p), "prob": float(pr)} for t, p, pr in zip(texts, pred, probs)]

    return final_model, predict


# final_model, predict_fn = run_refusal_pipeline(data_path="data/cleaned_train.csv", model_path="refusal.pkl", embed_model=sentence_model)
# print(predict_fn(["no will not provide you with this", "Sure, I will"]))



def Load_Model_MLflow(X_train, y_train, run_id):
    
    client = mlflow.MlflowClient()

    run_id = run_id

    run = client.get_run(run_id)

    params = run.data.params
    print("Parameters:", params)

    metrics = run.data.metrics
    print("Metrics:", metrics)
   
    xgb_params = {
        'objective': params['objective'],
        'eval_metric': params['eval_metric'],
        'max_depth': int(params['max_depth']),
        'learning_rate': float(params['learning_rate']),
        'n_estimators': int(params['n_estimators']),
        'subsample': float(params['subsample']),
        'colsample_bytree': float(params['colsample_bytree']),
        'gamma': float(params['gamma']),
        'reg_alpha': float(params['reg_alpha']),
        'reg_lambda': float(params['reg_lambda']),
        'scale_pos_weight': float(params['scale_pos_weight'])
    }

    threshold = float(params['threshold'])
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train, y_train)

    joblib.dump(model, "refusal_classifier1.pkl")
    

    return model, threshold   


