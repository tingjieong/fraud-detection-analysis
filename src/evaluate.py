from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
import numpy as np

def fit_eval(train, test, options, feature, seed):
    if options not in {"smote","weights","none"}:
        raise ValueError(f"Unknown option {options}")
    steps = [("sc", StandardScaler())]
    if options == "smote":
        steps.append(("sm", SMOTE(random_state=seed)))
    steps.append(("clf", LogisticRegression(max_iter=2000,
                  class_weight="balanced" if options == "smote" else None,
                  random_state=seed)))
    pipe = ImbPipeline(steps).fit(train[feature], train["Class"])
    p = pipe.predict_proba(test[feature])[:, 1]
    y_true = test["Class"].to_numpy()

    metrics = {"pr_auc": average_precision_score(y_true, p),
               "roc_auc": roc_auc_score(y_true, p),
               "recall_at_90": recall_at_precision(y_true, p, target=0.90)}
    return metrics, y_true, p

def recall_at_precision(y, p, target=0.90):
    prec, rec, _ = precision_recall_curve(y, p)
    mask = prec[:-1] >= target
    return rec[:-1][mask].max() if mask.any() else 0.0

def precision_at_k(y, p, k=500):
    idx = np.argsort(-p)[:k]
    return y.iloc[idx].mean(), y.iloc[idx].sum() / y.sum()