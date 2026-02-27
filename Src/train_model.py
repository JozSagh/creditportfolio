import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score


def split_oot(df, features):
    train_df = df[df["month"] <= 11]
    oot_df = df[df["month"] == 12]

    X_train = train_df[features]
    y_train = train_df["target"]

    X_oot = oot_df[features]
    y_oot = oot_df["target"]

    return train_df, X_train, y_train, X_oot, y_oot


def train_with_groupkfold(train_df, X_train, y_train, n_splits=5):

    gkf = GroupKFold(n_splits=n_splits)
    auc_scores = []

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        use_label_encoder=False
    )

    for train_idx, val_idx in gkf.split(
        X_train, y_train, groups=train_df["customer_id"]
    ):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_val)[:, 1]
        auc_scores.append(roc_auc_score(y_val, preds))

    print("Cross-Validated AUC:", np.mean(auc_scores))

    model.fit(X_train, y_train)
    return model