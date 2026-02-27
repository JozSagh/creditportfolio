import matplotlib.pyplot as plt
import shap
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report


def evaluate_oot(model, X_oot, y_oot):

    preds = model.predict_proba(X_oot)[:, 1]

    print("OOT AUC:", roc_auc_score(y_oot, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_oot, model.predict(X_oot)))
    print("Classification Report:")
    print(classification_report(y_oot, model.predict(X_oot)))

    return preds


def plot_roc(y_true, preds):
    fpr, tpr, _ = roc_curve(y_true, preds)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("OOT ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()


def shap_analysis(model, X_sample):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)

    shap.plots.bar(shap_values)
    shap.plots.beeswarm(shap_values)