from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt


def calibrate_model(model, X_train, y_train):

    calibrated_model = CalibratedClassifierCV(
        model, method="sigmoid", cv=5
    )
    calibrated_model.fit(X_train, y_train)

    return calibrated_model


def plot_calibration(y_true, preds):

    prob_true, prob_pred = calibration_curve(
        y_true, preds, n_bins=10
    )

    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("Calibration Curve")
    plt.xlabel("Predicted PD")
    plt.ylabel("Observed Default Rate")
    plt.show()