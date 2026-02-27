from src.data_generation import generate_grouped_data
from src.train_model import split_oot, train_with_groupkfold
from src.evaluation import evaluate_oot, plot_roc, shap_analysis
from src.calibration import calibrate_model, plot_calibration
from src.scoring import create_deciles, generate_score

features = ["income", "age", "existing_debt", "utilization_rate"]

# 1. Generate data
df = generate_grouped_data()

# 2. OOT split
train_df, X_train, y_train, X_oot, y_oot = split_oot(df, features)

# 3. Train model with leakage prevention
model = train_with_groupkfold(train_df, X_train, y_train)

# 4. OOT evaluation
oot_preds = evaluate_oot(model, X_oot, y_oot)
plot_roc(y_oot, oot_preds)

# 5. SHAP
shap_analysis(model, X_oot)

# 6. Calibration
calibrated_model = calibrate_model(model, X_train, y_train)
calib_preds = calibrated_model.predict_proba(X_oot)[:, 1]
plot_calibration(y_oot, calib_preds)

# 7. Deciles + scoring
import pandas as pd
oot_df = X_oot.copy()
oot_df["target"] = y_oot
oot_df["pd"] = calib_preds

oot_df, decile_summary = create_deciles(oot_df)
oot_df = generate_score(oot_df)

print(decile_summary)