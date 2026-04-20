import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Config ---
DATA_FOLDER = "datasets/x264"
REPEATS = 30

all_results = []

for filename in os.listdir(DATA_FOLDER):
    if not filename.endswith(".csv"):
        continue

    df = pd.read_csv(os.path.join(DATA_FOLDER, filename))
    
    X = df.drop(columns=["time"]).values
    y = df["time"].values

    mape_scores, mae_scores, rmse_scores = [], [], []

    for _ in range(REPEATS):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mape_scores.append(mean_absolute_percentage_error(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        rmse_scores.append(root_mean_squared_error(y_test, y_pred))

    all_results.append({
        "workload": filename,
        "MAPE": np.mean(mape_scores),
        "MAE":  np.mean(mae_scores),
        "RMSE": np.mean(rmse_scores)
    })
    print(f"{filename}: MAPE={np.mean(mape_scores):.4f}, MAE={np.mean(mae_scores):.4f}, RMSE={np.mean(rmse_scores):.4f}")

# Overall average across all workloads
print("\n--- Overall Average ---")
print(f"MAPE: {np.mean([r['MAPE'] for r in all_results]):.4f}")
print(f"MAE:  {np.mean([r['MAE']  for r in all_results]):.4f}")
print(f"RMSE: {np.mean([r['RMSE'] for r in all_results]):.4f}")