import os
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore", category=RuntimeWarning)

DATA_FOLDER = "datasets/z3"
REPEATS = 30

os.makedirs("results", exist_ok=True)
rows = []

for filename in sorted(os.listdir(DATA_FOLDER)):
    if not filename.endswith(".csv"):
        continue

    df = pd.read_csv(os.path.join(DATA_FOLDER, filename))
    X = df.drop(columns=["time"]).values
    y = df["time"].values

    lr_mape, lr_mae, lr_rmse = [], [], []
    rf_mape, rf_mae, rf_rmse = [], [], []

    for _ in range(REPEATS):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)

        rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)

        lr_mape.append(mean_absolute_percentage_error(y_test, lr_pred))
        lr_mae.append(mean_absolute_error(y_test, lr_pred))
        lr_rmse.append(root_mean_squared_error(y_test, lr_pred))

        rf_mape.append(mean_absolute_percentage_error(y_test, rf_pred))
        rf_mae.append(mean_absolute_error(y_test, rf_pred))
        rf_rmse.append(root_mean_squared_error(y_test, rf_pred))

    # Wilcoxon signed-rank test on MAPE scores
    try:
        _, p = wilcoxon(lr_mape, rf_mape, alternative="greater")
    except ValueError:
        p = float("nan")

    # Vargha-Delaney A12 effect size
    a12 = sum(1 if l > r else 0.5 if l == r else 0 for l, r in zip(lr_mape, rf_mape)) / REPEATS

    print(f"{filename}")
    print(f"  LR  -> MAPE={np.mean(lr_mape):.4f}  MAE={np.mean(lr_mae):.4f}  RMSE={np.mean(lr_rmse):.4f}")
    print(f"  RF  -> MAPE={np.mean(rf_mape):.4f}  MAE={np.mean(rf_mae):.4f}  RMSE={np.mean(rf_rmse):.4f}")
    print(f"  p={p:.4f}  A12={a12:.4f}\n")

    rows.append({
        "workload":    filename,
        "LR_MAPE":     np.mean(lr_mape),
        "RF_MAPE":     np.mean(rf_mape),
        "LR_MAE":      np.mean(lr_mae),
        "RF_MAE":      np.mean(rf_mae),
        "LR_RMSE":     np.mean(lr_rmse),
        "RF_RMSE":     np.mean(rf_rmse),
        "p_value":     p,
        "A12":         a12,
        "significant": "YES" if p < 0.05 else "NO"
    })

results = pd.DataFrame(rows)
results.to_csv("results/results.csv", index=False)

print("--- Overall Averages ---")
print(f"LR   MAPE={results['LR_MAPE'].mean():.4f}  MAE={results['LR_MAE'].mean():.4f}  RMSE={results['LR_RMSE'].mean():.4f}")
print(f"RF   MAPE={results['RF_MAPE'].mean():.4f}  MAE={results['RF_MAE'].mean():.4f}  RMSE={results['RF_RMSE'].mean():.4f}")
print(f"\nRF beats LR on MAPE: {(results['RF_MAPE'] < results['LR_MAPE']).sum()}/{len(results)} workloads")
print(f"Statistically significant (p<0.05): {(results['significant'] == 'YES').sum()}/{len(results)} workloads")