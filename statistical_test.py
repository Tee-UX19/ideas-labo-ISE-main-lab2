import json
import numpy as np
from scipy.stats import wilcoxon

with open("baseline_results.json") as f:
    baseline = json.load(f)

with open("random_forest_results.json") as f:
    rf = json.load(f)

print(f"{'Workload':<50} {'Baseline MAPE':>15} {'RF MAPE':>10} {'p-value':>10} {'Sig?':>6}")
print("-" * 95)

p_values = []

for workload in baseline:
    b_scores = baseline[workload]
    r_scores = rf[workload]
    
    stat, p = wilcoxon(b_scores, r_scores)
    p_values.append(p)
    
    sig = "✓" if p < 0.05 else "✗"
    print(f"{workload:<50} {np.mean(b_scores):>15.4f} {np.mean(r_scores):>10.4f} {p:>10.4f} {sig:>6}")

print(f"\nSignificant differences (p < 0.05): {sum(p < 0.05 for p in p_values)}/{len(p_values)}")