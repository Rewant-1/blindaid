import json
import numpy as np
from scipy.stats import wilcoxon

file_path = r'c:\blindaid\evaluation\results_video_evaluation.json'

with open(file_path, 'r') as f:
    data = json.load(f)

afp_coverage = []
static_coverage = []

for clip in data:
    if "afp" in clip and "static_skip" in clip:
        afp_cov = clip["afp"]["coverage"]["coverage"]
        static_cov = clip["static_skip"]["coverage"]["coverage"]
        afp_coverage.append(afp_cov)
        static_coverage.append(static_cov)

afp_coverage = np.array(afp_coverage)
static_coverage = np.array(static_coverage)

# Wilcoxon signed-rank test
stat, p_value = wilcoxon(afp_coverage, static_coverage)

print("--- Statistical Significance ---")
print(f"Number of clips: {len(afp_coverage)}")
print(f"AFP Mean Coverage: {np.mean(afp_coverage):.4f}")
print(f"Static 1/10 Mean Coverage: {np.mean(static_coverage):.4f}")
print(f"Wilcoxon statistic: {stat}")
print(f"p-value: {p_value:.6f}")
print(f"Difference (AFP - Static): {np.mean(afp_coverage - static_coverage):.4f}")

if p_value < 0.05:
    print("Result: Statistically Significant difference")
else:
    print("Result: NO statistically significant difference (they perform comparably)")
