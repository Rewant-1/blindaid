"""Verify all paper numbers against ground truth evaluation data."""
import json
from pathlib import Path
import statistics

DATA_DIR = Path(__file__).resolve().parent

# === 1. Real-World Video Evaluation (Table 6) ===
with open(DATA_DIR / "results_video_evaluation.json", "r") as f:
    video_data = json.load(f)

print("=" * 70)
print("REAL-WORLD VIDEO EVALUATION (Table 6 ground truth)")
print("=" * 70)

clips = video_data if isinstance(video_data, list) else [video_data]
total_frames_all = 0
total_duration = 0
ss_coverages = []
afp_coverages = []
afp_skips = []
afp_delays = []
ss_delays = []
afp_cpu_savings = []
gt_obstacles = []

for i, clip in enumerate(clips):
    total_frames_all += clip["total_frames"]
    total_duration += clip.get("duration_s", 0)
    
    gt_obs = clip["always_on"]["unique_obstacle_frames"]
    gt_obstacles.append(gt_obs)
    
    ss_cov = clip["static_skip"]["coverage"]["coverage"]
    ss_coverages.append(ss_cov)
    
    afp_cov = clip["afp"]["coverage"]["coverage"]
    afp_coverages.append(afp_cov)
    
    afp_skip = clip["afp"]["skip_ratio"]
    afp_skips.append(afp_skip)
    
    afp_delay = clip["afp"]["coverage"].get("avg_delay_frames", 0)
    afp_delays.append(afp_delay)
    
    ss_delay = clip["static_skip"]["coverage"].get("avg_delay_frames", 0)
    ss_delays.append(ss_delay)
    
    afp_cs = clip["afp"].get("cpu_savings_pct", 0)
    afp_cpu_savings.append(afp_cs)
    
    print(f"Clip {i+1:2d}: GT={gt_obs:4d}  SS_Cov={ss_cov*100:5.1f}%  AFP_Cov={afp_cov*100:5.1f}%  AFP_Skip={afp_skip*100:5.1f}%  AFP_Delay={afp_delay:.1f}f  SS_Delay={ss_delay:.1f}f")

print(f"\n--- AVERAGES ---")
print(f"Total clips: {len(clips)}")
print(f"Total frames: {total_frames_all}")
print(f"Total duration: {total_duration:.1f}s")
print(f"Avg SS Coverage:  {statistics.mean(ss_coverages)*100:.1f}%")
print(f"Avg AFP Coverage: {statistics.mean(afp_coverages)*100:.1f}%")
print(f"Avg AFP Skip:     {statistics.mean(afp_skips)*100:.1f}%")
print(f"Avg AFP Delay:    {statistics.mean(afp_delays):.1f} frames")
print(f"Avg SS Delay:     {statistics.mean(ss_delays):.1f} frames")
print(f"Avg AFP CPU Savings: {statistics.mean(afp_cpu_savings):.1f}%")

# === 2. Ablation Study (Table 7) ===
print("\n" + "=" * 70)
print("ABLATION STUDY (Table 7 ground truth)")
print("=" * 70)

with open(DATA_DIR / "results_ablation.json", "r") as f:
    ablation_data = json.load(f)

strategies = ["static_3", "static_5", "static_10", "static_15", 
              "random_skip", "optical_flow_skip", "motion_skip",
              "afp_stability_only", "afp_proximity_only", "afp_full"]

for strat in strategies:
    coverages = []
    skip_ratios = []
    for clip in ablation_data:
        if strat in clip:
            s = clip[strat]
            coverages.append(s["coverage"])
            skip_ratios.append(s["skip_ratio"])
    
    avg_cov = statistics.mean(coverages) * 100
    avg_skip = statistics.mean(skip_ratios) * 100
    # CEI = coverage / (1 - skip_ratio)
    cei = (statistics.mean(coverages)) / (1 - statistics.mean(skip_ratios))
    
    # 95% CI for coverage
    n = len(coverages)
    if n > 1:
        std = statistics.stdev(coverages) * 100
        margin = 1.96 * std / (n ** 0.5)
        ci_lo = avg_cov - margin
        ci_hi = avg_cov + margin
    else:
        ci_lo = ci_hi = avg_cov
    
    print(f"{strat:25s}: Cov={avg_cov:5.1f}%  Skip={avg_skip:5.1f}%  CEI={cei:.2f}  CI=[{ci_lo:.1f}, {ci_hi:.1f}]")

# === 3. Phase 7 Benchmarks ===
print("\n" + "=" * 70)
print("PHASE 7 BENCHMARKS (ground truth)")
print("=" * 70)

with open(DATA_DIR / "results_phase7_benchmarks.json", "r") as f:
    bench = json.load(f)

print("\nAFP Comparison (100 synthetic frames):")
for key in ["always_on", "static_skip", "afp"]:
    d = bench["afp_comparison"][key]
    print(f"  {d['name']:30s}: {d['frames_processed']:3d}/{d['frames_total']} frames, skip={d['skip_ratio']*100:.1f}%, total_cpu={d['total_cpu_ms']:.0f}ms")

print("\nModel Breakdown:")
for model, d in bench["model_breakdown"].items():
    print(f"  {model:25s}: median={d['median_ms']:.1f}ms, P95={d['p95_ms']:.1f}ms")

print("\nFull Pipeline:")
fp = bench["full_pipeline"]
print(f"  Total frames: {fp['total_frames']}")
print(f"  All-frame median: {fp['all_median_ms']:.4f}ms")
print(f"  All-frame mean: {fp.get('all_mean_ms', 'N/A')}")
# Check for processed frame data
for key in fp:
    if 'processed' in key.lower() or 'proc' in key.lower():
        print(f"  {key}: {fp[key]}")

# Print all full_pipeline keys
print("\n  All full_pipeline keys:")
for k, v in fp.items():
    print(f"    {k}: {v}")

# === 4. Static Skip ratio from video eval ===
print("\n" + "=" * 70)
print("STATIC SKIP DETAILS FROM VIDEO EVAL")
print("=" * 70)
ss_skip_ratios = []
for clip in clips:
    ss_sr = clip["static_skip"]["skip_ratio"]
    ss_skip_ratios.append(ss_sr)
print(f"Static Skip ratios per clip: {[f'{r*100:.1f}%' for r in ss_skip_ratios]}")
print(f"Average Static Skip ratio: {statistics.mean(ss_skip_ratios)*100:.1f}%")
