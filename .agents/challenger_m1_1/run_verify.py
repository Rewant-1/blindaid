import json
import statistics

# === 1. Real-World Video Evaluation (Table 6) ===
with open("c:/blindaid/evaluation/results_video_evaluation.json", "r") as f:
    video_data = json.load(f)

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

print("=" * 70)
print("REAL-WORLD VIDEO EVALUATION (Table 6 ground truth)")
print("=" * 70)
print(f"Total clips: {len(clips)}")
print(f"Total frames: {total_frames_all}")
print(f"Total duration: {total_duration:.1f}s")
print(f"Avg SS Coverage:  {statistics.mean(ss_coverages)*100:.6f}%")
print(f"Avg AFP Coverage: {statistics.mean(afp_coverages)*100:.6f}%")
print(f"Avg AFP Skip:     {statistics.mean(afp_skips)*100:.6f}%")
print(f"Avg AFP Delay:    {statistics.mean(afp_delays):.6f} frames")
print(f"Avg SS Delay:     {statistics.mean(ss_delays):.6f} frames")
print(f"Avg AFP CPU Savings: {statistics.mean(afp_cpu_savings):.6f}%")

# === 2. Ablation Study (Table 7) ===
print("\n" + "=" * 70)
print("ABLATION STUDY (Table 7 ground truth)")
print("=" * 70)

with open("c:/blindaid/evaluation/results_ablation.json", "r") as f:
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
    cei = (statistics.mean(coverages)) / (1 - statistics.mean(skip_ratios))
    
    n = len(coverages)
    if n > 1:
        std = statistics.stdev(coverages) * 100
        margin = 1.96 * std / (n ** 0.5)
        ci_lo = avg_cov - margin
        ci_hi = avg_cov + margin
    else:
        ci_lo = ci_hi = avg_cov
    
    print(f"{strat:25s}: Cov={avg_cov:.6f}%  Skip={avg_skip:.6f}%  CEI={cei:.6f}  CI=[{ci_lo:.6f}, {ci_hi:.6f}]")
