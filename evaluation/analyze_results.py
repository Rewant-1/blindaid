import json
from pathlib import Path
import statistics

DATA_FILE = Path(__file__).resolve().parent / "results_video_evaluation.json"

with open(DATA_FILE) as f:
    data = json.load(f)

print("=" * 80)
print("REAL-WORLD VIDEO EVALUATION SUMMARY")
print("=" * 80)

total_frames = sum(d["total_frames"] for d in data)
total_duration = sum(d["duration_s"] for d in data)
print(f"Total clips: {len(data)}")
print(f"Total frames evaluated: {total_frames}")
print(f"Total duration: {total_duration:.1f}s ({total_duration/60:.1f}min)")
print()

all_ao_cpu = 0
all_ss_cpu = 0
all_afp_cpu = 0
afp_coverages = []
ss_coverages = []
afp_skips = []
afp_delays = []
ss_delays = []

for d in data:
    ao = d["always_on"]
    ss = d["static_skip"]
    afp = d["afp"]

    gt_obs = ao["unique_obstacle_frames"]
    ss_cov = ss["coverage"]["coverage"]
    afp_cov = afp["coverage"]["coverage"]
    afp_skip = afp["skip_ratio"]

    all_ao_cpu += ao["total_cpu_ms"]
    all_ss_cpu += ss["total_cpu_ms"]
    all_afp_cpu += afp["total_cpu_ms"]
    afp_coverages.append(afp_cov)
    ss_coverages.append(ss_cov)
    afp_skips.append(afp_skip)
    afp_delays.append(afp["coverage"]["avg_delay_frames"])
    ss_delays.append(ss["coverage"]["avg_delay_frames"])

    name = d["video"]
    print(f"  {name}: GT={gt_obs}obs | SS_cov={ss_cov:.1%} AFP_cov={afp_cov:.1%} | AFP_skip={afp_skip:.1%} | AFP_delay={afp['coverage']['avg_delay_frames']:.1f}f")

print()
print("AGGREGATE RESULTS:")
print(f"  AFP CPU savings vs Always-On: {(1 - all_afp_cpu/all_ao_cpu)*100:.1f}%")
print(f"  Static-Skip CPU savings:      {(1 - all_ss_cpu/all_ao_cpu)*100:.1f}%")
print()
print(f"  Avg AFP detection coverage:   {statistics.mean(afp_coverages)*100:.1f}%")
print(f"  Avg Static-Skip coverage:     {statistics.mean(ss_coverages)*100:.1f}%")
print(f"  AFP coverage advantage:       +{(statistics.mean(afp_coverages) - statistics.mean(ss_coverages))*100:.1f}pp")
print()
print(f"  Avg AFP skip ratio:           {statistics.mean(afp_skips)*100:.1f}%")
print(f"  Avg AFP response delay:       {statistics.mean(afp_delays):.1f} frames")
print(f"  Avg SS response delay:        {statistics.mean(ss_delays):.1f} frames")
print()

wins = sum(1 for d in data if d["afp"]["coverage"]["coverage"] > d["static_skip"]["coverage"]["coverage"])
ties = sum(1 for d in data if d["afp"]["coverage"]["coverage"] == d["static_skip"]["coverage"]["coverage"])
losses = len(data) - wins - ties
print(f"  AFP beats Static-Skip: {wins}/{len(data)} clips")
print(f"  Ties: {ties}/{len(data)}")
print(f"  Static-Skip wins: {losses}/{len(data)}")
