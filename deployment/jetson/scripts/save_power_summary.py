import csv

rows = [
    ["15W", 28.4, 11.10, 11.35, 57.53, 59.75, 75.0, 3090],
    ["25W", 20.6, 13.55, 13.89, 60.45, 63.28, 66.6, 3068],
    ["MAXN_SUPER", 19.3, 15.44, 15.90, 62.76, 65.41, 68.8, 3063],
]

header = [
    "power_mode",
    "inference_ms",
    "avg_power_w",
    "peak_power_w",
    "avg_gpu_temp_c",
    "peak_gpu_temp_c",
    "avg_gpu_util_percent",
    "avg_ram_mb",
]

output_path = "benchmarks/summary/power_modes.csv"

with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"Saved to {output_path}")


