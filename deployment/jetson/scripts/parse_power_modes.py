import re
from pathlib import Path
from statistics import mean

MODES = {
    "15W": Path("benchmarks/power_modes/15w/tegrastats.log"),
    "25W": Path("benchmarks/power_modes/25w/tegrastats.log"),
    "MAXN_SUPER": Path("benchmarks/power_modes/maxn_super/tegrastats.log"),
}

ACTIVE_GPU_UTIL_THRESHOLD_PERCENT = 20

def parse_line(line):
    gpu_util = re.search(r"GR3D_FREQ\s+(\d+)%", line)
    gpu_temp = re.search(r"gpu@([\d.]+)C", line)
    ram = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
    power = re.search(r"VDD_IN\s+(\d+)mW", line)

    if not all([gpu_util, gpu_temp, ram, power]):
        return None

    return {
        "gpu_util": int(gpu_util.group(1)),
        "gpu_temp_c": float(gpu_temp.group(1)),
        "ram_mb": int(ram.group(1)),
        "power_mw": int(power.group(1)),
    }

for mode, path in MODES.items():
    samples = []

    for line in path.read_text().splitlines():
        sample = parse_line(line)

        if sample and sample["gpu_util"] > ACTIVE_GPU_UTIL_THRESHOLD_PERCENT:
            samples.append(sample)

    print(f"\n{mode}")
    print(f"Active samples: {len(samples)}")
    print(f"Avg active-sample power: {mean(x['power_mw'] for x in samples)/1000:.2f} W")
    print(f"Peak active-sample power: {max(x['power_mw'] for x in samples)/1000:.2f} W")
    print(f"Avg active-sample GPU temp: {mean(x['gpu_temp_c'] for x in samples):.2f} C")
    print(f"Peak active-sample GPU temp: {max(x['gpu_temp_c'] for x in samples):.2f} C")
    print(f"Avg active-sample GPU util: {mean(x['gpu_util'] for x in samples):.1f}%")
    print(f"Avg active-sample RAM: {mean(x['ram_mb'] for x in samples):.0f} MB")

