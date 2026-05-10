#!/usr/bin/env python3

from pathlib import Path
import csv
import re

pattern = re.compile(r"out_put_noise_(.*?)_ln_(.*?)\.txt")

data = {}

for path in sorted(Path(".").glob("out_put_noise_*_ln_*.txt")):
    match = pattern.match(path.name)
    if not match:
        continue

    with path.open() as f:
        header = f.readline().strip().split()

        # Header format:
        # noise 0.00999 ln 0.05
        noise = float(header[1])
        ln = float(header[3])

        values = [float(line.strip()) for line in f if line.strip()]

    data.setdefault(ln, {})[noise] = values

# Make one CSV for each learning rate, called ln here
for ln, noise_dict in sorted(data.items()):
    noises = sorted(noise_dict.keys())

    max_len = max(len(values) for values in noise_dict.values())

    out_name = f"combined_ln_{ln:g}.csv"

    with open(out_name, "w", newline="") as f:
        writer = csv.writer(f)

        header = ["instance"] + [f"noise_{noise:g}" for noise in noises]
        writer.writerow(header)

        for i in range(max_len):
            row = [i*2]

            for noise in noises:
                values = noise_dict[noise]
                row.append(values[i] if i < len(values) else "")

            writer.writerow(row)

    print(f"Wrote {out_name}")
