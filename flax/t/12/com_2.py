#!/usr/bin/env python3

from pathlib import Path
import csv
import re

pattern = re.compile(r"out_put_noise_(.*?)_ln_(.*?)\.txt")

data = []

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

    data.append([f"noise_{noise}_ln_{ln}", values])

out_name = f"combined.csv"
with open(out_name, "w", newline="") as f:
    writer = csv.writer(f)
    for i in data:
        print(i[0], end="\t", file=f)
    print("", file=f)
    for i in range(len(data[0][1])):
        for j in range(len(data)):
            print(data[j][1][i], end="\t", file=f)
        print("", file=f)

