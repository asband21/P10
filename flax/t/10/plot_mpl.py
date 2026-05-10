#!/usr/bin/env python3

from pathlib import Path
import re
import math
import matplotlib.pyplot as plt

pattern = re.compile(r"out_put_noise_(.*?)_ln_(.*?)\.txt")

data = {}

for path in sorted(Path(".").glob("out_put_noise_*_ln_*.txt")):
    match = pattern.match(path.name)
    if not match:
        continue

    with path.open() as f:
        header = f.readline().strip().split()

        noise = float(header[1])
        ln = float(header[3])

        values = []

        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                value = float(line)
            except ValueError:
                value = 0.0

            if math.isnan(value):
                value = 0.0

            values.append(value)

    data.setdefault(noise, {})[ln] = values


for noise, ln_dict in sorted(data.items()):
    plt.figure()

    for ln, values in sorted(ln_dict.items()):
        epochs = [(i + 1) * 4 for i in range(len(values))]
        plt.plot(epochs, values, marker="o", label=f"ln={ln:g}")

    plt.title(f"Results for noise={noise:g}")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_name = f"plot_noise_{noise:g}.png"
    plt.savefig(out_name, dpi=200)
    plt.close()

    print(f"Wrote {out_name}")
