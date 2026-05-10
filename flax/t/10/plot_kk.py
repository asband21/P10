#!/usr/bin/env python3

from pathlib import Path
import re
import math
import plotly.graph_objects as go

pattern = re.compile(r"out_put_noise_(.*?)_ln_(.*?)\.txt")

data = {}

# Read all files
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


# Show one dynamic plot at a time
for noise, ln_dict in sorted(data.items()):
    fig = go.Figure()

    for ln, values in sorted(ln_dict.items()):
        epochs = [(i + 1) * 4 for i in range(len(values))]

        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=values,
                mode="lines+markers",
                name=f"ln={ln:g}",
            )
        )

    fig.update_layout(
        title=f"Results for noise={noise:g}",
        xaxis_title="Epoch",
        yaxis_title="Value",
        template="plotly_white",
    )

    fig.show()

    input("Press Enter to show next noise level...")
