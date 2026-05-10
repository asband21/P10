from pathlib import Path
import csv

rows = []

for path in Path(".").glob("out_put_noise_*_ln_*.txt"):
    with path.open() as f:
        header = f.readline().strip().split()

        # Header format:
        # noise <value> ln <value>
        noise = float(header[1])
        ln = float(header[3])

        for epoch, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            rows.append({
                "noise": noise,
                "ln": ln,
                "epoch": epoch,
                "value": float(line),
                "file": path.name,
            })

with open("all_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["noise", "ln", "epoch", "value", "file"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to all_results.csv")
