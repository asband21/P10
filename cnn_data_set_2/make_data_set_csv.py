#!/usr/bin/env python3

from pathlib import Path
import csv
import re
import argparse


AUDIO_PATTERN = re.compile(r"chunk_.*__id_(?P<id>\d+)_num_(?P<num>\d+)__record_(?P<record>\d+)_.*\.wav$")
CSV_PATTERN = re.compile(r"time_s_(?P<time>\d+)_id__id_(?P<id>\d+)_num_(?P<num>\d+)_\.csv$")

def build_master_csv(folder: Path, output_file: Path):
    csv_files = {}
    audio_files = []
    
    for file in folder.iterdir():
        if not file.is_file():
            continue

        csv_match = CSV_PATTERN.match(file.name)
        if csv_match:
            key = (csv_match.group("id"), csv_match.group("num"), )
            csv_files[key] = {"time_s": csv_match.group("time"), "id": csv_match.group("id"), "num": csv_match.group("num"), "csv_file": file.name, }
            continue

        audio_match = AUDIO_PATTERN.match(file.name)
        if audio_match:
            key = (audio_match.group("id"), audio_match.group("num"),)
            audio_files.append({"key": key, "id": audio_match.group("id"), "num": audio_match.group("num"), "record": audio_match.group("record"), "audio_file": file.name,})

    rows = []

    for audio in audio_files:
        key = audio["key"]

        if key not in csv_files:
            print(f"Warning: no CSV match for audio file: {audio['audio_file']}")
            continue
        csv_info = csv_files[key]
        rows.append({ "time_s": csv_info["time_s"], "id": audio["id"], "num": audio["num"], "record": audio["record"], "csv_file": csv_info["csv_file"], "audio_file": audio["audio_file"],})

    rows.sort(key=lambda r: ( int(r["time_s"]), int(r["id"]), int(r["num"]), int(r["record"]), ))
    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f,fieldnames=["time_s","id","num","record","csv_file","audio_file",], delimiter="\t", )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created: {output_file}")
    print(f"Matched rows: {len(rows)}")

def main():
    parser = argparse.ArgumentParser(description="Create a master CSV matching WAV files to CSV files by id and num.")
    parser.add_argument("folder", help="Folder containing the WAV and CSV files.",)
    parser.add_argument("-o", "--output", default="master_dataset_index.csv", help="Output master CSV filename.",)
    args = parser.parse_args()
    folder = Path(args.folder)
    output_file = folder / args.output

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    build_master_csv(folder, output_file)


if __name__ == "__main__":
    main()
