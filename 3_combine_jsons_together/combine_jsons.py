#!/usr/bin/env python3
"""
merge_populated_jsons.py
Combine multiple “*_populated.json” files into one JSON file.

Usage (from the directory with your populated JSONs):
    python merge_populated_jsons.py
Optional flags:
    -o, --output   Name of the output file (default: combined_papers.json)
"""

import json 
import glob
import argparse
from pathlib import Path
import sys



def merge_json_files(output_name: str = "combined_curves.json") -> None:
    print("[INFO] Scanning for '*_populated.json' files …")
    json_paths = sorted(Path(".").glob("*_populated.json"))

    if not json_paths:
        print("[ERROR] No matching JSON files found. Abort.", file=sys.stderr)
        sys.exit(1)

    merged_curves = []
    curve_id_counter = 1

    for path in json_paths:
        print(f"[INFO] Reading → {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                paper_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[ERROR] {path} is not valid JSON: {e}", file=sys.stderr)
            sys.exit(1)

        graphs = paper_data.get("graphs", [])
        for graph in graphs:
            for curve in graph.get("curves", []):
                cleaned_curve = {
                    "id": curve_id_counter,
                    "curve_id": curve.get("curve_id"),
                    "curve_label": curve.get("curve_label"),
                    "alloy_composition": curve.get("alloy_composition_in_percentages_corresponding_to_this_curve"),
                    "curve_raw_data": curve.get("curve_raw_data"),
                    "Kocks–Mecking_hardening_parameters": curve.get("Kocks–Mecking_hardening_parameters") or curve.get("Kocks-Mecking_hardening_parameters")
                }
                merged_curves.append(cleaned_curve)
                curve_id_counter += 1

    print(f"[INFO] Writing {len(merged_curves)} curves to '{output_name}' …")
    with open(output_name, "w", encoding="utf-8") as f_out:
        json.dump({"curves": merged_curves}, f_out, indent=2, ensure_ascii=False)

    print("[SUCCESS] Merge complete!")

def main():
    parser = argparse.ArgumentParser(description="Merge populated JSON files.")
    parser.add_argument(
        "-o", "--output",
        default="combined_papers.json",
        help="output file name (default: combined_papers.json)"
    )
    args = parser.parse_args()
    merge_json_files(args.output)

if __name__ == "__main__":
    main()


# def merge_json_files(output_name: str = "combined_papers.json") -> None:
#     # 1. Detect candidate JSON files
#     print("[INFO] Scanning for '*_populated.json' files …")
#     json_paths = sorted(Path(".").glob("*_populated.json"))

#     if not json_paths:
#         print("[ERROR] No matching JSON files found. Abort.", file=sys.stderr)
#         sys.exit(1)

#     # 2. Load each file
#     merged_papers = []
#     for path in json_paths:
#         print(f"[INFO] Reading → {path}")
#         try:
#             with open(path, "r", encoding="utf-8") as f:
#                 paper_data = json.load(f)
#                 merged_papers.append(paper_data)
#         except json.JSONDecodeError as e:
#             print(f"[ERROR] {path} is not valid JSON: {e}", file=sys.stderr)
#             sys.exit(1)

#     # 3. Combine under a single top-level key
#     combined = {"papers": merged_papers}

#     # 4. Write the combined file
#     print(f"[INFO] Writing combined data to '{output_name}' …")
#     with open(output_name, "w", encoding="utf-8") as f_out:
#         json.dump(combined, f_out, indent=2, ensure_ascii=False)

#     print("[SUCCESS] Merge complete!")



