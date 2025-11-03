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


# def merge_json_files(output_name: str = "combined_curves.json") -> None:
#     """
#     This function merges multiple JSON files (ending with '_populated.json')
#     from the current directory into a single combined JSON file.
#     Each JSON file is expected to have a "graphs" list with curve information.
#     """

#     print("[INFO] Scanning for '*_populated.json' files …")

#     # Find all JSON files in the current directory that end with '_populated.json'
#     json_paths = sorted(Path(".").glob("*_populated.json"))

#     # Exit if no matching files are found
#     if not json_paths:
#         print("[ERROR] No matching JSON files found. Abort.", file=sys.stderr)
#         sys.exit(1)

#     merged_curves = []  # Stores all cleaned curves from all JSON files
#     curve_id_counter = 1  # Assigns unique incremental IDs to each curve

#     # Loop through each JSON file found
#     for path in json_paths:
#         print(f"[INFO] Reading → {path}")
#         try:
#             # Open and load JSON content
#             with open(path, "r", encoding="utf-8") as f:
#                 paper_data = json.load(f)
#         except json.JSONDecodeError as e:
#             # Stop execution if a file is not valid JSON
#             print(f"[ERROR] {path} is not valid JSON: {e}", file=sys.stderr)
#             sys.exit(1)

#         # Get the list of "graphs" from the file; default to empty list if missing
#         graphs = paper_data.get("graphs", [])

#         # --- Outer loop: iterate through each graph in this paper ---
#         for graph in graphs:
#             # --- Inner loop: iterate through each curve in this graph ---
#             for curve in graph.get("curves", []):
#                 # Extract and clean the relevant fields for the final output
#                 cleaned_curve = {
#                     "id": curve_id_counter,  # New global curve ID
#                     "curve_id": curve.get("curve_id"),  # Curve ID from the source JSON
#                     "curve_label": curve.get("curve_label"),  # Label describing the curve
#                     "alloy_composition": curve.get(
#                         "alloy_composition_in_percentages_corresponding_to_this_curve"
#                     ),  # Alloy composition (element %)
#                     "curve_raw_data": curve.get("curve_raw_data"),  # List of (x, y) points
#                     "Kocks–Mecking_hardening_parameters": curve.get(
#                         "Kocks–Mecking_hardening_parameters"
#                     ) or curve.get(
#                         "Kocks-Mecking_hardening_parameters"
#                     ),  # Model parameters (handle different spellings)
#                 }

#                 # Append this curve's data to the merged list
#                 merged_curves.append(cleaned_curve)

#                 # Increment global ID counter for next curve
#                 curve_id_counter += 1

#     # Write all merged and cleaned curves to a single output file
#     print(f"[INFO] Writing {len(merged_curves)} curves to '{output_name}' …")
#     with open(output_name, "w", encoding="utf-8") as f_out:
#         # Dump the list of curves as a JSON object with indentation
#         json.dump({"curves": merged_curves}, f_out, indent=2, ensure_ascii=False)

#     print("[SUCCESS] Merge complete!")



def merge_json_files(output_name: str = "combined_papers.json") -> None:
    # 1. Detect candidate JSON files
    print("[INFO] Scanning for '*_populated.json' files …")
    json_paths = sorted(Path(".").glob("*_populated.json"))

    if not json_paths:
        print("[ERROR] No matching JSON files found. Abort.", file=sys.stderr)
        sys.exit(1)

    # 2. Load each file
    merged_papers = []
    for path in json_paths:
        print(f"[INFO] Reading → {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                paper_data = json.load(f)
                merged_papers.append(paper_data)
        except json.JSONDecodeError as e:
            print(f"[ERROR] {path} is not valid JSON: {e}", file=sys.stderr)
            sys.exit(1)

    # 3. Combine under a single top-level key
    combined = {"papers": merged_papers}

    # 4. Write the combined file
    print(f"[INFO] Writing combined data to '{output_name}' …")
    with open(output_name, "w", encoding="utf-8") as f_out:
        json.dump(combined, f_out, indent=2, ensure_ascii=False)

    print("[SUCCESS] Merge complete!")

def main():
    """
    Command-line entry point:
    Allows the user to specify an output file name using -o or --output.
    Example usage:
        python merge_jsons.py -o combined_curves.json
    """
    parser = argparse.ArgumentParser(description="Merge populated JSON files.")
    parser.add_argument(
        "-o", "--output",
        default="combined_papers.json",
        help="output file name (default: combined_papers.json)"
    )
    args = parser.parse_args()

    # Call the main merge function with the provided output name
    merge_json_files(args.output)


# Entry point: this block only runs when the script is executed directly,
# not when imported as a module.
if __name__ == "__main__":
    main()




