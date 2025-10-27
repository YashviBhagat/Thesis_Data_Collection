import os
import json
import argparse
import glob

def read_csv(filepath):
    print(f"[1/4] Reading CSV file: {filepath}")
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("---"):
                continue
            if "," in line:
                parts = [p.strip() for p in line.split(",", 1)]
            else:
                parts = line.split()
            if len(parts) != 2:
                print(f"  Warning: skipping malformed line {line_num}: {line}")
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                print(f"  Warning: non-numeric data on line {line_num}, skipping: {line}")
                continue
            data.append({"x": x, "y": y})
    print(f"[2/4] Parsed {len(data)} valid (x,y) points.")
    return data

def build_json(data, x_label, y_label):
    print("[3/4] Building JSON structure.")
    return {
        "x_axis": x_label,
        "y_axis": y_label,
        "data": data
    }

def write_json(out_path, obj):
    print(f"[4/4] Writing JSON to file: {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print("Done.")

def pick_first_csv():
    files = sorted(glob.glob("*.csv"))
    return files[0] if files else None

def main():
    parser = argparse.ArgumentParser(description="Convert two-column CSV to JSON with x/y labels.")
    parser.add_argument("csv", nargs="?", help="Input CSV filename (two columns: x,y).")
    parser.add_argument("--xlabel", default="x", help="Name for x axis (default: x).")
    parser.add_argument("--ylabel", default="y", help="Name for y axis (default: y).")
    parser.add_argument("--out", help="Output JSON filename. If omitted, uses same basename as CSV with .json.")
    args = parser.parse_args()

    csv_file = args.csv
    if not csv_file:
        csv_file = pick_first_csv()
        if csv_file:
            print(f"No CSV provided, auto-detecting first CSV in directory: {csv_file}")
        else:
            print("Error: No CSV file provided and none found in current directory.")
            return

    if not os.path.isfile(csv_file):
        print(f"Error: CSV file '{csv_file}' not found.")
        return

    data = read_csv(csv_file)
    if not data:
        print("No valid data to write. Exiting.")
        return

    json_obj = build_json(data, args.xlabel, args.ylabel)
    out_fname = args.out if args.out else os.path.splitext(csv_file)[0] + ".json"
    write_json(out_fname, json_obj)

if __name__ == "__main__":
    main()
