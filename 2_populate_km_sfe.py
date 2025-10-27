#!/usr/bin/env python
"""
populate_km_params.py — robust version

Populate the “Kocks–Mecking_hardening_parameters” sections of every curve
in the single *.json file found in the working directory.

Improvements:
• Safer JSON loading (utf-8-sig, comment/trailing-comma tolerant, clear errors).
• Input validation and friendlier diagnostics.
• SG window selection guaranteed valid; guards for small datasets.
• Data cleanup (NaN removal, de-dup, sort by strain, monotonicity checks).
• More robust axis classification (tries units if labels are unclear).
"""

import json
import os
import re
import glob
from typing import Tuple, Dict, List, Optional

import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import linregress


# ──────────────────────────── utilities ──────────────────────────── #

def log(msg: str) -> None:
    print(f"[INFO] {msg}")


def fail(msg: str) -> None:
    print(f"[ERROR] {msg}")
    raise RuntimeError(msg)


def find_single_json() -> str:
    """Return the single .json file in cwd (excluding *_populated.json), or raise."""
    candidates = [f for f in glob.glob("*.json") if not f.endswith("_populated.json")]
    if not candidates:
        raise FileNotFoundError("No JSON file found in the current directory.")
    if len(candidates) > 1:
        raise RuntimeError(f"Expected one JSON file, found: {candidates}")
    return candidates[0]


# --- JSON loading that tolerates BOM, comments, and trailing commas ---

_COMMENT_RE = re.compile(
    r"""
    (//[^\n]*$)           |   # line comments
    (/\*.*?\*/)           |   # block comments
    (?P<trailing>,\s*[\]}])   # trailing comma before } or ]
    """,
    re.MULTILINE | re.DOTALL | re.VERBOSE,
)


def _strip_json_comments_and_trailing_commas(text: str) -> str:
    def _repl(m: re.Match) -> str:
        if m.group("trailing"):
            # remove comma but keep the closing bracket/brace
            return m.group("trailing")[1:]  # drop the leading comma
        return ""
    return _COMMENT_RE.sub(_repl, text)


def load_json_lenient(fname: str) -> dict:
    """Load JSON handling BOM, optional comments and trailing commas, with clear errors."""
    try:
        with open(fname, "r", encoding="utf-8-sig") as fh:
            raw = fh.read()
    except FileNotFoundError:
        fail(f"File not found: {fname}")

    if not raw.strip():
        fail(f"JSON appears empty: {fname}")

    # First try strict JSON
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e1:
        # Try lenient pass
        try:
            cleaned = _strip_json_comments_and_trailing_commas(raw)
            return json.loads(cleaned)
        except json.JSONDecodeError as e2:
            msg = (
                f"Invalid JSON in '{fname}'.\n"
                f"Strict parse error at line {e1.lineno}, col {e1.colno}: {e1.msg}\n"
                f"Lenient parse error at line {e2.lineno}, col {e2.colno}: {e2.msg}\n"
                "Tip: Ensure the file contains valid JSON (no Python code, not a PDF/CSV), "
                "and remove comments or trailing commas if present."
            )
            fail(msg)


# ────────────────────── domain-specific helpers ───────────────────── #

def classify_axis(label: str) -> Tuple[str, str]:
    """
    Return ('stress'|'strain'|'unknown', 'true'|'eng'|'unknown') based on label.
    Also tries unit hints: stress ~ MPa/GPa/Pa; strain ~ % or unitless.
    """
    lbl = (label or "").strip().lower()
    if not lbl:
        return "unknown", "unknown"

    kind = "unknown"
    if "stress" in lbl:
        kind = "stress"
    elif "strain" in lbl:
        kind = "strain"
    else:
        # unit heuristics
        if any(u in lbl for u in ("mpa", "gpa", "pa")):
            kind = "stress"
        if any(u in lbl for u in ("%", "percent")):
            kind = "strain"

    nature = "unknown"
    if "true" in lbl:
        nature = "true"
    elif any(k in lbl for k in ("eng", "engineering")):
        nature = "eng"

    return kind, nature


def eng_to_true(stress_eng: np.ndarray, strain_eng: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert engineering stress/strain arrays to true values."""
    strain_true = np.log1p(strain_eng)
    stress_true = stress_eng * (1.0 + strain_eng)
    return stress_true, strain_true


def choose_savgol_window(n_points: int, poly: int = 3) -> int:
    """
    Pick an odd window length roughly 1/10 of the data size (≥ poly+2).
    Ensure 5 ≤ window < n_points and window is odd.
    """
    if n_points < (poly + 2 + 2):  # need at least poly+2 for filter, plus headroom
        # too few points to smooth safely, caller will handle
        return 0
    win = max(5, int(max(5, n_points / 10)))
    if win % 2 == 0:
        win += 1
    if win >= n_points:
        win = n_points - 1 if (n_points - 1) % 2 == 1 else n_points - 2
    if win < (poly + 2):
        win = poly + 3 if (poly + 3) % 2 == 1 else poly + 2  # make it odd and >= poly+2
    return max(5, win)


def _clean_and_order(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remove NaNs/infs, drop duplicate x, sort by x ascending."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size == 0:
        return x, y
    # sort by x
    order = np.argsort(x)
    x, y = x[order], y[order]
    # drop duplicate x (keep first)
    if x.size > 1:
        uniq = np.concatenate([[True], np.diff(x) != 0])
        x, y = x[uniq], y[uniq]
    return x, y


def _detect_axes(
    x_arr: np.ndarray, y_arr: np.ndarray,
    x_label: str, y_label: str
) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """
    Decide which array is strain vs stress using labels; fallback to heuristics by scale.
    Returns (strain_eng, stress_eng, strain_nature, stress_nature).
    """
    x_kind, x_nat = classify_axis(x_label)
    y_kind, y_nat = classify_axis(y_label)

    if x_kind == "strain" and y_kind == "stress":
        return x_arr, y_arr, x_nat, y_nat
    if x_kind == "stress" and y_kind == "strain":
        return y_arr, x_arr, y_nat, x_nat

    # Heuristic fallback: strain is typically ≤ ~1.0 (0–100%), stress in MPa/GPa range
    x_span, y_span = np.nanmax(x_arr) - np.nanmin(x_arr), np.nanmax(y_arr) - np.nanmin(y_arr)
    if x_span <= 2.0 and y_span > 5.0:  # very rough heuristic
        return x_arr, y_arr, x_nat, y_nat
    if y_span <= 2.0 and x_span > 5.0:
        return y_arr, x_arr, y_nat, x_nat

    fail(
        "Could not determine which axis is strain/stress from labels or value ranges.\n"
        f"x_label='{x_label}', y_label='{y_label}'. "
        "Please ensure axis labels contain 'strain'/'stress' or appropriate units."
    )


def kocks_mecking_parameters(
    strain_true: np.ndarray, stress_true: np.ndarray
) -> Tuple[float, float, List[float], Dict[str, int], float, str]:
    """
    Compute θ₀, σ_sat, strain range, SG filter params, R² and a comment.
    """
    n = strain_true.size
    poly = 3
    win = choose_savgol_window(n, poly=poly)

    if win >= 5:
        stress_sm = savgol_filter(stress_true, win, poly)
        strain_sm = savgol_filter(strain_true, win, poly)
        sg_params = {"window_points": int(win), "poly_order": int(poly)}
    else:
        # no smoothing — too few points
        stress_sm = stress_true
        strain_sm = strain_true
        sg_params = {"window_points": 0, "poly_order": 0}

    # Numerical derivative θ = dσ/dε
    if np.all(np.diff(strain_sm) > 0) and np.ptp(strain_sm) > 0:
        theta = np.gradient(stress_sm, strain_sm)
    else:
        # fallback uniform gradient
        theta = np.gradient(stress_sm)

    # Pick fitting window: prefer 0.02–0.08 true strain; else central 30%
    mask = (strain_sm >= 0.02) & (strain_sm <= 0.08)
    if mask.sum() < 5:
        lo, hi = int(0.35 * n), int(0.65 * n)
        mask = np.zeros(n, dtype=bool)
        mask[lo:hi] = True

    if mask.sum() < 3:
        fail("Not enough points in selected fitting window to perform regression.")

    sigma_fit = stress_sm[mask]
    theta_fit = theta[mask]

    slope, intercept, r_val, *_ = linregress(sigma_fit, theta_fit)
    theta0 = float(intercept)
    sigma_sat = float("nan") if slope == 0 else -theta0 / float(slope)
    r2 = float(r_val ** 2)

    fit_strains = strain_sm[mask]
    fit_range = [float(fit_strains[0]), float(fit_strains[-1])]
    comment = (
        "Parameters obtained by linear regression on θ vs σ "
        f"in the strain window {fit_range[0]:.3f}–{fit_range[1]:.3f}."
    )

    return theta0, sigma_sat, fit_range, sg_params, r2, comment


# ────────────────────────── main processing ───────────────────────── #

def process_curve(curve: dict, g: dict) -> None:
    raw = (curve.get("curve_raw_data") or {}).get("data") or []
    if len(raw) < 5:
        fail(f"Curve '{curve.get('curve_id','<unknown>')}' in graph '{g.get('graph_id','<unknown>')}' "
             f"has too few points ({len(raw)}). Need ≥ 5.")

    x_arr = np.array([pt["x"] for pt in raw], dtype=float)
    y_arr = np.array([pt["y"] for pt in raw], dtype=float)

    x_arr, y_arr = _clean_and_order(x_arr, y_arr)
    if x_arr.size < 5:
        fail(f"After cleaning, curve '{curve.get('curve_id','<unknown>')}' has too few valid points.")

    # Determine which is strain vs stress
    x_label = g.get("x_axis_label", "")
    y_label = g.get("y_axis_label", "")
    strain_eng, stress_eng, strain_nat, stress_nat = _detect_axes(x_arr, y_arr, x_label, y_label)

    # Convert to true if any axis is engineering (unknown keeps as-is)
    if strain_nat == "eng" or stress_nat == "eng":
        stress_true, strain_true = eng_to_true(stress_eng, strain_eng)
    else:
        stress_true, strain_true = stress_eng, strain_eng

    # Final sanity: require at least some spread in strain
    if np.ptp(strain_true) <= 1e-9:
        fail(f"Strain range is ~0 for curve '{curve.get('curve_id','<unknown>')}'. Check input data.")

    theta0, sigma_sat, fit_range, sg_params, r2, comment = \
        kocks_mecking_parameters(strain_true, stress_true)

    curve["Kocks–Mecking_hardening_parameters"] = {
        "theta0_MPa": round(float(theta0), 3),
        "sigma_sat_MPa": round(float(sigma_sat), 3) if np.isfinite(sigma_sat) else None,
        "fit_strain_range": [round(fit_range[0], 5), round(fit_range[1], 5)],
        "savgol_filter": sg_params,
        "goodness_of_fit_R2": round(float(r2), 5),
        "comment": comment,
    }


def populate_file(fname: str) -> str:
    log(f"Loading input → {fname}")
    data = load_json_lenient(fname)

    if not isinstance(data, dict):
        fail("Top-level JSON must be an object.")
    graphs = data.get("graphs")
    if not graphs or not isinstance(graphs, list):
        fail("JSON must contain a 'graphs' array with at least one graph.")

    for g in graphs:
        if not isinstance(g, dict):
            fail("Each graph entry must be an object.")
        curves = g.get("curves") or []
        if not isinstance(curves, list) or not curves:
            fail(f"Graph '{g.get('graph_id','<unknown>')}' has no 'curves' array.")
        for curve in curves:
            process_curve(curve, g)
        log(f"Processed graph '{g.get('graph_id','<unknown>')}' with {len(curves)} curve(s).")

    out_name = os.path.splitext(fname)[0] + "_populated.json"
    log(f"Writing populated file → {out_name}")
    with open(out_name, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    return out_name


if __name__ == "__main__":
    try:
        in_file = find_single_json()
        populated = populate_file(in_file)
        log("All done ✔")
    except Exception as e:
        # Print the first line only to keep console readable
        msg = str(e).splitlines()[0]
        print(f"[ERROR] {msg}")
