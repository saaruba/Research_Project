#!/usr/bin/env python3
"""
Objective 3 validation, step 2: score the estimated O-space centre against
your hand-labelled ground truth.

Run this AFTER prepare_ospace_validation.py, and after you have filled in
label_x / label_y in labels_template.csv and saved it as labels.csv in the
same folder.

THE CRITERION (re-specified - see prepare_ospace_validation.py for the full
justification, and disclose this in your dissertation):
    Original proposal:  O-space centre within 0.3 m of the manual label,
                        for >= 70% of labelled frames.
    Used here:          O-space centre within 0.5 x mean person bounding-box
                        width of the manual label, for >= 70% of frames.
The change is forced by the dataset: the video is uncalibrated with no depth
channel, so metres are not recoverable from pixels. 0.5 x bbox width is a
scale-invariant stand-in of roughly the same magnitude (adult shoulder width
~0.45-0.50 m, so the tolerance is ~0.22-0.25 m equivalent).

Frames labelled `none` (no genuine conversational group visible) are
excluded from scoring and reported separately - that count is itself a
useful number, since it says how often the clustering proposes a "group"
where a human sees none.

Usage:
    python3 scripts/validate_ospace_estimate.py
    python3 scripts/validate_ospace_estimate.py --labels path/to/labels.csv
    python3 scripts/validate_ospace_estimate.py --tolerance-factor 0.75
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABELS = PROJECT_ROOT / "dataset" / "processed" / "ospace_labelling" / "labels.csv"

TOLERANCE_FACTOR = 0.5     # multiples of mean person bbox width
PASS_RATE_TARGET = 70.0    # percent of frames that must be within tolerance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--tolerance-factor", type=float, default=TOLERANCE_FACTOR,
                        help="tolerance as a multiple of mean person bbox width (default 0.5)")
    args = parser.parse_args()

    labels_path = args.labels.expanduser().resolve()
    if not labels_path.exists():
        print(f"{labels_path} not found.\n")
        print("Fill in label_x / label_y in labels_template.csv (in the same folder),")
        print("save it as labels.csv, then run this script again.")
        return

    df = pd.read_csv(labels_path)

    required = {"estimated_x", "estimated_y", "avg_bbox_width", "label_x", "label_y"}
    missing = required - set(df.columns)
    if missing:
        print(f"labels.csv is missing required column(s): {sorted(missing)}")
        return

    total_frames = len(df)

    # Rows the labeller marked as "no real group here".
    label_x_text = df["label_x"].astype(str).str.strip().str.lower()
    excluded = df[label_x_text.isin({"none", "n/a", "na", "-", ""}) | df["label_x"].isna()]
    scored = df.drop(excluded.index).copy()

    if scored.empty:
        print(f"All {total_frames} frames were marked 'none' - nothing to score.")
        return

    scored["label_x"] = pd.to_numeric(scored["label_x"], errors="coerce")
    scored["label_y"] = pd.to_numeric(scored["label_y"], errors="coerce")
    unparsed = scored[scored["label_x"].isna() | scored["label_y"].isna()]
    if not unparsed.empty:
        print(f"Warning: {len(unparsed)} row(s) had unreadable labels and were skipped:")
        for name in unparsed["frame_file"]:
            print(f"    {name}")
        scored = scored.dropna(subset=["label_x", "label_y"])

    error_px = np.hypot(scored["estimated_x"] - scored["label_x"],
                        scored["estimated_y"] - scored["label_y"])
    tolerance_px = args.tolerance_factor * scored["avg_bbox_width"]
    within = error_px <= tolerance_px

    # Error expressed in person-widths is the scale-free version, and is the
    # number worth quoting in the write-up alongside the raw pixel figure.
    error_in_person_widths = error_px / scored["avg_bbox_width"]

    pass_rate = float(within.mean() * 100)

    print("=" * 70)
    print("OBJECTIVE 3 - O-SPACE CENTRE VALIDATION")
    print("=" * 70)
    print(f"Frames prepared:          {total_frames}")
    print(f"Excluded ('none'):        {len(excluded)}  "
          f"({len(excluded) / total_frames * 100:.1f}% - clustering proposed a group where you saw none)")
    print(f"Frames scored:            {len(scored)}")
    print()
    print(f"Mean error:               {error_px.mean():.1f} px  "
          f"({error_in_person_widths.mean():.2f} person-widths)")
    print(f"Median error:             {np.median(error_px):.1f} px  "
          f"({np.median(error_in_person_widths):.2f} person-widths)")
    print(f"Tolerance:                {args.tolerance_factor} x person width  "
          f"(mean {tolerance_px.mean():.1f} px)")
    print()
    print(f"Within tolerance:         {int(within.sum())}/{len(scored)}  =  {pass_rate:.1f}%")
    print(f"Target:                   >= {PASS_RATE_TARGET:.0f}%")
    print()
    verdict = "PASS" if pass_rate >= PASS_RATE_TARGET else "FAIL"
    print(f"RESULT: {verdict}")
    if verdict == "FAIL":
        print("\nA fail here is still a reportable result - it quantifies how well a")
        print("centroid-based O-space estimate matches human judgement, which is exactly")
        print("what Objective 3 asked you to measure. Report it honestly.")

    worst = scored.assign(error_px=error_px).nlargest(min(5, len(scored)), "error_px")
    print("\nWorst frames (useful for the discussion section):")
    for _, row in worst.iterrows():
        print(f"    {row['frame_file']}: {row['error_px']:.1f} px off, "
              f"{int(row['num_people'])} people")

    summary = {
        "frames_prepared": int(total_frames),
        "frames_excluded_no_group": int(len(excluded)),
        "frames_scored": int(len(scored)),
        "tolerance_factor_person_widths": args.tolerance_factor,
        "mean_error_px": float(error_px.mean()),
        "median_error_px": float(np.median(error_px)),
        "mean_error_person_widths": float(error_in_person_widths.mean()),
        "pass_rate_percent": pass_rate,
        "target_percent": PASS_RATE_TARGET,
        "verdict": verdict,
    }
    output_path = labels_path.parent / "ospace_validation_result.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nWritten: {output_path}")


if __name__ == "__main__":
    main()
