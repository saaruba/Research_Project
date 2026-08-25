#!/usr/bin/env python3
"""
Fold the clicks from label_ospace.html into labels.csv for scoring.

    python3 scripts/merge_ospace_clicks.py ~/Downloads/ospace_clicks.csv

The HTML labeller writes a minimal three-column file (frame_file, label_x,
label_y). validate_ospace_estimate.py needs the full template columns -
estimated_x, estimated_y, avg_bbox_width and the rest - so this merges the
clicks back onto the template by frame_file and writes labels.csv beside it.

Matching is by frame_file, never by row order, so a partially completed file or
rows in a different order both merge correctly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABEL_DIR = PROJECT_ROOT / "dataset" / "processed" / "ospace_labelling"
TEMPLATE = LABEL_DIR / "labels_template.csv"
OUTPUT = LABEL_DIR / "labels.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clicks", type=Path,
                    help="ospace_clicks.csv downloaded from label_ospace.html")
    ap.add_argument("--out", type=Path, default=OUTPUT)
    args = ap.parse_args()

    if not args.clicks.exists():
        raise SystemExit(f"not found: {args.clicks}")

    template = pd.read_csv(TEMPLATE, dtype={"label_x": object, "label_y": object})
    clicks = pd.read_csv(args.clicks, dtype=str).fillna("")

    if "frame_file" not in clicks.columns:
        raise SystemExit("clicks file has no frame_file column")

    lookup = {r["frame_file"]: (r.get("label_x", ""), r.get("label_y", ""))
              for _, r in clicks.iterrows()}

    filled = 0
    unmatched = []
    for i, row in template.iterrows():
        name = row["frame_file"]
        if name not in lookup:
            unmatched.append(name)
            continue
        x, y = lookup[name]
        if str(x).strip():
            template.at[i, "label_x"] = x
            template.at[i, "label_y"] = y
            filled += 1

    template.to_csv(args.out, index=False)

    total = len(template)
    none_count = sum(1 for _, r in template.iterrows()
                     if str(r["label_x"]).strip().lower() == "none")
    print(f"Merged {filled}/{total} labels into {args.out}")
    print(f"  marked as a real group : {filled - none_count}")
    print(f"  marked 'no group'      : {none_count}")
    if unmatched:
        print(f"  frames with no click   : {len(unmatched)}")
        for n in unmatched[:5]:
            print(f"      {n}")
    if filled < total:
        print("\nSome frames are still unlabelled. Finish them in the browser,")
        print("download again, and re-run this - it merges by filename, not order.")
    else:
        print("\nAll frames labelled. Now run:")
        print("    python3 scripts/validate_ospace_estimate.py")


if __name__ == "__main__":
    main()
