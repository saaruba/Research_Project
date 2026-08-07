#!/usr/bin/env python3
"""
Validate the YOLOv8n detector against real ground truth, to get the actual
number for your proposal's Objective 2 target: "person-detection recall of
at least 80% on sessions 1 and 3, validated against the ground-truth face
annotations."

Prerequisite: run extract_person_detections.py on sessions 1 AND 3 as well
(they were skipped in the main batch loop since they already have real
facial_landmarks_uniface.csv for training purposes - but we still need the
detector's own output on those same sessions purely to check it against
that ground truth):

    python3 scripts/extract_person_detections.py --session dataset/1
    python3 scripts/extract_person_detections.py --session dataset/3

Then run this:
    python3 scripts/validate_detector_recall.py

What "recall" means here: of all the moments where the ground truth says at
least one face was actually present, what fraction did the detector also
find at least one person in? This directly answers "is our substitute for
LocateAnything-3B good enough to trust on the other 22 sessions."
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATCH_TOLERANCE_SEC = 1.0
VALIDATION_SESSIONS = ["1", "3"]


def normalise_timestamp(value: float) -> float:
    return value / 1e9 if abs(value) > 1e12 else value


def load_ground_truth(session_path: Path) -> pd.DataFrame:
    csv_path = session_path / "facial_landmarks_uniface.csv"
    df = pd.read_csv(csv_path)

    x_col = "x" if "x" in df.columns else "position_x"
    y_col = "y" if "y" in df.columns else "position_y"

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").apply(normalise_timestamp)
    df = df.dropna(subset=["timestamp", x_col, y_col])

    if "face_index" in df.columns:
        summary = df.groupby("timestamp")["face_index"].nunique().reset_index(name="num_faces_gt")
    else:
        summary = df.groupby("timestamp").size().reset_index(name="num_faces_gt")

    return summary.sort_values("timestamp").reset_index(drop=True)


def load_detections(session_path: Path) -> pd.DataFrame:
    csv_path = session_path / "detected_people.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run: python3 scripts/extract_person_detections.py --session {session_path}"
        )
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"num_faces": "num_faces_detected"})
    return df[["timestamp", "num_faces_detected"]].sort_values("timestamp").reset_index(drop=True)


def evaluate_session(session_name: str) -> dict:
    session_path = PROJECT_ROOT / "dataset" / session_name

    gt = load_ground_truth(session_path)
    det = load_detections(session_path)

    merged = pd.merge_asof(gt, det, on="timestamp", direction="nearest", tolerance=MATCH_TOLERANCE_SEC)
    merged["num_faces_detected"] = merged["num_faces_detected"].fillna(0)

    positives = merged[merged["num_faces_gt"] > 0]
    true_positives = (positives["num_faces_detected"] > 0).sum()
    recall = true_positives / len(positives) if len(positives) > 0 else float("nan")

    print(f"\nSession {session_name}:")
    print(f"  ground-truth moments with >=1 face: {len(positives)}")
    print(f"  of those, detector also found >=1 person: {true_positives}")
    print(f"  recall: {recall * 100:.1f}%")

    return {"session": session_name, "positives": len(positives), "true_positives": int(true_positives), "recall": recall}


def main() -> None:
    print("=" * 70)
    print("DETECTOR RECALL VALIDATION (Objective 2 target: >= 80%)")
    print("=" * 70)

    results = [evaluate_session(s) for s in VALIDATION_SESSIONS]

    total_positives = sum(r["positives"] for r in results)
    total_true_positives = sum(r["true_positives"] for r in results)
    overall_recall = total_true_positives / total_positives if total_positives > 0 else float("nan")

    print("\n" + "=" * 70)
    print(f"OVERALL RECALL across sessions {VALIDATION_SESSIONS}: {overall_recall * 100:.1f}%")
    if overall_recall >= 0.80:
        print("-> MEETS the >= 80% target from Objective 2. Good to proceed with confidence.")
    else:
        print("-> BELOW the 80% target. Worth documenting as a measured limitation either way -")
        print("   this is still useful, honest evidence for your methodology/results chapter.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
