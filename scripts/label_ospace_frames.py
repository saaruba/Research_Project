#!/usr/bin/env python3
"""
Objective 3, step 1c: hand-label the O-space centre on 30 validation frames.

    python3 scripts/extract_clean_ospace_frames.py    # once, if not done
    python3 scripts/label_ospace_frames.py            # then this
    python3 scripts/validate_ospace_estimate.py       # then score it

============================================================================
WHAT YOU ARE BEING ASKED TO MARK
============================================================================
Click the CENTRE OF THE GROUP AS IT APPEARS IN THE PICTURE - the middle of the
huddle, at roughly the same height as the people's bodies.

  * Two people facing each other     -> the gap between them, at body height
  * Three or more in a rough circle   -> the middle of that ring, at body height
  * People walking past, not talking  -> they are NOT part of the group
  * No genuine conversation in frame  -> press N

IMPORTANT - DO NOT CLICK THE FLOOR.

An earlier run of this labelling asked for the empty floor between the people.
Every one of the 30 clicks then landed on average 161 px BELOW the algorithm's
estimate, with a standard deviation of only 48 px - 30 out of 30 in the same
direction. That is not disagreement, it is two different definitions of
"centre": the pipeline computes the centroid of the person BOUNDING BOXES,
which sits at torso height, so a floor click can never match it however good
the clustering is. Horizontal agreement was already good (mean bias -7.6 px),
which is what showed the labelling itself was sound.

So: mark the middle of the group at the height of the people, not on the ground
in front of them. Judge it by eye as a person would - the detector's own
estimate is deliberately NOT drawn on these frames, so that this measures
accuracy rather than agreement with the algorithm.

============================================================================
CONTROLS
============================================================================
    left click   mark the O-space centre and advance
    N            no genuine conversational group in this frame
    B            go back one frame
    S            skip (leave unlabelled for now)
    Q            save and quit

Progress is saved after EVERY action, so it is safe to quit and resume at any
point. Output: dataset/processed/ospace_labelling/labels.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABEL_DIR = PROJECT_ROOT / "dataset" / "processed" / "ospace_labelling"
TEMPLATE = LABEL_DIR / "labels_template.csv"
OUTPUT = LABEL_DIR / "labels.csv"
CLEAN_DIR = LABEL_DIR / "clean"

# Seconds a click must be separated from the previous one to count.
MIN_CLICK_INTERVAL = 0.6


class Labeller:
    def __init__(self, df: pd.DataFrame, frames_dir: Path, out: Path):
        self.df, self.frames_dir, self.out = df, frames_dir, out
        self.i = self._first_unlabelled()
        self._last_click = 0.0
        self.fig, self.ax = plt.subplots(figsize=(11, 8.5))
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    # ---------------------------------------------------------------- state
    def _labelled(self, i: int) -> bool:
        v = str(self.df.at[i, "label_x"]).strip().lower()
        return v not in ("", "nan", "none-pending")

    def _first_unlabelled(self) -> int:
        for i in range(len(self.df)):
            if not self._labelled(i):
                return i
        return 0

    def _done(self) -> int:
        return sum(self._labelled(i) for i in range(len(self.df)))

    def save(self) -> None:
        self.df.to_csv(self.out, index=False)

    # ----------------------------------------------------------------- draw
    def show(self) -> None:
        self.ax.clear()
        row = self.df.iloc[self.i]
        path = self.frames_dir / row["frame_file"]
        if path.exists():
            self.ax.imshow(plt.imread(path))
        else:
            self.ax.text(0.5, 0.5, f"missing:\n{path.name}", ha="center",
                         va="center", transform=self.ax.transAxes)

        current = str(row["label_x"]).strip()
        if current and current.lower() not in ("nan", ""):
            if current.lower() == "none":
                mark = "marked: NO GROUP"
            else:
                mark = f"marked: ({float(row['label_x']):.0f}, {float(row['label_y']):.0f})"
                self.ax.plot(float(row["label_x"]), float(row["label_y"]),
                             marker="+", markersize=20, markeredgewidth=3, color="lime")
        else:
            mark = "not yet marked"

        self.ax.set_title(
            f"[{self.i + 1}/{len(self.df)}]  {row['frame_file']}   "
            f"session {row['session_id']},  {row['num_people']} detected   |   {mark}\n"
            f"CLICK the middle of the group AT BODY HEIGHT (not the floor)    "
            f"N = no group    B = back    S = skip    Q = save & quit",
            fontsize=10)
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.fig.canvas.draw_idle()

    # --------------------------------------------------------------- events
    def advance(self) -> None:
        if self.i + 1 < len(self.df):
            self.i += 1
            self.show()
        else:
            print(f"\nLast frame. {self._done()}/{len(self.df)} labelled.")
            self.show()

    def on_click(self, event) -> None:
        if event.inaxes is not self.ax or event.xdata is None:
            return

        # DEBOUNCE  (added Aug 2026)
        #
        # The first labelling run recorded 29 of 30 frames at the identical
        # pixel (234.3, 253.5): each click advanced the frame, and a rapid
        # series of clicks in one spot burned through the whole set in a few
        # seconds before any image had been looked at. The resulting labels
        # were worthless - and worse, they LOOKED like a completed dataset.
        #
        # A click arriving within MIN_CLICK_INTERVAL of the previous accepted
        # one cannot be a considered judgement about a new image, so it is
        # ignored and the reason is printed rather than silently swallowed.
        now = time.monotonic()
        if now - self._last_click < MIN_CLICK_INTERVAL:
            print(f"  (ignored a click {now - self._last_click:.2f}s after the "
                  f"last one - look at the image first)")
            return
        self._last_click = now

        self.df.at[self.i, "label_x"] = round(float(event.xdata), 1)
        self.df.at[self.i, "label_y"] = round(float(event.ydata), 1)
        self.save()
        print(f"  [{self.i + 1}/{len(self.df)}] {self.df.at[self.i, 'frame_file']} "
              f"-> ({event.xdata:.0f}, {event.ydata:.0f})")
        self.advance()

    def on_key(self, event) -> None:
        k = (event.key or "").lower()
        if k == "n":
            self.df.at[self.i, "label_x"] = "none"
            self.df.at[self.i, "label_y"] = "none"
            self.save()
            print(f"  [{self.i + 1}/{len(self.df)}] "
                  f"{self.df.at[self.i, 'frame_file']} -> NO GROUP")
            self.advance()
        elif k == "b":
            self.i = max(0, self.i - 1)
            self.show()
        elif k == "s":
            self.advance()
        elif k == "q":
            self.save()
            print(f"\nSaved {self.out}  ({self._done()}/{len(self.df)} labelled)")
            plt.close(self.fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, default=CLEAN_DIR)
    ap.add_argument("--out", type=Path, default=OUTPUT)
    ap.add_argument("--restart", action="store_true",
                    help="discard any existing labels.csv and start from scratch")
    args = ap.parse_args()

    if not args.frames.exists() or not list(args.frames.glob("*.jpg")):
        raise SystemExit(
            f"No clean frames in {args.frames}.\n"
            "Run: python3 scripts/extract_clean_ospace_frames.py")

    # Resume from labels.csv if it exists, otherwise start from the template.
    source = TEMPLATE if args.restart else (args.out if args.out.exists() else TEMPLATE)
    if args.restart:
        print('--restart: ignoring any existing labels.csv\n')
    df = pd.read_csv(source, dtype={"label_x": object, "label_y": object})
    for col in ("label_x", "label_y"):
        if col not in df.columns:
            df[col] = ""
    df[["label_x", "label_y"]] = df[["label_x", "label_y"]].fillna("")

    print(__doc__.split("CONTROLS")[0].split("WHAT YOU ARE BEING ASKED")[1])
    print(f"Loaded {len(df)} frames from {source.name}")

    app = Labeller(df, args.frames, args.out)
    print(f"Starting at frame {app.i + 1} ({app._done()}/{len(df)} already labelled)\n")
    app.show()
    plt.show()

    app.save()
    print(f"\nSaved: {args.out}   {app._done()}/{len(df)} labelled")
    if app._done() == len(df):
        print("All frames labelled. Now run:")
        print("    python3 scripts/validate_ospace_estimate.py")
    else:
        print("Run this script again to carry on where you left off.")


if __name__ == "__main__":
    main()
