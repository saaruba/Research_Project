"""
FIGURES FOR CHAPTER 5  -  approach accuracy by policy and detector

    pip install --user matplotlib          # once; it does NOT need scipy
    python3 scripts/plot_results.py --run dataset/processed/results_FINAL_20260824

Reads the approach_accuracy.csv written by summarise_sim_results.py in each
detector's folder and writes five figures to <run>/figures/, as PNG at 200 dpi
for pasting into Word and as SVG for anything that needs to scale.

    fig1_position_accuracy.png     mean position error, policy x detector
    fig2_orientation_accuracy.png  mean orientation error, policy x detector
    fig3_error_distribution.png    box plots - the spread behind the means
    fig4_tradeoff.png              position vs orientation, the key finding
    fig5_per_group.png             position error by group size

Figure 4 is the one to lead with. It shows the inversion directly: the policies
that place the robot most accurately are the ones that orient it least
accurately, which is the result the numbers alone make you work to see.

No seaborn, no pandas - matplotlib and the standard library only, to keep the
dependency surface as small as the rest of this project.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                    # headless: no display in the container
import matplotlib.pyplot as plt          # noqa: E402

# ---------------------------------------------------------------------------
# COLOUR PALETTE  -  edit this and every figure follows
#
# PALETTE is the ordered list of colours; COLOURS assigns the first three to
# the three policies. To recolour a policy, change the index it points at. To
# add a fourth policy later, give it PALETTE[3].
# ---------------------------------------------------------------------------
PALETTE = [
    "#000000",   # 0  black
    "#FF8C00",   # 1  orange
    "#5B4FCF",   # 2  purple-blue
    "#D62728",   # 3  red
    "#FFC000",   # 4  yellow
]
COLOURS = {"rule": PALETTE[0], "bc_ft": PALETTE[1], "mlp_ft": PALETTE[2]}
LABELS = {"rule": "Rule baseline", "bc_ft": "BC – Random Forest",
          "mlp_ft": "BC – MLP"}
DETECTORS = [("yolo", "YOLOv8n (2 Hz)"), ("locateanything", "LocateAnything-3B (0.5 Hz)")]
ORDER = ["rule", "bc_ft", "mlp_ft"]


def load(run: Path) -> dict:
    """{detector: {policy: [rows]}} from each approach_accuracy.csv."""
    data = {}
    for key, _ in DETECTORS:
        path = run / key / "approach_accuracy.csv"
        if not path.exists():
            print(f"  missing {path} - run summarise_sim_results.py first")
            continue
        by_policy: dict[str, list[dict]] = {}
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                for k in ("nearest_slot_error_m", "heading_error_deg", "slot_error_m"):
                    if k in row:
                        row[k] = float(row[k])
                row["within_distance"] = str(row.get("within_distance")).lower() == "true"
                row["group_id"] = int(row["group_id"])
                row["num_people"] = int(row["num_people"])
                by_policy.setdefault(row["policy"], []).append(row)
        data[key] = by_policy
    return data


def _bar_chart(data, field, only_in_band, title, ylabel, fname, out, fmt="{:.3f}"):
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    width = 0.35
    xs = range(len(ORDER))

    for di, (key, dlabel) in enumerate(DETECTORS):
        if key not in data:
            continue
        means, errs = [], []
        for policy in ORDER:
            rows = data[key].get(policy, [])
            vals = [r[field] for r in rows
                    if (not only_in_band) or r["within_distance"]]
            if vals:
                means.append(statistics.fmean(vals))
                # standard error of the mean
                errs.append(statistics.stdev(vals) / (len(vals) ** 0.5)
                            if len(vals) > 1 else 0.0)
            else:
                means.append(0.0)
                errs.append(0.0)

        offset = (di - 0.5) * width
        bars = ax.bar([x + offset for x in xs], means, width, yerr=errs,
                      capsize=4, label=dlabel,
                      color=[COLOURS[p] for p in ORDER],
                      alpha=1.0 if di == 0 else 0.55,
                      edgecolor="black", linewidth=0.6)
        for b, m, e in zip(bars, means, errs):
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + e + max(means) * 0.03,
                    fmt.format(m), ha="center", va="bottom", fontsize=9)

    ax.set_xticks(list(xs))
    ax.set_xticklabels([LABELS[p] for p in ORDER])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, pad=12)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)

    # Solid = YOLO, faded = LocateAnything. A colour legend would duplicate the
    # x axis, so the legend explains the SHADING instead.
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#555555", edgecolor="black", label=DETECTORS[0][1]),
                       Patch(facecolor="#555555", edgecolor="black", alpha=0.55,
                             label=DETECTORS[1][1])],
              loc="upper left", frameon=True, fontsize=9)
    ax.text(0.99, 0.97, "lower is better", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, style="italic", color="#555555")
    _save(fig, out, fname)


def _save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=200)
    fig.savefig(out / f"{name}.svg")
    plt.close(fig)
    print(f"  {out / (name + '.png')}")


def fig3_distribution(data, out: Path) -> None:
    """Box plots. Means hide the outliers that drive them."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, (key, dlabel) in zip(axes, DETECTORS):
        if key not in data:
            continue
        series = [[r["nearest_slot_error_m"] for r in data[key].get(p, [])]
                  for p in ORDER]
        bp = ax.boxplot(series, patch_artist=True, widths=0.55,
                        medianprops=dict(color="black", linewidth=2),
                        flierprops=dict(marker="o", markersize=4, alpha=0.6))
        for patch, p in zip(bp["boxes"], ORDER):
            patch.set_facecolor(COLOURS[p])
            patch.set_alpha(0.75)
        ax.set_xticklabels([LABELS[p] for p in ORDER], fontsize=9)
        ax.set_title(dlabel, fontsize=11)
        ax.grid(axis="y", alpha=0.3, linestyle=":")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Position error (m)")
    fig.suptitle("Distribution of position error across 30 group-visits per policy",
                 fontsize=12)
    _save(fig, out, "fig3_error_distribution")


def fig4_tradeoff(data, out: Path) -> None:
    """The headline: accurate placement and accurate facing pull apart."""
    fig, ax = plt.subplots(figsize=(9.0, 6.5))
    for di, (key, dlabel) in enumerate(DETECTORS):
        if key not in data:
            continue
        for policy in ORDER:
            rows = data[key].get(policy, [])
            if not rows:
                continue
            pos = statistics.fmean(r["nearest_slot_error_m"] for r in rows)
            head = [r["heading_error_deg"] for r in rows if r["within_distance"]]
            if not head:
                continue
            ax.scatter(pos, statistics.fmean(head), s=190, color=COLOURS[policy],
                       edgecolor="black", linewidth=1.0,
                       marker="o" if di == 0 else "^", zorder=3)
            ax.annotate(f"{LABELS[policy]}\n{dlabel.split(' ')[0]}",
                        (pos, statistics.fmean(head)),
                        textcoords="offset points", xytext=(11, -6),
                        fontsize=8, color="#333333")

    ax.set_xlabel("Position error (m)   ->  worse placement")
    ax.set_ylabel("Orientation error (°)   ->  worse facing")
    ax.set_title("The trade-off: policies that place well do not face well",
                 fontsize=12, pad=12)
    ax.grid(alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    ax.text(0.02, 0.03, "best corner", transform=ax.transAxes, fontsize=9,
            style="italic", color="#2ca02c")
    ax.margins(x=0.24, y=0.26)

    # Every point is annotated, so a six-entry legend would only repeat the
    # labels AND sit on top of the data. The legend explains the encoding -
    # colour is the policy, marker shape is the detector - and nothing more.
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", linestyle="", markersize=9,
                      markerfacecolor="#bbbbbb", markeredgecolor="black",
                      label=DETECTORS[0][1]),
               Line2D([], [], marker="^", linestyle="", markersize=9,
                      markerfacecolor="#bbbbbb", markeredgecolor="black",
                      label=DETECTORS[1][1])]
    handles += [Line2D([], [], marker="s", linestyle="", markersize=9,
                       markerfacecolor=COLOURS[p], markeredgecolor="black",
                       label=LABELS[p]) for p in ORDER]
    ax.legend(handles=handles, fontsize=8, loc="lower right", frameon=True,
              ncol=2, framealpha=0.95)
    _save(fig, out, "fig4_tradeoff")


def fig5_per_group(data, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, (key, dlabel) in zip(axes, DETECTORS):
        if key not in data:
            continue
        gids = sorted({r["group_id"] for p in ORDER for r in data[key].get(p, [])})
        sizes = {}
        for p in ORDER:
            for r in data[key].get(p, []):
                sizes[r["group_id"]] = r["num_people"]
        width = 0.26
        for pi, policy in enumerate(ORDER):
            vals = []
            for g in gids:
                v = [r["nearest_slot_error_m"] for r in data[key].get(policy, [])
                     if r["group_id"] == g]
                vals.append(statistics.fmean(v) if v else 0.0)
            ax.bar([i + (pi - 1) * width for i in range(len(gids))], vals, width,
                   label=LABELS[policy], color=COLOURS[policy],
                   edgecolor="black", linewidth=0.6)
        ax.set_xticks(range(len(gids)))
        ax.set_xticklabels([f"Group {g}\n({sizes[g]} people)" for g in gids], fontsize=9)
        ax.set_title(dlabel, fontsize=11)
        ax.grid(axis="y", alpha=0.3, linestyle=":")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Position error (m)")
    axes[0].legend(fontsize=9)
    fig.suptitle("Position error by group, showing which formations each policy handles",
                 fontsize=12)
    _save(fig, out, "fig5_per_group")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True,
                    help="the results_FINAL_* folder holding yolo/ and locateanything/")
    ap.add_argument("--out", default=None, help="figure directory (default <run>/figures)")
    args = ap.parse_args()

    run = Path(args.run).expanduser().resolve()
    out = Path(args.out) if args.out else run / "figures"
    out.mkdir(parents=True, exist_ok=True)

    data = load(run)
    if not data:
        raise SystemExit("no approach_accuracy.csv found - "
                         "run scripts/summarise_sim_results.py first")

    print(f"\nWriting figures to {out}")
    _bar_chart(data, "nearest_slot_error_m", False,
               "Approach position accuracy by policy and detector",
               "Mean position error (m)", "fig1_position_accuracy", out, "{:.3f}")
    _bar_chart(data, "heading_error_deg", True,
               "Approach orientation accuracy by policy and detector",
               "Mean orientation error (°)", "fig2_orientation_accuracy", out, "{:.1f}")
    fig3_distribution(data, out)
    fig4_tradeoff(data, out)
    fig5_per_group(data, out)
    print("\nError bars on the bar charts are the standard error of the mean "
          "over 30 group-visits.")
    print("Done.")


if __name__ == "__main__":
    main()
