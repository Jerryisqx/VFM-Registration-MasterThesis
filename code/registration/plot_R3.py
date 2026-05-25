"""Generate the R3 robustness curves: mean mTRE and success rate vs delta."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
R3_DIR = PROJECT_ROOT / "results" / "registration_R3"


def main():
    rows = list(csv.DictReader(open(R3_DIR / "results.csv")))
    deltas = sorted({float(r["delta"]) for r in rows})

    means, medians, p25s, p75s, succ_2, succ_5 = [], [], [], [], [], []
    for d in deltas:
        sel = [float(r["mtre_mm"]) for r in rows if float(r["delta"]) == d]
        means.append(np.mean(sel))
        medians.append(np.median(sel))
        p25s.append(np.percentile(sel, 25))
        p75s.append(np.percentile(sel, 75))
        succ_2.append(np.mean(np.array(sel) < 2.0))
        succ_5.append(np.mean(np.array(sel) < 5.0))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: mTRE vs delta with IQR band
    ax = axes[0]
    ax.fill_between(deltas, p25s, p75s, alpha=0.2, color="#3370b8", label="IQR (25-75%)")
    ax.plot(deltas, means, "o-", color="#3370b8", label="Mean mTRE", linewidth=2)
    ax.plot(deltas, medians, "s--", color="#cc4444", label="Median mTRE", linewidth=1.5, alpha=0.8)
    ax.axhline(2.0, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(5.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Initial perturbation $\\delta$ (mm / deg, paired)")
    ax.set_ylabel("Final mTRE (mm)")
    ax.set_title("R3: mTRE vs initial perturbation\n(dual-plane, 80 mm cube, n=10 per delta)")
    ax.set_xscale("log")
    ax.set_xticks(deltas)
    ax.set_xticklabels([f"{d:.0f}" for d in deltas])
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left")

    # Right: success rate vs delta
    ax = axes[1]
    ax.plot(deltas, [s * 100 for s in succ_5], "o-", color="#3370b8",
            label="Success @ 5 mm", linewidth=2)
    ax.plot(deltas, [s * 100 for s in succ_2], "s--", color="#cc4444",
            label="Success @ 2 mm", linewidth=1.5)
    ax.set_xlabel("Initial perturbation $\\delta$ (mm / deg, paired)")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("R3: success rate vs initial perturbation")
    ax.set_xscale("log")
    ax.set_xticks(deltas)
    ax.set_xticklabels([f"{d:.0f}" for d in deltas])
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower left")

    fig.tight_layout()
    out = R3_DIR / "robustness_curves.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
