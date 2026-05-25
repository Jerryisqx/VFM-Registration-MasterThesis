"""Generate the mTRE CDF plot for R1 results."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
R1_DIR = PROJECT_ROOT / "results" / "registration_R1"


def main():
    rows = list(csv.DictReader(open(R1_DIR / "results.csv")))
    single = sorted(float(r["final_mtre_mm"]) for r in rows if r["mode"] == "single")
    dual = sorted(float(r["final_mtre_mm"]) for r in rows if r["mode"] == "dual")

    def cdf(values):
        n = len(values)
        return np.array(values), np.arange(1, n + 1) / n

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    xs, ys = cdf(single)
    ax.step(xs, ys, where="post", label=f"Single-plane (n={len(single)})", color="#cc4444", linewidth=2)
    xs, ys = cdf(dual)
    ax.step(xs, ys, where="post", label=f"Dual-plane (n={len(dual)})", color="#3370b8", linewidth=2)

    ax.axvline(2.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(5.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(2.05, 0.04, "2 mm", color="gray", fontsize=8)
    ax.text(5.05, 0.04, "5 mm", color="gray", fontsize=8)

    ax.set_xlabel("Final mTRE (mm)")
    ax.set_ylabel("Cumulative fraction of trials")
    ax.set_title("R1: Single-plane vs Dual-plane registration accuracy\n(80 mm cube, 10 inits, ±5 mm / ±5°)")
    ax.set_xlim(0, max(max(single), max(dual)) * 1.05)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()

    out = R1_DIR / "cdf.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
