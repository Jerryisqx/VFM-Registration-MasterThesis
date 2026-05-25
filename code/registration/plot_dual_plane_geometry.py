"""§6.4 figure: dual-plane (BS + FS) C-arm geometry schematic.

Renders a top-down view of the dual-plane biplanar fluoroscopy setup:
  - BS intensifier at world origin, +Z out
  - FS intensifier at (333, 0, 233), normal rotated 70° about -Y
  - Per-camera X-ray source at intensifier center + focal_length * normal
  - Anatomy near (0, 0, 100) within both frustums
  - 70° angle annotation between the two normals
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/Users/jerrychen/Desktop/MasterThesis")
OUT = ROOT / "results/figure_dual_plane_geometry.png"

# World coords (top-down, X horizontal, Z vertical)
FOCAL = 972.0      # mm
BS_C = np.array([0.0, 0.0])
BS_N = np.array([0.0, 1.0])
FS_C = np.array([333.0, 233.0])
ANGLE_DEG = 70.0
ANGLE_RAD = np.deg2rad(ANGLE_DEG)
FS_N = np.array([np.sin(ANGLE_RAD), np.cos(ANGLE_RAD)])  # rotated +70° about -Y from BS

BS_SRC = BS_C + FOCAL * BS_N
FS_SRC = FS_C + FOCAL * FS_N

INTENSIFIER_HALF = 145.0   # mm (≈ kneefit 287.6 / 2)
ANATOMY_CENTER = np.array([60.0, 200.0])  # somewhere both cameras can image
ANATOMY_RADIUS = 50.0


def perp(v):
    """2D perpendicular vector (rotated +90°)."""
    return np.array([-v[1], v[0]])


def draw_intensifier(ax, center, normal, label, color):
    p = perp(normal) * INTENSIFIER_HALF
    a = center - p
    b = center + p
    ax.plot([a[0], b[0]], [a[1], b[1]], "-", color=color, linewidth=4, solid_capstyle="round")
    # Surface tick marks
    for t in np.linspace(-0.85, 0.85, 7):
        pt = center + t * p
        tick = pt - 0.04 * INTENSIFIER_HALF * normal
        ax.plot([pt[0], tick[0]], [pt[1], tick[1]], "-", color=color, linewidth=1)
    # Label
    off = -0.55 * normal * INTENSIFIER_HALF
    ax.text(center[0] + off[0] - 25, center[1] + off[1] - 35,
             label, fontsize=12, weight="bold", color=color)


def draw_source(ax, src, color, label):
    ax.plot(src[0], src[1], "*", markersize=18, markerfacecolor=color,
             markeredgecolor="black", markeredgewidth=0.6)
    ax.text(src[0] + 25, src[1] + 5, label, fontsize=10, color="black")


def draw_frustum(ax, src, center, normal, color):
    p = perp(normal) * INTENSIFIER_HALF
    a = center - p
    b = center + p
    poly = mpatches.Polygon([src, a, b], closed=True,
                             facecolor=color, edgecolor=color, alpha=0.10, linewidth=0)
    ax.add_patch(poly)
    for end in (a, b):
        ax.plot([src[0], end[0]], [src[1], end[1]], "--", color=color, linewidth=0.8, alpha=0.5)
    # Optical axis
    ax.plot([src[0], center[0]], [src[1], center[1]], ":", color=color, linewidth=1.0, alpha=0.5)


def draw_anatomy(ax):
    c = ANATOMY_CENTER
    ax.add_patch(mpatches.Circle(c, ANATOMY_RADIUS,
                                  facecolor="#e0e0e0", edgecolor="#444", linewidth=1.2, zorder=10))
    ax.text(c[0], c[1] + ANATOMY_RADIUS + 12, "anatomy\n(femur + tibia)",
             ha="center", va="bottom", fontsize=9, color="#333")


def draw_angle_arc(ax):
    # arc between BS_N and FS_N at origin
    radius = 90
    theta_bs = np.degrees(np.arctan2(BS_N[1], BS_N[0]))  # 90°
    theta_fs = np.degrees(np.arctan2(FS_N[1], FS_N[0]))  # < 90°
    arc = mpatches.Arc((0, 0), 2 * radius, 2 * radius,
                        theta1=min(theta_bs, theta_fs), theta2=max(theta_bs, theta_fs),
                        color="#222", linewidth=1.4)
    ax.add_patch(arc)
    # Angle label
    mid_angle = np.deg2rad((theta_bs + theta_fs) / 2)
    ax.text(radius * 0.65 * np.cos(mid_angle), radius * 0.65 * np.sin(mid_angle),
             "70°", fontsize=12, color="#222", weight="bold")


def draw_axes(ax):
    ax.annotate("", xy=(220, 0), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.0))
    ax.annotate("", xy=(0, 220), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.0))
    ax.text(225, -10, r"$X$", fontsize=11)
    ax.text(-25, 215, r"$Z$", fontsize=11)
    ax.plot(0, 0, "o", markersize=7, markerfacecolor="white", markeredgecolor="black")
    ax.text(-90, -25, "world origin\n(= BS intensifier center)",
             fontsize=8, color="#555")


def main():
    fig, ax = plt.subplots(figsize=(11, 7))
    # Frustums first (behind)
    draw_frustum(ax, BS_SRC, BS_C, BS_N, "#cc4444")
    draw_frustum(ax, FS_SRC, FS_C, FS_N, "#3370b8")
    # Intensifiers + sources
    draw_intensifier(ax, BS_C, BS_N, "BS intensifier", "#cc4444")
    draw_intensifier(ax, FS_C, FS_N, "FS intensifier", "#3370b8")
    draw_source(ax, BS_SRC, "#cc4444", "BS source")
    draw_source(ax, FS_SRC, "#3370b8", "FS source")
    # Anatomy
    draw_anatomy(ax)
    # Angle + axes
    draw_angle_arc(ax)
    draw_axes(ax)

    # Focal-length annotation on BS
    ax.annotate("", xy=BS_SRC, xytext=BS_C,
                 arrowprops=dict(arrowstyle="<->", color="#cc4444", lw=1.2))
    ax.text(BS_C[0] + 20, (BS_SRC[1] + BS_C[1]) / 2,
             f"focal length\n$f = {int(FOCAL)}$ mm", fontsize=9, color="#cc4444")

    # FS center label
    ax.text(FS_C[0] + 25, FS_C[1] + 25,
             f"FS center\n$(333, 0, 233)$ mm", fontsize=9, color="#3370b8")

    ax.set_xlim(-150, 1400)
    ax.set_ylim(-100, 1300)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_xlabel("World $X$ (mm)")
    ax.set_ylabel("World $Z$ (mm)")
    ax.set_title("Dual-plane fluoroscopy geometry (top-down view, $Y$ out of page)\n"
                  "BS plane at origin looking +$Z$; FS plane offset and rotated 70° about $-Y$",
                  fontsize=11)

    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
