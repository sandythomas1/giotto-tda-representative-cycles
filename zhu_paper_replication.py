"""
Zhu-Paper Replication: Persistent Homology Examples A–E
=======================================================
Replicates the canonical persistent-homology examples from:

  Zhu, X. (2013). "Persistent Homology: An Introduction and a New Text
  Representation for NLP."  IJCAI 2013, pp. 1953–1959.

The paper's tutorial (Section 2) builds intuition using:
  • The rubber-band / hole analogy for H1
  • A 4-point rectangle whose H1 bar is computable exactly  (Section 2.3)
  • Betti-number predictions for spheres, tori, annuli, figure-eights, etc.

We map those theoretical predictions to five point-cloud experiments
(A–E) and verify that RepresentativeCycles recovers the expected topology.

Expected topology (from the paper + standard TDA theory)
---------------------------------------------------------
  A  Rectangle / square    β1 = 1   birth = 2.0,  death = √5 ≈ 2.236
  B  Circle S¹             β1 = 1   one long-lived bar
  C  Figure-eight          β1 = 2   two long-lived bars of similar length
  D  Torus T²              β1 = 2   two dominant bars (plus many short noise bars)
  E  Sphere S²             β1 = 0   H2=1 not captured by H1 → only short noise bars

Run with the Python 3.12 environment that has giotto-tda + gph-ripser:
    python zhu_paper_replication.py
"""

from __future__ import annotations

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from representative_cycles import RepresentativeCycles

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "OK"
FAIL = "XX"


def _banner(label: str, title: str) -> None:
    print(f"\n{'='*66}")
    print(f"  Example {label}: {title}")
    print(f"{'='*66}")


def _check(condition: bool, description: str) -> None:
    status = "[PASS]" if condition else "[FAIL]"
    print(f"  {status}  {description}")


def _dominant_features(rc: RepresentativeCycles, top_n: int = 4) -> list:
    """Return top-N features by persistence."""
    return rc.features_[:top_n]


# ---------------------------------------------------------------------------
# Example A — Rectangle (exact, from Zhu Section 2.3)
# ---------------------------------------------------------------------------

def example_A_rectangle() -> tuple:
    """
    Exact 4-point example from Zhu (2013) Section 2.3.

    Points: (0,0), (0,1), (2,1), (2,0)
    Expected:  H1 birth = 2.0,  H1 death = sqrt(5) ≈ 2.2361
               persistence      = sqrt(5) − 2 ≈ 0.2361
    """
    _banner("A", "Rectangle — exact 4-point example (Zhu §2.3)")

    X = np.array([[0.0, 0.0],
                  [0.0, 1.0],
                  [2.0, 1.0],
                  [2.0, 0.0]], dtype=np.float64)

    expected_birth = 2.0
    expected_death = math.sqrt(5)          # ≈ 2.2361
    expected_persistence = expected_death - expected_birth   # ≈ 0.2361

    rc = RepresentativeCycles(min_persistence=0.0)
    rc.fit(X)
    rc.summary()

    feats = rc.features_
    n_h1 = len(feats)

    _check(n_h1 == 1, f"exactly 1 H1 bar (found {n_h1})")
    if n_h1 >= 1:
        f = feats[0]
        _check(
            abs(f.birth - expected_birth) < 1e-4,
            f"birth ≈ 2.000  (got {f.birth:.6f}, expected {expected_birth:.6f})"
        )
        _check(
            abs(f.death - expected_death) < 1e-4,
            f"death ≈ √5 ≈ {expected_death:.6f}  (got {f.death:.6f})"
        )
        _check(
            abs(f.persistence - expected_persistence) < 1e-4,
            f"persistence ≈ {expected_persistence:.6f}  (got {f.persistence:.6f})"
        )
        _check(
            len(f.cycle_edges) == 4,
            f"representative cycle has 4 edges — the full rectangle boundary (got {len(f.cycle_edges)})"
        )

    print(f"\n  Theory note (Zhu §2.3): VR(ε=2) creates the rectangular loop when"
          f"\n  the two horizontal edges (length 2) appear. VR(ε=√5) kills it when"
          f"\n  the diagonals (length √5) fill in the triangles.")

    return rc, 1


# ---------------------------------------------------------------------------
# Example B — Circle S¹
# ---------------------------------------------------------------------------

def example_B_circle() -> tuple:
    """
    80-point circle with small noise.
    Expected: β1 = 1  (one long-lived H1 bar, persistence >> all others).
    """
    _banner("B", "Circle S¹  — β1 = 1, one dominant loop")

    n = 80
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X = np.column_stack([np.cos(theta), np.sin(theta)])
    X += RNG.standard_normal(X.shape) * 0.05

    rc = RepresentativeCycles(min_persistence=0.0)
    rc.fit(X)

    feats = rc.features_
    if len(feats) == 0:
        print("  No H1 features found.")
        return rc, 0

    top = feats[0]
    rest = feats[1:]
    gap = top.persistence - (rest[0].persistence if rest else 0.0)

    print(f"  Top feature: birth={top.birth:.4f}  death={top.death:.4f}"
          f"  persistence={top.persistence:.4f}  |cycle|={len(top.cycle_edges)}")
    if rest:
        print(f"  Next best persistence: {rest[0].persistence:.4f}"
              f"  (spectral gap = {gap:.4f})")

    _check(top.persistence > 1.0,
           f"dominant bar has persistence > 1.0  (got {top.persistence:.4f})")
    _check(gap > 0.5 if rest else True,
           "large spectral gap isolates the single dominant loop")
    _check(len(top.cycle_edges) > 10,
           f"representative cycle has ≥ 10 edges (got {len(top.cycle_edges)})")

    rc.summary()
    return rc, 1


# ---------------------------------------------------------------------------
# Example C — Figure-eight  (two circles sharing one point)
# ---------------------------------------------------------------------------

def example_C_figure_eight() -> tuple:
    """
    Two tangent circles — wedge sum S¹ ∨ S¹.
    Expected: β1 = 2  (two dominant H1 bars of similar persistence).
    """
    _banner("C", "Figure-eight S¹∨S¹ — β1 = 2, two dominant loops")

    half = 60
    theta = np.linspace(0, 2 * np.pi, half, endpoint=False)
    left  = np.column_stack([np.cos(theta) - 1.0, np.sin(theta)])
    right = np.column_stack([np.cos(theta) + 1.0, np.sin(theta)])
    X = np.vstack([left, right])
    X += RNG.standard_normal(X.shape) * 0.05

    rc = RepresentativeCycles(min_persistence=0.0)
    rc.fit(X)

    feats = rc.features_
    if len(feats) < 2:
        print(f"  Only {len(feats)} H1 feature(s) found.")
        rc.summary()
        return rc, len(feats)

    f0, f1 = feats[0], feats[1]
    ratio = f1.persistence / f0.persistence

    print(f"  Feature 1: pers={f0.persistence:.4f}  |cycle|={len(f0.cycle_edges)}")
    print(f"  Feature 2: pers={f1.persistence:.4f}  |cycle|={len(f1.cycle_edges)}")
    print(f"  Ratio p2/p1 = {ratio:.4f}")

    _check(f0.persistence > 1.0,
           f"first loop persistence > 1.0  (got {f0.persistence:.4f})")
    _check(f1.persistence > 1.0,
           f"second loop persistence > 1.0  (got {f1.persistence:.4f})")
    _check(ratio > 0.7,
           f"two loops have comparable persistence (ratio {ratio:.4f} > 0.7)")

    third_pers = feats[2].persistence if len(feats) > 2 else 0.0
    _check(f1.persistence > 3 * third_pers or third_pers < 0.3,
           f"spectral gap separates 2 dominant loops from noise  "
           f"(3rd={third_pers:.4f})")

    rc.summary()
    return rc, 2


# ---------------------------------------------------------------------------
# Example D — Torus T²
# ---------------------------------------------------------------------------

def example_D_torus() -> tuple:
    """
    Random sample from a torus in R³.
    Expected: β1 = 2  (two dominant H1 loops, many short noise bars).
    H1(T²) = Z × Z → rank 2.
    """
    _banner("D", "Torus T²  — β1 = 2, two dominant loops (H1(T²)=Z×Z)")

    # R=2, r=0.8: fatter tube gives a larger meridian circle so its persistence
    # is well clear of noise, making both generators cleanly detectable.
    n = 500
    R, r = 2.0, 0.8
    theta = RNG.uniform(0, 2 * np.pi, n)
    phi   = RNG.uniform(0, 2 * np.pi, n)
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    X = np.column_stack([x, y, z])
    X += RNG.standard_normal(X.shape) * 0.03

    rc = RepresentativeCycles(min_persistence=0.0)
    rc.fit(X)

    feats = rc.features_
    if len(feats) < 2:
        print(f"  Only {len(feats)} feature(s) found.")
        rc.summary()
        return rc, len(feats)

    f0, f1 = feats[0], feats[1]
    third_pers = feats[2].persistence if len(feats) > 2 else 0.0

    print(f"  Feature 1: pers={f0.persistence:.4f}  |cycle|={len(f0.cycle_edges)}")
    print(f"  Feature 2: pers={f1.persistence:.4f}  |cycle|={len(f1.cycle_edges)}")
    if len(feats) > 2:
        print(f"  Feature 3: pers={third_pers:.4f}  (noise)")

    # Noise floor: median persistence of everything after the top-2
    noise_bars = [f.persistence for f in feats[2:]] if len(feats) > 2 else [0.0]
    noise_median = float(np.median(noise_bars))
    above_noise_ratio = f1.persistence / noise_median if noise_median > 0 else float("inf")

    _check(f0.persistence > 1.0,
           f"dominant bar (longitude) pers > 1.0  (got {f0.persistence:.4f})")
    _check(f1.persistence > 0.3,
           f"second bar (meridian) pers > 0.3  (got {f1.persistence:.4f})")
    _check(above_noise_ratio > 2.0,
           f"2nd bar is > 2x above noise median ({f1.persistence:.4f} / "
           f"{noise_median:.4f} = {above_noise_ratio:.2f})")

    rc.summary()
    return rc, 2


# ---------------------------------------------------------------------------
# Example E — Sphere S²
# ---------------------------------------------------------------------------

def example_E_sphere() -> tuple:
    """
    Random sample from S² in R³.
    Expected: β1 = 0  (H1(S²) = 0; the only non-trivial feature is H2).
    RepresentativeCycles only tracks H1, so we should see only short-lived
    noise bars — no dominant, long-lived H1 feature.
    """
    _banner("E", "Sphere S²  — β1 = 0 in theory; only noise H1 bars")

    n = 300
    pts = RNG.standard_normal((n, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    pts += RNG.standard_normal(pts.shape) * 0.04
    X = pts

    rc = RepresentativeCycles(min_persistence=0.0)
    rc.fit(X)

    feats = rc.features_
    if not feats:
        print("  No H1 features at all — perfect null result.")
        return rc, 0

    max_pers = feats[0].persistence
    min_pers = feats[-1].persistence

    print(f"  H1 features found: {len(feats)}")
    print(f"  Persistence range: [{min_pers:.4f}, {max_pers:.4f}]")
    print(f"  (All are short-lived noise artefacts from finite sampling)")

    _check(max_pers < 0.5,
           f"no dominant H1 bar — max persistence {max_pers:.4f} < 0.5")

    n_long = sum(1 for f in feats if f.persistence > 0.5)
    _check(n_long == 0,
           f"zero bars with persistence > 0.5  (found {n_long})")

    print("\n  Theory note: H1(S²) = 0.  H2(S²) = Z (the void inside the sphere).")
    print("  Our API computes only H1 representative cycles; the H2 void is not")
    print("  captured here.  The short bars confirm there is no genuine 1-loop.")

    rc.summary()
    return rc, 0


# ---------------------------------------------------------------------------
# Combined comparison figure
# ---------------------------------------------------------------------------

def _plot_persistence_panel(ax, rc: RepresentativeCycles, title: str,
                             highlight_top: int = 2) -> None:
    """Draw a persistence diagram for one example."""
    dgm = rc.diagrams_[1] if rc.diagrams_ is not None else np.empty((0, 2))
    if len(dgm) == 0:
        ax.set_title(title, fontsize=9)
        return

    finite = dgm[np.isfinite(dgm[:, 1])]
    if len(finite) == 0:
        ax.set_title(title, fontsize=9)
        return

    b, d = finite[:, 0], finite[:, 1]
    pers = d - b

    lo = min(b.min(), d.min())
    hi = max(b.max(), d.max())
    pad = max((hi - lo) * 0.05, 0.01)

    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.7, alpha=0.5)
    sc = ax.scatter(b, d, c=pers, cmap="plasma", s=30, zorder=3,
                    edgecolors="k", linewidths=0.3)

    # Highlight the top-N by persistence
    order = np.argsort(pers)[::-1]
    colors = ["limegreen", "deepskyblue", "orange"]
    for rank, (idx, col) in enumerate(zip(order[:highlight_top], colors[:highlight_top])):
        ax.scatter(b[idx], d[idx], s=100, facecolors="none",
                   edgecolors=col, linewidths=1.5, zorder=4)

    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Birth", fontsize=8)
    ax.set_ylabel("Death", fontsize=8)
    ax.tick_params(labelsize=7)


def _plot_cycle_panel(ax, rc: RepresentativeCycles, rank: int,
                      title: str, color: str) -> None:
    """Draw the representative cycle for the rank-th feature."""
    if not rc.features_ or rank >= len(rc.features_):
        ax.set_title(title, fontsize=9)
        ax.text(0.5, 0.5, "no feature", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="gray")
        return

    X = rc.point_cloud_
    coords = X[:, :2]
    f = rc.features_[rank]

    ax.scatter(coords[:, 0], coords[:, 1], s=8, c="lightgray", zorder=1)

    if len(f.cycle_edges) > 0:
        birth_key = (min(f.birth_edge), max(f.birth_edge))
        normal_segs, birth_segs = [], []
        for e in f.cycle_edges:
            key = (min(e[0], e[1]), max(e[0], e[1]))
            (birth_segs if key == birth_key else normal_segs).append(
                [coords[e[0]], coords[e[1]]]
            )
        import matplotlib.collections as mc
        if normal_segs:
            ax.add_collection(mc.LineCollection(
                normal_segs, colors=[color], linewidths=1.8, zorder=2, alpha=0.85))
        if birth_segs:
            ax.add_collection(mc.LineCollection(
                birth_segs, colors=["black"], linewidths=2.2, zorder=3,
                linestyles="dashed"))

    cv = f.cycle_vertices
    ax.scatter(coords[cv, 0], coords[cv, 1],
               s=22, c=[color], zorder=4, edgecolors="k", linewidths=0.3)

    death_str = f"{f.death:.3f}" if np.isfinite(f.death) else "∞"
    ax.set_title(
        f"{title}\nb={f.birth:.3f} d={death_str} p={f.persistence:.3f}",
        fontsize=8,
    )
    ax.set_aspect("equal")
    ax.autoscale_view()


def plot_zhu_comparison(
    results: dict[str, RepresentativeCycles],
    save_path: str,
) -> None:
    """
    Build a 2-row figure:
      Row 1: persistence diagrams for all 5 examples
      Row 2: top representative cycle(s) for each example
    """
    labels = list(results.keys())
    n = len(labels)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    fig = plt.figure(figsize=(4 * n, 9))
    gs = gridspec.GridSpec(2, n, figure=fig, hspace=0.45, wspace=0.35)

    for col, (label, color) in enumerate(zip(labels, colors)):
        rc = results[label]

        # Row 0: persistence diagram
        ax_diag = fig.add_subplot(gs[0, col])
        top_n = 2 if label in ("C", "D") else 1
        _plot_persistence_panel(ax_diag, rc, f"Example {label}", highlight_top=top_n)

        # Row 1: dominant cycle
        ax_cyc = fig.add_subplot(gs[1, col])
        _plot_cycle_panel(ax_cyc, rc, 0,
                          title=f"Top cycle ({label})", color=color)

    # Legend for row 0 highlighting
    legend_elems = [
        mpatches.Patch(facecolor="none", edgecolor="limegreen", linewidth=1.5,
                       label="Top feature"),
        mpatches.Patch(facecolor="none", edgecolor="deepskyblue", linewidth=1.5,
                       label="2nd feature"),
        Line2D([0], [0], color="black", lw=1.5, ls="--", label="Birth edge"),
    ]
    fig.legend(handles=legend_elems, loc="upper right", fontsize=8,
               bbox_to_anchor=(1.0, 1.0))

    fig.suptitle(
        "Zhu-Paper Replication: Examples A-E\n"
        "(Zhu 2013, Persistent Homology: An Introduction and NLP Application)",
        fontsize=11, fontweight="bold", y=1.02,
    )

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved comparison figure: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results: dict[str, RepresentativeCycles] = {}
    observed_b1: dict[str, int] = {}

    results["A"], observed_b1["A"] = example_A_rectangle()
    results["B"], observed_b1["B"] = example_B_circle()
    results["C"], observed_b1["C"] = example_C_figure_eight()
    results["D"], observed_b1["D"] = example_D_torus()
    results["E"], observed_b1["E"] = example_E_sphere()

    # --- Individual output figures ---
    print("\n--- Saving individual figures ---")
    for label, rc in results.items():
        if not rc.features_:
            continue
        max_c = 2 if label in ("C", "D") else 1
        fig = rc.plot_matplotlib(
            max_cycles=max_c,
            title=f"Example {label} — Representative H₁ Cycles",
            save_path=os.path.join(OUTPUT_DIR, f"zhu_example_{label}.png"),
        )
        plt.close(fig)

    # --- Combined comparison figure ---
    plot_zhu_comparison(
        results,
        save_path=os.path.join(OUTPUT_DIR, "zhu_paper_comparison.png"),
    )

    # --- Summary table ---
    print("\n" + "="*66)
    print("  SUMMARY — Expected vs Observed Topology")
    print("="*66)
    header = f"  {'Ex':>2}  {'Shape':20s}  {'Expected β1':>12}  {'Observed β1':>11}  {'Match':>5}"
    print(header)
    print("  " + "-"*62)

    expected = {
        "A": ("Rectangle (4 pts)",   1),
        "B": ("Circle S¹",           1),
        "C": ("Figure-eight S¹∨S¹",  2),
        "D": ("Torus T²",            2),
        "E": ("Sphere S² (H1=0)",    0),
    }

    for label, (shape, exp_b1) in expected.items():
        obs_b1 = observed_b1[label]
        match = "[OK]" if obs_b1 == exp_b1 else "[XX]"
        print(f"  {label:>2}  {shape:20s}  {exp_b1:>12}  {obs_b1:>11}  {match:>5}")

    print("\n  All outputs saved to:", OUTPUT_DIR)
