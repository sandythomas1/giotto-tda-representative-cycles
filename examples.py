"""
Example gallery for representative_cycles.py
=============================================
Runs seven examples and saves figures to ./output/:

  1. Circle       — one H1 loop
  2. Figure-eight — two H1 loops
  3. Torus        — two H1 loops (H1(T^2) = Z x Z)
  4. Annulus      — one H1 loop
  5. Sphere       — no H1 loops (H2 void, not captured here)
  6. Essential    — a loop that never dies (death = inf), under max_edge_length
  7. Overview     — plot_overview() and plot_cycle() on the torus

Run with:
    python examples.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from representative_cycles import RepresentativeCycles

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Dataset generators
# ---------------------------------------------------------------------------

def make_circle(n: int = 80, noise: float = 0.05) -> np.ndarray:
    """Uniformly-sampled circle in R^2 with Gaussian noise."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X = np.column_stack([np.cos(theta), np.sin(theta)])
    X += RNG.standard_normal(X.shape) * noise
    return X


def make_figure_eight(n: int = 120, noise: float = 0.05) -> np.ndarray:
    """Two circles sharing one point — two distinct H1 loops."""
    half = n // 2
    theta = np.linspace(0, 2 * np.pi, half, endpoint=False)
    left = np.column_stack([np.cos(theta) - 1.0, np.sin(theta)])
    right = np.column_stack([np.cos(theta) + 1.0, np.sin(theta)])
    X = np.vstack([left, right])
    X += RNG.standard_normal(X.shape) * noise
    return X


def make_torus(n: int = 400, R: float = 2.0, r: float = 0.5, noise: float = 0.05) -> np.ndarray:
    """Random sample from a torus in R^3.  Expected H1 rank = 2."""
    theta = RNG.uniform(0, 2 * np.pi, n)
    phi = RNG.uniform(0, 2 * np.pi, n)
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    X = np.column_stack([x, y, z])
    X += RNG.standard_normal(X.shape) * noise
    return X


def make_sphere(n: int = 300, noise: float = 0.05) -> np.ndarray:
    """Random sample from S^2 in R^3.  Expected H2 rank = 1."""
    pts = RNG.standard_normal((n, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    pts += RNG.standard_normal(pts.shape) * noise
    return pts


def make_annulus(n: int = 200, r_inner: float = 0.5, r_outer: float = 1.5,
                 noise: float = 0.04) -> np.ndarray:
    """Uniform sample from an annulus — one clear H1 loop."""
    angles = RNG.uniform(0, 2 * np.pi, n)
    radii = np.sqrt(RNG.uniform(r_inner**2, r_outer**2, n))
    X = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    X += RNG.standard_normal(X.shape) * noise
    return X


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

def example_circle():
    print("\n=== Example 1: Circle (H1 loop) ===")
    X = make_circle(n=80, noise=0.06)

    rc = RepresentativeCycles(min_persistence=0.2)
    rc.fit(X)
    rc.summary()

    fig = rc.plot_matplotlib(
        title="Circle — Representative H₁ Cycles",
        save_path=os.path.join(OUTPUT_DIR, "circle_cycles.png"),
    )
    plt.close(fig)

    fig = rc.plot_barcode(
        save_path=os.path.join(OUTPUT_DIR, "circle_barcode.png"),
    )
    plt.close(fig)

    rc.plot_plotly(title="Circle — Interactive Cycles").write_html(
        os.path.join(OUTPUT_DIR, "circle_cycles.html")
    )
    print("  Saved: circle_cycles.png, circle_barcode.png, circle_cycles.html")


def example_figure_eight():
    print("\n=== Example 2: Figure-eight (two H1 loops) ===")
    X = make_figure_eight(n=120, noise=0.06)

    rc = RepresentativeCycles(min_persistence=0.3)
    rc.fit(X)
    rc.summary()

    fig = rc.plot_matplotlib(
        max_cycles=3,
        title="Figure-Eight — Two Representative H₁ Loops",
        figsize=(16, 5),
        save_path=os.path.join(OUTPUT_DIR, "figure_eight_cycles.png"),
    )
    plt.close(fig)

    rc.plot_plotly(title="Figure-Eight — Interactive Cycles").write_html(
        os.path.join(OUTPUT_DIR, "figure_eight_cycles.html")
    )
    print("  Saved: figure_eight_cycles.png, figure_eight_cycles.html")


def example_torus():
    print("\n=== Example 3: Torus (H1 rank 2) ===")
    X = make_torus(n=400, noise=0.04)

    rc = RepresentativeCycles(min_persistence=0.15)
    rc.fit(X)
    rc.summary()

    fig = rc.plot_matplotlib(
        max_cycles=4,
        figsize=(18, 5),
        title="Torus — Representative H₁ Cycles (each on its own best-fit plane)",
        save_path=os.path.join(OUTPUT_DIR, "torus_cycles.png"),
    )
    plt.close(fig)

    rc.plot_plotly(title="Torus — Interactive 3D Cycles").write_html(
        os.path.join(OUTPUT_DIR, "torus_cycles.html")
    )
    print("  Saved: torus_cycles.png, torus_cycles.html")


def example_annulus():
    print("\n=== Example 4: Annulus (one H1 loop) ===")
    X = make_annulus(n=200)

    rc = RepresentativeCycles(min_persistence=0.3)
    rc.fit(X)
    rc.summary()

    fig = rc.plot_matplotlib(
        title="Annulus — Representative H₁ Cycle",
        save_path=os.path.join(OUTPUT_DIR, "annulus_cycles.png"),
    )
    plt.close(fig)

    rc.plot_plotly(title="Annulus — Interactive Cycles").write_html(
        os.path.join(OUTPUT_DIR, "annulus_cycles.html")
    )
    print("  Saved: annulus_cycles.png, annulus_cycles.html")


def example_sphere():
    print("\n=== Example 5: Sphere (H2 void — shown as H1 boundary) ===")
    X = make_sphere(n=300, noise=0.04)

    rc = RepresentativeCycles(min_persistence=0.2)
    rc.fit(X)
    rc.summary()

    fig = rc.plot_matplotlib(
        max_cycles=4,
        figsize=(18, 5),
        title="Sphere — H₁ Cycles (each on its own best-fit plane)",
        save_path=os.path.join(OUTPUT_DIR, "sphere_cycles.png"),
    )
    plt.close(fig)

    rc.plot_plotly(title="Sphere — Interactive 3D Cycles").write_html(
        os.path.join(OUTPUT_DIR, "sphere_cycles.html")
    )
    print("  Saved: sphere_cycles.png, sphere_cycles.html")


def example_essential_bar():
    """A loop that never dies within the filtration.

    Capping the filtration at `max_edge_length=1.0` stops the Rips complex
    ever filling the circle in, so the loop survives to the end and its bar
    has `death = inf`.  Earlier releases dropped exactly these classes — the
    most persistent loop in the data — and reported nothing at all here.
    """
    print("\n=== Example 6: Essential bar (death = ∞) ===")
    X = make_circle(n=80, noise=0.03)

    rc = RepresentativeCycles(max_edge_length=1.0, min_persistence=0.3)
    rc.fit(X)
    rc.summary()

    essential = [f for f in rc.features_ if f.is_essential]
    print(f"  essential features: {len(essential)} of {len(rc.features_)}")

    fig = rc.plot_matplotlib(
        title="Circle under max_edge_length=1.0 — an essential H₁ cycle",
        save_path=os.path.join(OUTPUT_DIR, "essential_cycles.png"),
    )
    plt.close(fig)

    fig = rc.plot_barcode(
        save_path=os.path.join(OUTPUT_DIR, "essential_barcode.png"),
    )
    plt.close(fig)

    try:
        frame = rc.to_dataframe()
    except ImportError:
        print("  (install pandas to see to_dataframe())")
    else:
        print(frame[["birth", "death", "is_essential", "is_verified"]])

    print("  Saved: essential_cycles.png, essential_barcode.png")


def example_overview_figure():
    """The composite view, and one cycle at full size."""
    print("\n=== Example 7: Overview and single-cycle figures ===")
    X = make_torus(n=400, noise=0.04)

    rc = RepresentativeCycles(min_persistence=0.15)
    rc.fit(X)

    fig = rc.plot_overview(
        max_cycles=4,
        title="Torus — Overview",
        save_path=os.path.join(OUTPUT_DIR, "torus_overview.png"),
    )
    plt.close(fig)

    fig = rc.plot_cycle(
        0,
        save_path=os.path.join(OUTPUT_DIR, "torus_cycle_0.png"),
    )
    plt.close(fig)
    print("  Saved: torus_overview.png, torus_cycle_0.png")


# ---------------------------------------------------------------------------
# Combined diagram gallery
# ---------------------------------------------------------------------------

def example_diagram_gallery():
    """Side-by-side persistence diagrams for all datasets."""
    print("\n=== Gallery: Persistence Diagrams ===")
    datasets = {
        "Circle": make_circle(80),
        "Figure-Eight": make_figure_eight(120),
        "Annulus": make_annulus(200),
        "Torus (proj)": make_torus(400)[:, :2],
    }

    fig, axes = plt.subplots(1, len(datasets), figsize=(16, 4))
    for ax, (name, X) in zip(axes, datasets.items()):
        rc = RepresentativeCycles(min_persistence=0.0)
        rc.fit(X)

        dgm = rc.diagrams_[1]
        if len(dgm) > 0:
            finite = dgm[np.isfinite(dgm[:, 1])]
            if len(finite) > 0:
                b, d = finite[:, 0], finite[:, 1]
                pers = d - b
                lo, hi = min(b.min(), d.min()), max(b.max(), d.max())
                pad = (hi - lo) * 0.05
                ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.7)
                sc = ax.scatter(b, d, c=pers, cmap="plasma", s=40,
                                edgecolors="k", lw=0.4, zorder=3)
                plt.colorbar(sc, ax=ax, label="Persistence")
                ax.set_xlim(lo - pad, hi + pad)
                ax.set_ylim(lo - pad, hi + pad)
                ax.set_aspect("equal")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")

    fig.suptitle("H₁ Persistence Diagrams", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "overview_diagrams.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    example_circle()
    example_figure_eight()
    example_torus()
    example_annulus()
    example_sphere()
    example_essential_bar()
    example_overview_figure()
    example_diagram_gallery()
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
