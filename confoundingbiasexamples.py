"""
Sensitivity analysis demo:
1) Book single-U confounding example
2) True two-U example, but sensitivity analysis incorrectly assumes one U

Outputs exactly TWO figure files:
    plots/confoundingbias/book_single_u_v2.png
    plots/confoundingbias/two_u_misspecified_v2.png

Each figure has 4 panels:
    top-left:  true causal graph
    top-right: assumed causal graph used in sensitivity analysis
    bottom-left:  bias sensitivity contours
    bottom-right: possible true ATE contours
"""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.linear_model import LinearRegression

PLOT_DIR = Path("plots/confoundingbias")
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Graph drawing helpers
# ============================================================

def draw_graph_book(ax, title):
    G = nx.DiGraph()
    G.add_edges_from([
        ("W", "T"),
        ("W", "Y"),
        ("U", "T"),
        ("U", "Y"),
        ("T", "Y"),
    ])

    pos = {
        "W": (0.0, 1.0),
        "U": (2.0, 1.0),
        "T": (0.7, 0.0),
        "Y": (1.7, 0.0),
    }

    nx.draw_networkx(
        G,
        pos=pos,
        ax=ax,
        node_size=1900,
        font_size=11,
        arrowsize=20,
        width=1.7,
    )

    ax.set_title(title)
    ax.axis("off")


def draw_graph_two_u_true(ax, title):
    G = nx.DiGraph()
    G.add_edges_from([
        ("W", "T"),
        ("W", "Y"),
        ("U1", "T"),
        ("U1", "Y"),
        ("U2", "T"),
        ("U2", "Y"),
        ("T", "Y"),
    ])

    pos = {
        "W": (0.0, 1.0),
        "U1": (1.3, 1.8),
        "U2": (2.6, 1.0),
        "T": (0.9, 0.0),
        "Y": (2.1, 0.0),
    }

    nx.draw_networkx(
        G,
        pos=pos,
        ax=ax,
        node_size=1750,
        font_size=10,
        arrowsize=18,
        width=1.6,
    )

    ax.set_title(title)
    ax.axis("off")


def add_text_box(ax, lines, x=0.02, y=0.02):
    text = "\n".join(lines)
    ax.text(
        x, y, text,
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", alpha=0.12),
    )


# ============================================================
# Shared contour plotting helper
# ============================================================

def plot_assumed_one_u_sensitivity(
    ax_bias,
    ax_ate,
    observed_effect,
    assumed_alpha,
    assumed_beta,
    actual_true_ate,
    actual_total_bias=None,
):
    """
    Assumed one-U sensitivity model:
        assumed bias = beta_u / alpha_u

    x-axis is 1/alpha_u, y-axis is beta_u
    """
    inv_alpha = np.linspace(0.01, 5.0, 800)
    beta = np.linspace(0.0, 10.0, 800)
    X, B = np.meshgrid(inv_alpha, beta)

    assumed_bias_grid = B * X  # beta / alpha

    # Bottom-left: bias sensitivity
    levels_bias = [1, 2, 3, 5, 10, 15, 25]
    levels_bias = [
        level for level in levels_bias
        if assumed_bias_grid.min() < level < assumed_bias_grid.max()
    ]

    cs_bias = ax_bias.contour(
        X, B, assumed_bias_grid,
        levels=levels_bias,
        linewidths=1.6,
    )
    ax_bias.clabel(cs_bias, inline=True, fontsize=8, fmt="%g")

    assumed_x = 1 / assumed_alpha
    assumed_y = assumed_beta
    assumed_bias = assumed_beta / assumed_alpha

    ax_bias.scatter(
        [assumed_x], [assumed_y],
        s=85,
        edgecolors="black",
        linewidths=1.0,
        zorder=5,
    )

    if actual_total_bias is None:
        bias_note = (
            f"chosen point:\n"
            f"assumed bias = {assumed_bias:.1f}"
        )
    else:
        bias_note = (
            f"chosen point:\n"
            f"assumed one-U bias = {assumed_bias:.1f}\n"
            f"actual total bias = {actual_total_bias:.1f}"
        )

    ax_bias.annotate(
        bias_note,
        xy=(assumed_x, assumed_y),
        xytext=(assumed_x + 0.55, assumed_y + 0.9),
        arrowprops={"arrowstyle": "->"},
        fontsize=8.7,
    )

    ax_bias.set_xlabel(r"$1/\alpha_u$")
    ax_bias.set_ylabel(r"$\beta_u$")
    ax_bias.set_title("Bias sensitivity")
    ax_bias.text(
        0.02, 0.98,
        "Contours show assumed bias = βu / αu",
        transform=ax_bias.transAxes,
        fontsize=8.8,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.25", alpha=0.10),
    )

    # Bottom-right: possible true ATE
    possible_true_ate = observed_effect - assumed_bias_grid

    candidate_levels = sorted(set([
        -5, 0, 1, 2, 3, 4, 5,
        float(actual_true_ate),
        float(round(observed_effect, 1)),
    ]))
    levels_ate = [
        level for level in candidate_levels
        if possible_true_ate.min() < level < possible_true_ate.max()
    ]

    cs_ate = ax_ate.contour(
        X, B, possible_true_ate,
        levels=levels_ate,
        linewidths=1.6,
    )
    ax_ate.clabel(cs_ate, inline=True, fontsize=8, fmt="%.1f")

    ax_ate.scatter(
        [assumed_x], [assumed_y],
        s=85,
        edgecolors="black",
        linewidths=1.0,
        zorder=5,
    )

    assumed_corrected_ate = observed_effect - assumed_bias

    note = (
        f"hidden-U estimate = {observed_effect:.1f}\n"
        f"assumed bias = {assumed_bias:.1f}\n"
        f"assumed corrected ATE = {assumed_corrected_ate:.1f}\n"
        f"actual true ATE = {actual_true_ate:.1f}"
    )

    ax_ate.annotate(
        note,
        xy=(assumed_x, assumed_y),
        xytext=(assumed_x + 0.55, assumed_y + 0.9),
        arrowprops={"arrowstyle": "->"},
        fontsize=8.7,
    )

    ax_ate.set_xlabel(r"$1/\alpha_u$")
    ax_ate.set_ylabel(r"$\beta_u$")
    ax_ate.set_title("Possible true ATE")
    ax_ate.text(
        0.02, 0.98,
        "Contours show: true ATE = hidden-U estimate - assumed bias",
        transform=ax_ate.transAxes,
        fontsize=8.8,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.25", alpha=0.10),
    )


# ============================================================
# Example 1: book single-U
# ============================================================

def make_book_example():
    alpha_w = 1.2
    alpha_u = 0.8
    beta_w = 0.7
    beta_u = 1.6
    delta = 2.0

    rng = np.random.default_rng(7)
    n = 20_000

    W = rng.normal(size=n)
    U = rng.normal(size=n)

    T = alpha_w * W + alpha_u * U
    Y = beta_w * W + beta_u * U + delta * T

    # True ATE
    true_ate = delta

    # Hidden-U estimate from Y ~ T + W
    model_hidden = LinearRegression().fit(
        np.column_stack([T, W]),
        Y,
    )
    hidden_u_estimate = model_hidden.coef_[0]

    actual_bias = hidden_u_estimate - true_ate
    assumed_bias = beta_u / alpha_u

    print("\n" + "=" * 74)
    print("BOOK EXAMPLE — SINGLE HIDDEN U")
    print("=" * 74)
    print(f"True ATE delta                  = {true_ate:.6f}")
    print(f"Hidden-U estimate               = {hidden_u_estimate:.6f}")
    print(f"Actual bias                     = {actual_bias:.6f}")
    print(f"Assumed one-U bias beta/alpha   = {assumed_bias:.6f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    draw_graph_book(axes[0, 0], "True causal graph")
    add_text_box(
        axes[0, 0],
        [
            "True DGP:",
            "T = 1.2·W + 0.8·U",
            "Y = 0.7·W + 1.6·U + 2.0·T",
        ],
    )

    draw_graph_book(axes[0, 1], "Assumed graph used in sensitivity analysis")
    add_text_box(
        axes[0, 1],
        [
            "Assumed hidden-confounding model:",
            "one hidden confounder U",
            "assumed bias formula = βu / αu",
            "here: 1.6 / 0.8 = 2.0",
        ],
    )

    plot_assumed_one_u_sensitivity(
        axes[1, 0],
        axes[1, 1],
        observed_effect=hidden_u_estimate,
        assumed_alpha=alpha_u,
        assumed_beta=beta_u,
        actual_true_ate=true_ate,
        actual_total_bias=actual_bias,
    )

    fig.suptitle(
        "Example 1 — Book single-U confounding bias",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output = PLOT_DIR / "book_single_u_v2.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output}")


# ============================================================
# Example 2: true two-U, assumed one-U
# ============================================================

def make_two_u_example():
    alpha_w = 1.2

    alpha_1 = 0.8
    beta_1 = 1.6

    alpha_2 = 0.8
    beta_2 = 3.2

    beta_w = 0.7
    delta = 2.0

    rng = np.random.default_rng(13)
    n = 100_000

    W = rng.normal(size=n)
    U1 = rng.normal(size=n)
    U2 = rng.normal(size=n)

    T = alpha_w * W + alpha_1 * U1 + alpha_2 * U2
    Y = beta_w * W + beta_1 * U1 + beta_2 * U2 + delta * T

    true_ate = delta

    model_hidden = LinearRegression().fit(
        np.column_stack([T, W]),
        Y,
    )
    hidden_u_estimate = model_hidden.coef_[0]

    actual_total_bias = hidden_u_estimate - true_ate

    # Correct total bias formula for this exact two-U noiseless DGP
    correct_two_u_bias = (
        alpha_1 * beta_1 + alpha_2 * beta_2
    ) / (
        alpha_1**2 + alpha_2**2
    )

    # Misspecified one-U sensitivity assumption
    assumed_bias = beta_1 / alpha_1
    assumed_corrected_ate = hidden_u_estimate - assumed_bias

    print("\n" + "=" * 74)
    print("TWO-U EXAMPLE — TRUE DGP HAS TWO HIDDEN Us, BUT WE ASSUME ONE")
    print("=" * 74)
    print(f"True ATE delta                  = {true_ate:.6f}")
    print(f"Hidden-U estimate               = {hidden_u_estimate:.6f}")
    print(f"Actual total bias               = {actual_total_bias:.6f}")
    print(f"Correct 2-U bias formula        = {correct_two_u_bias:.6f}")
    print(f"Assumed one-U bias              = {assumed_bias:.6f}")
    print(f"Assumed corrected ATE           = {assumed_corrected_ate:.6f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    draw_graph_two_u_true(axes[0, 0], "True causal graph")
    add_text_box(
        axes[0, 0],
        [
            "True DGP:",
            "T = 1.2·W + 0.8·U1 + 0.8·U2",
            "Y = 0.7·W + 1.6·U1 + 3.2·U2 + 2.0·T",
            "true ATE = 2.0",
        ],
    )

    draw_graph_book(axes[0, 1], "Assumed graph used in sensitivity analysis")
    add_text_box(
        axes[0, 1],
        [
            "Wrong simplifying assumption:",
            "pretend there is only one hidden U",
            "and use bias = βu / αu",
            "chosen one-U values: βu=1.6, αu=0.8",
            "assumed bias = 2.0",
        ],
    )

    plot_assumed_one_u_sensitivity(
        axes[1, 0],
        axes[1, 1],
        observed_effect=hidden_u_estimate,
        assumed_alpha=alpha_1,
        assumed_beta=beta_1,
        actual_true_ate=true_ate,
        actual_total_bias=actual_total_bias,
    )

    fig.suptitle(
        "Example 2 — True two-U confounding, but sensitivity analysis assumes one U",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output = PLOT_DIR / "two_u_misspecified_v2.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output}")


if __name__ == "__main__":
    make_book_example()
    make_two_u_example()

    print("\nDone. Wrote:")
    print(PLOT_DIR / "book_single_u_v2.png")
    print(PLOT_DIR / "two_u_misspecified_v2.png")
