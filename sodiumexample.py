"""
Simulated sodium -> blood pressure causal regression example.

Variables:
    S = sodium
    A = age
    P = proteinuria
    Y = blood pressure

Models:
    Yhat = b + c_S S
    Yhat = b + c_S S + c_A A
    Yhat = b + c_S S + c_A A + c_P P

Produces:
    plots/binary_treatment_models.png
    plots/continuous_treatment_models.png
    plots/causal_summary.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# Configuration
# ============================================================

PLOT_DIR = Path("plots/sodium")
PLOT_DIR.mkdir(exist_ok=True)

TRUE_EFFECT = 1.05
N = 200

MODEL_SPECS = [
    (
        ["sodium"],
        r"$\hat{Y}=b+c_S S$",
    ),
    (
        ["sodium", "age"],
        r"$\hat{Y}=b+c_S S+c_A A$",
    ),
    (
        ["sodium", "age", "proteinuria"],
        r"$\hat{Y}=b+c_S S+c_A A+c_P P$",
    ),
]

SYMBOL = {
    "sodium": "S",
    "age": "A",
    "proteinuria": "P",
}

COEF_SYMBOL = {
    "sodium": "c_S",
    "age": "c_A",
    "proteinuria": "c_P",
}


# ============================================================
# Data generation
# ============================================================

def generate_data(
    n=N,
    seed=0,
    beta1=TRUE_EFFECT,
    alpha1=0.4,
    alpha2=0.3,
    binary_treatment=True,
    binary_cutoff=3.5,
):
    rng = np.random.default_rng(seed)

    # A
    age = rng.normal(65, 5, n)

    # A -> S
    sodium = age / 18 + rng.normal(size=n)

    if binary_treatment:
        if binary_cutoff is None:
            binary_cutoff = sodium.mean()

        sodium = (sodium > binary_cutoff).astype(int)

    # S -> Y and A -> Y
    blood_pressure = (
        beta1 * sodium
        + 2 * age
        + rng.normal(size=n)
    )

    # S -> P and Y -> P
    proteinuria = (
        alpha1 * sodium
        + alpha2 * blood_pressure
        + rng.normal(size=n)
    )

    return pd.DataFrame({
        "sodium": sodium,
        "age": age,
        "proteinuria": proteinuria,
        "blood_pressure": blood_pressure,
    })


# ============================================================
# Fit models once
# ============================================================

def fit_models(df):
    y = df["blood_pressure"]
    results = []

    for features, label in MODEL_SPECS:

        model = LinearRegression().fit(
            df[features],
            y,
        )

        results.append({
            "model": model,
            "features": features,
            "label": label,
        })

    return results


# ============================================================
# ATE estimate
# ============================================================

def estimate_ate(df, model, features):
    """
    Predict every individual twice:
        S = 1
        S = 0

    while keeping all other covariates at that individual's
    actual values.

    For these additive linear models this equals c_S.
    """

    X1 = df[features].copy()
    X0 = df[features].copy()

    X1["sodium"] = 1
    X0["sodium"] = 0

    return np.mean(
        model.predict(X1)
        - model.predict(X0)
    )


# ============================================================
# Print results
# ============================================================

def print_results(df, title, models):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    y = df["blood_pressure"]

    for result in models:

        model = result["model"]
        features = result["features"]
        label = result["label"]

        prediction = model.predict(
            df[features]
        )

        ate = estimate_ate(
            df,
            model,
            features,
        )

        print(f"\nModel: {label}")
        print(f"b   = {model.intercept_:.4f}")

        for feature, coef in zip(
            features,
            model.coef_,
        ):
            print(
                f"{COEF_SYMBOL[feature]:3s} = "
                f"{coef:.4f}"
            )

        print(f"ATE = {ate:.4f}")

        print(
            f"R²  = "
            f"{r2_score(y, prediction):.4f}"
        )

        print(
            f"MSE = "
            f"{mean_squared_error(y, prediction):.4f}"
        )


def print_binary_age_difference(df):

    a0 = df.loc[
        df["sodium"] == 0,
        "age"
    ].mean()

    a1 = df.loc[
        df["sodium"] == 1,
        "age"
    ].mean()

    diff = a1 - a0

    print("\nBinary treatment age groups")

    print(
        f"E[A | S=0] = {a0:.4f}"
    )

    print(
        f"E[A | S=1] = {a1:.4f}"
    )

    print(
        f"Age difference = {diff:.4f}"
    )

    print(
        f"Expected naive c_S ≈ "
        f"{TRUE_EFFECT:.2f} + 2({diff:.4f}) "
        f"= {TRUE_EFFECT + 2 * diff:.4f}"
    )


# ============================================================
# Plot helpers
# ============================================================

def make_grid(df, variable, n=200):

    if (
        variable == "sodium"
        and df["sodium"].nunique() <= 2
    ):
        return np.array([0, 1])

    return np.linspace(
        df[variable].quantile(0.01),
        df[variable].quantile(0.99),
        n,
    )


def jitter_binary(x, seed=0):

    x = np.asarray(x)

    if np.unique(x).size > 2:
        return x

    rng = np.random.default_rng(seed)

    return (
        x
        + rng.uniform(
            -0.025,
            0.025,
            len(x),
        )
    )


def projection(
    model,
    features,
    df,
    variable,
    grid,
):
    """
    Recalculate Yhat while varying one included variable.

    Every OTHER included variable is fixed at its sample mean.
    """

    X = pd.DataFrame({
        feature: np.full(
            len(grid),
            df[feature].mean(),
        )
        for feature in features
    })

    X[variable] = grid

    return model.predict(X)


def projection_equation(
    features,
    variable,
):
    """
    Example:

        model:
            Yhat = b + c_S S + c_A A

        plotting A:
            Yhat(A) = b + c_S Sbar + c_A A
    """

    terms = ["b"]

    for feature in features:

        c = COEF_SYMBOL[feature]
        x = SYMBOL[feature]

        if feature == variable:
            terms.append(f"{c}{x}")

        else:
            terms.append(
                rf"{c}\bar{{{x}}}"
            )

    lhs = (
        rf"\hat{{Y}}({SYMBOL[variable]})"
    )

    return (
        "$"
        + lhs
        + "="
        + "+".join(terms)
        + "$"
    )


# ============================================================
# Main regression figure
# ============================================================

def save_plot(
    df,
    title,
    filename,
    models,
):

    variables = [
        ("sodium", "Sodium S"),
        ("age", "Age A"),
        ("proteinuria", "Proteinuria P"),
    ]

    grids = {
        variable: make_grid(df, variable)
        for variable, _ in variables
    }

    fig, axes = plt.subplots(
        3,
        5,
        figsize=(24, 14),
        gridspec_kw={
            "width_ratios": [
                1, 1, 1, 1, 0.65
            ]
        },
    )

    fig.suptitle(
        f"{title}\n"
        f"Ground-truth causal effect "
        f"$S \\rightarrow Y$ = "
        f"{TRUE_EFFECT:.2f}",
        fontsize=17,
    )

    # ========================================================
    # One row per model
    # ========================================================

    for row, result in enumerate(models):

        model = result["model"]
        features = result["features"]
        label = result["label"]

        # ====================================================
        # Columns 0-2:
        # included variables vs ground-truth Y
        # ====================================================

        for col, (
            variable,
            xlabel,
        ) in enumerate(variables):

            ax = axes[row, col]

            # -----------------------------------------------
            # Do not plot variables absent from this model
            # -----------------------------------------------

            if variable not in features:
                ax.axis("off")
                continue

            grid = grids[variable]

            x = df[variable]

            if variable == "sodium":
                x = jitter_binary(x)

            # -----------------------------------------------
            # Scatter = ground-truth Y
            # -----------------------------------------------

            ax.scatter(
                x,
                df["blood_pressure"],
                alpha=0.45,
                s=16,
            )

            # -----------------------------------------------
            # Line = predicted Yhat
            #
            # Other included variables are held at means
            # -----------------------------------------------

            yhat = projection(
                model,
                features,
                df,
                variable,
                grid,
            )

            ax.plot(
                grid,
                yhat,
                linewidth=3,
            )

            equation = projection_equation(
                features,
                variable,
            )

            ax.set_title(
                f"Scatter: ground-truth $Y$\n"
                f"Line: predicted {equation}\n"
                f"other included variables "
                f"held at mean",
                fontsize=10,
            )

            ax.set_xlabel(xlabel)

            ax.set_ylabel(
                r"Ground-truth blood pressure $Y$"
            )

        # ====================================================
        # Column 3:
        # Ground-truth Y vs predicted Yhat
        # ====================================================

        ax = axes[row, 3]

        ground_truth = (
            df["blood_pressure"]
            .to_numpy()
        )

        # Every person's ACTUAL covariates are used here.
        # Nothing is held at a mean.
        predicted = model.predict(
            df[features]
        )

        ax.scatter(
            ground_truth,
            predicted,
            alpha=0.45,
            s=16,
        )

        lo = min(
            ground_truth.min(),
            predicted.min(),
        )

        hi = max(
            ground_truth.max(),
            predicted.max(),
        )

        # Perfect-prediction reference
        ax.plot(
            [lo, hi],
            [lo, hi],
            linewidth=2,
        )

        terms = [r"b", r"c_S S"]

        if "age" in features:
            terms.append(r"c_A A")

        if "proteinuria" in features:
            terms.append(r"c_P P")

        yhat_formula = r"$\hat{Y} = " + " + ".join(terms) + r"$"

        ax.set_title(
            r"Scatter: ground-truth $Y$ "
            r"vs predicted $\hat{Y}$"
            "\n"
            r"Line: perfect prediction "
            r"$\hat{Y}=Y$"
            "\n"
            + yhat_formula,
            fontsize=10,
        )

        ax.set_xlabel(
            r"Ground-truth $Y$"
        )

        ax.set_ylabel(
            r"Predicted $\hat{Y}$"
        )

        # ====================================================
        # Column 4:
        # Model equation + ATE
        # ====================================================

        ax = axes[row, 4]
        ax.axis("off")

        ate = estimate_ate(
            df,
            model,
            features,
        )

        ax.text(
            0.5,
            0.68,
            "Model",
            ha="center",
            fontsize=12,
        )

        ax.text(
            0.5,
            0.56,
            label,
            ha="center",
            fontsize=15,
        )

        ax.text(
            0.5,
            0.36,
            "Estimated ATE",
            ha="center",
            fontsize=12,
        )

        ax.text(
            0.5,
            0.25,
            f"{ate:.4f}",
            ha="center",
            fontsize=19,
        )

        ax.text(
            0.5,
            0.13,
            f"Error = "
            f"{ate - TRUE_EFFECT:+.4f}",
            ha="center",
            fontsize=11,
        )

    # ========================================================
    # Match scales only between plots that actually exist
    # ========================================================

    for col in range(4):

        active_axes = [
            axes[row, col]
            for row in range(3)
            if axes[row, col].axison
        ]

        if not active_axes:
            continue

        xmin = min(
            ax.get_xlim()[0]
            for ax in active_axes
        )

        xmax = max(
            ax.get_xlim()[1]
            for ax in active_axes
        )

        ymin = min(
            ax.get_ylim()[0]
            for ax in active_axes
        )

        ymax = max(
            ax.get_ylim()[1]
            for ax in active_axes
        )

        for ax in active_axes:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

    plt.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    output = PLOT_DIR / filename

    plt.savefig(
        output,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    print(f"\nSaved: {output}")


# ============================================================
# Causal summary figure
# ============================================================

def save_causal_summary_png(filename="causal_summary.png"):

    fig = plt.figure(figsize=(15, 8.5))

    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[3.2, 1.35],
        width_ratios=[1, 1.4],
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    # ========================================================
    # Left: true DAG
    # ========================================================

    ax1.axis("off")
    ax1.set_title(
        "True data-generating dependencies",
        fontsize=14,
    )

    pos = {
        "A": (0.15, 0.75),
        "S": (0.50, 0.75),
        "Y": (0.50, 0.30),
        "P": (0.85, 0.52),
    }

    radius = 0.10

    colors = {
        "A": "#d9ead3",
        "S": "#cfe2f3",
        "Y": "#f4cccc",
        "P": "#fce5cd",
    }

    labels = {
        "A": "A\n(age)",
        "S": "S\n(sodium)",
        "Y": "Y\n(blood pressure)",
        "P": "P\n(proteinuria)",
    }

    for node, (x, y) in pos.items():

        circle = plt.Circle(
            (x, y),
            radius,
            color=colors[node],
            ec="black",
            lw=1.5,
        )

        ax1.add_patch(circle)

        ax1.text(
            x,
            y,
            labels[node],
            ha="center",
            va="center",
            fontsize=10,
        )

    def arrow(a, b):

        x1, y1 = pos[a]
        x2, y2 = pos[b]

        dx = x2 - x1
        dy = y2 - y1
        dist = np.sqrt(dx**2 + dy**2)

        ux = dx / dist
        uy = dy / dist

        start = (
            x1 + radius * ux,
            y1 + radius * uy,
        )

        end = (
            x2 - radius * ux,
            y2 - radius * uy,
        )

        ax1.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(
                arrowstyle="->",
                lw=2,
            ),
        )

    arrow("A", "S")
    arrow("A", "Y")
    arrow("S", "Y")
    arrow("S", "P")
    arrow("Y", "P")

    ax1.text(
        0.03,
        0.03,
        "Backdoor path:  S ← A → Y\n"
        "Condition on A to block it.\n\n"
        "Collider/post-treatment path:  S → P ← Y\n"
        "Conditioning on P can introduce bias.",
        fontsize=10,
        va="bottom",
    )

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # ========================================================
    # Right: equations + correct adjustment
    # ========================================================

    ax2.axis("off")

    ax2.set_title(
        "Equations and correct adjustment",
        fontsize=14,
    )

    text = (
        r"Data-generating process" "\n"
        r"$S = \frac{A}{18} + e_S$" "\n"
        r"$Y = 1.05\,S + 2\,A + e_Y$" "\n"
        r"$P = 0.4\,S + 0.3\,Y + e_P$"
        "\n\n"

        r"Target causal effect" "\n"
        r"$ATE = E[Y(1)-Y(0)]$"
        "\n\n"

        r"Valid adjustment set for estimating $S \rightarrow Y$" "\n"
        r"$\{A\}$"
        "\n\n"

        r"Here, adjusting for $A$ means including it as a covariate" "\n"
        r"and estimating it alongside $S$:" "\n"
        r"$\hat{Y}=b+c_S S+c_A A$"
        "\n\n"

        r"For every individual, predict:" "\n"
        r"$\hat{Y}_i(S=1,A_i)$  and  "
        r"$\hat{Y}_i(S=0,A_i)$"
        "\n\n"

        r"Then average the individual contrasts:" "\n"
        r"$\widehat{ATE}="
        r"\frac{1}{n}\sum_i"
        r"\left[\hat{Y}_i(1,A_i)-\hat{Y}_i(0,A_i)\right]$"
    )

    ax2.text(
        0.02,
        0.98,
        text,
        va="top",
        fontsize=11.5,
    )

    # ========================================================
    # Bottom: TL;DR intuition
    # ========================================================

    ax3.axis("off")

    ax3.set_title(
        "  Intuition",
        fontsize=14,
        loc="left",
    )

    tldr = (
        r"$\bf{Y\sim S:}$  "
        "Because S is the only predictor, variation in Y associated with "
        "differences between the S groups is attributed to S. "
        "Since A is a common cause of S and Y, age-related variation is "
        r"partly absorbed into $c_S$, producing an overestimated ATE."
        "\n\n"

        r"$\bf{Y\sim S+A:}$  "
        "Conditioning on A lets the model separately account for "
        "age-related variation in Y, removing the confounding path "
        r"$S\leftarrow A\rightarrow Y$ and recovering the intended "
        "S–Y comparison."
        "\n\n"

        r"$\bf{Y\sim S+A+P:}$  "
        "P is downstream of S and Y. Conditioning on P introduces "
        "post-treatment/collider bias and lets P absorb variation in Y "
        r"that would otherwise contribute to $c_S$ and $c_A$. "
        "In this simulation this shifts the estimated effect of S downward, "
        "leading to an underestimated ATE."
    )

    ax3.text(
        0.01,
        0.90,
        tldr,
        va="top",
        fontsize=10.5,
        wrap=True,
    )

    plt.tight_layout()

    output = PLOT_DIR / filename

    plt.savefig(
        output,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    print(f"\nSaved: {output}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Binary treatment
    # --------------------------------------------------------

    binary_df = generate_data(
        binary_treatment=True,
        n=N,
        seed=0,
    )

    binary_models = fit_models(
        binary_df
    )

    print_binary_age_difference(
        binary_df
    )

    print_results(
        binary_df,
        "BINARY TREATMENT",
        binary_models,
    )

    save_plot(
        binary_df,
        "Binary Treatment",
        "binary_treatment_models.png",
        binary_models,
    )

    # --------------------------------------------------------
    # Continuous treatment
    # --------------------------------------------------------

    continuous_df = generate_data(
        binary_treatment=False,
        n=N,
        seed=0,
    )

    continuous_models = fit_models(
        continuous_df
    )

    print_results(
        continuous_df,
        "CONTINUOUS TREATMENT",
        continuous_models,
    )

    save_plot(
        continuous_df,
        "Continuous Treatment",
        "continuous_treatment_models.png",
        continuous_models,
    )

    save_causal_summary_png()