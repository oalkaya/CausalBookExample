"""
Augmented sodium -> blood pressure causal regression example.

Variables
---------
S = sodium treatment
A = age
B = BMI
P = proteinuria (post-treatment / collider-style variable)
Y = blood pressure

True binary-treatment DAG
-------------------------
A -> S <- B
A -> Y <- B
S -> Y
S -> P <- Y

Hence the total effect S -> Y has two open backdoor paths:
    S <- A -> Y
    S <- B -> Y
and the valid observed adjustment set is {A, B}.

For the binary treatment we also estimate a propensity score
    e(A, B) = P(S=1 | A, B)
and compare the direct adjustment model against a propensity-score model.

Produces
--------
plots/binary_treatment_models_two_confounders.png
plots/continuous_treatment_models_two_confounders.png
plots/propensity_score_diagnostics.png
plots/causal_summary_two_confounders.png
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

PLOT_DIR = Path("plots/propensity")
PLOT_DIR.mkdir(exist_ok=True)

TRUE_EFFECT = 1.05
N = 1000

SYMBOL = {
    "sodium": "S",
    "age": "A",
    "bmi": "B",
    "proteinuria": "P",
    "propensity_score": r"\hat e(A,B)",
}

COEF_SYMBOL = {
    "sodium": "c_S",
    "age": "c_A",
    "bmi": "c_B",
    "proteinuria": "c_P",
    "propensity_score": "c_e",
}


# ============================================================
# Data generation
# ============================================================

def generate_data(
    n=N,
    seed=0,
    beta1=TRUE_EFFECT,
    binary_treatment=True,
):
    """
    Generate a simple causal DGP with TWO required confounders.

    A = age and B = BMI both cause treatment S and outcome Y.
    Neither A nor B alone is a sufficient adjustment set.

    Binary treatment:
        e(A,B) = P(S=1 | A,B)
               = 0.50 + 0.18*A_z + 0.18*B_z

    A_z and B_z are bounded in [-1, 1], so the true propensity
    stays in [0.14, 0.86] and positivity is deliberately good.

    The outcome is chosen to be linear in A_z and B_z. Because
    the same standardized combination enters the propensity score,
    a simple linear outcome model using e(A,B) is also well behaved.
    This is intentionally pedagogical; in real data E[Y|S,e(X)]
    need not be linear in e(X).
    """
    rng = np.random.default_rng(seed)

    # Two pre-treatment confounders.
    age = rng.uniform(50, 80, n)
    bmi = rng.uniform(20, 35, n)

    # Convenient bounded standardized versions in [-1, 1].
    age_z = (age - 65.0) / 15.0
    bmi_z = (bmi - 27.5) / 7.5

    if binary_treatment:
        # A -> S and B -> S.
        true_propensity = 0.50 + 0.18 * age_z + 0.18 * bmi_z
        sodium = rng.binomial(1, true_propensity, size=n)
    else:
        # Continuous analogue used only to preserve the old example.
        # Standard binary propensity scores are not used here.
        true_propensity = np.full(n, np.nan)
        sodium = (
            3.5
            + 0.8 * age_z
            + 0.8 * bmi_z
            + rng.normal(0, 0.8, n)
        )

    # S -> Y, A -> Y, B -> Y.
    blood_pressure = (
        120.0
        + beta1 * sodium
        + 10.0 * age_z
        + 10.0 * bmi_z
        + rng.normal(0, 2.0, n)
    )

    # S -> P and Y -> P: P is post-treatment and a collider on S -> P <- Y.
    proteinuria = (
        0.4 * sodium
        + 0.3 * blood_pressure
        + rng.normal(0, 1.0, n)
    )

    return pd.DataFrame({
        "sodium": sodium,
        "age": age,
        "bmi": bmi,
        "proteinuria": proteinuria,
        "blood_pressure": blood_pressure,
        "true_propensity": true_propensity,
    })


# ============================================================
# Propensity score model
# ============================================================

def add_propensity_score(df):
    """
    Estimate e(A,B) = P(S=1 | A,B) and add it as one scalar column.

    We deliberately use a linear probability model here because the
    simulated true propensity is linear in A and B and bounded away
    from 0/1. This keeps the example parallel to the linear-regression
    examples from before.

    Logistic regression is much more common in real applications.
    """
    ps_model = LinearRegression().fit(
        df[["age", "bmi"]],
        df["sodium"],
    )

    estimated = ps_model.predict(df[["age", "bmi"]])
    estimated = np.clip(estimated, 0.01, 0.99) # to avoid issue with div by 0 when 1/e(X) or 1/(1-e(X)

    out = df.copy()
    out["propensity_score"] = estimated
    return out, ps_model


# ============================================================
# Model specifications
# ============================================================

def binary_model_specs():
    return [
        {
            "name": "Naive",
            "features": ["sodium"],
            "label": r"$\hat{Y}=b+c_S S$",
        },
        {
            "name": "Direct valid adjustment",
            "features": ["sodium", "age", "bmi"],
            "label": r"$\hat{Y}=b+c_S S+c_A A+c_B B$",
        },
        {
            "name": "Propensity-score adjustment",
            "features": ["sodium", "propensity_score"],
            "label": r"$\hat{Y}=b+c_S S+c_e\hat e(A,B)$",
        },
        {
            "name": "Bad post-treatment adjustment",
            "features": ["sodium", "age", "bmi", "proteinuria"],
            "label": r"$\hat{Y}=b+c_S S+c_A A+c_B B+c_P P$",
        },
    ]


def continuous_model_specs():
    return [
        {
            "name": "Naive",
            "features": ["sodium"],
            "label": r"$\hat{Y}=b+c_S S$",
        },
        {
            "name": "Direct valid adjustment",
            "features": ["sodium", "age", "bmi"],
            "label": r"$\hat{Y}=b+c_S S+c_A A+c_B B$",
        },
        {
            "name": "Bad post-treatment adjustment",
            "features": ["sodium", "age", "bmi", "proteinuria"],
            "label": r"$\hat{Y}=b+c_S S+c_A A+c_B B+c_P P$",
        },
    ]


# ============================================================
# Fit models
# ============================================================

def fit_models(df, specs):
    y = df["blood_pressure"]
    results = []

    for spec in specs:
        model = LinearRegression().fit(df[spec["features"]], y)
        results.append({
            **spec,
            "model": model,
        })

    return results


# ============================================================
# ATE estimate
# ============================================================

def estimate_ate(df, model, features):
    """
    Predict every individual twice, once with S=1 and once with S=0,
    while keeping all other included covariates at that individual's
    actual values.

    For these additive linear outcome models, this equals c_S.
    """
    X1 = df[features].copy()
    X0 = df[features].copy()
    X1["sodium"] = 1
    X0["sodium"] = 0

    return np.mean(model.predict(X1) - model.predict(X0))


# ============================================================
# Console summaries
# ============================================================

def print_results(df, title, models):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

    y = df["blood_pressure"]

    for result in models:
        model = result["model"]
        features = result["features"]
        prediction = model.predict(df[features])
        ate = estimate_ate(df, model, features)

        print(f"\n{result['name']}")
        print(f"  {result['label']}")
        print(f"  b   = {model.intercept_:.4f}")

        for feature, coef in zip(features, model.coef_):
            print(f"  {COEF_SYMBOL[feature]:3s} = {coef:.4f}")

        print(f"  ATE = {ate:.4f}")
        print(f"  ATE error = {ate - TRUE_EFFECT:+.4f}")
        print(f"  R²  = {r2_score(y, prediction):.4f}")
        print(f"  MSE = {mean_squared_error(y, prediction):.4f}")


def print_binary_group_differences(df):
    print("\nBinary treatment group differences")
    for variable, symbol in [("age", "A"), ("bmi", "B")]:
        x0 = df.loc[df["sodium"] == 0, variable].mean()
        x1 = df.loc[df["sodium"] == 1, variable].mean()
        print(f"  E[{symbol} | S=0] = {x0:.4f}")
        print(f"  E[{symbol} | S=1] = {x1:.4f}")
        print(f"  difference        = {x1 - x0:+.4f}\n")


def print_propensity_diagnostics(df, ps_model):
    corr = np.corrcoef(
        df["true_propensity"],
        df["propensity_score"],
    )[0, 1]
    mse = mean_squared_error(
        df["true_propensity"],
        df["propensity_score"],
    )

    print("\nPropensity model")
    print("  e(A,B) = P(S=1 | A,B)")
    print(f"  intercept = {ps_model.intercept_:.4f}")
    print(f"  age coef  = {ps_model.coef_[0]:.4f}")
    print(f"  BMI coef  = {ps_model.coef_[1]:.4f}")
    print(f"  corr(true e, estimated e) = {corr:.4f}")
    print(f"  MSE(true e, estimated e)  = {mse:.6f}")


# ============================================================
# Plot helpers
# ============================================================

def make_grid(df, variable, n=200):
    if variable == "sodium" and df["sodium"].nunique() <= 2:
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
    return x + rng.uniform(-0.025, 0.025, len(x))


def projection(model, features, df, variable, grid):
    """
    Recalculate Yhat while varying one included variable.
    Every OTHER included variable is fixed at its sample mean.
    """
    X = pd.DataFrame({
        feature: np.full(len(grid), df[feature].mean())
        for feature in features
    })
    X[variable] = grid
    return model.predict(X)


def projection_equation(features, variable):
    terms = ["b"]

    for feature in features:
        c = COEF_SYMBOL[feature]
        x = SYMBOL[feature]
        if feature == variable:
            terms.append(f"{c}{x}")
        else:
            terms.append(rf"{c}\overline{{{x}}}")

    lhs = rf"\hat{{Y}}({SYMBOL[variable]})"
    return "$" + lhs + "=" + "+".join(terms) + "$"


# ============================================================
# Main regression figure
# ============================================================

def save_model_plot(df, title, filename, models, binary=False):
    variables = [
        ("sodium", "Sodium S"),
        ("age", "Age A"),
        ("bmi", "BMI B"),
        ("proteinuria", "Proteinuria P"),
    ]

    if binary and "propensity_score" in df.columns:
        variables.append(("propensity_score", r"Propensity score $\hat e(A,B)$"))

    n_rows = len(models)
    n_cols = len(variables) + 2

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.4 * n_cols, 3.9 * n_rows),
        squeeze=False,
        gridspec_kw={
            "width_ratios": [1] * len(variables) + [1, 0.72],
        },
    )

    fig.suptitle(
        f"{title}\nGround-truth causal effect S → Y = {TRUE_EFFECT:.2f}",
        fontsize=17,
    )

    grids = {
        variable: make_grid(df, variable)
        for variable, _ in variables
    }

    for row, result in enumerate(models):
        model = result["model"]
        features = result["features"]
        label = result["label"]

        # Feature projection panels.
        for col, (variable, xlabel) in enumerate(variables):
            ax = axes[row, col]

            if variable not in features:
                ax.axis("off")
                continue

            grid = grids[variable]
            x = df[variable]
            if variable == "sodium" and binary:
                x = jitter_binary(x, seed=row)

            ax.scatter(
                x,
                df["blood_pressure"],
                alpha=0.25,
                s=12,
            )

            yhat = projection(
                model,
                features,
                df,
                variable,
                grid,
            )
            ax.plot(grid, yhat, linewidth=3)

            equation = projection_equation(features, variable)
            ax.set_title(
                "Scatter: observed Y\n"
                f"Line: {equation}\n"
                "other included variables held at mean",
                fontsize=9,
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Blood pressure Y")

        # Actual Y vs predicted Yhat.
        ax = axes[row, len(variables)]
        actual = df["blood_pressure"].to_numpy()
        predicted = model.predict(df[features])
        ax.scatter(actual, predicted, alpha=0.25, s=12)

        lo = min(actual.min(), predicted.min())
        hi = max(actual.max(), predicted.max())
        ax.plot([lo, hi], [lo, hi], linewidth=2)
        ax.set_title(
            "Observed Y vs predicted $\\hat{Y}$\n"
            "line = perfect prediction",
            fontsize=9,
        )
        ax.set_xlabel("Observed Y")
        ax.set_ylabel(r"Predicted $\hat{Y}$")

        # Model + ATE summary.
        ax = axes[row, len(variables) + 1]
        ax.axis("off")
        ate = estimate_ate(df, model, features)
        prediction = model.predict(df[features])

        ax.text(0.5, 0.91, result["name"], ha="center", va="top", fontsize=12)
        ax.text(0.5, 0.76, label, ha="center", va="top", fontsize=13)
        ax.text(0.5, 0.56, "Estimated ATE", ha="center", fontsize=11)
        ax.text(0.5, 0.47, f"{ate:.4f}", ha="center", fontsize=19)
        ax.text(
            0.5,
            0.37,
            f"error = {ate - TRUE_EFFECT:+.4f}",
            ha="center",
            fontsize=10,
        )
        ax.text(
            0.5,
            0.22,
            f"R² = {r2_score(actual, prediction):.3f}\n"
            f"MSE = {mean_squared_error(actual, prediction):.3f}",
            ha="center",
            fontsize=10,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output = PLOT_DIR / filename
    plt.savefig(output, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")


# ============================================================
# Propensity diagnostics
# ============================================================

def save_propensity_diagnostics(df, filename="propensity_score_diagnostics.png"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # True vs estimated propensity.
    axes[0].scatter(
        df["true_propensity"],
        df["propensity_score"],
        alpha=0.30,
        s=16,
    )
    lo = min(df["true_propensity"].min(), df["propensity_score"].min())
    hi = max(df["true_propensity"].max(), df["propensity_score"].max())
    axes[0].plot([lo, hi], [lo, hi], linewidth=2)
    axes[0].set_xlabel("True propensity e(A,B)")
    axes[0].set_ylabel(r"Estimated propensity $\hat e(A,B)$")
    axes[0].set_title("Propensity model accuracy")

    # Distribution / overlap by treatment group.
    axes[1].hist(
        df.loc[df["sodium"] == 0, "propensity_score"],
        bins=20,
        alpha=0.55,
        label="S=0",
    )
    axes[1].hist(
        df.loc[df["sodium"] == 1, "propensity_score"],
        bins=20,
        alpha=0.55,
        label="S=1",
    )
    axes[1].set_xlabel(r"Estimated propensity $\hat e(A,B)$")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Overlap / positivity")
    axes[1].legend()

    # Treatment rate by propensity-score bin.
    tmp = df.copy()
    tmp["ps_bin"] = pd.qcut(
        tmp["propensity_score"],
        q=10,
        duplicates="drop",
    )
    grouped = tmp.groupby("ps_bin", observed=True).agg(
        mean_ps=("propensity_score", "mean"),
        treated_rate=("sodium", "mean"),
    )
    axes[2].scatter(grouped["mean_ps"], grouped["treated_rate"], s=45)
    axes[2].plot([0, 1], [0, 1], linewidth=2)
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel(r"Mean estimated $\hat e(A,B)$ in bin")
    axes[2].set_ylabel("Observed treatment rate")
    axes[2].set_title("Calibration by propensity decile")

    fig.suptitle(
        "Propensity score: compressing the valid adjustment set {A, B} to one scalar",
        fontsize=15,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    output = PLOT_DIR / filename
    plt.savefig(output, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")


# ============================================================
# Causal summary figure
# ============================================================

def save_causal_summary_png(filename="causal_summary_two_confounders.png"):
    fig = plt.figure(figsize=(17, 9.5))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[3.4, 1.7],
        width_ratios=[1, 1.45],
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    # --------------------------
    # Left: true DAG
    # --------------------------
    ax1.axis("off")
    ax1.set_title("True data-generating dependencies", fontsize=14)

    pos = {
        "A": (0.12, 0.78),
        "B": (0.12, 0.32),
        "S": (0.50, 0.62),
        "Y": (0.77, 0.62),
        "P": (0.77, 0.22),
    }
    labels = {
        "A": "A\n(age)",
        "B": "B\n(BMI)",
        "S": "S\n(sodium)",
        "Y": "Y\n(BP)",
        "P": "P\n(proteinuria)",
    }
    radius = 0.09

    for node, (x, y) in pos.items():
        circle = plt.Circle((x, y), radius, ec="black", fc="white", lw=1.5)
        ax1.add_patch(circle)
        ax1.text(x, y, labels[node], ha="center", va="center", fontsize=10)

    def arrow(a, b):
        x1, y1 = pos[a]
        x2, y2 = pos[b]
        dx = x2 - x1
        dy = y2 - y1
        dist = np.sqrt(dx**2 + dy**2)
        ux = dx / dist
        uy = dy / dist
        start = (x1 + radius * ux, y1 + radius * uy)
        end = (x2 - radius * ux, y2 - radius * uy)
        ax1.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="->", lw=2),
        )

    for edge in [
        ("A", "S"),
        ("A", "Y"),
        ("B", "S"),
        ("B", "Y"),
        ("S", "Y"),
        ("S", "P"),
        ("Y", "P"),
    ]:
        arrow(*edge)

    ax1.text(
        0.02,
        0.02,
        "Open backdoor paths:\n"
        "  S ← A → Y\n"
        "  S ← B → Y\n\n"
        "Valid adjustment set: {A, B}\n"
        "Neither {A} nor {B} alone is sufficient.\n\n"
        "P is post-treatment and a collider on S → P ← Y.",
        fontsize=10,
        va="bottom",
    )
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # --------------------------
    # Right: equations
    # --------------------------
    ax2.axis("off")
    ax2.set_title("Direct adjustment vs propensity-score adjustment", fontsize=14)

    text = (
        "Binary treatment mechanism\n"
        r"$e(A,B)=P(S=1\mid A,B)$" "\n"
        r"$e(A,B)=0.50+0.18A_z+0.18B_z$" "\n\n"
        "Outcome mechanism\n"
        r"$Y=120+1.05S+10A_z+10B_z+e_Y$" "\n\n"
        "Direct adjustment\n"
        r"$\hat Y=b+c_SS+c_AA+c_BB$" "\n"
        r"$\widehat{ATE}=\frac{1}{n}\sum_i[\hat Y_i(1,A_i,B_i)-\hat Y_i(0,A_i,B_i)]$"
        "\n\n"
        "Propensity-score route\n"
        r"$(A,B)\longrightarrow \hat e(A,B)$" "\n"
        r"$\hat Y=b+c_SS+c_e\hat e(A,B)$" "\n"
        r"$\widehat{ATE}=\frac{1}{n}\sum_i[\hat Y_i(1,\hat e_i)-\hat Y_i(0,\hat e_i)]$"
    )
    ax2.text(0.02, 0.98, text, va="top", fontsize=11.5)

    # --------------------------
    # Bottom: intuition
    # --------------------------
    ax3.axis("off")
    ax3.set_title("Intuition", fontsize=14, loc="left")
    tldr = (
        r"$\bf{Y\sim S:}$ "
        "A and B both shift treatment assignment and blood pressure, so the treatment coefficient absorbs confounding from both backdoor paths.\n\n"
        r"$\bf{Y\sim S+A+B:}$ "
        "Conditioning on both pre-treatment confounders blocks S←A→Y and S←B→Y and recovers the intended causal comparison.\n\n"
        r"$\bf{Y\sim S+\hat e(A,B):}$ "
        "The propensity model compresses the two-variable adjustment information into one scalar. In this deliberately simple DGP, a linear outcome model in the propensity score is sufficient.\n\n"
        r"$\bf{Y\sim S+A+B+P:}$ "
        "P is downstream of S and Y. Adding it can improve prediction while biasing the treatment-effect estimate."
    )
    ax3.text(0.01, 0.93, tldr, va="top", fontsize=10.7, wrap=True)

    plt.tight_layout()
    output = PLOT_DIR / filename
    plt.savefig(output, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # --------------------------------------------------------
    # Binary treatment + propensity score
    # --------------------------------------------------------
    binary_df = generate_data(
        binary_treatment=True,
        n=N,
        seed=0,
    )
    binary_df, ps_model = add_propensity_score(binary_df)

    binary_models = fit_models(
        binary_df,
        binary_model_specs(),
    )

    print_binary_group_differences(binary_df)
    print_propensity_diagnostics(binary_df, ps_model)
    print_results(binary_df, "BINARY TREATMENT", binary_models)

    save_model_plot(
        binary_df,
        "Binary Treatment: direct adjustment vs propensity-score adjustment",
        "binary_treatment_models_two_confounders.png",
        binary_models,
        binary=True,
    )

    save_propensity_diagnostics(binary_df)

    # --------------------------------------------------------
    # Continuous treatment: same old idea, now with A and B
    # --------------------------------------------------------
    continuous_df = generate_data(
        binary_treatment=False,
        n=N,
        seed=0,
    )

    continuous_models = fit_models(
        continuous_df,
        continuous_model_specs(),
    )

    # Cont case does not have the propensity score, cant just use logits for a cont S.
    print_results(
        continuous_df,
        "CONTINUOUS TREATMENT",
        continuous_models,
    )

    save_model_plot(
        continuous_df,
        "Continuous Treatment: two-confounder adjustment",
        "continuous_treatment_models_two_confounders.png",
        continuous_models,
        binary=False,
    )

    save_causal_summary_png()