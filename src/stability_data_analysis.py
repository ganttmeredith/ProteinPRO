"""
Stability data analysis pipeline for polymer-protein formulation experiments.

Processes Excel uploads (e.g. Bayesian Optimization round data) and produces
automated figures: distribution by round, improvement trends, enrichment,
fold-change, best-so-far discovery, PCA/UMAP projections, correlation heatmap,
stacked composition bars, convex hull, and SHAP. Adapted from gen-on-gen lipase pipeline.
"""

import io
import os
import re
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Optional: seaborn for heatmap, umap for UMAP, shap for interpretability
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# Styling (Rutgers-inspired)
RUTGERS_RED = "#CC0033"
RUTGERS_DARK = "#111111"
RUTGERS_GRAY = "#6E6E6E"
RUTGERS_LIGHT = "#D9D9D9"
RUTGERS_SOFT = "#F6D6DE"
RUTGERS_SOFT3 = "#EAA5B8"

PERF_CANDIDATES = [
    "Average_REA_across_days",
    "Average REA across days",
    "Avg_REA",
    "Average_REA",
    "REA",
    "Performance",
]
STD_CANDIDATES = ["STD", "Std", "std", "Average_STD", "Run1 STD", "Run2 STD"]
MONOMER_COLS = ["DEAEMA", "HPMA", "BMA", "MMA", "DMAPMA", "PEGMA", "SPMA", "TMAEMA", "EHMA", "GMA"]
DP_COL = "Degree of Polymerization"


def detect_round_number(filename: str) -> Optional[int]:
    """Extract round number from filename like '5_1_Lip_Round1_run1_to_2combinedREA_08032025.xlsx'."""
    match = re.search(r"round\s*[_\- ]?(\d+)", filename.lower())
    return int(match.group(1)) if match else None


def read_round_file(
    file_bytes: bytes,
    filename: str,
    round_override: Optional[int] = None,
) -> pd.DataFrame:
    """
    Read one Excel file and standardize columns.
    file_bytes: raw file content (from Streamlit upload).
    filename: original name (for round detection).
    round_override: if set, use this instead of detecting from filename.
    """
    df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    perf_col = None
    for c in PERF_CANDIDATES:
        if c in df.columns:
            perf_col = c
            break
    if perf_col is None:
        raise ValueError(
            f"Could not find performance column. Looked for: {PERF_CANDIDATES}"
        )

    std_col = None
    for c in STD_CANDIDATES:
        if c in df.columns:
            std_col = c
            break

    round_num = round_override if round_override is not None else detect_round_number(filename)
    if round_num is None:
        round_num = 1

    temp = df.copy()
    temp["Round"] = round_num
    temp["Performance"] = pd.to_numeric(temp[perf_col], errors="coerce")

    if std_col:
        temp["STD_Used"] = pd.to_numeric(temp[std_col], errors="coerce")
    else:
        temp["STD_Used"] = np.nan

    keep_cols = ["Round", "Performance", "STD_Used"]
    optional = [DP_COL] + MONOMER_COLS
    for c in optional:
        if c in temp.columns:
            keep_cols.append(c)
    temp = temp[[c for c in keep_cols if c in temp.columns]].copy()
    temp["Source_File"] = filename
    return temp


def _shannon_entropy(vals):
    """Shannon entropy of positive values (for composition diversity)."""
    vals = np.array(vals, dtype=float)
    vals = vals[vals > 0]
    if len(vals) == 0:
        return 0.0
    p = vals / vals.sum()
    return float(-(p * np.log2(p + 1e-10)).sum())


def bootstrap_ci_mean(arr, n_boot=3000, ci=95, random_state=42):
    """Bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(random_state)
    arr = np.array(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    if len(arr) == 1:
        return arr[0], arr[0]
    means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    lower = np.percentile(means, (100 - ci) / 2)
    upper = np.percentile(means, 100 - (100 - ci) / 2)
    return lower, upper


def run_analysis(
    data: pd.DataFrame,
    performance_label: str = "Average REA Across Days",
) -> Tuple[pd.DataFrame, List[Tuple[str, bytes]]]:
    """
    Run the full analysis pipeline on combined round data.
    Returns (summary_df, list of (title, png_bytes) for each figure).
    """
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "grid.alpha": 0.25,
    })

    rounds = sorted(data["Round"].dropna().unique())
    if len(rounds) == 0:
        raise ValueError("No rounds found in data.")

    # 1. Round summary
    summary_rows = []
    for r in rounds:
        df_r = data[data["Round"] == r].copy()
        perf = df_r["Performance"].dropna().values
        top10_n = max(1, int(np.ceil(len(perf) * 0.10)))
        top25_n = max(1, int(np.ceil(len(perf) * 0.25)))
        perf_sorted = np.sort(perf)[::-1]
        mean_ci_low, mean_ci_high = bootstrap_ci_mean(perf)
        summary_rows.append({
            "Round": r,
            "N": len(perf),
            "Mean_REA": np.mean(perf),
            "Median_REA": np.median(perf),
            "Std_REA": np.std(perf, ddof=1) if len(perf) > 1 else 0.0,
            "Best_REA": np.max(perf),
            "Top10pct_Mean_REA": np.mean(perf_sorted[:top10_n]),
            "Top25pct_Mean_REA": np.mean(perf_sorted[:top25_n]),
            "Mean_CI_Lower": mean_ci_low,
            "Mean_CI_Upper": mean_ci_high,
        })
    summary = pd.DataFrame(summary_rows).sort_values("Round").reset_index(drop=True)

    for metric in ["Mean_REA", "Median_REA", "Best_REA", "Top10pct_Mean_REA", "Top25pct_Mean_REA"]:
        summary[f"{metric}_Delta"] = summary[metric].diff()
        summary[f"{metric}_PctChange"] = summary[metric].pct_change() * 100

    round_perf = {
        int(r): data.loc[data["Round"] == r, "Performance"].dropna().sort_values(ascending=False).reset_index(drop=True)
        for r in rounds
    }
    figures: List[Tuple[str, bytes]] = []

    def save_fig_bytes(fig, title: str):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=600, bbox_inches="tight", facecolor="white")
        buf.seek(0)
        figures.append((title, buf.read()))
        plt.close(fig)

    # Plot 1: Distribution by round
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_data = [data.loc[data["Round"] == r, "Performance"].dropna().values for r in rounds]
    bp = ax.boxplot(plot_data, patch_artist=True, labels=[f"Round {int(r)}" for r in rounds], showfliers=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(RUTGERS_SOFT)
        patch.set_alpha(0.7)
    rng = np.random.default_rng(123)
    for i, r in enumerate(rounds, start=1):
        y = data.loc[data["Round"] == r, "Performance"].dropna().values
        x = rng.normal(i, 0.05, size=len(y))
        ax.scatter(x, y, alpha=0.45, s=25)
    ax.set_title("Performance Distribution Across Rounds")
    ax.set_xlabel("Bayesian Optimization Round")
    ax.set_ylabel(performance_label)
    ax.grid(True, alpha=0.25)
    save_fig_bytes(fig, "Distribution by round")

    # Plot 2: Mean / median / best improvement
    fig, ax = plt.subplots(figsize=(10, 5))
    x = summary["Round"].values
    ax.plot(x, summary["Mean_REA"].values, marker="o", linewidth=2.5, label="Mean REA")
    ax.plot(x, summary["Median_REA"].values, marker="s", linewidth=2.5, label="Median REA")
    ax.plot(x, summary["Best_REA"].values, marker="^", linewidth=2.5, label="Best REA")
    ax.fill_between(x, summary["Mean_CI_Lower"].values, summary["Mean_CI_Upper"].values, alpha=0.18, label="Mean 95% CI")
    for xi, yi in zip(x, summary["Mean_REA"].values):
        ax.text(xi, yi, f"{yi:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Generation-on-Generation Improvement")
    ax.set_xlabel("Bayesian Optimization Round")
    ax.set_ylabel("REA")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    save_fig_bytes(fig, "Improvement trends")

    # Plot 3: Top fraction enrichment
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(summary["Round"], summary["Top10pct_Mean_REA"], marker="o", linewidth=2.5, label="Top 10% Mean")
    ax.plot(summary["Round"], summary["Top25pct_Mean_REA"], marker="s", linewidth=2.5, label="Top 25% Mean")
    ax.set_title("Enrichment of High-Performing Polymers")
    ax.set_xlabel("Bayesian Optimization Round")
    ax.set_ylabel("Top-Fraction Mean REA")
    ax.set_xticks(summary["Round"])
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    save_fig_bytes(fig, "Top fraction enrichment")

    # Plot 4: Percent change (only if multiple rounds)
    if len(rounds) >= 2:
        delta_df = summary.dropna(subset=["Mean_REA_PctChange"])
        if len(delta_df) > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(
                [f"R{int(r-1)}→R{int(r)}" for r in delta_df["Round"]],
                delta_df["Mean_REA_PctChange"],
            )
            for bar, val in zip(bars, delta_df["Mean_REA_PctChange"]):
                y = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, y, f"{val:+.1f}%", ha="center", va="bottom" if y >= 0 else "top", fontsize=10)
            ax.axhline(0, linewidth=1)
            ax.set_title("Round-to-Round % Change in Mean Performance")
            ax.set_xlabel("Transition")
            ax.set_ylabel("% Change")
            ax.grid(True, axis="y", alpha=0.25)
            save_fig_bytes(fig, "Percent change round-to-round")

    # Plot 5: Best-so-far discovery
    best_by_round = summary[["Round", "Best_REA"]].copy()
    best_by_round["Best_So_Far"] = best_by_round["Best_REA"].cummax()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(best_by_round["Round"], best_by_round["Best_REA"], marker="o", color=RUTGERS_GRAY, linewidth=2, label="Best in round")
    ax.plot(best_by_round["Round"], best_by_round["Best_So_Far"], marker="o", color=RUTGERS_RED, linewidth=2.5, label="Best discovered so far")
    ax.set_title("Best-So-Far Discovery Across Rounds")
    ax.set_xlabel("Bayesian Optimization Round")
    ax.set_ylabel("Best REA")
    ax.set_xticks(best_by_round["Round"])
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    save_fig_bytes(fig, "Best-so-far discovery")

    # Plot 6: Fold-change vs Round 1 (only if multiple rounds)
    if len(rounds) >= 2:
        baseline_mean = summary.loc[summary["Round"] == rounds[0], "Mean_REA"].values[0]
        baseline_top10 = summary.loc[summary["Round"] == rounds[0], "Top10pct_Mean_REA"].values[0]
        baseline_best = summary.loc[summary["Round"] == rounds[0], "Best_REA"].values[0]
        fold_df = summary.copy()
        fold_df["Mean_Fold"] = fold_df["Mean_REA"] / baseline_mean
        fold_df["Top10_Fold"] = fold_df["Top10pct_Mean_REA"] / baseline_top10
        fold_df["Best_Fold"] = fold_df["Best_REA"] / baseline_best
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(fold_df["Round"], fold_df["Mean_Fold"], marker="o", color=RUTGERS_RED, linewidth=2.5, label="Mean")
        ax.plot(fold_df["Round"], fold_df["Top10_Fold"], marker="s", color=RUTGERS_GRAY, linewidth=2.5, label="Top 10%")
        ax.plot(fold_df["Round"], fold_df["Best_Fold"], marker="^", color=RUTGERS_DARK, linewidth=2.5, label="Best")
        ax.axhline(1.0, linestyle="--", color=RUTGERS_LIGHT, linewidth=1.5)
        ax.set_title("Fold-Change vs Round 1")
        ax.set_xlabel("Bayesian Optimization Round")
        ax.set_ylabel("Fold-change")
        ax.set_xticks(fold_df["Round"])
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        save_fig_bytes(fig, "Fold-change vs Round 1")

    # Plot 7: Rank-ordered curves
    fig, ax = plt.subplots(figsize=(10, 5))
    curve_colors = {1: RUTGERS_LIGHT, 2: RUTGERS_RED, 3: RUTGERS_GRAY, 4: RUTGERS_DARK}
    for r in rounds:
        s = round_perf[int(r)].reset_index(drop=True)
        ranks = np.arange(1, len(s) + 1)
        ax.plot(ranks, s.values, linewidth=2, color=curve_colors.get(int(r), RUTGERS_GRAY), label=f"Round {int(r)}")
    ax.set_title("Rank-Ordered Performance by Round")
    ax.set_xlabel("Rank within round")
    ax.set_ylabel(performance_label)
    ax.set_xlim(1, max(len(v) for v in round_perf.values()))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    save_fig_bytes(fig, "Rank-ordered curves")

    # Plot 8: Top 5 per round
    top_n = 5
    top_rows = []
    for r in sorted(data["Round"].unique()):
        df_r = data[data["Round"] == r].copy().sort_values("Performance", ascending=False).head(top_n)
        df_r["Rank_In_Round"] = np.arange(1, len(df_r) + 1)
        top_rows.append(df_r)
    top_df = pd.concat(top_rows, ignore_index=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in sorted(top_df["Round"].unique()):
        sub = top_df[top_df["Round"] == r].sort_values("Rank_In_Round")
        ax.plot(sub["Rank_In_Round"], sub["Performance"], marker="o", linewidth=2, label=f"Round {int(r)}")
    ax.set_title(f"Top {top_n} Performers Within Each Round")
    ax.set_xlabel("Rank Within Round")
    ax.set_ylabel(performance_label)
    ax.set_xticks(range(1, top_n + 1))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    save_fig_bytes(fig, "Top 5 per round")

    # --- Extended analysis: feature space, PCA, UMAP, heatmap, SHAP ---
    data = data.copy()
    monomer_cols_present = [c for c in MONOMER_COLS if c in data.columns]
    feature_cols = list(monomer_cols_present)
    if DP_COL in data.columns:
        feature_cols.append(DP_COL)
    # Derived features (only if component cols exist)
    if all(c in data.columns for c in ["BMA", "MMA", "EHMA"]):
        data["_Hydrophobic_sum"] = data[["BMA", "MMA", "EHMA"]].sum(axis=1)
        feature_cols.append("_Hydrophobic_sum")
    if all(c in data.columns for c in ["DEAEMA", "DMAPMA", "TMAEMA"]):
        data["_Cationic_sum"] = data[["DEAEMA", "DMAPMA", "TMAEMA"]].sum(axis=1)
        feature_cols.append("_Cationic_sum")
    if "SPMA" in data.columns:
        data["_Anionic_sum"] = data["SPMA"]
        feature_cols.append("_Anionic_sum")
    if all(c in data.columns for c in ["HPMA", "PEGMA", "GMA"]):
        data["_Neutral_hydrophilic_sum"] = data[["HPMA", "PEGMA", "GMA"]].sum(axis=1)
        feature_cols.append("_Neutral_hydrophilic_sum")
    if monomer_cols_present:
        data["_Nonzero_monomers"] = (data[monomer_cols_present] > 0).sum(axis=1)
        data["_Composition_entropy"] = data[monomer_cols_present].apply(_shannon_entropy, axis=1)
        feature_cols.extend(["_Nonzero_monomers", "_Composition_entropy"])

    if len(feature_cols) >= 2:
        X_raw = data[feature_cols].fillna(0).values
        valid_mask = ~np.any(np.isnan(X_raw), axis=1)
        if valid_mask.sum() >= 10:
            X_clean = X_raw[valid_mask]
            scaler_f = StandardScaler()
            X_scaled = scaler_f.fit_transform(X_clean)
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            data = data.copy()
            data.loc[valid_mask, "PCA1"] = X_pca[:, 0]
            data.loc[valid_mask, "PCA2"] = X_pca[:, 1]
            data_valid = data[valid_mask].copy()

            # Correlation heatmap
            heat_cols = [c for c in feature_cols + ["Performance"] if c in data_valid.columns]
            if len(heat_cols) >= 2:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    corr = data_valid[heat_cols].corr()
                    short_labels = [c.replace("_", " ")[:12] for c in corr.columns]
                    if HAS_SEABORN:
                        corr_plot = corr.rename(columns=dict(zip(corr.columns, short_labels)), index=dict(zip(corr.index, short_labels)))
                        sns.heatmap(corr_plot, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, vmin=-1, vmax=1)
                    else:
                        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
                        ax.set_xticks(range(len(short_labels)))
                        ax.set_yticks(range(len(short_labels)))
                        ax.set_xticklabels(short_labels, rotation=45, ha="right")
                        ax.set_yticklabels(short_labels)
                        plt.colorbar(im, ax=ax)
                    ax.set_title("Feature Correlation Matrix")
                    save_fig_bytes(fig, "Correlation heatmap")

            # Stacked bar: average monomer composition by round
            if monomer_cols_present:
                stack_df = data_valid.groupby("Round")[monomer_cols_present].mean()
                fig, ax = plt.subplots(figsize=(10, 5))
                bottom = np.zeros(len(stack_df))
                for mon in monomer_cols_present:
                    vals = stack_df[mon].values
                    ax.bar(stack_df.index.astype(str), vals, bottom=bottom, label=mon)
                    bottom += vals
                ax.set_title("Average Monomer Composition by Round")
                ax.set_ylabel("Mean fraction")
                ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
                ax.grid(True, axis="y", alpha=0.25)
                save_fig_bytes(fig, "Stacked composition by round")

            round_color_map = {1: RUTGERS_LIGHT, 2: RUTGERS_RED, 3: RUTGERS_SOFT3, 4: RUTGERS_DARK}

            # PCA by round
            fig, ax = plt.subplots(figsize=(10, 6))
            for r in sorted(data_valid["Round"].unique()):
                sub = data_valid[data_valid["Round"] == r]
                ax.scatter(sub["PCA1"], sub["PCA2"], s=45, alpha=0.78, color=round_color_map.get(int(r), RUTGERS_GRAY), label=f"Round {int(r)}")
            ax.set_title("Design Space Projection by Round (PCA)")
            ax.set_xlabel("PCA1")
            ax.set_ylabel("PCA2")
            ax.legend(frameon=False)
            ax.grid(True, alpha=0.25)
            save_fig_bytes(fig, "PCA by round")

            # PCA by performance
            fig, ax = plt.subplots(figsize=(10, 6))
            sc = ax.scatter(data_valid["PCA1"], data_valid["PCA2"], c=data_valid["Performance"], s=50, cmap="Reds", alpha=0.86, edgecolor="k", linewidth=0.2)
            ax.set_title("Design Space Colored by Observed REA (PCA)")
            ax.set_xlabel("PCA1")
            ax.set_ylabel("PCA2")
            plt.colorbar(sc, ax=ax, label="Observed REA")
            ax.grid(True, alpha=0.25)
            save_fig_bytes(fig, "PCA by performance")

            # UMAP (if available)
            if HAS_UMAP and X_scaled.shape[0] >= 15:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    umap_model = umap.UMAP(n_neighbors=min(15, X_scaled.shape[0] - 1), min_dist=0.15, metric="euclidean", random_state=42)
                    X_umap = umap_model.fit_transform(X_scaled)
                data_valid["UMAP1"] = X_umap[:, 0]
                data_valid["UMAP2"] = X_umap[:, 1]
                fig, ax = plt.subplots(figsize=(10, 6))
                for r in sorted(data_valid["Round"].unique()):
                    sub = data_valid[data_valid["Round"] == r]
                    ax.scatter(sub["UMAP1"], sub["UMAP2"], s=45, alpha=0.78, color=round_color_map.get(int(r), RUTGERS_GRAY), label=f"Round {int(r)}")
                ax.set_title("Design Space Projection by Round (UMAP)")
                ax.set_xlabel("UMAP1")
                ax.set_ylabel("UMAP2")
                ax.legend(frameon=False)
                ax.grid(True, alpha=0.25)
                save_fig_bytes(fig, "UMAP by round")
                fig, ax = plt.subplots(figsize=(10, 6))
                sc = ax.scatter(data_valid["UMAP1"], data_valid["UMAP2"], c=data_valid["Performance"], s=50, cmap="Reds", alpha=0.86, edgecolor="k", linewidth=0.2)
                ax.set_title("Design Space Colored by Observed REA (UMAP)")
                ax.set_xlabel("UMAP1")
                ax.set_ylabel("UMAP2")
                plt.colorbar(sc, ax=ax, label="Observed REA")
                ax.grid(True, alpha=0.25)
                save_fig_bytes(fig, "UMAP by performance")

                # Convex hull of top performers on UMAP
                try:
                    from scipy.spatial import ConvexHull
                    thresh = data_valid["Performance"].quantile(0.90)
                    top_pts = data_valid[data_valid["Performance"] >= thresh][["UMAP1", "UMAP2"]].values
                    if len(top_pts) >= 3:
                        hull = ConvexHull(top_pts)
                        hull_pts = np.vstack([top_pts[hull.vertices], top_pts[hull.vertices[0]]])
                        fig, ax = plt.subplots(figsize=(10, 6))
                        top_df = data_valid[data_valid["Performance"] >= thresh]
                        ax.scatter(data_valid["UMAP1"], data_valid["UMAP2"], s=40, color=RUTGERS_LIGHT, alpha=0.7, label="All polymers")
                        ax.scatter(top_df["UMAP1"], top_df["UMAP2"], s=70, color=RUTGERS_RED, alpha=0.9, label="Top 10%")
                        ax.plot(hull_pts[:, 0], hull_pts[:, 1], "--", color=RUTGERS_DARK, linewidth=2, label="Convex hull (top 10%)")
                        ax.set_title("UMAP Edge Structure and Top-Performer Enrichment")
                        ax.set_xlabel("UMAP1")
                        ax.set_ylabel("UMAP2")
                        ax.legend(frameon=False)
                        ax.grid(True, alpha=0.25)
                        save_fig_bytes(fig, "UMAP convex hull top performers")
                except Exception:
                    pass

            # SHAP 1: Monomers + Degree of Polymerization only
            monomer_dp_cols = list(monomer_cols_present)
            if DP_COL in data_valid.columns:
                monomer_dp_cols.append(DP_COL)
            if HAS_SHAP and len(monomer_dp_cols) >= 2 and X_scaled.shape[0] >= 20:
                try:
                    X_mono = data_valid[monomer_dp_cols].fillna(0).values
                    X_mono_scaled = StandardScaler().fit_transform(X_mono)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FutureWarning)
                        from sklearn.ensemble import RandomForestRegressor
                        model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
                        model.fit(X_mono_scaled, data_valid["Performance"].values)
                        explainer = shap.TreeExplainer(model)
                        shap_vals = explainer.shap_values(X_mono_scaled)
                    if isinstance(shap_vals, list):
                        shap_vals = shap_vals[0]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        shap.summary_plot(shap_vals, X_mono_scaled, feature_names=monomer_dp_cols, show=False, max_display=min(15, len(monomer_dp_cols)))
                    buf = io.BytesIO()
                    plt.tight_layout()
                    plt.savefig(buf, format="png", dpi=600, bbox_inches="tight", facecolor="white")
                    buf.seek(0)
                    figures.append(("SHAP: monomers + Degree of Polymerization", buf.read()))
                    plt.close("all")
                except Exception:
                    pass

            # SHAP 2: Extended chemical features (derived from monomer fractions)
            derived_cols = [c for c in feature_cols if c not in monomer_cols_present and c != DP_COL]
            if HAS_SHAP and len(feature_cols) >= 3 and len(derived_cols) >= 1 and X_scaled.shape[0] >= 20:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FutureWarning)
                        from sklearn.ensemble import RandomForestRegressor
                        model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
                        model.fit(X_scaled, data_valid["Performance"].values)
                        explainer = shap.TreeExplainer(model)
                        shap_vals = explainer.shap_values(X_scaled)
                    if isinstance(shap_vals, list):
                        shap_vals = shap_vals[0]
                    # Human-readable labels for derived features
                    feat_labels = [c.replace("_", " ").replace("  ", " ").strip(" _") for c in feature_cols]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        shap.summary_plot(shap_vals, X_scaled, feature_names=feat_labels, show=False, max_display=min(20, len(feature_cols)))
                    buf = io.BytesIO()
                    plt.tight_layout()
                    plt.savefig(buf, format="png", dpi=600, bbox_inches="tight", facecolor="white")
                    buf.seek(0)
                    figures.append(("SHAP: monomers + DP + derived chemical features", buf.read()))
                    plt.close("all")
                except Exception:
                    pass

    return summary, figures
