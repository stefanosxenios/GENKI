"""
Predicted vs observed flux comparison plot — ORACLE vs GENKI.

The figure has three panels:
    - top-left:  scatter of predicted vs observed flux for ORACLE, coloured by
                 KPI compartment.
    - top-right: same for the GENKI / generated ensemble.
    - bottom:    per-reaction error score (zeta, SMAPE, or MAPE).

For the ζ-score, an optional `sigma_floor` can be applied to avoid blown-up
scores on near-constant-residual reactions; see the function docstring.
"""

from __future__ import annotations


def plot_predicted_vs_observed(
    oracle_file: str,
    genki_file: str,
    exp_file: str,
    perturbed_reaction: str,
    scoring: str = "zeta",
    *,
    perturbation_column: str | None = None,
    exclude_from_scatter=("PGM", "CO2t"),
    compartment_of_map: dict | None = None,
    label_top_n: int = 5,
    save_path: str | None = None,
    figsize: tuple = (14, 6),
    dpi: int = 450,
    eps: float = 1e-8,
    mape_eps: float = 0.025,
    use_sigma_floor: bool = False,
    sigma_floor: float | str | None = None,
    show: bool = True,
):
    """
    Predicted-vs-observed flux comparison — ORACLE vs GENKI.

    Parameters
    ----------
    use_sigma_floor : bool, default False
        Whether to apply a lower bound to sigma_r in zeta scoring.

    sigma_floor : float, "auto", or None
        Floor applied to sigma_r when scoring="zeta" and use_sigma_floor=True.

        - None:
            No floor is applied unless use_sigma_floor=True, in which case
            an automatic floor is used.
        - "auto":
            Uses 0.05 * median(sigma_r).
        - float:
            Uses the provided value as the lower bound for sigma_r.

    compartment_of_map : dict or None
        Custom reaction → compartment mapping that overrides the default
        E. coli map.  Any reaction not present in the dict falls back to
        "Intracellular".  Pass a yeast-specific dict to colour the KPI
        reactions distinctly, e.g.::

            compartment_of_map={
                "GLCt1":               "Uptake",
                "LMPD_s_0450_c_1_256": "Growth",
                "ETOHt":               "Secretion",
                "CO2t":                "Secretion",
            }

    Notes
    -----
    sigma_floor only affects scoring="zeta".
    It prevents tiny oracle residual variance values from producing
    artificially huge zeta scores.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns

    scoring = scoring.lower()
    if scoring not in {"zeta", "smape", "mape"}:
        raise ValueError(f"Unknown scoring {scoring!r}. Use 'zeta', 'smape', or 'mape'.")

    if perturbation_column is None:
        perturbation_column = perturbed_reaction.lower()
    if save_path is None:
        save_path = f"{perturbation_column}_predicted_vs_observed.png"

    exclude_from_scatter = set(exclude_from_scatter)

    # ----- colour & compartment maps ----------------------------------
    COLOR_ORACLE = "#ff7f0e"
    COLOR_GENKI  = "#1f77b4"
    COMPARTMENT_COLORS = {
        "Uptake":        "#1f77b4",
        "Growth":        "#2ca02c",
        "Perturbed":     "#9467bd",
        "Secretion":     "#d62728",
        "Intracellular": "#7f7f7f",
    }
    _default_compartment_map = {
        "GLCpts": "Uptake",
        "BIOMASS_Ecoli_core_w_GAM": "Growth",
        "SUCCt2_2": "Secretion",
        "SUCCt3":   "Secretion",
        perturbed_reaction: "Perturbed",
    }
    COMPARTMENT_OF = compartment_of_map if compartment_of_map is not None \
        else _default_compartment_map
    COMPARTMENT_ORDER = ["Uptake", "Growth", "Perturbed", "Secretion", "Intracellular"]
    compartment_of = lambda r: COMPARTMENT_OF.get(r, "Intracellular")

    # ----- data --------------------------------------------------------
    exp_df    = pd.read_csv(exp_file,    index_col=0)
    oracle_df = pd.read_csv(oracle_file, index_col=0)
    genki_df  = pd.read_csv(genki_file,  index_col=0)

    if perturbation_column not in exp_df.columns:
        raise ValueError(
            f"Column '{perturbation_column}' not in {exp_file}. "
            f"Available: {list(exp_df.columns)}"
        )

    exp_series = exp_df[perturbation_column]

    oracle_models = oracle_df.drop(
        columns=[c for c in ["Steady"] if c in oracle_df.columns]
    ).T

    genki_models = genki_df.drop(
        columns=[c for c in ["Steady"] if c in genki_df.columns]
    ).T

    reactions = (
        exp_series.index
        .intersection(oracle_models.columns)
        .intersection(genki_models.columns)
        .tolist()
    )

    reactions = sorted(
        reactions,
        key=lambda r: (COMPARTMENT_ORDER.index(compartment_of(r)), r)
    )

    exp_vals         = exp_series[reactions]
    oracle_models    = oracle_models[reactions]
    genki_models     = genki_models[reactions]
    rxn_compartments = [compartment_of(r) for r in reactions]

    # ----- stats -------------------------------------------------------
    oracle_mean = oracle_models.mean()
    oracle_std  = oracle_models.std()

    genki_mean = genki_models.mean()
    genki_std  = genki_models.std()

    oracle_abs_res = oracle_models.sub(exp_vals, axis=1).abs()
    genki_abs_res  = genki_models.sub(exp_vals,  axis=1).abs()

    sigma_floor_used = None

    if scoring == "zeta":
        sigma_r_raw = oracle_abs_res.std(axis=0)

        if use_sigma_floor:
            if sigma_floor is None or sigma_floor == "auto":
                sigma_floor_used = 0.05 * float(sigma_r_raw.median())
            else:
                sigma_floor_used = float(sigma_floor)

            if sigma_floor_used < 0:
                raise ValueError("sigma_floor must be non-negative.")

            sigma_r = sigma_r_raw.clip(lower=sigma_floor_used)
        else:
            sigma_r = sigma_r_raw

        oracle_score = oracle_abs_res.mean(axis=0) / (sigma_r + eps)
        genki_score  = genki_abs_res.mean(axis=0)  / (sigma_r + eps)

    elif scoring == "smape":
        denom_o = (oracle_models.abs().add(exp_vals.abs(), axis=1)) / 2.0
        denom_g = (genki_models.abs().add(exp_vals.abs(), axis=1)) / 2.0

        oracle_score = (oracle_abs_res.div(denom_o + eps)).mean(axis=0)
        genki_score  = (genki_abs_res.div(denom_g + eps)).mean(axis=0)

    else:  # mape
        denom = np.maximum(exp_vals.abs(), mape_eps)

        oracle_score = oracle_abs_res.div(denom, axis=1).mean(axis=0)
        genki_score  = genki_abs_res.div(denom, axis=1).mean(axis=0)

    # ----- figure ------------------------------------------------------
    sns.set(style="whitegrid")

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(
        1, 2,
        hspace=0.40,
        wspace=0.35
    )

    ax_oracle = fig.add_subplot(gs[0, 0])
    ax_genki  = fig.add_subplot(gs[0, 1])

    # Robust axis limits over the scatter-included reactions only.
    scatter_reactions = [r for r in reactions if r not in exclude_from_scatter]

    pool = np.concatenate([
        oracle_mean.loc[scatter_reactions].values,
        genki_mean.loc[scatter_reactions].values,
        exp_vals.loc[scatter_reactions].values,
    ])

    lo, hi = np.percentile(pool, [2, 98])
    pad = 0.10 * (hi - lo)

    ax_lo = float(np.floor(lo - pad))
    ax_hi = float(np.ceil(hi + pad))

    for ax, mean_pred, std_pred, label in [
        (ax_oracle, oracle_mean, oracle_std, "ORACLE"),
        (ax_genki,  genki_mean,  genki_std,  "GENKI"),
    ]:
        for comp in COMPARTMENT_ORDER:
            mask = [
                (c == comp) and (r in scatter_reactions)
                for r, c in zip(reactions, rxn_compartments)
            ]

            if not any(mask):
                continue

            xs = exp_vals[mask].values
            ys = mean_pred[mask].values
            es = std_pred[mask].values

            is_highlighted = comp != "Intracellular"
            ax.errorbar(
                xs,
                ys,
                yerr=es,
                fmt="o",
                color=COMPARTMENT_COLORS[comp],
                ecolor=COMPARTMENT_COLORS[comp],
                alpha=0.90 if is_highlighted else 0.70,
                markersize=9 if is_highlighted else 6,
                elinewidth=1.5 if is_highlighted else 1.0,
                capsize=3,
                zorder=4 if is_highlighted else 3,
                label=comp,
                linestyle="none",
            )

        ax.plot(
            [ax_lo, ax_hi],
            [ax_lo, ax_hi],
            color="gray",
            linestyle="--",
            linewidth=1.2,
            alpha=0.65,
            zorder=2,
        )

        ax.set_xlim(ax_lo, ax_hi)
        ax.set_ylim(ax_lo, ax_hi)
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel("Observed flux (mmol gDW$^{-1}$ h$^{-1}$)", fontsize=11)
        ax.set_ylabel(
            "Predicted flux — mean $\\pm$ std (mmol gDW$^{-1}$ h$^{-1}$)",
            fontsize=11,
        )

        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.40)

        # Label KPI/highlighted reactions + top-N outliers (by distance from diagonal)
        scatter_rxns  = [r for r in reactions if r not in exclude_from_scatter]
        scatter_xs    = exp_vals[scatter_rxns].values
        scatter_ys    = mean_pred[scatter_rxns].values
        dist_from_diag = np.abs(scatter_ys - scatter_xs)

        kpi_rxns = {r for r in scatter_rxns
                    if compartment_of(r) != "Intracellular"}
        top_outlier_idx = set(
            np.argsort(dist_from_diag)[-label_top_n:]
        )
        label_set = kpi_rxns | {scatter_rxns[i] for i in top_outlier_idx}

        for rxn, xv, yv in zip(scatter_rxns, scatter_xs, scatter_ys):
            if rxn not in label_set:
                continue
            is_kpi = rxn in kpi_rxns
            ax.annotate(
                rxn,
                (xv, yv),
                fontsize=7.5 if is_kpi else 6.5,
                fontweight="bold" if is_kpi else "normal",
                color=COMPARTMENT_COLORS[compartment_of(rxn)] if is_kpi else "dimgray",
                xytext=(5, 5),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="-", color="lightgray",
                                lw=0.6) if is_kpi else None,
            )

        ax.legend(
            loc="lower right",
            fontsize=9,
            framealpha=0.9,
            title="Compartment",
        )

        if exclude_from_scatter:
            ax.text(
                0.05,
                0.05,
                "Off-scale: " + ", ".join(sorted(exclude_from_scatter)),
                transform=ax.transAxes,
                fontsize=8,
                color="dimgray",
                va="bottom",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc="white",
                    ec="lightgray",
                    alpha=0.85,
                ),
            )

    title = f"{perturbed_reaction} downregulation — predicted vs observed fluxes"

    if scoring == "zeta" and use_sigma_floor:
        title += rf"  ($\sigma_{{floor}}={sigma_floor_used:.3g}$)"

    fig.suptitle(
        title,
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Per-model R² distribution across perturbations
# ---------------------------------------------------------------------------
def plot_per_model_r2_kde(
    perturbations,
    exp_file: str,
    *,
    perturbation_columns: dict | None = None,
    exclude_reactions=("PGM", "CO2t"),
    save_path: str | None = None,
    figsize: tuple | None = None,
    dpi: int = 450,
    r2_clip: tuple = (-1.0, 1.0),
    r2_threshold: float = 0.5,
    n_cols: int = 3,
    show: bool = True,
):
    """
    KDE of per-model R² — ORACLE vs GENKI — across multiple perturbations.

    For each perturbation we compute one R² value per model in each ensemble:

        R^2_k = 1 - sum_r (exp_r - sim_{k,r})^2 / sum_r (exp_r - mean(exp))^2

    and then plot a kernel density estimate of those per-model R² values for
    ORACLE (orange) and GENKI (blue), one panel per perturbation.

    Unlike the ensemble-mean scatter, this view is immune to ensemble-mean
    cancellation: it shows how concentrated the per-model fits are around the
    experimental truth, which is the quantity a conditional generative model
    is actually trained to improve.

    Parameters
    ----------
    perturbations : list of (str, str, str)
        List of (mutant, oracle_csv_path, genki_csv_path) tuples.
    exp_file : str
        Experimental flux CSV with one lowercase column per perturbation.
    perturbation_columns : dict or None
        Override mutant -> exp column name. Defaults to mutant.lower() per
        entry.
    exclude_reactions : iterable of str
        Reactions excluded from the R² computation (typically off-scale
        outliers such as PGM and CO2t).
    save_path : str or None
        Output PNG path. None disables saving.
    figsize : tuple or None
        Defaults to (5.5 * n_cols, 4.0 * n_rows).
    dpi : int
        Resolution of the saved PNG.
    r2_clip : tuple (lo, hi)
        Visualisation clipping range for R² (R² can be very negative for bad
        models; clipping keeps the KDE readable). Models outside the range are
        clipped for plotting but counted in the legend.
    r2_threshold : float
        Threshold used in the per-panel annotation ("fraction of models with
        R² >= r2_threshold").
    n_cols : int
        Number of columns in the panel grid.
    show : bool
        Whether to call plt.show() at the end.

    Returns
    -------
    (matplotlib.figure.Figure, pandas.DataFrame)
        The figure and a per-perturbation summary table (also useful for
        saving to a CSV alongside the figure).
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    COLOR_ORACLE = "#ff7f0e"
    COLOR_GENKI  = "#1f77b4"

    perturbation_columns = perturbation_columns or {}
    exclude_reactions = set(exclude_reactions)

    exp_df = pd.read_csv(exp_file, index_col=0)

    n = len(perturbations)
    n_rows = int(np.ceil(n / n_cols))
    if figsize is None:
        figsize = (5.5 * n_cols, 4.0 * n_rows)

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    lo, hi = r2_clip
    summary_rows = []

    for ax, (mutant, oracle_csv, genki_csv) in zip(axes, perturbations):
        exp_col = perturbation_columns.get(mutant, mutant.lower())
        if exp_col not in exp_df.columns:
            ax.text(0.5, 0.5, f"Column '{exp_col}'\nnot in exp file",
                    transform=ax.transAxes, ha="center", va="center",
                    color="dimgray")
            ax.set_title(f"{mutant} (skipped)", fontsize=12, fontweight="bold")
            continue

        exp_series = exp_df[exp_col]
        oracle_df_m = pd.read_csv(oracle_csv, index_col=0)
        genki_df_m  = pd.read_csv(genki_csv,  index_col=0)

        # Drop 'Steady' column if present (ORACLE CSVs include it).
        oracle_df_m = oracle_df_m.drop(
            columns=[c for c in ["Steady"] if c in oracle_df_m.columns]
        )
        genki_df_m = genki_df_m.drop(
            columns=[c for c in ["Steady"] if c in genki_df_m.columns]
        )

        # Common reactions across the three sources, minus any excluded ones.
        common = (exp_series.index
                  .intersection(oracle_df_m.index)
                  .intersection(genki_df_m.index))
        common = [r for r in common if r not in exclude_reactions]

        exp_vec = exp_series.loc[common].values
        obs_centered_ss = float(np.sum((exp_vec - exp_vec.mean()) ** 2))

        def per_model_r2(df_models):
            mat = df_models.loc[common].values   # (n_reactions, n_models)
            ss_res = np.sum((mat - exp_vec[:, None]) ** 2, axis=0)
            r2 = 1.0 - ss_res / (obs_centered_ss + 1e-12)
            return pd.Series(r2, index=df_models.columns).dropna()

        r2_oracle = per_model_r2(oracle_df_m)
        r2_genki  = per_model_r2(genki_df_m)

        # Count clipped models for the annotation, then clip for KDE plotting.
        n_or_below = int((r2_oracle < lo).sum())
        n_ge_below = int((r2_genki  < lo).sum())
        r2_oracle_c = r2_oracle.clip(lo, hi)
        r2_genki_c  = r2_genki .clip(lo, hi)

        sns.kdeplot(r2_oracle_c, ax=ax, color=COLOR_ORACLE, fill=True,
                    alpha=0.40, linewidth=1.5, label="ORACLE", clip=(lo, hi))
        sns.kdeplot(r2_genki_c, ax=ax, color=COLOR_GENKI, fill=True,
                    alpha=0.40, linewidth=1.5, label="GENKI",  clip=(lo, hi))

        med_or = float(r2_oracle.median())
        med_ge = float(r2_genki .median())
        ax.axvline(med_or, color=COLOR_ORACLE, linestyle="--", linewidth=1.0, alpha=0.8)
        ax.axvline(med_ge, color=COLOR_GENKI,  linestyle="--", linewidth=1.0, alpha=0.8)

        frac_or = float((r2_oracle >= r2_threshold).mean())
        frac_ge = float((r2_genki  >= r2_threshold).mean())

        info = (f"median $R^2$:  O = {med_or:.3f}\n"
                f"                G = {med_ge:.3f}\n"
                f"$R^2 \\geq {r2_threshold}$:  O = {frac_or:.0%}\n"
                f"                G = {frac_ge:.0%}")
        if n_or_below or n_ge_below:
            info += f"\n(<{lo}: O={n_or_below}, G={n_ge_below})"
        ax.text(0.03, 0.97, info, transform=ax.transAxes,
                fontsize=8.5, va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", alpha=0.9))

        ax.set_xlim(lo, hi)
        ax.set_xlabel(r"per-model $R^2$", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"{mutant}", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

        summary_rows.append({
            "mutant": mutant,
            "median_r2_oracle": med_or,
            "median_r2_genki":  med_ge,
            f"frac_r2_ge_{r2_threshold}_oracle": frac_or,
            f"frac_r2_ge_{r2_threshold}_genki":  frac_ge,
            "n_models_oracle": int(len(r2_oracle)),
            "n_models_genki":  int(len(r2_genki)),
            "n_below_clip_oracle": n_or_below,
            "n_below_clip_genki":  n_ge_below,
        })

    # Hide unused panels.
    for ax in axes[len(perturbations):]:
        ax.axis("off")

    fig.suptitle(r"Per-model $R^2$ distribution — ORACLE vs GENKI",
                 fontsize=14, fontweight="bold", y=1.005)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    summary = (pd.DataFrame(summary_rows).set_index("mutant")
               if summary_rows else None)
    return fig, summary
