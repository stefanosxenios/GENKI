import matplotlib.pyplot as plt

def plot_error_histograms(datasets: dict,
                          bins: int = 50,
                          density: bool = True,
                          alpha: float = 0.5,
                          figsize=(10, 6),
                          title: str = "Error Distribution Comparison"):
    """
    Plot histograms of Error columns from one or many datasets.

    Parameters
    ----------
    datasets : dict
        A dictionary mapping labels to dataframes, e.g.:
        {
            "VAE": df_vae,
            "ORACLE": df_oracle,
            "Synthetic": df_synth
        }

    bins : int
        Number of histogram bins.

    density : bool
        If True, normalize histograms (probability distributions).

    alpha : float
        Transparency of histogram bars.

    figsize : tuple
        Size of the figure.

    title : str
        Title of the plot.
    """

    plt.figure(figsize=figsize)

    for label, df in datasets.items():
        if "Error" not in df.columns:
            raise KeyError(f"Dataframe '{label}' has no 'Error' column.")

        plt.hist(
            df["Error"].dropna(),
            bins=bins,
            density=density,
            alpha=alpha,
            label=label
        )

    plt.xlabel("Error")
    plt.ylabel("Density" if density else "Count")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()





import numpy as np
import pandas as pd
from scipy.stats import entropy


def kl_divergence_per_column(
    df_p: pd.DataFrame,
    df_q: pd.DataFrame,
    bins: int = 50,
    top_n: int = 20,
    epsilon: float = 1e-10
) -> pd.DataFrame:
    """
    Compute KL divergence per column between two dataframes.

    Parameters
    ----------
    df_p, df_q : pd.DataFrame
        DataFrames with identical columns
    bins : int
        Number of histogram bins
    top_n : int
        Number of top differing columns to return
    epsilon : float
        Small value to avoid log(0)

    Returns
    -------
    pd.DataFrame
        Sorted KL divergence per column
    """

    assert list(df_p.columns) == list(df_q.columns), "Columns must match"

    kl_scores = {}

    for col in df_p.columns:
        p = df_p[col].dropna().values
        q = df_q[col].dropna().values

        # Shared binning
        hist_range = (min(p.min(), q.min()), max(p.max(), q.max()))

        p_hist, _ = np.histogram(p, bins=bins, range=hist_range, density=True)
        q_hist, _ = np.histogram(q, bins=bins, range=hist_range, density=True)

        # Numerical stability
        p_hist += epsilon
        q_hist += epsilon

        kl_scores[col] = entropy(p_hist, q_hist)

    kl_df = (
        pd.DataFrame.from_dict(kl_scores, orient="index", columns=["KL_divergence"])
        .sort_values("KL_divergence", ascending=False)
    )

    return kl_df.head(top_n)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def compare_km_means_violin(
    df1, df2, df3=None, 
    name1="Dataset 1", name2="Dataset 2", name3="Dataset 3",
    cols=None, top=20, plot=True
):
    """
    Compare mean log(Km) values across datasets and plot ONLY top-k differing parameters using violin plots.

    Parameters
    ----------
    df1, df2 : DataFrames (required)
    df3 : DataFrame or None (optional third dataset)
    name1, name2, name3 : str
        Labels to use for the datasets (for plot & legend)
    cols : list, slice, or None
        Parameter columns to evaluate. If None, intersect common columns.
    top : int
        Number of top-difference Km parameters to show.
    plot : bool
        If True, draw violin plots of only top parameters.
    """

    # --- Column selection ---
    if cols is None:
        cols = df1.columns.intersection(df2.columns)
        if df3 is not None:
            cols = cols.intersection(df3.columns)

    # --- Compute log-means ---
    mean1 = np.log(df1[cols]).mean()
    mean2 = np.log(df2[cols]).mean()

    if df3 is not None:
        mean3 = np.log(df3[cols]).mean()

    # --- Compute pairwise absolute differences ---
    distances = {}

    distances[f"{name1}–{name2}"] = (mean1 - mean2).abs()

    if df3 is not None:
        distances[f"{name1}–{name3}"] = (mean1 - mean3).abs()
        distances[f"{name2}–{name3}"] = (mean2 - mean3).abs()

    dist_table = pd.DataFrame(distances)

    # --- Pick top parameters ---
    top_params = dist_table.max(axis=1).sort_values(ascending=False).head(top).index

    print(f"\n🔝 Top {top} parameters with largest differences in mean log(Km):")
    display(dist_table.loc[top_params])

    # --- Prepare data for plotting ---
    if plot:
        df_list = []

        for df, name in zip([df1, df2, df3], [name1, name2, name3]):
            if df is not None:
                melted = df[top_params].melt(var_name="Parameter", value_name="Value")
                melted["Value"] = np.log(melted["Value"])
                melted["Dataset"] = name
                df_list.append(melted)

        df_combined = pd.concat(df_list, ignore_index=True)

        # --- Violin plot ---
        plt.figure(figsize=(18, 8))
        sns.violinplot(data=df_combined, x="Parameter", y="Value", hue="Dataset", split=True)
        plt.xticks(rotation=90)
        plt.title(f"Top {top} Parameters with Biggest Mean Differences (Violin): {name1}, {name2}" + 
                  (f", {name3}" if df3 is not None else ""))
        plt.tight_layout()
        plt.show()

    return dist_table.loc[top_params]

def compare_km_means_kde(
    df1, df2, df3=None, 
    name1="Dataset 1", name2="Dataset 2", name3="Dataset 3",
    cols=None, top=20, log_transform=True
):
    """
    Compare mean log(Km) values across datasets and plot overlaid KDEs for top-k differing parameters.

    Parameters
    ----------
    df1, df2 : DataFrames (required)
    df3 : DataFrame or None (optional third dataset)
    name1, name2, name3 : str
        Labels to use for the datasets (for plot & legend)
    cols : list, slice, or None
        Parameter columns to evaluate. If None, intersect common columns.
    top : int
        Number of top-difference Km parameters to show.
    log_transform : bool
        Whether to log-transform Km values before plotting.
    """

    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    if cols is None:
        cols = df1.columns.intersection(df2.columns)
        if df3 is not None:
            cols = cols.intersection(df3.columns)

    mean1 = np.log(df1[cols]).mean() if log_transform else df1[cols].mean()
    mean2 = np.log(df2[cols]).mean() if log_transform else df2[cols].mean()
    if df3 is not None:
        mean3 = np.log(df3[cols]).mean() if log_transform else df3[cols].mean()

    # Compute pairwise absolute differences
    distances = {f"{name1}–{name2}": (mean1 - mean2).abs()}
    if df3 is not None:
        distances[f"{name1}–{name3}"] = (mean1 - mean3).abs()
        distances[f"{name2}–{name3}"] = (mean2 - mean3).abs()

    dist_table = pd.DataFrame(distances)
    top_params = dist_table.max(axis=1).sort_values(ascending=False).head(top).index

    print(f"\n🔝 Top {top} parameters with largest differences in mean log(Km):")
    display(dist_table.loc[top_params])

    # Plot overlaid KDEs
    ncols = 5
    nrows = int(np.ceil(top / ncols))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3.5 * nrows), constrained_layout=True)

    axs = axs.flatten()

    for i, param in enumerate(top_params):
        ax = axs[i]

        for df, name in zip([df1, df2, df3], [name1, name2, name3]):
            if df is not None:
                values = df[param].dropna()
                if log_transform:
                    values = np.log(values)
                sns.kdeplot(values, ax=ax, label=name, linewidth=2)

        ax.set_title(param, fontsize=10)
        ax.set_xlabel("log(Km)")
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    plt.suptitle("Overlaid KDEs for Top-Differing Km Parameters", fontsize=14)
    plt.show()

    return dist_table.loc[top_params]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_kde_with_arrows(df1, df2, top_params, name1="new_synthetic", name2="ORACLE"):
    """
    Overlaid KDEs with mean arrows and tight x-limits for top differing parameters.
    """
    num_plots = len(top_params)
    fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4), sharey=True)

    for i, param in enumerate(top_params):
        ax = axes[i] if num_plots > 1 else axes

        data1 = np.log(df1[param].dropna())
        data2 = np.log(df2[param].dropna())

        sns.kdeplot(data1, ax=ax, label=name1, linewidth=2)
        sns.kdeplot(data2, ax=ax, label=name2, linewidth=2)

        # Arrow annotation
        mean1 = data1.mean()
        mean2 = data2.mean()
        ymin, ymax = ax.get_ylim()
        ymid = ymax * 0.8

        ax.annotate('', xy=(mean2, ymid), xytext=(mean1, ymid),
                    arrowprops=dict(arrowstyle='<->', color='gray', lw=2, alpha=0.6))
        ax.text((mean1 + mean2) / 2, ymid * 1.05, 'Δmean', ha='center', fontsize=9)

        ax.set_title(param)
        ax.set_xlabel("log(Km)")

        # Tight x-limits
        all_data = np.concatenate([data1, data2])
        margin = 0.01 * (all_data.max() - all_data.min())
        ax.set_xlim(all_data.min() - margin, all_data.max() + margin)

    plt.suptitle("Overlaid KDEs for Top-Differing Km Parameters", fontsize=14)
    plt.tight_layout()
    plt.legend()
    plt.show()



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def compare_km_means(
    df1, df2, df3=None,
    name1="Dataset 1", name2="Dataset 2", name3="Dataset 3",
    cols=None, top=20, plot=True, figsize=(10.5, 7.62),
    savepath=None,
    title=None,
    orient="v",
    palette=None,
    km_mapping_path="./data/FCCs/km_mapping.csv",
):
    """
    Compare mean log(Km) values across datasets and plot top-k differing parameters.

    Parameters
    ----------
    df1, df2 : DataFrames (required)
    df3 : DataFrame or None (optional third dataset)
    name1, name2, name3 : str
        Labels for the datasets.
    cols : list or None
        Columns to evaluate. If None, uses intersection.
    top : int
        Number of top-differing parameters to show.
    plot : bool
        If True, draw the boxplot.
    figsize : tuple
        Figure size.
    savepath : str or None
        Path to save the figure. If None, figure is not saved.
    title : str or None
        Custom plot title. Defaults to an auto-generated title.
    orient : str
        Boxplot orientation: 'v' (vertical) or 'h' (horizontal).
    palette : dict or None
        Custom colour palette. If None, uses blue/orange/green defaults.
    km_mapping_path : str
        Path to CSV mapping raw column names to readable parameter names.
        Expected columns: 'parameter' (raw) and 'name' (readable).
    """
    # --- Column selection ---
    if cols is None:
        cols = df1.columns.intersection(df2.columns)
        if df3 is not None:
            cols = cols.intersection(df3.columns)

    # --- Compute log-means ---
    mean1 = np.log(df1[cols]).mean()
    mean2 = np.log(df2[cols]).mean()
    if df3 is not None:
        mean3 = np.log(df3[cols]).mean()

    # --- Compute pairwise absolute differences ---
    distances = {}
    distances[f"{name1}–{name2}"] = (mean1 - mean2).abs()
    if df3 is not None:
        distances[f"{name1}–{name3}"] = (mean1 - mean3).abs()
        distances[f"{name2}–{name3}"] = (mean2 - mean3).abs()
    dist_table = pd.DataFrame(distances)

    # --- Sort and select top parameters ---
    top_params = dist_table.max(axis=1).sort_values(ascending=False).head(top).index

    # --- Load km mapping (if available) ---
    # Build LaTeX labels of the form  $K^{reaction}_{m,metabolite}$
    # from a CSV with columns: parameter, reaction, metabolite
    km_mapping = {}
    if km_mapping_path is not None:
        try:
            km_map_df = pd.read_csv(km_mapping_path)
            cols_lower = {c.lower(): c for c in km_map_df.columns}
            _pcol = cols_lower.get("parameter") or cols_lower.get("param")
            _rcol = cols_lower.get("reaction")
            _mcol = cols_lower.get("metabolite")

            # Human-readable aliases applied before LaTeX escaping
            _REACTION_ALIASES = {
                "BIOMASS_Ecoli_core_w_GAM": "Growth",
            }

            if _pcol and _rcol and _mcol:
                # Build LaTeX label for each row
                for _, row in km_map_df.iterrows():
                    param   = str(row[_pcol])
                    rxn     = _REACTION_ALIASES.get(str(row[_rcol]), str(row[_rcol]))
                    met     = str(row[_mcol]).lstrip("_")  # strip leading underscore
                    # Escape underscores so matplotlib mathtext treats them as literals
                    rxn_tex = rxn.replace("_", r"\_")
                    met_tex = met.replace("_", r"\_")
                    label   = r"$\mathbf{K}^{\mathbf{" + rxn_tex + r"}}_{\mathbf{m," + met_tex + r"}}$"
                    km_mapping[param] = label
            else:
                # Fallback: look for a plain 'name'/'label' column
                _ncol = cols_lower.get("name") or cols_lower.get("label") or cols_lower.get("readable")
                if _pcol and _ncol:
                    km_mapping = dict(zip(km_map_df[_pcol].astype(str),
                                         km_map_df[_ncol].astype(str)))
        except Exception as e:
            print(f"[compare_km_means] Could not load km_mapping from {km_mapping_path}: {e}")

    # Readable labels for the selected parameters
    readable_labels = [km_mapping.get(str(p), str(p)) for p in top_params]

    print(f"\n🔝 Top {top} parameters with largest differences in mean log(Km):")
    try:
        display(dist_table.loc[top_params])
    except Exception:
        print(dist_table.loc[top_params].to_string())

    # --- Plot setup ---
    if plot:
        df_list = []
        for df, name in zip([df1, df2, df3], [name1, name2, name3]):
            if df is None:
                continue
            melted = df[top_params].copy()
            melted.columns = readable_labels  # rename to human-readable
            melted = melted.melt(var_name="Parameter", value_name="Value")
            melted["Value"] = np.log(melted["Value"])
            melted["Dataset"] = name
            df_list.append(melted)

        df_combined = pd.concat(df_list, ignore_index=True)

        # Colour palette
        if palette is None:
            palette = {
                name1: "#1f77b4",   # blue
                name2: "#ff7f0e",   # orange
            }
            if df3 is not None:
                palette[name3] = "#2ca02c"  # green

        sns.set(style="whitegrid")
        fig, ax = plt.subplots(figsize=figsize)

        box_kwargs = dict(
            data=df_combined,
            hue="Dataset",
            palette=palette,
            showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"},
            ax=ax,
        )
        if orient == "h":
            box_kwargs.update({"x": "Value", "y": "Parameter"})
            sns.boxplot(**box_kwargs)
            ax.set_xlabel(r"$\mathbf{log}(\mathbf{K_m})$ Value", fontsize=11, fontweight="bold")
            ax.set_ylabel("")
            ax.grid(axis="x", linestyle="--", alpha=0.6)
        else:
            box_kwargs.update({"x": "Parameter", "y": "Value"})
            sns.boxplot(**box_kwargs)
            ax.set_xlabel("")   # remove "Parameter" label
            ax.set_ylabel(r"$\mathbf{log}(\mathbf{K_m})$ Value", fontsize=11, fontweight="bold")
            ax.tick_params(axis="x", rotation=45, labelsize=10)
            ax.tick_params(axis="y", labelsize=10)
            ax.grid(axis="y", linestyle="--", alpha=0.6)

        plot_title = title if title is not None else f"Top {top} Differing log(Km) Parameters"
        ax.set_title(plot_title, fontsize=13, fontweight="bold")
        sns.despine()
        ax.legend(title="Dataset", title_fontsize=9, fontsize=9)
        fig.tight_layout()

        if savepath is not None:
            fig.savefig(savepath, dpi=450, bbox_inches="tight")
            print(f"[saved] {savepath}")

        plt.show()

    return dist_table.loc[top_params]

import matplotlib.pyplot as plt
import seaborn as sns

def plot_error_histograms_vibrant(datasets: dict,
                                   bins: int = 40,
                                   density: bool = True,
                                   alpha: float = 0.6,
                                   figsize=(10, 6),
                                   title: str = "Error Distribution Comparison",
                                   xlim: tuple = None,
                                   savepath: str = None):
    """
    Plot error histograms with KDE overlays using vibrant colors.
    """
    plt.figure(figsize=figsize)

    palette = {
        "GENKI": "#1f77b4",    # Blue
        "ORACLE": "#ff7f0e",   # Orange
        "Rules":  "#2ca02c",   # Green (if needed)
    }

    for label, df in datasets.items():
        # if "Error" not in df.columns:
        #     raise KeyError(f"Dataframe '{label}' has no 'Error' column.")

        sns.histplot(df.dropna(),
                     bins=bins,
                     kde=True,
                     stat='density' if density else 'count',
                     alpha=alpha,
                     label=f"{label}",
                     color=palette.get(label, None),
                     edgecolor='black')

        # Mean line
        mean_val = df.mean()
        plt.axvline(mean_val, linestyle='--', color=palette.get(label, 'gray'), linewidth=2)

    plt.xlabel("KPI", fontsize=11, fontweight="bold")
    plt.ylabel("Density" if density else "Count", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold")
    plt.legend(title="Model Type")
    plt.grid(True, linestyle="--", alpha=0.3)

    if xlim:
        plt.xlim(*xlim)

    plt.tight_layout()
    if savepath is not None:
        import os
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=450, bbox_inches='tight')


def plot_thresholds_per_reaction(
    data,
    thresholds,
    output_dir,
    oracle_name="ORACLE",
    add_oracle_best=True,
    title=None,
):
    import os
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    default_perc = [0.20, 0.10, 0.01]

    plot_thresholds = thresholds.get("plot", {}) if thresholds else {}

    for reaction, models in data["kpi"].items():

        if oracle_name not in models:
            raise ValueError(f"{oracle_name} not found in {reaction}")

        oracle_kpi = models[oracle_name]

        perc_list = plot_thresholds.get(reaction, default_perc)

        # ---- compute thresholds FROM ORACLE ----
        thr_values = [oracle_kpi.quantile(p) for p in perc_list]
        labels     = [f"{int(p*100)}%" for p in perc_list]

        # Optionally add a "Best Model" column: % of each ensemble that beats
        # the single best oracle model (oracle minimum KPI).
        if add_oracle_best:
            thr_values.append(float(oracle_kpi.min()))
            labels.append("Best\nModel")

        # ---- compute results ----
        results = {name: [] for name in models.keys()}

        for name, kpi in models.items():
            for thr in thr_values:
                results[name].append((kpi < thr).mean() * 100)

        # ---- plotting ----
        x     = np.arange(len(labels))
        width = 0.8 / len(models)

        # Colour palette: ORACLE = orange, everything else = steelblue shades
        COLORS = {"ORACLE": "#ff7f0e"}
        _blues = ["#1f77b4", "#4a90d9", "#6baed6", "#9ecae1"]
        _bi = 0
        for name in models:
            if name not in COLORS:
                COLORS[name] = _blues[_bi % len(_blues)]
                _bi += 1

        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 4.5))

        for i, (name, values) in enumerate(results.items()):
            bars = ax.bar(
                x + i * width, values, width,
                label=name,
                color=COLORS.get(name),
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
            )
            for xi, yi in zip(x + i * width, values):
                ax.text(xi, yi + 0.5, f"{yi:.1f}%",
                        ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("% below threshold", fontsize=11)
        ax.set_title(title if title is not None else f"{reaction} — performance thresholds", fontsize=11)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_ylim(bottom=0)


        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{reaction}_threshold.png"), dpi=450,
                    bbox_inches="tight")
        plt.show()
        plt.close(fig)