import pandas as pd
import numpy as np

def get_key_reactions(cut_reaction: str) -> list:
    """
    Automatically generate key reactions based on the cut reaction.
    You only specify the cut reaction; the rest is fixed.
    """
    return [
        cut_reaction,
        "BIOMASS_Ecoli_core_w_GAM",
        "GLCpts",
        "SUCCt2_2",
        "SUCCt3",
        "CO2t",
    ]


def compute_mape(simulated: pd.DataFrame, experimental: pd.Series, epsilon: float = 0.1) -> pd.DataFrame:
    exp_aligned = experimental.reindex(simulated.index)
    
    # Using numpy's element-wise maximum
    div_term = np.maximum(epsilon, exp_aligned.abs())
    
    return (simulated.sub(exp_aligned, axis=0).abs()).div(div_term, axis=0)


def compute_smape(simulated: pd.DataFrame,
                  experimental: pd.Series,
                  eps: float = 1e-12) -> pd.DataFrame:
    """
    Symmetric Mean Absolute Percentage Error, computed per reaction (rows)
    and per model (columns).

        SMAPE = |sim - exp| / ((|sim| + |exp|) / 2)

    Bounded in [0, 2], symmetric in sim/exp, well-defined when exp is close
    to zero (unlike MAPE). `eps` is added to the denominator only to avoid
    a division by zero in the degenerate case sim = exp = 0.
    """
    exp_aligned = experimental.reindex(simulated.index)
    numerator = simulated.sub(exp_aligned, axis=0).abs()
    denominator = (simulated.abs().add(exp_aligned.abs(), axis=0)) / 2.0
    return numerator.div(denominator + eps, axis=0)


def compute_zscore_vs_oracle(simulated: pd.DataFrame,
                             experimental: pd.Series,
                             oracle_simulated: pd.DataFrame,
                             eps: float = 1e-12) -> pd.DataFrame:
    """
    Per-reaction absolute residual rescaled by the spread of the initial
    oracle ensemble (cancellation-free variant of an oracle-referenced
    z-score).

    For each reaction r:
        sigma_r       = std over oracle models of |sim_oracle_r - exp_r|
        score_{r,m}   = |sim_m_r - exp_r| / sigma_r

    The score is always >= 0 (so aggregating across reactions with
    :func:`compute_KPI` cannot cancel a bad fit against a great one). The
    unit is "oracle-error standard deviations": a score of 2 on reaction r
    means the model's absolute residual is 2x the oracle ensemble's typical
    spread on that reaction. Lower is always better.

    Parameters
    ----------
    simulated : DataFrame (reactions x models)
        Simulated fluxes for the models we want to score.
    experimental : Series (reactions,)
        Experimental flux vector to fit.
    oracle_simulated : DataFrame (reactions x oracle_models)
        Simulated fluxes from the initial oracle ensemble. Used purely to
        define the per-reaction scale sigma_r; not scored itself.
    eps : float
        Small constant added to sigma_r to avoid division by zero when a
        reaction is essentially flat across the oracle ensemble.
    """
    exp_aligned = experimental.reindex(simulated.index)
    oracle_aligned = oracle_simulated.reindex(simulated.index)

    oracle_abs_res = oracle_aligned.sub(exp_aligned, axis=0).abs()
    sigma = oracle_abs_res.std(axis=1)

    abs_res = simulated.sub(exp_aligned, axis=0).abs()
    return abs_res.div(sigma + eps, axis=0)


def compute_KPI(mape: pd.DataFrame, key_reactions: list) -> pd.Series:
    key_reactions = [r for r in key_reactions if r in mape.index]
    other_reactions = [r for r in mape.index if r not in key_reactions]

    key_sum = mape.loc[key_reactions].sum(axis=0)
    other_avg = mape.loc[other_reactions].mean(axis=0)

    return key_sum + other_avg


def build_dataset(km_file: str,
                  flux_exp_file: str,
                  enzyme_file: str,
                  cut_reaction: str,
                  epsilon: float = 0.1):  # <-- Added epsilon parameter with a default value
    """
    Build dataset with MAPE + KPI.
    User specifies ONLY the cut reaction (e.g., 'PGL').
    """

    # Load input data
    fluxes_exp = pd.read_csv(flux_exp_file, index_col=0)
    enzyme_df = pd.read_csv(enzyme_file, index_col=0)
    Km = pd.read_csv(km_file, index_col=0)

    # Hard-coded corrections
    fluxes_exp.loc['BIOMASS_Ecoli_core_w_GAM'] = 0.17
    fluxes_exp.loc['SUCCt3'] = 1.555
    fluxes_exp.loc['SUCCt2_2'] = 1.550

    # Clean & align enzyme DF
    if "CO2t" in enzyme_df.index:
        enzyme_df.loc["EX_co2_e"] = enzyme_df.loc["CO2t"]

    enzyme_df = enzyme_df[enzyme_df.index.isin(fluxes_exp.index)]
    enzyme_df = enzyme_df.iloc[:, 1:]     # drop unnecessary first column (model IDs)

    # EXPERIMENTAL VECTOR for the cut reaction
    exp_column = cut_reaction.lower()
    if exp_column not in fluxes_exp.columns:
        raise ValueError(f"Experimental column '{exp_column}' not found in fluxes_exp.csv")

    exp_vec = fluxes_exp[exp_column].reindex(enzyme_df.index)

    # Compute MAPE (Passing epsilon here)
    mape = compute_mape(enzyme_df, exp_vec, epsilon=epsilon)  # <-- Updated this line

    # Key reactions (automatically generated)
    if cut_reaction == "G6PDH2r":
        cut_reaction = "PGL"
    key_rxns = get_key_reactions(cut_reaction)

    # Filter Km to included models
    Km = Km.loc[Km.index.isin(enzyme_df.columns)]

    # Compute KPI
    KPI = compute_KPI(mape, key_rxns)

    # Build final dataset
    dataset = Km.copy()
    dataset["Error"] = KPI

    return dataset, mape, KPI


def _load_enzyme_df(enzyme_file: str, fluxes_exp_index) -> pd.DataFrame:
    """Helper shared by the SMAPE/Z-score builders: load and clean an
    enzyme-perturbation flux CSV the same way ``build_dataset`` does."""
    enzyme_df = pd.read_csv(enzyme_file, index_col=0)

    if "CO2t" in enzyme_df.index:
        enzyme_df.loc["EX_co2_e"] = enzyme_df.loc["CO2t"]

    enzyme_df = enzyme_df[enzyme_df.index.isin(fluxes_exp_index)]
    enzyme_df = enzyme_df.iloc[:, 1:]   # drop the leading model-ID column
    return enzyme_df


def _load_fluxes_exp(flux_exp_file: str) -> pd.DataFrame:
    """Apply the same hard-coded corrections as ``build_dataset``."""
    fluxes_exp = pd.read_csv(flux_exp_file, index_col=0)
    fluxes_exp.loc['BIOMASS_Ecoli_core_w_GAM'] = 0.17
    fluxes_exp.loc['SUCCt3'] = 1.555
    fluxes_exp.loc['SUCCt2_2'] = 1.550
    return fluxes_exp


def build_dataset_smape(km_file: str,
                        flux_exp_file: str,
                        enzyme_file: str,
                        cut_reaction: str):
    """
    Build a training dataset (Km parameters + Error column) using SMAPE
    instead of MAPE as the per-reaction error.

    Same inputs/outputs as :func:`build_dataset`. The ``Error`` column is
    aggregated by :func:`compute_KPI` (sum on key reactions, mean on the
    rest), so the rest of the pipeline (plots, CVAE training) is unaffected.
    """
    fluxes_exp = _load_fluxes_exp(flux_exp_file)
    enzyme_df = _load_enzyme_df(enzyme_file, fluxes_exp.index)
    Km = pd.read_csv(km_file, index_col=0)

    exp_column = cut_reaction.lower()
    if exp_column not in fluxes_exp.columns:
        raise ValueError(f"Experimental column '{exp_column}' not found in {flux_exp_file}")

    exp_vec = fluxes_exp[exp_column].reindex(enzyme_df.index)

    smape = compute_smape(enzyme_df, exp_vec)
    if cut_reaction=="G6PDH2r":
        cut_reaction="PGL"
    key_rxns = get_key_reactions(cut_reaction)
    Km = Km.loc[Km.index.isin(enzyme_df.columns)]
    KPI = compute_KPI(smape, key_rxns)

    dataset = Km.copy()
    dataset["Error"] = KPI

    return dataset, smape, KPI


def build_dataset_yeast(
    Km_df: pd.DataFrame,
    enzyme_df: pd.DataFrame,
    flux_exp_df: pd.DataFrame,
    important_rxns: list = None,
    condition: str = 'microaerobic',
    epsilon: float = 0.1,
) -> tuple:
    """
    Build a training dataset (Km parameters + per-reaction MAPE columns + KPI)
    for a yeast model with oxygen-level experiments.

    This is the ``flux_utlis``-native version of ``build_dataset_oxygen_levels``
    from ``main_yeast.ipynb``.  It delegates MAPE computation to the module-level
    :func:`compute_mape` (epsilon-clipped denominator, result in [0, ∞)) and
    uses the same KPI aggregation logic as :func:`compute_KPI`:

        KPI = Σ MAPE(important reactions) + mean MAPE(all other reactions)

    Parameters
    ----------
    Km_df : DataFrame
        Km parameters, rows = models.
    enzyme_df : DataFrame
        Simulated fluxes, rows = reactions, columns = model indices.
    flux_exp_df : DataFrame
        Experimental fluxes, rows = reactions, columns = oxygen conditions.
    important_rxns : list, optional
        Reactions tracked individually and summed into KPI.
        Defaults to ["GLCt1", "LMPD_s_0450_c_1_256", "ETOHt", "CO2t"].
    condition : str
        Column in ``flux_exp_df`` to use as the experimental reference.
        Defaults to ``'microaerobic'``.
    epsilon : float
        Passed to :func:`compute_mape`; clips denominator away from zero.

    Returns
    -------
    dataset : DataFrame
        Km parameters with ``MAPE_{rxn}`` columns for each important reaction,
        ``MAPE_OTHER`` (mean over remaining reactions), and ``KPI``.
    mape_df : DataFrame
        Full per-reaction MAPE table, shape (reactions × models).
    KPI : Series
        Aggregated error per model (lower is better).
    """
    if important_rxns is None:
        important_rxns = ["GLCt1", "LMPD_s_0450_c_1_256", "ETOHt", "CO2t"]

    # Align reactions shared by simulated and experimental data
    common_rxns = enzyme_df.index.intersection(flux_exp_df.index)
    enzyme_aligned = enzyme_df.loc[common_rxns]

    if condition not in flux_exp_df.columns:
        raise ValueError(
            f"Condition '{condition}' not found in flux_exp_df. "
            f"Available columns: {flux_exp_df.columns.tolist()}"
        )
    exp_vec = flux_exp_df.loc[common_rxns, condition]

    # compute_mape → DataFrame (reactions × models)
    mape_df = compute_mape(enzyme_aligned, exp_vec, epsilon=epsilon)

    missing = [r for r in important_rxns if r not in mape_df.index]
    if missing:
        raise ValueError(f"Important reactions not found in mape index: {missing}")

    # Transpose to (models × reactions) for row-wise dataset assembly
    mape_T = mape_df.T

    mape_imp = mape_T[important_rxns]
    other_rxns = [r for r in mape_T.columns if r not in important_rxns]
    mape_other = mape_T[other_rxns].mean(axis=1)

    KPI = mape_imp.sum(axis=1) + mape_other

    # Build final dataset — Km rows are a subset of model indices
    valid_idx = Km_df.index.intersection(mape_T.index)
    dataset = Km_df.loc[valid_idx].copy()
    for rxn in important_rxns:
        dataset[f"MAPE_{rxn}"] = mape_imp.loc[valid_idx, rxn]
    dataset["MAPE_OTHER"] = mape_other.loc[valid_idx]
    dataset["KPI"] = KPI.loc[valid_idx]

    return dataset, mape_df, KPI


def build_dataset_yeast_zscore(
    Km_df: pd.DataFrame,
    enzyme_df: pd.DataFrame,
    flux_exp_df: pd.DataFrame,
    oracle_enzyme_df: pd.DataFrame,
    important_rxns: list = None,
    condition: str = 'microaerobic',
    eps: float = 1e-12,
    reaction_scaling: dict = None,
) -> tuple:
    """
    Build a training dataset for yeast using oracle-referenced z-scores
    instead of MAPE as the per-reaction error metric.

    Delegates per-reaction scoring to :func:`compute_zscore_vs_oracle`:

        sigma_r = std over oracle models of |sim_oracle_r - exp_r|
        score_{r,m} = |sim_m_r - exp_r| / (sigma_r + eps)

    KPI aggregation mirrors :func:`build_dataset_yeast`:

        KPI = Σ ZSCORE(important reactions) + mean ZSCORE(all other reactions)

    Signal amplification via ``reaction_scaling``
    ---------------------------------------------
    Reactions with very small absolute flux values (e.g. LMPD_s_0450_c_1_256
    ~ 1e-7) can produce near-zero z-scores if the oracle ensemble is also
    diverse at that scale, burying the reaction's contribution in the KPI.

    ``reaction_scaling`` rescales the **candidate simulations and the
    experimental target only** — the oracle sigma is intentionally left in
    original units.  The resulting z-score for reaction r becomes:

        z_amplified_r = scale_r × z_original_r

    For example, ``reaction_scaling={"LMPD_s_0450_c_1_256": 1e6}`` converts
    the LMPD flux to growth-rate units before scoring, amplifying its KPI
    contribution by 10⁶ relative to reactions that are not scaled.  The oracle
    sigma acts as the reference in the original flux units, so the z-score is
    no longer dimensionless but the relative ranking of models is preserved.

    Parameters
    ----------
    Km_df : DataFrame
        Km parameters, rows = models.
    enzyme_df : DataFrame
        Simulated fluxes of the *candidate* models, rows = reactions,
        columns = model indices.
    flux_exp_df : DataFrame
        Experimental fluxes, rows = reactions, columns = oxygen conditions.
    oracle_enzyme_df : DataFrame
        Simulated fluxes of the *initial oracle ensemble*, used only to define
        the per-reaction scale sigma_r (not scored itself).
    important_rxns : list, optional
        Reactions tracked individually and summed into KPI.
        Defaults to ["GLCt1", "LMPD_s_0450_c_1_256", "ETOHt", "CO2t"].
    condition : str
        Column in ``flux_exp_df`` to use as the experimental reference.
        Defaults to ``'microaerobic'``.
    eps : float
        Small constant added to sigma_r to guard against division by zero.
    reaction_scaling : dict, optional
        Mapping of reaction name → scale factor.  Only ``enzyme_df`` and
        ``exp_vec`` rows are scaled (oracle is untouched), so the z-score for
        each listed reaction is multiplied by its scale factor.
        Example: ``{"LMPD_s_0450_c_1_256": 1e6}``

    Returns
    -------
    dataset : DataFrame
        Km parameters with ``ZSCORE_{rxn}`` columns for each important
        reaction, ``ZSCORE_OTHER``, and ``KPI``.
    zscore_df : DataFrame
        Full per-reaction z-score table, shape (reactions × models).
    KPI : Series
        Aggregated error per model (lower is better).
    """
    if important_rxns is None:
        important_rxns = ["GLCt1", "LMPD_s_0450_c_1_256", "ETOHt", "CO2t"]

    # Align reactions shared by simulated and experimental data
    common_rxns = enzyme_df.index.intersection(flux_exp_df.index)
    enzyme_aligned = enzyme_df.loc[common_rxns].copy()
    oracle_aligned = oracle_enzyme_df.reindex(common_rxns)

    if condition not in flux_exp_df.columns:
        raise ValueError(
            f"Condition '{condition}' not found in flux_exp_df. "
            f"Available columns: {flux_exp_df.columns.tolist()}"
        )
    exp_vec = flux_exp_df.loc[common_rxns, condition].copy()

    # Pre-scale predicted and experimental fluxes for specified reactions.
    # Oracle is left in original units so sigma_r reflects the oracle spread
    # before the unit conversion.  Scaling both sim and exp by the same factor
    # propagates the unit change consistently into the numerator:
    #
    #   numerator  : |scale × sim  −  scale × exp|  =  scale × |sim − exp|
    #   sigma_r    : std(|oracle_raw − scale × exp|)   (mixed-unit reference)
    #
    # Use this when the physical observable is scale × flux (e.g. growth rate
    # = 1e6 × LMPD flux) and you want the z-score computed in those units.
    if reaction_scaling:
        for rxn, scale in reaction_scaling.items():
            if rxn in enzyme_aligned.index:
                enzyme_aligned.loc[rxn] = enzyme_aligned.loc[rxn] * scale
            if rxn in exp_vec.index:
                exp_vec.loc[rxn] = exp_vec.loc[rxn] * scale

    # compute_zscore_vs_oracle → DataFrame (reactions × models)
    zscore_df = compute_zscore_vs_oracle(enzyme_aligned, exp_vec, oracle_aligned, eps=eps)

    missing = [r for r in important_rxns if r not in zscore_df.index]
    if missing:
        raise ValueError(f"Important reactions not found in zscore index: {missing}")

    # Transpose to (models × reactions)
    zscore_T = zscore_df.T

    zscore_imp = zscore_T[important_rxns]
    other_rxns = [r for r in zscore_T.columns if r not in important_rxns]
    zscore_other = zscore_T[other_rxns].mean(axis=1)

    KPI = zscore_imp.sum(axis=1) + zscore_other

    # Build final dataset
    valid_idx = Km_df.index.intersection(zscore_T.index)
    dataset = Km_df.loc[valid_idx].copy()
    for rxn in important_rxns:
        dataset[f"ZSCORE_{rxn}"] = zscore_imp.loc[valid_idx, rxn]
    dataset["ZSCORE_OTHER"] = zscore_other.loc[valid_idx]
    dataset["KPI"] = KPI.loc[valid_idx]

    return dataset, zscore_df, KPI


def build_dataset_zscore(km_file: str,
                         flux_exp_file: str,
                         enzyme_file: str,
                         oracle_enzyme_file: str,
                         cut_reaction: str):
    """
    Build a training dataset (Km parameters + Error column) where the
    per-reaction error is the z-score of the absolute residual relative to
    the initial oracle ensemble — see :func:`compute_zscore_vs_oracle`.

    Parameters
    ----------
    km_file : str
        CSV of Km parameters (rows = models, cols = parameters).
    flux_exp_file : str
        CSV of experimental fluxes (rows = reactions, cols = perturbations).
    enzyme_file : str
        Simulated fluxes of the candidate models we want to score
        (rows = reactions, cols = models).
    oracle_enzyme_file : str
        Simulated fluxes of the *initial oracle ensemble*, used to define
        the reference distribution (mu_r, sigma_r) for the z-score. For a
        cut at ``cut_reaction``, this is typically
        ``./data/enzyme_perturbations/<cut_reaction>.csv``.
    cut_reaction : str
        Reaction that was cut (e.g. 'PGL').

    Returns
    -------
    dataset : DataFrame
        Km parameters with an added ``Error`` column (lower = better fit).
    zscore : DataFrame
        Per-reaction z-scores (rows = reactions, cols = models).
    KPI : Series
        Aggregated error per model (key-reaction sum + other-reaction mean).
    """
    fluxes_exp = _load_fluxes_exp(flux_exp_file)
    enzyme_df = _load_enzyme_df(enzyme_file, fluxes_exp.index)
    oracle_df = _load_enzyme_df(oracle_enzyme_file, fluxes_exp.index)
    Km = pd.read_csv(km_file, index_col=0)

    exp_column = cut_reaction.lower()
    if exp_column not in fluxes_exp.columns:
        raise ValueError(f"Experimental column '{exp_column}' not found in {flux_exp_file}")

    exp_vec = fluxes_exp[exp_column].reindex(enzyme_df.index)

    # Align oracle on the same reaction set used for scoring.
    oracle_df = oracle_df.reindex(enzyme_df.index)

    zscore = compute_zscore_vs_oracle(enzyme_df, exp_vec, oracle_df)
    if cut_reaction=="G6PDH2r":
        cut_reaction="PGL"
    key_rxns = get_key_reactions(cut_reaction)
    Km = Km.loc[Km.index.isin(enzyme_df.columns)]
    KPI = compute_KPI(zscore, key_rxns)

    dataset = Km.copy()
    dataset["Error"] = KPI

    return dataset, zscore, KPI

