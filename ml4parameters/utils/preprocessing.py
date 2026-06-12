# utils/preprocessing.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ============================================================
#  Align DataFrames by common indices
# ============================================================

def align_dataframes(df1: pd.DataFrame, df2: pd.DataFrame):
    """
    Returns df1 and df2 restricted to the intersection of their indices,
    sorted in the same order.
    """
    common_index = df1.index.intersection(df2.index)
    common_index = common_index.sort_values()

    df1_aligned = df1.loc[common_index].copy()
    df2_aligned = df2.loc[common_index].copy()

    return df1_aligned, df2_aligned


# ============================================================
#  Generic scaling helper
# ============================================================

def generate_scaled_versions(X: np.ndarray, log_transform: bool = True):
    """
    Given a feature matrix X, returns several scaled versions and scalers.
    - X_standard: StandardScaler
    - X_minmax: MinMaxScaler
    - X_log_standard: log(X) then StandardScaler
    """
    # Standard scaling
    scaler_standard = StandardScaler()
    X_standard = scaler_standard.fit_transform(X)

    # MinMax scaling
    scaler_minmax = MinMaxScaler()
    X_minmax = scaler_minmax.fit_transform(X)

    # Log + standard
    if log_transform:
        X_log = np.log(X + 1e-8)
    else:
        X_log = X.copy()

    scaler_log_standard = StandardScaler()
    X_log_standard = scaler_log_standard.fit_transform(X_log)

    return {
        "X_standard": X_standard,
        "X_minmax": X_minmax,
        "X_log_standard": X_log_standard,
        "scalers": {
            "standard": scaler_standard,
            "minmax": scaler_minmax,
            "log_standard": scaler_log_standard,
        },
    }


# ============================================================
#  Single-label preprocessing (one df with KMs + Error)
# ============================================================

def preprocess_single_label(
    df: pd.DataFrame,
    error_col: str = "Error",
    percentile_cutoff: float = 10.0,
    log_transform: bool = True,
):
    """
    For a single table with KMs and an Error column.

    - X = all columns except error_col
    - y_binary = 1 for samples in the lowest `percentile_cutoff` % Error,
                 0 otherwise.
    - Returns multiple scaled versions of X and the binary label.

    Returns dict:
        {
          "X_standard",
          "X_minmax",
          "X_log_standard",
          "y_binary",
          "scalers",
          "threshold",
        }
    """
    if error_col not in df.columns:
        raise ValueError(f"Error column '{error_col}' not found in dataframe.")

    X = df.drop(columns=[error_col]).values
    y = df[error_col].values

    threshold = np.percentile(y, percentile_cutoff)
    y_binary = (y <= threshold).astype(int).reshape(-1, 1)

    scaled = generate_scaled_versions(X, log_transform=log_transform)

    return {
        "X_standard": scaled["X_standard"],
        "X_minmax": scaled["X_minmax"],
        "X_log_standard": scaled["X_log_standard"],
        "y_binary": y_binary,
        "scalers": scaled["scalers"],
        "threshold": threshold,
    }


# ============================================================
#  Multi-label preprocessing from TWO dfs
# ============================================================

def preprocess_multilabel_two_dfs(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    error_col: str = "Error",
    percentile_cutoff=(10.0, 10.0),   # NOW accepts tuple/list
    log_transform: bool = True,
):
    """
    Multi-label preprocessing for TWO perturbations.

    Now supports DIFFERENT cutoff percentiles for each perturbation.
    Example:
        percentile_cutoff = (10, 15)

    Steps:
        1. Align df1 and df2 on common indices.
        2. Extract KM columns.
        3. Compute separate percentile thresholds.
        4. Create binary labels.
        5. Scale X using multiple scalers.

    Returns dict with:
        X_standard, X_minmax, X_log_standard
        y_multilabel
        scalers
        thresholds   (actual error values)
        df1_aligned
        df2_aligned
    """

    # 0. Ensure we have two cutoff values
    if isinstance(percentile_cutoff, (int, float)):
        p1 = p2 = float(percentile_cutoff)
    else:
        if len(percentile_cutoff) != 2:
            raise ValueError("percentile_cutoff must be a single value or a tuple/list of two values.")
        p1, p2 = float(percentile_cutoff[0]), float(percentile_cutoff[1])

    # 1. Align rows
    df1_aligned, df2_aligned = align_dataframes(df1, df2)

    # 2. Check KM column consistency
    km_cols1 = [c for c in df1_aligned.columns if c != error_col]
    km_cols2 = [c for c in df2_aligned.columns if c != error_col]

    if km_cols1 != km_cols2:
        raise ValueError(
            "KM columns do not match between the two dataframes.\n"
            f"df1 KMs: {km_cols1}\n"
            f"df2 KMs: {km_cols2}"
        )

    # 3. Features (X)
    X = df1_aligned[km_cols1].values

    # 4. Labels & thresholds
    y_err1 = df1_aligned[error_col].values
    y_err2 = df2_aligned[error_col].values

    thr1 = np.percentile(y_err1, p1)
    thr2 = np.percentile(y_err2, p2)

    print(f"Cut 1 threshold: Error <= {thr1:.6f}  (percentile {p1})")
    print(f"Cut 2 threshold: Error <= {thr2:.6f}  (percentile {p2})")

    y1 = (y_err1 <= thr1).astype(int)
    y2 = (y_err2 <= thr2).astype(int)

    y_multilabel = np.stack([y1, y2], axis=1)

    # 5. Scaling
    scaled = generate_scaled_versions(X, log_transform=log_transform)

    return {
        "X_standard": scaled["X_standard"],
        "X_minmax": scaled["X_minmax"],
        "X_log_standard": scaled["X_log_standard"],
        "y_multilabel": y_multilabel.astype(int),
        "scalers": scaled["scalers"],
        "thresholds": [thr1, thr2],
        "df1_aligned": df1_aligned,
        "df2_aligned": df2_aligned,
    }

import pandas as pd
import numpy as np

def preprocess_multilabel_dfs(
    df_list,
    error_col="Error",
    percentile_cutoffs=10.0,
    log_transform=True,
):
    """
    Multi-label preprocessing for ANY number of DataFrames.
    """

    import numpy as np

    N = len(df_list)

    # -----------------------
    # Handle cutoff input
    # -----------------------
    if isinstance(percentile_cutoffs, (int, float)):
        pct_list = [float(percentile_cutoffs)] * N
    else:
        if len(percentile_cutoffs) != N:
            raise ValueError(
                f"percentile_cutoffs must be a single value or a list of {N} values."
            )
        pct_list = list(map(float, percentile_cutoffs))

    # -----------------------
    # Align all dataframes on the SAME index
    # -----------------------
    common_index = df_list[0].index
    for df in df_list[1:]:
        common_index = common_index.intersection(df.index)

    common_index = common_index.sort_values()
    df_aligned = [df.loc[common_index].copy() for df in df_list]

    # -----------------------
    # Check KM column consistency
    # -----------------------
    km_cols = [c for c in df_aligned[0].columns if c != error_col]
    for i, df in enumerate(df_aligned):
        km_cols_i = [c for c in df.columns if c != error_col]
        if km_cols_i != km_cols:
            raise ValueError(
                f"KM columns mismatch in df {i}. "
                f"Expected {km_cols}, got {km_cols_i}"
            )

    # -----------------------
    # Extract X (same for all)
    # -----------------------
    X = df_aligned[0][km_cols].values

    # -----------------------
    # Compute labels and thresholds
    # -----------------------
    thresholds = []
    labels = []

    print("\nThresholds per label:")
    print("-" * 40)

    for i, (df, pct) in enumerate(zip(df_aligned, pct_list)):
        y_err = df[error_col].values
        thr = np.percentile(y_err, pct)

        thresholds.append(thr)

        print(f"Label {i}: Error ≤ {thr:.6f}  (percentile {pct})")

        y = (y_err <= thr).astype(int)
        labels.append(y)

    print("-" * 40)

    # Stack into multilabel matrix
    y_multilabel = np.vstack(labels).T   # shape (n_samples, N_labels)

    # -----------------------
    # Scale X using your existing scaler function
    # -----------------------
    scaled = generate_scaled_versions(X, log_transform=log_transform)

    return {
        "X_standard": scaled["X_standard"],
        "X_minmax": scaled["X_minmax"],
        "X_log_standard": scaled["X_log_standard"],
        "y_multilabel": y_multilabel,
        "thresholds": thresholds,
        "df_aligned_list": df_aligned,
        "scalers": scaled["scalers"]
    }





