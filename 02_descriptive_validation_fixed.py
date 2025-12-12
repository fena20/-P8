"""
02_descriptive_validation_fixed.py
=================================

Descriptive Statistics and Macro-Level Validation (RECS 2020)

What this module does
---------------------
1) Loads the processed dataset from Step 1 (e.g., 03_gas_heated_clean.csv)
2) Computes WEIGHTED descriptive statistics (point estimates with NWEIGHT)
3) Provides diagnostics / validation scaffolding against official RECS tables
4) Produces tables and figures for thesis/paper

Key fixes vs. the original 02_descriptive_validation.py
-------------------------------------------------------
- Robust handling of RECS missing/special codes (-1, -2, -3, -7, -8, -9) on load.
- Safer weighted estimators (handle empty masks, nonpositive weights).
- Weighted quantiles use interpolation (more stable than step-function cutoff).
- Replicate-weight RSE:
    * If NWEIGHT1..NWEIGHT60 exist, uses the RECS 2020 Jackknife replicate formula:
          Var_hat = (R-1)/R * sum_r (theta_r - theta)^2
      (per EIA RECS 2020 microdata guide).
    * If BRRWT* exist (older RECS), uses Fay's BRR with rho=0.5 by default.
- Table 1 variable dictionary updated to allow BOTH BTUSPH and TOTALBTUSPH.

Author: Fafa (GitHub: Fateme9977)
Institution: K. N. Toosi University of Technology
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -------------------------
# Paths
# -------------------------
PROJECT_ROOT = Path(r"C:\Users\FATEME\Desktop\Energy")
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# -------------------------
# Plotting style (safe)
# -------------------------
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    logger.warning("Could not load seaborn-v0_8-whitegrid style; using matplotlib defaults.")

sns.set_palette("husl")

# -------------------------
# RECS special / missing codes
# -------------------------
RECS_MISSING_CODES = {-1, -2, -3, -7, -8, -9}


def replace_recs_missing_codes(
    df: pd.DataFrame,
    exclude_prefixes: Sequence[str] = ("NWEIGHT", "BRRWT"),
    extra_exclude: Sequence[str] = (),
) -> pd.DataFrame:
    """
    Replace common RECS special codes (-1, -2, -3, -7, -8, -9) with NaN.

    Notes:
      - We avoid touching weight columns (NWEIGHT, NWEIGHT1.., BRRWT..).
      - For object columns, we attempt numeric coercion when it looks mostly numeric.
    """
    df = df.copy()
    exclude_prefixes = tuple(exclude_prefixes) + tuple(extra_exclude)

    for col in df.columns:
        col_str = str(col)
        if any(col_str.startswith(p) for p in exclude_prefixes):
            continue

        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            df[col] = s.replace(list(RECS_MISSING_CODES), np.nan)
        elif pd.api.types.is_object_dtype(s):
            coerced = pd.to_numeric(s, errors="coerce")
            # if it looks numeric enough, use numeric coercion
            if coerced.notna().sum() >= max(10, int(0.05 * len(coerced))):
                df[col] = coerced.replace(list(RECS_MISSING_CODES), np.nan)

    return df


# ==========================================================
# Data loading
# ==========================================================
def load_processed_data(use_clean: bool = True) -> pd.DataFrame:
    """Load the processed dataset from Step 1."""
    filepath = OUTPUT_DIR / ("03_gas_heated_clean.csv" if use_clean else "02_gas_heated_full.csv")

    if not filepath.exists():
        raise FileNotFoundError(
            f"Processed data not found at {filepath}. Run 01_data_prep.py (or the fixed version) first."
        )

    df = pd.read_csv(filepath)
    df = replace_recs_missing_codes(df)
    logger.info(f"Loaded {len(df):,} households from {filepath}")
    return df


# ==========================================================
# Weighted estimators (point estimates)
# ==========================================================
def _valid_weight_mask(df: pd.DataFrame, var: str, weight: str) -> np.ndarray:
    if var not in df.columns or weight not in df.columns:
        return np.zeros(len(df), dtype=bool)
    m = df[var].notna() & df[weight].notna()
    # treat nonpositive weights as invalid
    m &= df[weight].astype(float) > 0
    return m.values


def weighted_mean(df: pd.DataFrame, var: str, weight: str = "NWEIGHT") -> float:
    """Weighted mean (NaN-safe)."""
    m = _valid_weight_mask(df, var, weight)
    if not m.any():
        return np.nan
    x = df.loc[m, var].astype(float).values
    w = df.loc[m, weight].astype(float).values
    return float(np.average(x, weights=w))


def weighted_std(df: pd.DataFrame, var: str, weight: str = "NWEIGHT") -> float:
    """
    Weighted population standard deviation (NaN-safe).

    Note: This is NOT an unbiased estimator under complex survey design;
    for inference, prefer replicate weights / survey packages.
    """
    m = _valid_weight_mask(df, var, weight)
    if not m.any():
        return np.nan
    x = df.loc[m, var].astype(float).values
    w = df.loc[m, weight].astype(float).values
    mu = np.average(x, weights=w)
    var_w = np.average((x - mu) ** 2, weights=w)
    return float(np.sqrt(var_w))


def weighted_quantile(
    df: pd.DataFrame, var: str, q: float, weight: str = "NWEIGHT"
) -> float:
    """
    Weighted quantile with linear interpolation over the weighted CDF.

    q in [0, 1].
    """
    if q < 0 or q > 1:
        raise ValueError("q must be in [0, 1].")

    m = _valid_weight_mask(df, var, weight)
    if not m.any():
        return np.nan

    x = df.loc[m, var].astype(float).values
    w = df.loc[m, weight].astype(float).values

    sorter = np.argsort(x)
    x = x[sorter]
    w = w[sorter]

    cum_w = np.cumsum(w)
    total = cum_w[-1]
    if total <= 0:
        return np.nan

    cdf = cum_w / total

    # Interpolate; clamp q to [cdf[0], cdf[-1]] to avoid edge issues.
    q_eff = min(max(q, float(cdf[0])), float(cdf[-1]))
    return float(np.interp(q_eff, cdf, x))


def weighted_proportion(
    df: pd.DataFrame,
    var: str,
    category: Union[Any, Sequence[Any]],
    weight: str = "NWEIGHT",
) -> float:
    """Weighted proportion for a category (or multiple categories)."""
    if var not in df.columns or weight not in df.columns:
        return np.nan

    m = df[var].notna() & df[weight].notna() & (df[weight].astype(float) > 0)
    if not m.any():
        return np.nan

    total_w = float(df.loc[m, weight].astype(float).sum())
    if total_w <= 0:
        return np.nan

    if isinstance(category, (list, tuple, set, np.ndarray)):
        in_cat = df[var].isin(category)
    else:
        in_cat = df[var] == category

    cat_w = float(df.loc[m & in_cat, weight].astype(float).sum())
    return cat_w / total_w


# ==========================================================
# Batch stats helper
# ==========================================================
def compute_weighted_stats(
    df: pd.DataFrame,
    numeric_vars: List[str],
    categorical_vars: List[str],
    groupby: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute a dictionary of weighted stats for overall and optional groups."""
    results: Dict[str, Dict[str, float]] = {}

    if groupby and groupby in df.columns:
        groups = pd.Series(df[groupby]).dropna().unique().tolist()
        for g in groups:
            gdf = df[df[groupby] == g]
            results[str(g)] = _compute_stats_for_group(gdf, numeric_vars, categorical_vars)
    else:
        results["overall"] = _compute_stats_for_group(df, numeric_vars, categorical_vars)

    return results


def _compute_stats_for_group(
    df: pd.DataFrame, numeric_vars: List[str], categorical_vars: List[str]
) -> Dict[str, float]:
    out: Dict[str, float] = {
        "n_sample": float(len(df)),
        "n_weighted": float(df["NWEIGHT"].sum()) if "NWEIGHT" in df.columns else float(len(df)),
    }

    for v in numeric_vars:
        if v in df.columns:
            out[f"{v}_mean"] = weighted_mean(df, v)
            out[f"{v}_std"] = weighted_std(df, v)
            out[f"{v}_median"] = weighted_quantile(df, v, 0.5)
            out[f"{v}_q25"] = weighted_quantile(df, v, 0.25)
            out[f"{v}_q75"] = weighted_quantile(df, v, 0.75)

    for v in categorical_vars:
        if v in df.columns:
            cats = pd.Series(df[v]).dropna().unique().tolist()
            for c in cats:
                p = weighted_proportion(df, v, c)
                out[f"{v}_{c}_pct"] = p * 100 if not np.isnan(p) else np.nan

    return out


# ==========================================================
# Tables
# ==========================================================
def generate_table1_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Generate Table 1: Overview of variables used in analysis."""
    logger.info("Generating Table 1: Variable definitions")

    variables = [
        # Identification & weights
        ("DOEID", "ID", "-", "Unique household identifier", "Identification"),
        ("NWEIGHT", "w", "-", "Sample weight for national estimates", "Weight"),

        # Climate
        ("HDD65", "HDD₆₅", "°F·day/yr", "Heating degree days (base 65°F)", "Climate"),
        ("CDD65", "CDD₆₅", "°F·day/yr", "Cooling degree days (base 65°F)", "Climate"),

        # Building geometry (raw RECS fields)
        ("TOTHSQFT", "A_total", "ft²", "Reported total floor area of the housing unit (survey)", "Building"),
        ("TOTSQFT_EN", "A_cond", "ft²", "EIA-adjusted conditioned floor area (if provided)", "Building"),

        # Building geometry (derived field)
        ("A_heated", "A_heated", "ft²", "Heated/conditioned floor area used in intensity calculations (derived)", "Derived"),

        # Building type & vintage
        ("YEARMADERANGE", "-", "category", "Year built range (RECS code)", "Building"),
        ("TYPEHUQ", "-", "category", "Housing type (RECS code)", "Building"),

        # Envelope
        ("DRAFTY", "-", "ordinal", "Draftiness level (RECS code)", "Envelope"),
        ("ADQINSUL", "-", "ordinal", "Insulation adequacy (RECS code)", "Envelope"),
        ("TYPEGLASS", "-", "category", "Window glass type (RECS code)", "Envelope"),

        # Heating system
        ("FUELHEAT", "-", "category", "Main heating fuel (RECS code)", "Heating"),
        ("EQUIPM", "-", "category", "Heating equipment type (RECS code)", "Heating"),
        ("EQUIPAGE", "-", "category", "Heating equipment age category (RECS code)", "Heating"),

        # Raw RECS energy quantities (commonly kBTU in microdata)
        ("TOTALBTUSPH", "E_heat_raw", "kBTU/yr", "Annual space heating energy (survey; if present)", "Consumption"),
        ("BTUSPH", "E_heat_raw", "kBTU/yr", "Annual space heating energy (survey; alternate naming)", "Consumption"),
        ("BTUNG", "E_gas_raw", "kBTU/yr", "Annual natural gas use (survey; thousand BTU)", "Consumption"),

        # Derived energy quantities used in analysis
        ("E_heat_btu", "E_heat", "BTU/yr", "Annual space heating energy (derived from space-heating and/or gas totals)", "Derived"),

        # Expenditures & geography
        ("TOTALDOLLARSPH", "$_heat", "USD/yr", "Annual heating expenditure (if present)", "Expenditure"),
        ("DOLLARSPH", "$_heat", "USD/yr", "Annual heating expenditure (alternate naming)", "Expenditure"),
        ("REGIONC", "-", "category", "Census region (1–4)", "Geography"),
        ("DIVISION", "-", "category", "Census division code (1–10)", "Geography"),
        ("division_name", "-", "category", "Census division (name; derived)", "Geography"),

        # Final analysis variables
        ("Thermal_Intensity_I", "I", "BTU/(ft²·HDD)", "Heating thermal intensity", "Derived"),
        ("envelope_class", "-", "category", "Envelope efficiency class (poor / medium / good)", "Derived"),
        ("climate_zone", "-", "category", "Climate zone (mild / mixed / cold)", "Derived"),
    ]

    table1 = pd.DataFrame(variables, columns=["Variable", "Symbol", "Unit", "Description", "Category"])
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    table1.to_csv(TABLES_DIR / "table1_variable_definitions.csv", index=False)
    try:
        table1.to_latex(
            TABLES_DIR / "table1_variable_definitions.tex",
            index=False,
            caption="Definition, units, and categories of main variables used in the analysis.",
            label="tab:variables",
        )
    except Exception as e:
        logger.warning(f"Failed to export LaTeX for Table 1: {e}")

    return table1


def generate_table2_sample_characteristics(df: pd.DataFrame) -> pd.DataFrame:
    """Generate Table 2: Weighted sample characteristics by division and envelope class."""
    logger.info("Generating Table 2: Sample characteristics")

    required = {"NWEIGHT", "HDD65", "A_heated", "Thermal_Intensity_I", "division_name", "envelope_class"}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        logger.warning(f"Table 2 skipped (missing columns): {missing}")
        return pd.DataFrame()

    results: List[Dict[str, Any]] = []

    for division in df["division_name"].dropna().unique():
        div_df = df[df["division_name"] == division]

        for env_class in ["poor", "medium", "good"]:
            subset = div_df[div_df["envelope_class"] == env_class]
            if len(subset) == 0:
                continue

            row = {
                "Division": division,
                "Envelope Class": env_class,
                "N (sample)": int(len(subset)),
                "N (weighted, millions)": float(subset["NWEIGHT"].sum() / 1e6),
                "Mean HDD65": weighted_mean(subset, "HDD65"),
                "Mean Sqft": weighted_mean(subset, "A_heated"),
                "Mean Thermal Intensity": weighted_mean(subset, "Thermal_Intensity_I"),
            }
            results.append(row)

    table2 = pd.DataFrame(results)
    if len(table2) > 0:
        table2["N (weighted, millions)"] = table2["N (weighted, millions)"].round(2)
        table2["Mean HDD65"] = table2["Mean HDD65"].round(0)
        table2["Mean Sqft"] = table2["Mean Sqft"].round(0)
        table2["Mean Thermal Intensity"] = table2["Mean Thermal Intensity"].round(3)

    table2.to_csv(TABLES_DIR / "table2_sample_characteristics.csv", index=False)
    return table2


# ==========================================================
# Figures
# ==========================================================
def generate_figure2_climate_envelope(df: pd.DataFrame) -> None:
    """
    Figure 2:
      (a) HDD65 by division – weighted median and IQR using NWEIGHT
      (b) Envelope class shares – weighted distribution with sample sizes
    """
    logger.info("Generating Figure 2: Climate and envelope overview")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) HDD65 by division
    ax1 = axes[0]
    if {"division_name", "HDD65", "NWEIGHT"}.issubset(df.columns):
        rows = []
        for division in df["division_name"].dropna().unique():
            sub = df[df["division_name"] == division]
            if len(sub) == 0:
                continue
            rows.append(
                {
                    "division_name": division,
                    "q25": weighted_quantile(sub, "HDD65", 0.25),
                    "median": weighted_quantile(sub, "HDD65", 0.50),
                    "q75": weighted_quantile(sub, "HDD65", 0.75),
                }
            )

        if rows:
            stats_df = pd.DataFrame(rows).sort_values("median", ascending=False)
            x_positions = np.arange(len(stats_df))
            ax1.set_xticks(x_positions)
            ax1.set_xticklabels(stats_df["division_name"], rotation=45, ha="right")

            for idx, row in enumerate(stats_df.itertuples(index=False)):
                ax1.vlines(
                    x=idx,
                    ymin=row.q25,
                    ymax=row.q75,
                    linewidth=6,
                    color="steelblue",
                    alpha=0.8,
                )
                ax1.scatter(idx, row.median, s=40, color="black", zorder=3)

            ax1.set_xlabel("Census Division", fontsize=12)
            ax1.set_ylabel("Heating degree days, HDD65 (°F·day/year)", fontsize=12)
            ax1.set_title("(a) HDD65 (weighted median and IQR) by division", fontsize=14)
    else:
        ax1.axis("off")
        ax1.set_title("(a) HDD65 by division (missing required columns)", fontsize=12)

    # (b) Envelope class shares
    ax2 = axes[1]
    if {"envelope_class", "NWEIGHT"}.issubset(df.columns):
        env_shares = df.groupby("envelope_class")["NWEIGHT"].sum()
        env_shares = env_shares / env_shares.sum() * 100

        env_counts = df["envelope_class"].value_counts()

        colors = {"poor": "#d62728", "medium": "#ff7f0e", "good": "#2ca02c"}
        bars = ax2.bar(env_shares.index, env_shares.values, color=[colors.get(x, "gray") for x in env_shares.index])

        for bar, pct, cls in zip(bars, env_shares.values, env_shares.index):
            n = env_counts.get(cls, np.nan)
            label = f"{pct:.1f}%"
            if not pd.isna(n):
                label += f"\n(n={int(n):,})"
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                label,
                ha="center",
                va="bottom",
                fontsize=11,
            )

        ax2.set_xlabel("Envelope efficiency class", fontsize=12)
        ax2.set_ylabel("Share of gas-heated homes (%) – weighted", fontsize=12)
        ax2.set_title("(b) Envelope class distribution (weighted)", fontsize=14)
        ax2.set_ylim(0, 100)
    else:
        ax2.axis("off")
        ax2.set_title("(b) Envelope distribution (missing required columns)", fontsize=12)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure2_climate_envelope.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "figure2_climate_envelope.pdf", bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Figure 2 to {FIGURES_DIR}")


def generate_figure3_thermal_intensity_distribution(df: pd.DataFrame) -> None:
    """
    Figure 3: Distribution of thermal intensity by envelope and climate.

    Important:
        Boxplots here are UNWEIGHTED and visualize the *sample* distribution.
        Weighted summaries are reported in tables.
    """
    logger.info("Generating Figure 3: Thermal intensity distribution")

    if not {"Thermal_Intensity_I", "envelope_class"}.issubset(df.columns):
        logger.warning("Figure 3 skipped (missing required columns).")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    sns.boxplot(
        data=df,
        x="envelope_class",
        y="Thermal_Intensity_I",
        order=["poor", "medium", "good"],
        palette={"poor": "#d62728", "medium": "#ff7f0e", "good": "#2ca02c"},
        ax=ax1,
    )
    ax1.set_xlabel("Envelope efficiency class", fontsize=12)
    ax1.set_ylabel("Thermal intensity, I (BTU/(ft²·HDD))", fontsize=12)
    ax1.set_title("(a) By envelope class (unweighted)", fontsize=14)

    ax2 = axes[1]
    if "climate_zone" in df.columns:
        sns.boxplot(
            data=df,
            x="climate_zone",
            y="Thermal_Intensity_I",
            order=["mild", "mixed", "cold"],
            palette="coolwarm",
            ax=ax2,
        )
        ax2.set_xlabel("Climate zone", fontsize=12)
        ax2.set_ylabel("Thermal intensity, I (BTU/(ft²·HDD))", fontsize=12)
        ax2.set_title("(b) By climate zone (unweighted)", fontsize=14)
    else:
        ax2.axis("off")
        ax2.set_title("(b) By climate zone (missing climate_zone column)", fontsize=12)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure3_thermal_intensity_distribution.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "figure3_thermal_intensity_distribution.pdf", bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Figure 3 to {FIGURES_DIR}")


# ==========================================================
# Validation scaffolding
# ==========================================================
def validate_against_hc_tables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate microdata aggregates against official RECS HC tables (SCaffold).

    NOTE: This function currently stores *table references* (e.g., "HC6.1") rather than
    hard-coding official values. To do numeric validation, provide a local copy of the HC
    tables and implement the official-value lookup.
    """
    logger.info("Creating validation scaffolding against official RECS tables...")

    validation_results: List[Dict[str, Any]] = []

    if "NWEIGHT" in df.columns:
        total_units = df["NWEIGHT"].sum() / 1e6
        validation_results.append(
            {
                "Metric": "Total gas-heated units (millions)",
                "Microdata": round(float(total_units), 2),
                "Official_table_ref": "HC6.1",
                "Notes": "Compare with natural gas heating row (total units).",
            }
        )

    if "A_heated" in df.columns:
        mean_sqft = weighted_mean(df, "A_heated")
        validation_results.append(
            {
                "Metric": "Mean heated floor area (ft²)",
                "Microdata": round(float(mean_sqft), 0) if not np.isnan(mean_sqft) else np.nan,
                "Official_table_ref": "HC10.1",
                "Notes": "Compare with overall mean (or relevant subgroup).",
            }
        )

    if "HDD65" in df.columns:
        mean_hdd = weighted_mean(df, "HDD65")
        validation_results.append(
            {
                "Metric": "Mean HDD65",
                "Microdata": round(float(mean_hdd), 0) if not np.isnan(mean_hdd) else np.nan,
                "Official_table_ref": "-",
                "Notes": "Reference only (not always in HC tables).",
            }
        )

    # Housing type distribution
    if "housing_type" in df.columns:
        for htype in pd.Series(df["housing_type"]).dropna().unique():
            pct = weighted_proportion(df, "housing_type", htype) * 100
            validation_results.append(
                {
                    "Metric": f"Housing type: {htype} (%)",
                    "Microdata": round(float(pct), 1) if not np.isnan(pct) else np.nan,
                    "Official_table_ref": "HC2.1",
                    "Notes": "Compare with housing type rows.",
                }
            )

    # Year built distribution
    if "year_built_cat" in df.columns:
        for yb in pd.Series(df["year_built_cat"]).dropna().unique():
            pct = weighted_proportion(df, "year_built_cat", yb) * 100
            validation_results.append(
                {
                    "Metric": f"Year built: {yb} (%)",
                    "Microdata": round(float(pct), 1) if not np.isnan(pct) else np.nan,
                    "Official_table_ref": "HC2.1",
                    "Notes": "Compare with year built rows.",
                }
            )

    # Division distribution
    if "division_name" in df.columns:
        for div in pd.Series(df["division_name"]).dropna().unique():
            pct = weighted_proportion(df, "division_name", div) * 100
            validation_results.append(
                {
                    "Metric": f"Division: {div} (%)",
                    "Microdata": round(float(pct), 1) if not np.isnan(pct) else np.nan,
                    "Official_table_ref": "HC6.1",
                    "Notes": "Compare with division columns.",
                }
            )

    validation_df = pd.DataFrame(validation_results)
    validation_df.to_csv(TABLES_DIR / "validation_against_official.csv", index=False)
    return validation_df


def generate_figure4_validation(df: pd.DataFrame, validation_df: Optional[pd.DataFrame] = None) -> None:
    """
    Figure 4:
      (a) Weighted mean heating energy by division
      (b) Weighted housing-type distribution with sample sizes
    """
    logger.info("Generating Figure 4: Validation comparison")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) mean heating energy by division (weighted)
    ax1 = axes[0]
    if {"division_name", "E_heat_btu", "NWEIGHT"}.issubset(df.columns):
        rows = []
        for div in df["division_name"].dropna().unique():
            sub = df[df["division_name"] == div]
            rows.append(
                {
                    "division_name": div,
                    "mean_btu": weighted_mean(sub, "E_heat_btu"),
                    "n_weighted": float(sub["NWEIGHT"].sum()),
                }
            )
        div_stats = pd.DataFrame(rows).sort_values("mean_btu", ascending=False)
        ax1.barh(div_stats["division_name"], div_stats["mean_btu"] / 1e6)
        ax1.set_xlabel("Mean heating energy, E_heat (million BTU/year – weighted)", fontsize=12)
        ax1.set_ylabel("Census Division", fontsize=12)
        ax1.set_title("(a) Mean heating energy by division", fontsize=14)
    else:
        ax1.axis("off")
        ax1.set_title("(a) Mean heating energy (missing required columns)", fontsize=12)

    # (b) housing type distribution (weighted)
    ax2 = axes[1]
    if {"housing_type", "NWEIGHT"}.issubset(df.columns):
        htype_shares = df.groupby("housing_type")["NWEIGHT"].sum()
        htype_shares = (htype_shares / htype_shares.sum() * 100).sort_values(ascending=True)

        htype_counts = df["housing_type"].value_counts()

        bars = ax2.barh(htype_shares.index.astype(str), htype_shares.values, color="steelblue")

        for bar, pct, hname in zip(bars, htype_shares.values, htype_shares.index):
            n = htype_counts.get(hname, np.nan)
            label = f"{pct:.1f}%"
            if not pd.isna(n):
                label += f" (n={int(n):,})"
            ax2.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                label,
                ha="left",
                va="center",
                fontsize=10,
            )

        ax2.set_xlabel("Share of gas-heated homes (%) – weighted", fontsize=12)
        ax2.set_ylabel("Housing type", fontsize=12)
        ax2.set_title("(b) Housing type distribution (weighted)", fontsize=14)
        ax2.set_xlim(0, min(100, max(60, float(htype_shares.max()) + 10)))
    else:
        ax2.axis("off")
        ax2.set_title("(b) Housing type distribution (missing required columns)", fontsize=12)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure4_validation.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "figure4_validation.pdf", bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Figure 4 to {FIGURES_DIR}")


# ==========================================================
# Replicate-weight RSE (diagnostic)
# ==========================================================
def compute_replicate_weight_rse(df: pd.DataFrame, var: str) -> float:
    """
    Compute approximate Relative Standard Error (RSE, %) for a *weighted mean*
    using RECS replicate weights.

    Replicate-weight detection:
      - If NWEIGHT1..NWEIGHT60 exist: Jackknife replicates (RECS 2020 public microdata).
        Variance estimate:
            Var_hat = (R-1)/R * sum_r (theta_r - theta)^2
      - Else if BRRWT* exist: Fay's BRR (older RECS releases); variance:
            Var_hat = 1/(R*(1-rho)^2) * sum_r (theta_r - theta)^2
        with rho=0.5 by default (common in RECS examples).

    Returns:
        RSE (%) = 100 * SE / |theta|
    """
    if "NWEIGHT" not in df.columns or var not in df.columns:
        return np.nan

    # prefer JK1 (NWEIGHT1..), otherwise Fay BRR (BRRWT..)
    jk_cols = [c for c in df.columns if c.startswith("NWEIGHT") and c != "NWEIGHT"]
    brr_cols = [c for c in df.columns if c.startswith("BRRWT")]

    if len(jk_cols) > 0:
        rep_cols = jk_cols
        rep_type = "JK1"
    elif len(brr_cols) > 0:
        rep_cols = brr_cols
        rep_type = "FAY_BRR"
    else:
        logger.warning("No replicate weight columns (NWEIGHT1.. or BRRWT..) found for RSE calculation.")
        return np.nan

    # main estimate
    m_main = _valid_weight_mask(df, var, "NWEIGHT")
    if not m_main.any():
        return np.nan

    theta = float(np.average(df.loc[m_main, var].astype(float), weights=df.loc[m_main, "NWEIGHT"].astype(float)))
    if theta == 0 or np.isnan(theta):
        return np.nan

    # replicate estimates
    thetas = []
    for rw in rep_cols:
        if rw not in df.columns:
            continue
        m = _valid_weight_mask(df, var, rw)
        if not m.any():
            continue
        th = float(np.average(df.loc[m, var].astype(float), weights=df.loc[m, rw].astype(float)))
        if not np.isnan(th):
            thetas.append(th)

    R = len(thetas)
    if R == 0:
        logger.warning("No valid replicate estimates for RSE calculation.")
        return np.nan

    thetas = np.asarray(thetas, dtype=float)

    if rep_type == "JK1":
        variance = (R - 1) / R * np.sum((thetas - theta) ** 2)
    else:
        rho = 0.5
        variance = 1.0 / (R * (1.0 - rho) ** 2.0) * np.sum((thetas - theta) ** 2)

    se = float(np.sqrt(variance))
    rse = float(se / abs(theta) * 100.0)
    return rse


# ==========================================================
# Summary tables
# ==========================================================
def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics with replicate-weight RSE diagnostics."""
    logger.info("Generating summary statistics with RSEs...")

    numeric_vars = ["HDD65", "A_heated", "E_heat_btu", "Thermal_Intensity_I"]
    rows = []

    for v in numeric_vars:
        if v not in df.columns:
            continue

        row = {
            "Variable": v,
            "N": int(df[v].notna().sum()),
            "Mean (weighted)": weighted_mean(df, v),
            "Std (weighted)": weighted_std(df, v),
            "Min (unweighted)": float(df[v].min(skipna=True)) if df[v].notna().any() else np.nan,
            "Q25 (weighted)": weighted_quantile(df, v, 0.25),
            "Median (weighted)": weighted_quantile(df, v, 0.5),
            "Q75 (weighted)": weighted_quantile(df, v, 0.75),
            "Max (unweighted)": float(df[v].max(skipna=True)) if df[v].notna().any() else np.nan,
            "RSE (%)": compute_replicate_weight_rse(df, v),
        }
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(TABLES_DIR / "summary_statistics.csv", index=False)
    return summary_df


# ==========================================================
# Orchestrator
# ==========================================================
def run_descriptive_validation() -> Dict[str, Any]:
    """Run the full descriptive stats + validation pipeline."""
    logger.info("=" * 60)
    logger.info("Descriptive Statistics and Validation")
    logger.info("=" * 60)

    df = load_processed_data(use_clean=True)

    table1 = generate_table1_variables(df)
    logger.info(f"Table 1: {len(table1)} variables documented")

    table2 = generate_table2_sample_characteristics(df)
    logger.info(f"Table 2: {len(table2)} rows")

    summary = generate_summary_statistics(df)
    logger.info(f"Summary statistics generated for {len(summary)} variables")

    validation = validate_against_hc_tables(df)
    logger.info(f"Validation scaffolding: {len(validation)} metrics listed")

    generate_figure2_climate_envelope(df)
    generate_figure3_thermal_intensity_distribution(df)
    generate_figure4_validation(df, validation)

    logger.info("=" * 60)
    logger.info("Descriptive statistics and validation complete!")
    logger.info("=" * 60)

    return {"table1": table1, "table2": table2, "summary": summary, "validation": validation, "data": df}


if __name__ == "__main__":
    results = run_descriptive_validation()
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY (scaffolding)")
    print("=" * 60)
    print(results["validation"].to_string(index=False))
