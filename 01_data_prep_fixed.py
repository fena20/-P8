"""
01_data_prep.py
================
RECS 2020 Data Preparation for Heat Pump Retrofit Analysis

This module loads and cleans the RECS 2020 public-use microdata,
filters for gas-heated homes, constructs the thermal intensity metric,
and creates envelope efficiency classes.

Author: Fafa (GitHub: Fateme9977)
Institution: K. N. Toosi University of Technology
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import logging
from typing import Optional, Sequence, Iterable, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(r"C:\Users\FATEME\Desktop\Energy")
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Constants for analysis
BTU_PER_THERM = 100000  # 1 therm = 100,000 BTU
BTU_PER_KWH = 3412      # 1 kWh = 3,412 BTU
BTU_PER_GAL_PROPANE = 91500
BTU_PER_GAL_FUEL_OIL = 138500

# RECS 2020 fuel codes for natural gas heating
GAS_FUEL_CODES = [1]  # FUELHEAT=1 is natural gas

# Common RECS missing value codes (negative integers). We treat these as NaN for analysis.
# (EIA uses codes like -2 = Not applicable; -1/-9 often used for Don't know / Refused in some fields.)
RECS_MISSING_CODES = {-1, -2, -3, -7, -8, -9}

def replace_recs_missing_codes(df: pd.DataFrame,
                               exclude_prefixes: Sequence[str] = ("NWEIGHT",)) -> pd.DataFrame:
    """Replace common RECS negative missing-code sentinels with NaN for numeric-like columns.

    This is intentionally conservative: it only targets a small set of negative codes.
    """
    df = df.copy()
    for col in df.columns:
        if any(str(col).startswith(p) for p in exclude_prefixes):
            continue
        s = df[col]
        # Only attempt replacement for numeric dtypes or object columns that look numeric.
        if pd.api.types.is_numeric_dtype(s):
            df[col] = s.replace(list(RECS_MISSING_CODES), np.nan)
        elif pd.api.types.is_object_dtype(s):
            # Try coercion; if many values coerce, apply replacement on coerced view
            coerced = pd.to_numeric(s, errors="coerce")
            if coerced.notna().sum() >= max(10, int(0.05 * len(coerced))):
                df[col] = coerced.replace(list(RECS_MISSING_CODES), np.nan)
    return df

def weighted_quantile(values: Union[pd.Series, np.ndarray],
                      quantiles: Union[float, Sequence[float]],
                      sample_weight: Optional[Union[pd.Series, np.ndarray]] = None
                      ) -> Union[float, np.ndarray]:
    """Compute (one or many) weighted quantiles of *values*.

    If sample_weight is None, falls back to np.quantile.
    """
    v = np.asarray(values, dtype=float)
    if sample_weight is None:
        return np.quantile(v[~np.isnan(v)], quantiles)

    w = np.asarray(sample_weight, dtype=float)
    mask = (~np.isnan(v)) & (~np.isnan(w)) & (w > 0)
    v = v[mask]
    w = w[mask]
    if v.size == 0:
        return np.nan if np.isscalar(quantiles) else np.full(len(quantiles), np.nan)

    sorter = np.argsort(v)
    v = v[sorter]
    w = w[sorter]
    cum_w = np.cumsum(w)
    cum_w = cum_w / cum_w[-1]

    qs = np.array([quantiles], dtype=float) if np.isscalar(quantiles) else np.asarray(quantiles, dtype=float)
    out = np.interp(qs, cum_w, v)
    return float(out[0]) if np.isscalar(quantiles) else out

def weighted_median(values: Union[pd.Series, np.ndarray],
                    sample_weight: Optional[Union[pd.Series, np.ndarray]] = None) -> float:
    """Weighted median convenience wrapper."""
    return float(weighted_quantile(values, 0.5, sample_weight))




def find_microdata_file(data_dir: Path) -> Path:
    """Find the RECS 2020 microdata CSV file in the data directory.

    Priority:
    1) Match common filename patterns.
    2) Choose the *most recently modified* file among matches (more robust than lexical sort).
    """
    patterns = ["recs2020_public*.csv", "RECS2020*.csv"]

    for pattern in patterns:
        matches = list(data_dir.glob(pattern))
        if matches:
            return max(matches, key=lambda p: p.stat().st_mtime)

    raise FileNotFoundError(
        f"Could not find RECS 2020 microdata CSV in {data_dir}. "
        "Expected filename like 'recs2020_public_v7.csv'."
    )



def load_raw_microdata(filepath: Path) -> pd.DataFrame:
    """
    Load the raw RECS 2020 microdata.

    Notes
    -----
    - Applies conservative replacement of common RECS negative missing codes (-2, -1, …) with NaN
      to reduce silent bias in downstream scoring / calibration.
    """
    logger.info(f"Loading microdata from {filepath}")

    df = pd.read_csv(filepath, low_memory=False)
    logger.info(f"Loaded {len(df):,} households with {len(df.columns)} variables")

    df = replace_recs_missing_codes(df)

    return df



def select_key_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select key variables needed for analysis.
    
    Categories:
    - Identification and weights
    - Geographic/climate
    - Building characteristics
    - Heating system
    - Energy consumption
    - Energy expenditure
    """
    
    # Define variable groups
    id_weight_vars = ['DOEID', 'NWEIGHT']
    
    # Add replicate weights for RSE calculation
    replicate_weights = [f'NWEIGHT{i}' for i in range(1, 61)]
    
    geographic_vars = [
        'REGIONC',      # Census region (4 categories)
        'DIVISION',     # Census division (9 categories + 10 for territories)
        'UATYP10',      # Urban/rural type
        'CLIMATE_REGION_PUB',  # BA climate zone
        'HDD65',        # Heating degree days (base 65°F)
        'CDD65',        # Cooling degree days (base 65°F)
        'HDD30YR',      # 30-year normal HDD
        'CDD30YR',      # 30-year normal CDD
    ]
    
    building_vars = [
        'TYPEHUQ',      # Housing unit type
        'YEARMADERANGE', # Year built range
        'TOTSQFT_EN',   # Total square footage (conditioned)
        'TOTHSQFT',     # Heated square footage
        'TOTCSQFT',     # Cooled square footage
        'STORIES',      # Number of stories
        'NCOMBATH',     # Number of full bathrooms
        'NHAFBATH',     # Number of half bathrooms
        'BEDROOMS',     # Number of bedrooms
        'TOTROOMS',     # Total rooms
        'GARESSION',    # Garage/carport attached
        'PRKGPLC1',     # Garage attachment type
        'WINDOWS',      # Number of windows
        'WINFRAME',     # Window frame material
        'TYPEGLASS',    # Window glass type
        'ADQINSUL',     # Adequate insulation
        'DRAFTY',       # How drafty
        'HIGHCEIL',     # High ceilings
        'NUMFLRS',      # Number of floors
        'CELLAR',       # Basement type
        'CRAWL',        # Crawl space
        'CONCRETE',     # Slab foundation
        'BASEFIN',      # Basement finished
        'ATTIC',        # Attic type
        'ATTICFIN',     # Attic finished
        'WALLTYPE',     # Exterior wall material
        'ROOFTYPE',     # Roof material
    ]
    
    heating_vars = [
        'FUELHEAT',     # Main heating fuel
        'EQUIPM',       # Main heating equipment type
        'EQUIPMUSE',    # Main equipment used (yes/no)
        'EQUIPAGE',     # Main equipment age
        'FUELAUX',      # Secondary heating fuel (if any)
        'NOHEATBROKE',  # Heating broken in last year
        'NOHEATBULK',   # No heat - couldn't afford bulk fuel
        'NOHEATDAYS',   # Days without heat
        'THERMESSION',  # Programmable thermostat
        'HEESSION',     # Home energy management system
        'SMARTTHERM',   # Smart thermostat
        'HEATCNTL',     # Heating control type
        'TEMPHOME',     # Temperature when home
        'TEMPNITE',     # Temperature at night
        'TEMPGONE',     # Temperature when away
    ]
    
    cooling_vars = [
        'COOLTYPE',     # Cooling equipment type
        'FUELCOOL',     # Cooling fuel (electricity assumed)
        'ACEQUIPM_PUB', # AC equipment type (public)
        'ACEQUIPAGE',   # AC equipment age
        'CENACHP',      # Central AC is heat pump
    ]
    
    # Energy consumption variables (annual, in original units)
    consumption_vars = [
        'BTUEL',        # Electricity consumption (BTU)
        'BTUNG',        # Natural gas consumption (BTU)
        'BTULP',        # Propane/LPG consumption (BTU)
        'BTUFO',        # Fuel oil consumption (BTU)
        'BTUWOOD',      # Wood consumption (BTU)
        'TOTALBTU',     # Total consumption (BTU)
        'KWH',          # Electricity (kWh)
        'CUFEETNG',     # Natural gas (cubic feet)
        'GALLONLP',     # Propane (gallons)
        'GALLONFO',     # Fuel oil (gallons)
        'CORDS',        # Wood (cords)
        'PELESSION',    # Wood pellets (tons)
    ]
    
    # End-use consumption (where available)
    enduse_vars = [
        # Space-heating end-use naming varies across RECS public microdata releases.
        'TOTALBTUSPH',   # Total space-heating energy (kBtu) (often used in EIA examples)
        'BTUSPH',        # Space-heating energy (kBtu) (alternate naming)
        'TOTALDOLLARSPH',# Total space-heating expenditure ($) (alternate naming)
        'DOLLARSPH',     # Space-heating expenditure ($)
        'BTUCOL',       # Space cooling BTU
        'BTUWTH',       # Water heating BTU
        'BTUOTH',       # Other end uses BTU
    ]
    
    # Expenditure variables
    expenditure_vars = [
        'DOLLAREL',     # Electricity expenditure ($)
        'DOLLARNG',     # Natural gas expenditure ($)
        'DOLLARLP',     # Propane expenditure ($)
        'DOLLARFO',     # Fuel oil expenditure ($)
        'DOLLARWOOD',   # Wood expenditure ($)
        'TOTALDOL',     # Total expenditure ($)
    ]
    
    # Income and demographic
    demographic_vars = [
        'NHSLDMEM',     # Number of household members
        'HHAGE',        # Age of householder
        'HHSEX',        # Sex of householder
        'EDUCATION',    # Education level
        'EMPLOESSION',  # Employment status
        'MONESSION',    # Income
        'ELPAY',        # Who pays electric bill
        'NGPAY',        # Who pays gas bill
        'PAYHELP',      # Received energy assistance
        'SCALEE',       # LIHEAP electricity
        'SCALEG',       # LIHEAP gas
        'SCALEEB',      # LIHEAP other
    ]
    
    # Combine all variable lists
    all_vars = (
        id_weight_vars + 
        geographic_vars + 
        building_vars + 
        heating_vars + 
        cooling_vars + 
        consumption_vars + 
        enduse_vars + 
        expenditure_vars + 
        demographic_vars +
        replicate_weights
    )
    
    # Filter to only variables that exist in the dataframe
    available_vars = [v for v in all_vars if v in df.columns]
    missing_vars = [v for v in all_vars if v not in df.columns]
    
    if missing_vars:
        logger.warning(f"Missing {len(missing_vars)} variables: {missing_vars[:10]}...")
    
    logger.info(f"Selected {len(available_vars)} variables for analysis")
    
    return df[available_vars].copy()


def filter_gas_heated_homes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataset to include only gas-heated homes suitable for retrofit analysis.

    Criteria:
    - Main heating fuel is natural gas (FUELHEAT in GAS_FUEL_CODES)
    - Has valid heated floor area (A_heated > 0)
    - Has valid HDD65 (>0)
    - Has non-zero relevant energy signal: prefer BTUNG>0; otherwise fall back to space-heating end-use
      (BTUSPH or TOTALBTUSPH) if gas totals are unavailable.
    """
    logger.info("Filtering for gas-heated homes...")

    initial_count = len(df)

    # 1) Filter for natural gas as main heating fuel
    if 'FUELHEAT' in df.columns:
        mask_gas = df['FUELHEAT'].isin(GAS_FUEL_CODES)
        df = df[mask_gas].copy()
        logger.info(
            f"After gas fuel filter: {len(df):,} households "
            f"({len(df) / max(initial_count, 1) * 100:.1f}%)"
        )
    else:
        logger.warning("FUELHEAT column not found - skipping fuel filter")

    # 2) Construct heated floor area (A_heated)
    area = pd.Series(np.nan, index=df.index)
    if 'TOTHSQFT' in df.columns:
        area = pd.to_numeric(df['TOTHSQFT'], errors='coerce')
    if 'TOTSQFT_EN' in df.columns:
        area = area.fillna(pd.to_numeric(df['TOTSQFT_EN'], errors='coerce'))

    if area.notna().any():
        mask_area = (area > 0)
        dropped = len(df) - int(mask_area.sum())
        if dropped > 0:
            logger.info(f"Dropping {dropped:,} households with non-positive or missing heated area")
        df = df[mask_area].copy()
        df['A_heated'] = area[mask_area]
        logger.info(f"After heated area filter: {len(df):,} households")
    else:
        logger.error("No heated area information (TOTHSQFT/TOTSQFT_EN) available; cannot compute thermal intensity.")
        return df.iloc[0:0].copy()

    # 3) Filter for valid HDD65
    if 'HDD65' in df.columns:
        df['HDD65'] = pd.to_numeric(df['HDD65'], errors='coerce')
        mask_hdd = (df['HDD65'] > 0) & df['HDD65'].notna()
        df = df[mask_hdd].copy()
        logger.info(f"After HDD filter: {len(df):,} households")
    else:
        logger.warning("HDD65 column not found - skipping HDD filter")

    # 4) Filter for non-zero energy signal
    if 'BTUNG' in df.columns:
        df['BTUNG'] = pd.to_numeric(df['BTUNG'], errors='coerce')
        mask_btu = (df['BTUNG'] > 0) & df['BTUNG'].notna()
        df = df[mask_btu].copy()
        logger.info(f"After gas consumption filter (BTUNG): {len(df):,} households")
    else:
        # Fall back to any available space-heating end-use column
        space_heat_col = None
        for cand in ('BTUSPH', 'TOTALBTUSPH'):
            if cand in df.columns:
                space_heat_col = cand
                break

        if space_heat_col is not None:
            df[space_heat_col] = pd.to_numeric(df[space_heat_col], errors='coerce')
            mask_btu = (df[space_heat_col] > 0) & df[space_heat_col].notna()
            df = df[mask_btu].copy()
            logger.info(f"After space-heating energy filter ({space_heat_col}): {len(df):,} households")
        else:
            logger.warning("Neither BTUNG nor any space-heating end-use column found; cannot filter on heating consumption.")

    final_count = len(df)
    logger.info(
        f"Final sample: {final_count:,} households "
        f"({final_count / max(initial_count, 1) * 100:.1f}% of total input)"
    )

    return df



def compute_heating_energy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute annual space-heating energy for each household.

    Key improvements vs. the original implementation
    ------------------------------------------------
    1) Robustly supports alternative RECS column names for space-heating end-use:
       - BTUSPH (alternate)
       - TOTALBTUSPH (often used in EIA examples)
    2) Uses *weighted medians* (NWEIGHT) when calibrating BTUSPH/BTUNG fractions.
    3) Applies conservative bounds and climate-specific calibration when possible.
    """
    logger.info("Computing annual space heating energy...")

    df = df.copy()

    # Ensure numeric where present
    for col in ('BTUNG', 'BTUSPH', 'TOTALBTUSPH', 'HDD65', 'NWEIGHT'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    weight_col = 'NWEIGHT' if 'NWEIGHT' in df.columns else None
    w = df[weight_col] if weight_col else None

    # --- Choose the best available "space heating end-use" column ---
    space_heat_candidates = [c for c in ('BTUSPH', 'TOTALBTUSPH') if c in df.columns]
    space_heat_col: Optional[str] = None

    if space_heat_candidates:
        if 'BTUNG' in df.columns and df['BTUNG'].notna().any():
            # If gas totals exist, pick the candidate whose ratio to BTUNG looks most plausible
            best = None
            best_score = np.inf
            for cand in space_heat_candidates:
                mask = df[cand].notna() & df['BTUNG'].notna() & (df['BTUNG'] > 0) & (df[cand] > 0)
                if mask.sum() < 50:
                    continue
                ratio = (df.loc[mask, cand] / df.loc[mask, 'BTUNG']).astype(float)
                ww = w.loc[mask] if w is not None else None
                med = weighted_median(ratio, ww)
                q95 = weighted_quantile(ratio, 0.95, ww)
                # Penalize medians far from [0.1, 1.2] and very large upper tails
                score = abs(med - 0.7) + max(0.0, q95 - 1.5) * 2.0
                if score < best_score:
                    best_score = score
                    best = cand
            space_heat_col = best or space_heat_candidates[0]
        else:
            space_heat_col = space_heat_candidates[0]

    if space_heat_col:
        logger.info(f"Using '{space_heat_col}' as the space-heating end-use column (kBtu).")
    else:
        logger.warning("No space-heating end-use column found (BTUSPH/TOTALBTUSPH). Will rely on BTUNG heuristic if possible.")

    # --- Initialize E_heat_kbtu ---
    df['E_heat_kbtu'] = np.nan

    # 1) Direct space-heating data where available
    if space_heat_col is not None:
        mask_direct = df[space_heat_col].notna() & (df[space_heat_col] > 0)
        df.loc[mask_direct, 'E_heat_kbtu'] = df.loc[mask_direct, space_heat_col]
        logger.info(f"Directly set E_heat_kbtu from {space_heat_col} for {int(mask_direct.sum()):,} households.")

    # 2) Calibrate fraction of BTUNG (if BTUNG exists) using weighted medians
    calibrated = False
    if 'BTUNG' in df.columns:
        if space_heat_col is not None:
            mask_both = df[space_heat_col].notna() & df['BTUNG'].notna() & (df['BTUNG'] > 0) & (df[space_heat_col] > 0)
        else:
            mask_both = pd.Series(False, index=df.index)

        if mask_both.any():
            ratio_raw = (df.loc[mask_both, space_heat_col] / df.loc[mask_both, 'BTUNG']).astype(float)
            ww = w.loc[mask_both] if w is not None else None

            # Conservative "reasonable" bounds: allow >1 in case the space-heating column is "total"
            valid_ratio_mask = (ratio_raw > 0) & (ratio_raw <= 2.0)
            ratio = ratio_raw[valid_ratio_mask]
            ww_valid = ww[valid_ratio_mask] if ww is not None else None

            if ratio.size >= 50:
                global_ratio = weighted_median(ratio, ww_valid)
                logger.info(
                    f"Calibrated global space-heating fraction ({space_heat_col}/BTUNG) "
                    f"from {ratio.size:,} households (weighted median={global_ratio:.2f})."
                )

                ratio_by_zone = None
                bins = None

                # Optional climate segmentation if HDD is present
                if 'HDD65' in df.columns and df['HDD65'].notna().any():
                    bins = [0, 3000, 6000, np.inf]
                    zones = pd.cut(df.loc[mask_both, 'HDD65'], bins=bins, labels=['mild', 'mixed', 'cold'])
                    # Compute weighted median per zone
                    tmp = pd.DataFrame({'ratio': ratio_raw, 'zone': zones, 'w': ww if ww is not None else 1.0})
                    tmp = tmp.loc[valid_ratio_mask.values].copy()
                    ratio_by_zone = {}
                    for z in ['mild', 'mixed', 'cold']:
                        sub = tmp[tmp['zone'] == z]
                        if len(sub) >= 50:
                            ratio_by_zone[z] = weighted_median(sub['ratio'], sub['w'])
                    if ratio_by_zone:
                        logger.info("Calibrated climate-specific space-heating fractions (weighted medians):")
                        for z, v in ratio_by_zone.items():
                            logger.info(f"  {z}: {v:.2f}")

                # Impute where direct is missing
                mask_impute = df['E_heat_kbtu'].isna() & df['BTUNG'].notna() & (df['BTUNG'] > 0)
                if mask_impute.any():
                    if ratio_by_zone and ('HDD65' in df.columns):
                        zones_for_impute = pd.cut(df.loc[mask_impute, 'HDD65'], bins=bins, labels=['mild', 'mixed', 'cold'])
                        zone_ratio = zones_for_impute.map(ratio_by_zone).fillna(global_ratio).astype(float)
                        df.loc[mask_impute, 'E_heat_kbtu'] = df.loc[mask_impute, 'BTUNG'] * zone_ratio
                        logger.info("Imputed E_heat_kbtu from BTUNG using climate-specific fractions.")
                    else:
                        df.loc[mask_impute, 'E_heat_kbtu'] = df.loc[mask_impute, 'BTUNG'] * global_ratio
                        logger.info("Imputed E_heat_kbtu from BTUNG using global fraction.")
                calibrated = True

    # 3) Fallback heuristic if still missing
    if 'BTUNG' in df.columns and not calibrated:
        mask_impute = df['E_heat_kbtu'].isna() & df['BTUNG'].notna() & (df['BTUNG'] > 0)
        n_impute = int(mask_impute.sum())
        if n_impute > 0:
            df.loc[mask_impute, 'E_heat_kbtu'] = df.loc[mask_impute, 'BTUNG'] * 0.70
            logger.warning(
                f"Space-heating end-use not available/usable for calibration; using 70% of BTUNG as E_heat_kbtu for {n_impute:,} households."
            )

    if df['E_heat_kbtu'].notna().sum() == 0:
        logger.error("E_heat_kbtu could not be computed for any household. Check column names and filters.")

    # --- Convert kBtu → Btu, therm, kWh-equivalent ---
    df['E_heat_btu'] = df['E_heat_kbtu'] * 1000.0
    df['E_heat_therm'] = df['E_heat_btu'] / BTU_PER_THERM
    df['E_heat_kwh_equiv'] = df['E_heat_btu'] / BTU_PER_KWH

    return df



def compute_thermal_intensity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute thermal intensity:

        I = E_heat_btu / (A_heated * HDD65)

    where
    - E_heat_btu: annual space heating energy [BTU]
    - A_heated  : heated floor area [ft²]
    - HDD65     : annual heating degree days [°F·day]

    Units:
    -------
    I has units of BTU / (ft²·HDD).
    For interpretation, values are typically O(1–10) BTU/(ft²·HDD).
    """
    logger.info("Computing thermal intensity...")

    df = df.copy()

    # Heated area:
    # - If A_heated already exists (from previous steps), use it.
    # - Otherwise, construct it from TOTHSQFT with row-wise fallback to TOTSQFT_EN.
    if 'A_heated' in df.columns and df['A_heated'].notna().any():
        area = pd.to_numeric(df['A_heated'], errors='coerce')
        logger.info("Using existing A_heated as heated area")
    else:
        area = pd.Series(np.nan, index=df.index)
        if 'TOTHSQFT' in df.columns:
            area = pd.to_numeric(df['TOTHSQFT'], errors='coerce')
        if 'TOTSQFT_EN' in df.columns:
            area = area.fillna(pd.to_numeric(df['TOTSQFT_EN'], errors='coerce'))

        if not area.notna().any():
            logger.error(
                "No heated area information (A_heated/TOTHSQFT/TOTSQFT_EN) "
                "available; cannot compute thermal intensity."
            )
            df['Thermal_Intensity_I'] = np.nan
            return df

        df['A_heated'] = area
        logger.info("Constructed A_heated from TOTHSQFT/TOTSQFT_EN")

    # Ensure numeric HDD and energy
    df['HDD65'] = pd.to_numeric(df.get('HDD65'), errors='coerce')
    df['E_heat_btu'] = pd.to_numeric(df.get('E_heat_btu'), errors='coerce')

    valid_mask = (
        df['E_heat_btu'].notna() & (df['E_heat_btu'] > 0) &
        df['A_heated'].notna()  & (df['A_heated'] > 0) &
        df['HDD65'].notna()     & (df['HDD65'] > 0)
    )

    df['Thermal_Intensity_I'] = np.nan
    df.loc[valid_mask, 'Thermal_Intensity_I'] = (
        df.loc[valid_mask, 'E_heat_btu'] /
        (df.loc[valid_mask, 'A_heated'] * df.loc[valid_mask, 'HDD65'])
    )

    # Summary stats (unweighted) – sanity check
    valid_I = df['Thermal_Intensity_I'].dropna()
    if not valid_I.empty:
        logger.info("Thermal intensity I stats [BTU/(ft²·HDD)]:")
        logger.info(f"  Mean:   {valid_I.mean():.4f}")
        logger.info(f"  Median: {valid_I.median():.4f}")
        logger.info(f"  Std:    {valid_I.std():.4f}")

    return df




def create_envelope_classes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create envelope efficiency classes based on building characteristics.

    Classification is based on:
    - Year built (YEARMADERANGE)
    - Draftiness (DRAFTY)
    - Insulation adequacy (ADQINSUL)
    - Window quality (TYPEGLASS)

    Classes: 'poor', 'medium', 'good'

    Higher envelope_score = better envelope.
    """
    logger.info("Creating envelope efficiency classes...")

    df = df.copy()
    df['envelope_score'] = 0

    # Year built scoring
    # YEARMADERANGE: 1=Before 1950, ..., 9=2016-2020
    if 'YEARMADERANGE' in df.columns:
        year_score = df['YEARMADERANGE'].map({
            1: 0,   # Before 1950 - poorest
            2: 1,   # 1950-1959
            3: 1,   # 1960-1969
            4: 2,   # 1970-1979
            5: 2,   # 1980-1989
            6: 3,   # 1990-1999
            7: 3,   # 2000-2009
            8: 4,   # 2010-2015
            9: 4,   # 2016-2020
        }).fillna(2)
        df['envelope_score'] += year_score

    # Draftiness scoring
    # DRAFTY: 1=Very drafty, 2=Somewhat drafty, 3=Not drafty at all
    if 'DRAFTY' in df.columns:
        draft_score = df['DRAFTY'].map({
            1: 0,   # Very drafty
            2: 2,   # Somewhat drafty
            3: 4,   # Not drafty
        }).fillna(2)
        df['envelope_score'] += draft_score

    # Insulation adequacy scoring
    # ADQINSUL: 1=Well insulated, 2=Adequately insulated,
    #           3=Poorly insulated, 4=Not insulated
    if 'ADQINSUL' in df.columns:
        insul_score = df['ADQINSUL'].map({
            1: 4,   # Well insulated
            2: 3,   # Adequate
            3: 1,   # Poorly insulated
            4: 0,   # Not insulated
        }).fillna(2)
        df['envelope_score'] += insul_score

    # Window glass type scoring
    # TYPEGLASS: 1=Single-pane, 2=Double-pane, 3=Triple-pane
    if 'TYPEGLASS' in df.columns:
        glass_score = df['TYPEGLASS'].map({
            1: 0,   # Single pane
            2: 2,   # Double pane
            3: 4,   # Triple pane
        }).fillna(1)
        df['envelope_score'] += glass_score

    # Classify into envelope classes based on total score
    # Score range: 0-16 (4 variables × 0-4 each)
    df['envelope_class'] = pd.cut(
        df['envelope_score'],
        bins=[-1, 5, 10, 20],
        labels=['poor', 'medium', 'good']
    )

    # Log distribution (unweighted)
    class_dist = df['envelope_class'].value_counts(normalize=True) * 100
    logger.info("Envelope class distribution (unweighted):")
    for cls, pct in class_dist.sort_index().items():
        logger.info(f"  {cls}: {pct:.1f}%")

    # اگر شدت حرارتی را قبلاً محاسبه کرده‌ایم، میانگین آن را در هر کلاس لاگ کن
    if 'Thermal_Intensity_I' in df.columns:
        logger.info("Mean thermal intensity by envelope class (BTU/sqft/HDD):")
        mean_I_by_class = df.groupby('envelope_class')['Thermal_Intensity_I'].mean()
        for cls, val in mean_I_by_class.sort_index().items():
            logger.info(f"  {cls}: {val:.2f}")
        # این لاگ کمک می‌کند چک کنی که واقعاً poor > medium > good باشد.

    return df



def create_climate_zones(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create climate zone categories based on HDD65.
    
    Categories:
    - Cold: HDD65 >= 6000
    - Mixed: 3000 <= HDD65 < 6000
    - Mild: HDD65 < 3000
    """
    logger.info("Creating climate zone categories...")
    
    df['climate_zone'] = pd.cut(
        df['HDD65'],
        bins=[0, 3000, 6000, float('inf')],
        labels=['mild', 'mixed', 'cold']
    )
    
    # Also create HDD bands for more granular analysis
    df['hdd_band'] = pd.cut(
        df['HDD65'],
        bins=[0, 2000, 4000, 6000, 8000, float('inf')],
        labels=['<2000', '2000-4000', '4000-6000', '6000-8000', '>8000']
    )
    
    # Log distribution
    zone_dist = df['climate_zone'].value_counts(normalize=True) * 100
    logger.info("Climate zone distribution:")
    for zone, pct in zone_dist.items():
        logger.info(f"  {zone}: {pct:.1f}%")
    
    return df


def create_year_built_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create readable year built categories.
    """
    if 'YEARMADERANGE' in df.columns:
        df['year_built_cat'] = df['YEARMADERANGE'].map({
            1: 'Before 1950',
            2: '1950-1959',
            3: '1960-1969',
            4: '1970-1979',
            5: '1980-1989',
            6: '1990-1999',
            7: '2000-2009',
            8: '2010-2015',
            9: '2016-2020',
        })
    return df


def create_housing_type_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create readable housing type categories.
    """
    if 'TYPEHUQ' in df.columns:
        df['housing_type'] = df['TYPEHUQ'].map({
            1: 'Mobile home',
            2: 'Single-family detached',
            3: 'Single-family attached',
            4: 'Apartment (2-4 units)',
            5: 'Apartment (5+ units)',
        })
    return df


def create_division_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create readable census division labels.
    """
    if 'DIVISION' in df.columns:
        # Check if DIVISION is already string (newer RECS format)
        if df['DIVISION'].dtype == 'object':
            df['division_name'] = df['DIVISION']
        else:
            df['division_name'] = df['DIVISION'].map({
                1: 'New England',
                2: 'Middle Atlantic',
                3: 'East North Central',
                4: 'West North Central',
                5: 'South Atlantic',
                6: 'East South Central',
                7: 'West South Central',
                8: 'Mountain',
                9: 'Pacific',
                10: 'US Territories',
            })
    return df


def create_region_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create readable census region labels.
    """
    if 'REGIONC' in df.columns:
        # Check if REGIONC is already string (newer RECS format)
        if df['REGIONC'].dtype == 'object':
            df['region_name'] = df['REGIONC']
        else:
            df['region_name'] = df['REGIONC'].map({
                1: 'Northeast',
                2: 'Midwest',
                3: 'South',
                4: 'West',
            })
    return df


def remove_outliers(df: pd.DataFrame,
                    column: str = 'Thermal_Intensity_I',
                    method: str = 'iqr',
                    threshold: float = 1.5,
                    use_weights: bool = False,
                    weight_col: str = 'NWEIGHT') -> pd.DataFrame:
    """
    Remove outliers from the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    column : str
        Column to use for outlier detection
    method : str
        'iqr' for interquartile range, 'zscore' for z-score
    threshold : float
        Threshold for outlier detection (IQR multiplier or z-score)
    use_weights : bool
        If True and weight_col exists, compute IQR using weighted quantiles.
    weight_col : str
        Final sampling weight column (default: NWEIGHT)
    """
    logger.info(f"Removing outliers using {method} method...")

    df = df.copy()
    initial_count = len(df)

    if column not in df.columns:
        logger.warning(f"Outlier column '{column}' not found; skipping outlier removal.")
        return df

    x = pd.to_numeric(df[column], errors='coerce')
    mask_valid = x.notna()
    if mask_valid.sum() < 10:
        logger.warning("Too few valid values for outlier detection; skipping.")
        return df

    if method == 'iqr':
        if use_weights and (weight_col in df.columns):
            w = pd.to_numeric(df.loc[mask_valid, weight_col], errors='coerce')
            Q1 = weighted_quantile(x.loc[mask_valid], 0.25, w)
            Q3 = weighted_quantile(x.loc[mask_valid], 0.75, w)
        else:
            Q1 = x.loc[mask_valid].quantile(0.25)
            Q3 = x.loc[mask_valid].quantile(0.75)

        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        keep = mask_valid & (x >= lower) & (x <= upper)

    elif method == 'zscore':
        mu = x.loc[mask_valid].mean()
        sd = x.loc[mask_valid].std()
        if sd == 0 or np.isnan(sd):
            logger.warning("Std is zero/NaN in z-score outlier removal; skipping.")
            return df
        z = (x - mu) / sd
        keep = mask_valid & (np.abs(z) <= threshold)

    else:
        raise ValueError("method must be one of {'iqr', 'zscore'}")

    df = df[keep].copy()

    removed = initial_count - len(df)
    logger.info(f"Removed {removed:,} outliers ({removed / max(initial_count, 1) * 100:.2f}%)")

    return df



def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer additional features for modeling.
    """
    logger.info("Engineering features...")
    
    # Building age (approximate)
    year_mid = {
        1: 1940, 2: 1955, 3: 1965, 4: 1975,
        5: 1985, 6: 1995, 7: 2005, 8: 2013, 9: 2018
    }
    if 'YEARMADERANGE' in df.columns:
        df['building_age'] = 2020 - df['YEARMADERANGE'].map(year_mid)
    
    # Equipment age
    if 'EQUIPAGE' in df.columns:
        # Map known categories to approximate mid-point age;
        # leave unknown codes (e.g. 42, 43) as NaN instead of forcing 25 years.
        equip_age_mid = {
            1: 1,   # <2 years
            2: 4,   # 2–5 years
            3: 8,   # 6–10 years
            4: 13,  # 11–15 years
            5: 18,  # >15 years
        }
        df['heating_equip_age'] = df['EQUIPAGE'].map(equip_age_mid)
        # Optional flag for unknown ages (could be used as a categorical feature)
        df['heating_equip_age_unknown'] = df['heating_equip_age'].isna().astype(int)
    
    # Log-transformed floor area
    if 'A_heated' in df.columns:
        df['log_sqft'] = np.log1p(df['A_heated'])
    
    # Interaction features
    if 'HDD65' in df.columns and 'A_heated' in df.columns:
        df['hdd_sqft_interaction'] = df['HDD65'] * df['A_heated'] / 1_000_000.0
    
    if 'E_heat_btu' in df.columns and 'A_heated' in df.columns:
        df['heating_per_sqft'] = df['E_heat_btu'] / df['A_heated']
    
    return df


def prepare_analysis_dataset(save_intermediate: bool = True) -> pd.DataFrame:
    """
    Main function to prepare the analysis dataset.
    
    Steps:
    1. Load raw microdata
    2. Select key variables
    3. Filter for gas-heated homes
    4. Compute heating energy
    5. Compute thermal intensity
    6. Create envelope classes
    7. Create climate zones
    8. Engineer features
    9. Remove outliers
    
    Returns
    -------
    pd.DataFrame
        Cleaned, processed dataset ready for analysis
    """
    logger.info("=" * 60)
    logger.info("RECS 2020 Data Preparation Pipeline")
    logger.info("=" * 60)
    
    # Step 1: Load data
    microdata_path = find_microdata_file(DATA_DIR)
    df = load_raw_microdata(microdata_path)
    
    # Step 2: Select variables
    df = select_key_variables(df)
    
    if save_intermediate:
        out_path = OUTPUT_DIR / "01_selected_variables.csv"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info(f"Saved selected variables to {out_path}")
    
    # Step 3: Filter for gas-heated homes
    df = filter_gas_heated_homes(df)
    
    # Step 4: Compute heating energy
    df = compute_heating_energy(df)
    
    # Step 5: Compute thermal intensity
    df = compute_thermal_intensity(df)
    
    # Step 6: Create categorical variables
    df = create_envelope_classes(df)
    df = create_climate_zones(df)
    df = create_year_built_categories(df)
    df = create_housing_type_categories(df)
    df = create_division_labels(df)
    df = create_region_labels(df)
    
    # Step 7: Engineer features
    df = engineer_features(df)
    
    # Step 8: Remove outliers (optional - can be skipped for validation)
    df_clean = remove_outliers(df.copy(), 'Thermal_Intensity_I', method='iqr', threshold=3.0)
    
    # Save final dataset
    if save_intermediate:
        # Full dataset (with outliers)
        full_path = OUTPUT_DIR / "02_gas_heated_full.csv"
        df.to_csv(full_path, index=False)
        logger.info(f"Saved full dataset to {full_path}")
        
        # Clean dataset (outliers removed)
        clean_path = OUTPUT_DIR / "03_gas_heated_clean.csv"
        df_clean.to_csv(clean_path, index=False)
        logger.info(f"Saved clean dataset to {clean_path}")
    
    logger.info("=" * 60)
    logger.info("Data preparation complete!")
    logger.info(f"Final sample size: {len(df_clean):,} households")
    logger.info("=" * 60)
    
    return df_clean


def summarize_dataset(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for the prepared dataset and run
    a few sanity checks on thermal intensity.

    Notes:
    - All statistics here are UNWEIGHTED unless explicitly stated.
      Weighted summaries for publication should be computed in a
      separate analysis script / notebook.
    """
    summary: dict[str, object] = {}

    # --- Basic sample stats ---
    summary['n_households'] = len(df)
    summary['n_weighted'] = df['NWEIGHT'].sum() if 'NWEIGHT' in df.columns else None

    # --- Thermal intensity level ---
    if 'Thermal_Intensity_I' in df.columns:
        I = pd.to_numeric(df['Thermal_Intensity_I'], errors='coerce')
        summary['thermal_intensity_mean'] = float(I.mean())
        summary['thermal_intensity_median'] = float(I.median())
        summary['thermal_intensity_std'] = float(I.std())
    else:
        summary['thermal_intensity_mean'] = None
        summary['thermal_intensity_median'] = None
        summary['thermal_intensity_std'] = None

    # --- HDD and area ---
    if 'HDD65' in df.columns:
        summary['hdd_mean'] = float(pd.to_numeric(df['HDD65'], errors='coerce').mean())
    else:
        summary['hdd_mean'] = None

    if 'A_heated' in df.columns:
        summary['sqft_mean'] = float(pd.to_numeric(df['A_heated'], errors='coerce').mean())
    else:
        summary['sqft_mean'] = None

    # --- Discrete distributions ---
    if 'envelope_class' in df.columns:
        summary['envelope_distribution'] = df['envelope_class'].value_counts().to_dict()
    else:
        summary['envelope_distribution'] = None

    if 'climate_zone' in df.columns:
        summary['climate_zone_distribution'] = df['climate_zone'].value_counts().to_dict()
    else:
        summary['climate_zone_distribution'] = None

    if 'division_name' in df.columns:
        summary['division_distribution'] = df['division_name'].value_counts().to_dict()
    else:
        summary['division_distribution'] = None

    # =========================================================
    # 1) Mean intensity by envelope class (monotonicity check)
    # =========================================================
    if {'envelope_class', 'Thermal_Intensity_I'} <= set(df.columns):
        I = pd.to_numeric(df['Thermal_Intensity_I'], errors='coerce')
        mean_by_env = (
            df.assign(_I=I)
              .groupby('envelope_class')['_I']
              .mean()
              .sort_index()
        )
        # dict for quick inspection in console
        summary['intensity_by_envelope_mean'] = {
            str(k): float(v) for k, v in mean_by_env.to_dict().items()
        }

    # =========================================================
    # 2) Mean intensity by climate zone (pattern vs HDD bands)
    # =========================================================
    if {'climate_zone', 'Thermal_Intensity_I'} <= set(df.columns):
        I = pd.to_numeric(df['Thermal_Intensity_I'], errors='coerce')
        mean_by_climate = (
            df.assign(_I=I)
              .groupby('climate_zone')['_I']
              .mean()
              .sort_index()
        )
        summary['intensity_by_climate_mean'] = {
            str(k): float(v) for k, v in mean_by_climate.to_dict().items()
        }

    # =========================================================
    # 3) Correlation between intensity and HDD
    #    (ideally small, since HDD is in the denominator)
    # =========================================================
    if {'Thermal_Intensity_I', 'HDD65'} <= set(df.columns):
        tmp = df[['Thermal_Intensity_I', 'HDD65']].apply(
            pd.to_numeric, errors='coerce'
        ).dropna()
        if len(tmp) > 2:
            corr = float(tmp['Thermal_Intensity_I'].corr(tmp['HDD65']))
        else:
            corr = None
        summary['corr_intensity_hdd'] = corr
    else:
        summary['corr_intensity_hdd'] = None

    # =========================================================
    # 4) Key percentiles of thermal intensity
    # =========================================================
    if 'Thermal_Intensity_I' in df.columns:
        I = pd.to_numeric(df['Thermal_Intensity_I'], errors='coerce').dropna()
        if not I.empty:
            q = I.quantile([0.05, 0.50, 0.95, 0.99])
            summary['thermal_intensity_percentiles'] = {
                f"q{int(p*100)}": float(v) for p, v in q.items()
            }
        else:
            summary['thermal_intensity_percentiles'] = None
    else:
        summary['thermal_intensity_percentiles'] = None

    return summary


if __name__ == "__main__":
    # Run the data preparation pipeline
    df = prepare_analysis_dataset(save_intermediate=True)
    
    # Print summary
    summary = summarize_dataset(df)
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    for key, value in summary.items():
        print(f"{key}: {value}")
