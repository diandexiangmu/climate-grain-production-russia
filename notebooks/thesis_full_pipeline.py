# -*- coding: utf-8 -*-
"""
One-shot thesis pipeline for Russian regional grain production and climate analysis.

Purpose
-------
This script combines the previous modelling scripts and the revised thesis structure.
It runs, as far as the available data and packages allow:

1. Data loading and preprocessing
2. Descriptive statistics table
3. Two-way fixed effects models
4. Nonlinear fixed effects model
5. Post-2014 climate-sensitivity interaction model
6. Spatial lag construction, SAR-style and SDM-style panel models
7. Quantile regression
8. Panel ARDL-style distributed lag model
9. Random forest models with and without lagged production
10. Cross-validation for random forest
11. SHAP interpretation for random forest if the shap package is installed
12. Maps and all thesis figures expected by the revised LaTeX draft
13. Model comparison summary table
14. ZIP package of the generated outputs

Default project layout
----------------------
/Users/littlestars/Desktop/grain_project
    data/final/final_panel_yield_spi_spei_subsidy.csv
    data/large_files/gadm41_RUS_shp/gadm41_RUS_1.shp

Run
---
python thesis_full_pipeline.py

If paths differ, edit BASE_DIR, PANEL_PATH, and GADM_SHP_PATH below.
"""

from __future__ import annotations

import os
import sys
import math
import json
import shutil
import zipfile
import warnings
import difflib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import geopandas as gpd
except Exception:
    gpd = None

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
except Exception:
    sm = None
    smf = None

try:
    from linearmodels.panel import PanelOLS
except Exception:
    PanelOLS = None

try:
    from libpysal.weights import Queen
except Exception:
    Queen = None

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
except Exception:
    RandomForestRegressor = None
    GroupShuffleSplit = None
    GroupKFold = None
    cross_val_score = None
    mean_squared_error = None
    mean_absolute_error = None
    r2_score = None
    OneHotEncoder = None
    ColumnTransformer = None
    Pipeline = None

try:
    import shap
except Exception:
    shap = None


# =============================================================================
# 0. User configuration
# =============================================================================

BASE_DIR = Path("/Users/littlestars/Desktop/grain_project")

PANEL_PATH = BASE_DIR / "data" / "final" / "final_panel_yield_spi_spei_subsidy.csv"
GADM_SHP_PATH = BASE_DIR / "data" / "large_files" / "gadm41_RUS_shp" / "gadm41_RUS_1.shp"

# All outputs go here. This avoids mixing new and old results.
OUTPUT_ROOT = BASE_DIR / "thesis_outputs"
RESULTS_DIR = OUTPUT_ROOT / "results"
FIG_DIR = OUTPUT_ROOT / "figures"
TABLE_DIR = OUTPUT_ROOT / "tables"
LOG_DIR = OUTPUT_ROOT / "logs"

# If True, fail when a core model cannot run. If False, skip failed parts and continue.
STRICT_MODE = False

# Random forest settings. 500 is stable; reduce to 200 if the run is slow.
RF_N_ESTIMATORS = 500
RF_RANDOM_STATE = 42
RF_MIN_SAMPLES_LEAF = 3

# If SHAP is slow, set this lower. SHAP will use at most this many test rows.
SHAP_MAX_ROWS = 250

# Main column names used in your existing data.
Y_COL = "ln_yield"
REGION_COL_CANDIDATES = ["region_match", "matched_region", "region_std", "region", "region_name", "region_en", "name"]
YEAR_COL = "year"

# Candidate explanatory variables. Missing ones are skipped automatically.
BASE_X_CANDIDATES = [
    "temp_grow_mean",
    "prec_grow_sum",
    "spi_grow_mean",
    "spei_grow_mean",
    "ln_subsidy_main",
    "ln_subsidy_per_area",
    "ln_area",
    "ln_yield_lag1",
]

NONLINEAR_X_CANDIDATES = [
    "temp_grow_mean",
    "prec_grow_sum",
    "spi_grow_mean",
    "spei_grow_mean",
    "ln_subsidy_main",
    "ln_subsidy_per_area",
    "ln_area",
    "ln_yield_lag1",
    "temp_grow_sq",
    "spi_sq",
    "spei_sq",
    "temp_x_subsidy",
    "spi_x_subsidy",
    "spei_x_subsidy",
]

POST2014_CLIMATE_VARS = ["temp_grow_mean", "spi_grow_mean", "spei_grow_mean"]
ARDL_LAG_VARS = ["temp_grow_mean", "prec_grow_sum", "spi_grow_mean", "spei_grow_mean"]
QUANTILES = [0.25, 0.50, 0.75]


# =============================================================================
# 1. Utilities
# =============================================================================


def ensure_dirs() -> None:
    for d in [OUTPUT_ROOT, RESULTS_DIR, FIG_DIR, TABLE_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(msg, flush=True)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def safe_to_latex(df: pd.DataFrame, path: Path, caption: str, label: str, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        tex = df.to_latex(index=index, escape=False, caption=caption, label=label)
    except Exception:
        tex = "% Failed to export table to LaTeX.\n"
    path.write_text(tex, encoding="utf-8")


def read_csv_flexible(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "gbk", "cp1251", "latin1"]
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_error = e
    raise last_error


def find_file(filename: str, root: Path) -> Optional[Path]:
    for p in root.rglob(filename):
        if ".ipynb_checkpoints" not in str(p):
            return p
    return None


def find_first_existing(paths: Sequence[Path], fallback_name: Optional[str] = None) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    if fallback_name is not None and BASE_DIR.exists():
        return find_file(fallback_name, BASE_DIR)
    return None


def find_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for cand in candidates:
        cand_low = cand.lower()
        for c in df.columns:
            if cand_low in str(c).lower():
                return c
    return None


def rmse_score(y_true, y_pred) -> float:
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    log(f"Saved figure: {path}")


def get_feature_names_from_pipeline(model: Pipeline) -> np.ndarray:
    pre = model.named_steps["preprocess"]
    try:
        return pre.get_feature_names_out()
    except Exception:
        names = []
        for name, trans, cols in pre.transformers_:
            if name == "num":
                names.extend(cols)
            elif name == "cat" and hasattr(trans, "get_feature_names_out"):
                names.extend(trans.get_feature_names_out(cols))
        return np.array(names)


def readable_feature_name(name: str) -> str:
    mapping = {
        "num__ln_yield_lag1": r"$\ln Y_{t-1}$",
        "ln_yield_lag1": r"$\ln Y_{t-1}$",
        "num__temp_grow_mean": "Growing-season temperature",
        "temp_grow_mean": "Growing-season temperature",
        "num__prec_grow_sum": "Growing-season precipitation",
        "prec_grow_sum": "Growing-season precipitation",
        "num__spi_grow_mean": "SPI",
        "spi_grow_mean": "SPI",
        "num__spei_grow_mean": "SPEI",
        "spei_grow_mean": "SPEI",
        "num__ln_subsidy_main": r"$\ln$ Subsidy",
        "ln_subsidy_main": r"$\ln$ Subsidy",
        "num__ln_subsidy_per_area": r"$\ln$ Subsidy / Area",
        "ln_subsidy_per_area": r"$\ln$ Subsidy / Area",
        "num__ln_area": r"$\ln$ Sown area",
        "ln_area": r"$\ln$ Sown area",
        "num__temp_grow_sq": r"Temperature$^2$",
        "temp_grow_sq": r"Temperature$^2$",
        "num__spi_sq": r"SPI$^2$",
        "spi_sq": r"SPI$^2$",
        "num__spei_sq": r"SPEI$^2$",
        "spei_sq": r"SPEI$^2$",
        "num__temp_x_subsidy": "Temperature × Subsidy",
        "temp_x_subsidy": "Temperature × Subsidy",
        "num__spi_x_subsidy": "SPI × Subsidy",
        "spi_x_subsidy": "SPI × Subsidy",
        "num__spei_x_subsidy": "SPEI × Subsidy",
        "spei_x_subsidy": "SPEI × Subsidy",
        "cat__agro_zone_black_soil": "Agro zone: Black soil",
        "cat__agro_zone_volga_dry": "Agro zone: Volga dry",
        "cat__agro_zone_risky_farming": "Agro zone: Risky farming",
        "cat__agro_zone_other": "Agro zone: Other",
    }
    return mapping.get(name, name.replace("num__", "").replace("cat__", ""))


def model_result_table(res, model_name: str) -> pd.DataFrame:
    rows = []
    try:
        params = res.params
        se = res.std_errors
        pvals = res.pvalues
        for var in params.index:
            rows.append({
                "model": model_name,
                "variable": var,
                "coefficient": params[var],
                "std_error": se.get(var, np.nan),
                "pvalue": pvals.get(var, np.nan),
                "significance": significance_stars(pvals.get(var, np.nan)),
            })
    except Exception:
        pass
    return pd.DataFrame(rows)


def significance_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def panel_summary_row(name: str, res, purpose: str = "") -> Dict[str, object]:
    return {
        "Model": name,
        "Purpose": purpose,
        "N": getattr(res, "nobs", np.nan),
        "R2_within": getattr(res, "rsquared_within", np.nan),
        "R2_between": getattr(res, "rsquared_between", np.nan),
        "R2_overall": getattr(res, "rsquared_overall", np.nan),
        "RMSE": np.nan,
        "MAE": np.nan,
        "Main finding": "",
    }


# =============================================================================
# 2. Region matching helpers
# =============================================================================


MANUAL_REGION_MAP = {
    "Алтайский край": "Altay",
    "Амурская область": "Amur",
    "Архангельская область": "Arkhangel'sk",
    "Астраханская область": "Astrakhan'",
    "Белгородская область": "Belgorod",
    "Брянская область": "Bryansk",
    "Владимирская область": "Vladimir",
    "Волгоградская область": "Volgograd",
    "Вологодская область": "Vologda",
    "Воронежская область": "Voronezh",
    "Забайкальский край": "Zabaykal'ye",
    "Ивановская область": "Ivanovo",
    "Иркутская область": "Irkutsk",
    "Кабардино-Балкарская Республика": "Kabardin-Balkar",
    "Калининградская область": "Kaliningrad",
    "Калужская область": "Kaluga",
    "Камчатский край": "Kamchatka",
    "Карачаево-Черкесская Республика": "Karachay-Cherkess",
    "Кировская область": "Kirov",
    "Костромская область": "Kostroma",
    "Краснодарский край": "Krasnodar",
    "Красноярский край": "Krasnoyarsk",
    "Курганская область": "Kurgan",
    "Курская область": "Kursk",
    "Ленинградская область": "Leningrad",
    "Липецкая область": "Lipetsk",
    "Московская область": "Moskva",
    "Нижегородская область": "Nizhny Novgorod",
    "Новгородская область": "Novgorod",
    "Новосибирская область": "Novosibirsk",
    "Омская область": "Omsk",
    "Оренбургская область": "Orenburg",
    "Орловская область": "Orël",
    "Пензенская область": "Penza",
    "Пермский край": "Perm'",
    "Приморский край": "Primor'ye",
    "Псковская область": "Pskov",
    "Республика Адыгея": "Adygey",
    "Республика Алтай": "Altay",
    "Республика Башкортостан": "Bashkortostan",
    "Республика Бурятия": "Buryat",
    "Республика Дагестан": "Dagestan",
    "Республика Ингушетия": "Ingush",
    "Республика Калмыкия": "Kalmyk",
    "Республика Карелия": "Karelia",
    "Республика Коми": "Komi",
    "Республика Марий Эл": "Mariy-El",
    "Республика Мордовия": "Mordovia",
    "Республика Саха (Якутия)": "Sakha",
    "Республика Северная Осетия-Алания": "North Ossetia",
    "Республика Татарстан": "Tatarstan",
    "Республика Тыва": "Tuva",
    "Республика Хакасия": "Khakass",
    "Ростовская область": "Rostov",
    "Рязанская область": "Ryazan'",
    "Самарская область": "Samara",
    "Саратовская область": "Saratov",
    "Свердловская область": "Sverdlovsk",
    "Смоленская область": "Smolensk",
    "Ставропольский край": "Stavropol'",
    "Тамбовская область": "Tambov",
    "Тверская область": "Tver'",
    "Томская область": "Tomsk",
    "Тульская область": "Tula",
    "Тюменская область": "Tyumen'",
    "Удмуртская Республика": "Udmurt",
    "Ульяновская область": "Ul'yanovsk",
    "Хабаровский край": "Khabarovsk",
    "Челябинская область": "Chelyabinsk",
    "Чеченская Республика": "Chechnya",
    "Ярославская область": "Yaroslavl'",
}


def cyrillic_to_latin(text: object) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    table = {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
        "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
        "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
        "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "",
        "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
        "А": "a", "Б": "b", "В": "v", "Г": "g", "Д": "d", "Е": "e", "Ё": "e",
        "Ж": "zh", "З": "z", "И": "i", "Й": "y", "К": "k", "Л": "l", "М": "m",
        "Н": "n", "О": "o", "П": "p", "Р": "r", "С": "s", "Т": "t", "У": "u",
        "Ф": "f", "Х": "kh", "Ц": "ts", "Ч": "ch", "Ш": "sh", "Щ": "shch", "Ъ": "",
        "Ы": "y", "Ь": "", "Э": "e", "Ю": "yu", "Я": "ya",
    }
    return "".join(table.get(ch, ch) for ch in text)


def clean_region_name(x: object) -> str:
    if pd.isna(x):
        return ""
    s = cyrillic_to_latin(str(x).strip()).lower()
    replacements = {"’": "", "'": "", "`": "", "´": "", "-": " ", "_": " ", ".": "", ",": ""}
    for a, b in replacements.items():
        s = s.replace(a, b)
    s = " ".join(s.split())

    suffixes = [
        " oblast", " region", " republic", " krai", " kray", " autonomous okrug",
        " autonomous oblast", " federal city", " city", " resp", " respublika",
        " avtonomnyy okrug", " avtonomnyi okrug", " avtonomnaya oblast", " gorod",
    ]
    for suf in suffixes:
        if s.endswith(suf):
            s = s[: -len(suf)]
            s = " ".join(s.split())

    manual = {
        "moskva": "moscow",
        "moskovskaya": "moscow",
        "sankt peterburg": "saint petersburg",
        "st petersburg": "saint petersburg",
        "nizhegorodskaya": "nizhny novgorod",
        "nizhnij novgorod": "nizhny novgorod",
        "nizhniy novgorod": "nizhny novgorod",
        "orlovskaya": "oryol",
        "orel": "oryol",
        "orël": "oryol",
        "tverskaya": "tver",
        "tyumenskaya": "tyumen",
        "chelyabinskaya": "chelyabinsk",
        "ulyanovskaya": "ulyanovsk",
        "yaroslavskaya": "yaroslavl",
        "samarskaya": "samara",
        "saratovskaya": "saratov",
        "tambovskaya": "tambov",
        "belgorodskaya": "belgorod",
        "bryanskaya": "bryansk",
        "vladimirskaya": "vladimir",
        "volgogradskaya": "volgograd",
        "vologodskaya": "vologda",
        "voronezhskaya": "voronezh",
        "ivanovskaya": "ivanovo",
        "irkutskaya": "irkutsk",
        "kaluzhskaya": "kaluga",
        "kostromskaya": "kostroma",
        "kurganskaya": "kurgan",
        "kurskaya": "kursk",
        "leningradskaya": "leningrad",
        "lipetskaya": "lipetsk",
        "novgorodskaya": "novgorod",
        "novosibirskaya": "novosibirsk",
        "omskaya": "omsk",
        "orenburgskaya": "orenburg",
        "penzenskaya": "penza",
        "pskovskaya": "pskov",
        "rostovskaya": "rostov",
        "ryazanskaya": "ryazan",
        "smolenskaya": "smolensk",
        "sverdlovskaya": "sverdlovsk",
        "tomskaya": "tomsk",
        "tulskaya": "tula",
        "amurskaya": "amur",
        "arkhangelskaya": "arkhangelsk",
        "astrakhanskaya": "astrakhan",
        "kaliningradskaya": "kaliningrad",
        "kirovskaya": "kirov",
        "krasnodarskiy": "krasnodar",
        "krasnoyarskiy": "krasnoyarsk",
        "permskiy": "perm",
        "primorskiy": "primorye",
        "stavropolskiy": "stavropol",
        "zabaykalskiy": "zabaykalye",
        "altayskiy": "altay",
        "mariy el": "mariy el",
        "mari el": "mariy el",
        "saha yakutiya": "sakha",
        "yakutiya": "sakha",
        "severnaya osetiya alaniya": "north ossetia",
        "kabardino balkarskaya": "kabardin balkar",
        "karachaevo cherkesskaya": "karachay cherkess",
    }
    return manual.get(s, s)


def fuzzy_match_one(name: str, candidates: Sequence[str], cutoff: float = 0.38) -> Tuple[Optional[str], float]:
    if not name:
        return None, 0.0
    result = difflib.get_close_matches(name, candidates, n=1, cutoff=cutoff)
    if not result:
        return None, 0.0
    best = result[0]
    score = difflib.SequenceMatcher(None, name, best).ratio()
    return best, score


# =============================================================================
# 3. Loading and preprocessing
# =============================================================================


def load_panel() -> Tuple[pd.DataFrame, str]:
    panel_path = find_first_existing(
        [PANEL_PATH],
        fallback_name="final_panel_yield_spi_spei_subsidy.csv",
    )
    if panel_path is None or not panel_path.exists():
        raise FileNotFoundError(
            f"Panel file not found. Edit PANEL_PATH. Current value: {PANEL_PATH}"
        )
    log("=" * 80)
    log("LOAD PANEL DATA")
    log("=" * 80)
    log(f"Panel path: {panel_path}")
    df = read_csv_flexible(panel_path)
    log(f"Panel shape: {df.shape}")
    log("Columns: " + ", ".join(map(str, df.columns)))

    region_col = find_col(df, REGION_COL_CANDIDATES)
    if region_col is None:
        raise ValueError(f"No region column found. Candidates: {REGION_COL_CANDIDATES}")
    if YEAR_COL not in df.columns:
        raise ValueError(f"No '{YEAR_COL}' column found in panel data.")
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce").astype("Int64")
    df = df.dropna(subset=[YEAR_COL]).copy()
    df[YEAR_COL] = df[YEAR_COL].astype(int)
    return df, region_col


def add_derived_variables(df: pd.DataFrame, region_col: str) -> pd.DataFrame:
    df = df.copy()

    # Dependent variable.
    if Y_COL not in df.columns:
        prod_col = find_col(df, ["grain_production", "production", "yield", "grain_yield", "y"])
        if prod_col is None:
            raise ValueError(f"Dependent variable '{Y_COL}' not found and no production column can be inferred.")
        df[prod_col] = pd.to_numeric(df[prod_col], errors="coerce")
        df[Y_COL] = np.where(df[prod_col] > 0, np.log(df[prod_col]), np.nan)
        log(f"Created {Y_COL} from {prod_col}.")

    # Area logarithm.
    if "ln_area" not in df.columns:
        area_col = find_col(df, ["sown_area", "area", "grain_area", "crop_area", "grain_crop_sown_area"])
        if area_col is not None:
            df[area_col] = pd.to_numeric(df[area_col], errors="coerce")
            df["ln_area"] = np.where(df[area_col] > 0, np.log(df[area_col]), np.nan)
            log(f"Created ln_area from {area_col}.")

    # Subsidy logarithm.
    if "ln_subsidy_main" not in df.columns:
        sub_col = find_col(df, ["subsidy", "subsidies", "agri_subsidy", "total_subsidy", "support"])
        if sub_col is not None:
            df[sub_col] = pd.to_numeric(df[sub_col], errors="coerce")
            df["ln_subsidy_main"] = np.log1p(df[sub_col].clip(lower=0))
            log(f"Created ln_subsidy_main from {sub_col} using log1p.")

    # Numeric conversion for common variables.
    for c in set(BASE_X_CANDIDATES + NONLINEAR_X_CANDIDATES + [Y_COL]):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Lagged dependent variable.
    if "ln_yield_lag1" not in df.columns:
        df = df.sort_values([region_col, YEAR_COL]).copy()
        df["ln_yield_lag1"] = df.groupby(region_col)[Y_COL].shift(1)
        log("Created ln_yield_lag1 by region-year sorting.")

    # Quadratic terms.
    if "temp_grow_sq" not in df.columns and "temp_grow_mean" in df.columns:
        df["temp_grow_sq"] = df["temp_grow_mean"] ** 2
    if "spi_sq" not in df.columns and "spi_grow_mean" in df.columns:
        df["spi_sq"] = df["spi_grow_mean"] ** 2
    if "spei_sq" not in df.columns and "spei_grow_mean" in df.columns:
        df["spei_sq"] = df["spei_grow_mean"] ** 2

    # Climate × subsidy interactions.
    if "ln_subsidy_main" in df.columns:
        if "temp_x_subsidy" not in df.columns and "temp_grow_mean" in df.columns:
            df["temp_x_subsidy"] = df["temp_grow_mean"] * df["ln_subsidy_main"]
        if "spi_x_subsidy" not in df.columns and "spi_grow_mean" in df.columns:
            df["spi_x_subsidy"] = df["spi_grow_mean"] * df["ln_subsidy_main"]
        if "spei_x_subsidy" not in df.columns and "spei_grow_mean" in df.columns:
            df["spei_x_subsidy"] = df["spei_grow_mean"] * df["ln_subsidy_main"]

    # post2014 and interactions.
    df["post2014"] = (df[YEAR_COL] >= 2014).astype(int)
    for v in POST2014_CLIMATE_VARS:
        if v in df.columns:
            df[f"post2014_x_{v}"] = df["post2014"] * df[v]

    return df


def load_shapefile() -> Optional[pd.DataFrame]:
    if gpd is None:
        log("geopandas is not installed. Map and spatial models will be skipped.")
        return None
    shp_path = find_first_existing([GADM_SHP_PATH], fallback_name="gadm41_RUS_1.shp")
    if shp_path is None or not shp_path.exists():
        log(f"Shapefile not found. Map and spatial models will be skipped. Current path: {GADM_SHP_PATH}")
        return None
    try:
        gdf = gpd.read_file(shp_path)
    except Exception as e:
        log(f"Failed to read shapefile: {e}")
        return None
    log("=" * 80)
    log("LOAD SHAPEFILE")
    log("=" * 80)
    log(f"Shapefile path: {shp_path}")
    log(f"Shapefile shape: {gdf.shape}")
    return gdf


def prepare_region_matching(df: pd.DataFrame, region_col: str, gdf: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[str]]:
    df = df.copy()
    df[region_col] = df[region_col].astype(str).str.strip()

    if gdf is None:
        df["region_match"] = df[region_col]
        return df, None, None

    if "NAME_1" in gdf.columns:
        shape_name_col = "NAME_1"
    else:
        shape_name_col = find_col(gdf, ["name_1", "name", "region", "NAME", "NAME_EN"])
        if shape_name_col is None:
            log("No suitable region name column found in shapefile. Spatial matching skipped.")
            df["region_match"] = df[region_col]
            return df, None, None

    gdf = gdf.to_crs("EPSG:4326").copy()
    gdf["region_shape"] = gdf[shape_name_col].astype(str).str.strip()
    gdf["_shape_clean"] = gdf["region_shape"].apply(clean_region_name)

    shape_lookup = (
        gdf[["region_shape", "_shape_clean"]]
        .drop_duplicates("_shape_clean")
        .set_index("_shape_clean")
    )
    clean_shape_candidates = sorted(shape_lookup.index.dropna().tolist())

    rows = []
    region_match_values = []

    for raw in df[region_col].astype(str):
        if raw in MANUAL_REGION_MAP:
            target = MANUAL_REGION_MAP[raw]
            method = "manual"
            score = 1.0
        elif raw in set(gdf["region_shape"]):
            target = raw
            method = "exact_raw"
            score = 1.0
        else:
            clean_raw = clean_region_name(raw)
            if clean_raw in shape_lookup.index:
                target = shape_lookup.loc[clean_raw, "region_shape"]
                method = "exact_clean"
                score = 1.0
            else:
                best_clean, score = fuzzy_match_one(clean_raw, clean_shape_candidates, cutoff=0.38)
                if best_clean is not None:
                    target = shape_lookup.loc[best_clean, "region_shape"]
                    method = "fuzzy"
                else:
                    target = raw
                    method = "unmatched"
                    score = 0.0
        region_match_values.append(target)
        rows.append({
            "panel_region_original": raw,
            "region_match": target,
            "match_method": method,
            "match_score": score,
        })

    df["region_match"] = region_match_values
    match_df = pd.DataFrame(rows).drop_duplicates()
    match_df.to_csv(RESULTS_DIR / "region_map_match_check.csv", index=False, encoding="utf-8-sig")
    log("Region matching summary:")
    log(str(match_df["match_method"].value_counts(dropna=False)))

    matched = sorted(set(df["region_match"]) & set(gdf["region_shape"]))
    log(f"Matched regions with shapefile: {len(matched)}")

    if len(matched) < 30:
        log("Warning: fewer than 30 regions matched. Spatial models may be skipped or unreliable.")

    return df, gdf, shape_name_col


def build_model_data(df: pd.DataFrame, x_candidates: Sequence[str]) -> Tuple[pd.DataFrame, List[str]]:
    x_cols = [c for c in x_candidates if c in df.columns]
    for c in [Y_COL] + x_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    keep_cols = ["region_match", YEAR_COL, Y_COL] + x_cols
    if "agro_zone" in df.columns:
        keep_cols.append("agro_zone")
    data = df[keep_cols].copy()
    before = len(data)
    data = data.dropna(subset=[Y_COL] + x_cols).copy()
    log(f"Model data: {data.shape}; dropped {before - len(data)} rows with missing model variables.")

    dup = data[data.duplicated(subset=["region_match", YEAR_COL], keep=False)]
    if not dup.empty:
        log("Duplicate region-year rows found. Aggregating numeric variables by mean and categorical variables by first.")
        numeric_cols = [Y_COL] + x_cols
        other_cols = [c for c in data.columns if c not in ["region_match", YEAR_COL] + numeric_cols]
        agg = {c: "mean" for c in numeric_cols}
        for c in other_cols:
            agg[c] = "first"
        data = data.groupby(["region_match", YEAR_COL], as_index=False).agg(agg)
        log(f"After duplicate aggregation: {data.shape}")
    return data, x_cols


# =============================================================================
# 4. Tables and figures
# =============================================================================


def make_descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    desc_map = {
        Y_COL: r"$\ln Y$",
        "ln_yield_lag1": r"$\ln Y_{t-1}$",
        "ln_area": r"$\ln A$",
        "temp_grow_mean": "Growing-season temperature",
        "prec_grow_sum": "Growing-season precipitation",
        "spi_grow_mean": "SPI",
        "spei_grow_mean": "SPEI",
        "ln_subsidy_main": r"$\ln Sub$",
        "ln_subsidy_per_area": r"$\ln Sub/A$",
    }
    cols = [c for c in desc_map if c in df.columns]
    desc = df[cols].apply(pd.to_numeric, errors="coerce").describe().T
    out = pd.DataFrame({
        "Variable": [desc_map[c] for c in desc.index],
        "Observations": desc["count"].astype(int),
        "Mean": desc["mean"].round(3),
        "Std. Dev.": desc["std"].round(3),
        "Min": desc["min"].round(3),
        "Median": df[cols].median(numeric_only=True).reindex(desc.index).round(3).values,
        "Max": desc["max"].round(3),
    })
    out.to_csv(TABLE_DIR / "descriptive_statistics.csv", index=False, encoding="utf-8-sig")
    safe_to_latex(out, TABLE_DIR / "descriptive_statistics.tex", "Descriptive statistics of the main variables", "tab:descriptive")
    return out


def draw_workflow() -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axis("off")
    boxes = [
        (0.04, 0.60, "ERA5-Land\nmonthly grids"),
        (0.04, 0.20, "Russian regional\nboundary polygons"),
        (0.28, 0.40, "Polygon mask\n(regionmask)"),
        (0.50, 0.40, "Region-month\nclimate panel"),
        (0.72, 0.40, "Region-year\nclimate variables"),
        (0.72, 0.08, "Merged\nproduction panel"),
    ]
    for x, y, text in boxes:
        rect = plt.Rectangle((x, y), 0.17, 0.18, fill=False, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + 0.085, y + 0.09, text, ha="center", va="center", fontsize=9)

    arrows = [
        ((0.21, 0.69), (0.28, 0.49)),
        ((0.21, 0.29), (0.28, 0.49)),
        ((0.45, 0.49), (0.50, 0.49)),
        ((0.67, 0.49), (0.72, 0.49)),
        ((0.805, 0.40), (0.805, 0.26)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.1))
    ax.text(0.50, 0.73, "2m temperature, total precipitation\nmonthly aggregation and unit conversion", ha="center", fontsize=8)
    ax.text(0.46, 0.20, "Annual temperature, annual precipitation,\ngrowing-season temperature, growing-season precipitation,\nSPI / SPEI and lagged variables", ha="center", fontsize=8)
    save_fig(FIG_DIR / "fig1_climate_data_workflow.png")


def yearly_line(df: pd.DataFrame, col: str, ylabel: str, title: str, filename: str) -> None:
    if col not in df.columns:
        log(f"Skip {filename}: missing column {col}")
        return
    d = df[[YEAR_COL, col]].dropna().copy()
    if d.empty:
        return
    g = d.groupby(YEAR_COL)[col].mean().reset_index()
    plt.figure(figsize=(8, 5))
    plt.plot(g[YEAR_COL], g[col], marker="o")
    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.title(title)
    save_fig(FIG_DIR / filename)


def scatter_quad(df: pd.DataFrame, xcol: str, ycol: str, xlabel: str, ylabel: str, title: str, filename: str) -> None:
    if xcol not in df.columns or ycol not in df.columns:
        log(f"Skip {filename}: missing {xcol} or {ycol}")
        return
    d = df[[xcol, ycol]].dropna().copy()
    if len(d) < 5:
        log(f"Skip {filename}: too few observations")
        return
    x = d[xcol].astype(float).values
    y = d[ycol].astype(float).values
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, alpha=0.45, s=18)
    try:
        coef = np.polyfit(x, y, 2)
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 200)
        ys = coef[0] * xs ** 2 + coef[1] * xs + coef[2]
        plt.plot(xs, ys, linewidth=2)
    except Exception:
        pass
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    save_fig(FIG_DIR / filename)


def quantile_plot(qdf: pd.DataFrame, variable: str, title: str, filename: str) -> None:
    if qdf is None or qdf.empty:
        return
    if not {"quantile", "variable", "coef"}.issubset(qdf.columns):
        return
    d = qdf[qdf["variable"] == variable].copy().sort_values("quantile")
    if d.empty:
        return
    plt.figure(figsize=(7, 5))
    plt.plot(d["quantile"], d["coef"], marker="o")
    plt.axhline(0, linewidth=1)
    plt.xlabel("Quantile")
    plt.ylabel("Coefficient")
    plt.title(title)
    save_fig(FIG_DIR / filename)


def plot_feature_importance(importance: pd.DataFrame, filename: str, title: str, top_n: int = 20) -> None:
    if importance is None or importance.empty:
        return
    d = importance.copy().sort_values("importance", ascending=False).head(top_n)
    d["feature_readable"] = d["feature"].map(readable_feature_name)
    d = d.sort_values("importance", ascending=True)
    plt.figure(figsize=(8.5, 6))
    plt.barh(d["feature_readable"], d["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(title)
    save_fig(FIG_DIR / filename)


def make_maps(df: pd.DataFrame, gdf: Optional[pd.DataFrame]) -> None:
    if gdf is None or gpd is None:
        log("Maps skipped: no shapefile/geopandas.")
        return
    if "region_shape" not in gdf.columns:
        log("Maps skipped: shapefile missing region_shape.")
        return
    if "region_match" not in df.columns or Y_COL not in df.columns:
        return

    map_data = gdf.copy()
    panel_mean = df.groupby("region_match", as_index=False)[Y_COL].mean()
    map_data = map_data.merge(panel_mean, left_on="region_shape", right_on="region_match", how="left")
    matched = map_data.dropna(subset=[Y_COL]).copy()
    if matched.empty:
        log("Maps skipped: no matched regions with data.")
        return

    centroids_projected = matched.to_crs("EPSG:3857").geometry.centroid
    centroids_wgs84 = gpd.GeoSeries(centroids_projected, crs="EPSG:3857").to_crs("EPSG:4326")
    points_gdf = gpd.GeoDataFrame(matched[["region_shape", Y_COL]].copy(), geometry=centroids_wgs84, crs="EPSG:4326")

    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    gdf.boundary.plot(ax=ax, linewidth=0.38)
    points_gdf.plot(ax=ax, markersize=14, alpha=0.85, edgecolor="black", linewidth=0.20)
    minx, miny, maxx, maxy = gdf.total_bounds
    ax.set_xlim(minx - 2, maxx + 2)
    ax.set_ylim(miny - 1, maxy + 1)
    ax.set_title("Sample Regions within Russia", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", alpha=0.25)
    save_fig(FIG_DIR / "sample_regions_map.png")

    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    map_data.plot(
        ax=ax,
        column=Y_COL,
        legend=True,
        missing_kwds={"color": "lightgrey", "edgecolor": "white", "hatch": "///", "label": "No data"},
        edgecolor="black",
        linewidth=0.30,
        legend_kwds={"shrink": 0.65, "label": "Mean log grain production"},
    )
    ax.set_title("Regional Mean Log Grain Production within Russia", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.set_xlim(minx - 2, maxx + 2)
    ax.set_ylim(miny - 1, maxy + 1)
    save_fig(FIG_DIR / "mean_log_yield_map.png")


# =============================================================================
# 5. Spatial lags and models
# =============================================================================


def spatial_lag_for_year(group: pd.DataFrame, variable: str, w_obj, region_order: List[str]) -> pd.DataFrame:
    group = group.copy()
    group["region_match"] = group["region_match"].astype(str)
    temp = group.set_index("region_match").reindex(region_order)
    values = temp[variable].astype(float).values
    lag_values = np.full(len(values), np.nan, dtype=float)

    for i in range(len(values)):
        neighs = w_obj.neighbors.get(i, [])
        weights = w_obj.weights.get(i, [])
        if len(neighs) == 0:
            continue
        neighs = np.array(neighs, dtype=int)
        weights = np.array(weights, dtype=float)
        valid = neighs < len(values)
        neighs = neighs[valid]
        weights = weights[valid]
        if len(neighs) == 0:
            continue
        neigh_values = values[neighs]
        ok = ~np.isnan(neigh_values)
        if not np.any(ok):
            continue
        neigh_values = neigh_values[ok]
        weights = weights[ok]
        if weights.sum() != 0:
            weights = weights / weights.sum()
            lag_values[i] = np.sum(weights * neigh_values)

    return pd.DataFrame({
        "region_match": region_order,
        YEAR_COL: group[YEAR_COL].iloc[0],
        f"W_{variable}": lag_values,
    })


def construct_spatial_lags(data: pd.DataFrame, gdf: Optional[pd.DataFrame], variables_to_lag: Sequence[str]) -> Tuple[pd.DataFrame, Optional[object], List[str]]:
    if gdf is None or Queen is None:
        log("Spatial lag construction skipped: missing shapefile or libpysal.")
        return data.copy(), None, []
    if "region_shape" not in gdf.columns:
        log("Spatial lag construction skipped: missing region_shape in shapefile.")
        return data.copy(), None, []

    common_regions = sorted(set(data["region_match"].astype(str)) & set(gdf["region_shape"].astype(str)))
    if len(common_regions) < 30:
        log("Spatial lag construction skipped: too few matched regions.")
        return data.copy(), None, []

    data2 = data[data["region_match"].astype(str).isin(common_regions)].copy()
    gdf2 = gdf[gdf["region_shape"].astype(str).isin(common_regions)].copy()
    gdf2 = gdf2.sort_values("region_shape").reset_index(drop=True)
    region_order = gdf2["region_shape"].astype(str).tolist()

    data2["region_match"] = data2["region_match"].astype(str)
    data2 = data2[data2["region_match"].isin(region_order)].copy()
    data2["region_match"] = pd.Categorical(data2["region_match"], categories=region_order, ordered=True)
    data2 = data2.sort_values([YEAR_COL, "region_match"]).reset_index(drop=True)

    try:
        w = Queen.from_dataframe(gdf2)
        w.transform = "R"
    except Exception as e:
        log(f"Failed to construct Queen weights: {e}")
        return data.copy(), None, []

    variables_to_lag = [v for v in dict.fromkeys(variables_to_lag) if v in data2.columns]
    year_tables = []
    for year, group in data2.groupby(YEAR_COL, observed=True):
        year_table = pd.DataFrame({"region_match": region_order, YEAR_COL: year})
        for var in variables_to_lag:
            lag_table = spatial_lag_for_year(group, var, w, region_order)
            year_table = year_table.merge(lag_table, on=["region_match", YEAR_COL], how="left")
        year_tables.append(year_table)

    spatial_lag_data = pd.concat(year_tables, ignore_index=True)
    data2["region_match"] = data2["region_match"].astype(str)
    data_spatial = data2.merge(spatial_lag_data, on=["region_match", YEAR_COL], how="left")

    needed_spatial_cols = [f"W_{v}" for v in variables_to_lag]
    before = len(data_spatial)
    data_spatial = data_spatial.dropna(subset=[Y_COL] + list(variables_to_lag) + needed_spatial_cols).copy()
    log(f"Spatial data: {data_spatial.shape}; dropped {before - len(data_spatial)} rows with missing spatial lags.")
    data_spatial.to_csv(RESULTS_DIR / "panel_with_spatial_lags.csv", index=False, encoding="utf-8-sig")
    return data_spatial, w, region_order


def run_panelols(name: str, data: pd.DataFrame, x_cols: Sequence[str], purpose: str = "") -> Tuple[Optional[object], Optional[pd.DataFrame], Optional[Dict[str, object]]]:
    if PanelOLS is None or sm is None:
        log(f"{name} skipped: linearmodels/statsmodels not installed.")
        return None, None, None
    x_cols = [c for c in x_cols if c in data.columns]
    d = data.dropna(subset=[Y_COL] + x_cols + ["region_match", YEAR_COL]).copy()
    if d.empty or len(x_cols) == 0:
        log(f"{name} skipped: no usable data or predictors.")
        return None, None, None
    d = d.set_index(["region_match", YEAR_COL])
    Y = d[Y_COL]
    X = d[x_cols]
    try:
        X = sm.add_constant(X, has_constant="add")
        model = PanelOLS(Y, X, entity_effects=True, time_effects=True, drop_absorbed=True, check_rank=False)
        res = model.fit(cov_type="clustered", cluster_entity=True)
        save_text(RESULTS_DIR / f"{name}_results.txt", str(res))
        coef_table = model_result_table(res, name)
        coef_table.to_csv(TABLE_DIR / f"{name}_coefficients.csv", index=False, encoding="utf-8-sig")
        safe_to_latex(coef_table, TABLE_DIR / f"{name}_coefficients.tex", f"{name} coefficients", f"tab:{name}_coef")
        row = panel_summary_row(name, res, purpose)
        return res, coef_table, row
    except Exception as e:
        log(f"{name} failed: {e}")
        if STRICT_MODE:
            raise
        return None, None, None


def run_quantile_regression(data: pd.DataFrame, x_cols: Sequence[str]) -> pd.DataFrame:
    if smf is None:
        log("Quantile regression skipped: statsmodels not installed.")
        return pd.DataFrame()
    x_cols = [c for c in x_cols if c in data.columns]
    d = data.dropna(subset=[Y_COL] + x_cols + ["region_match", YEAR_COL]).copy()
    if d.empty:
        return pd.DataFrame()
    formula = Y_COL + " ~ " + " + ".join(x_cols) + " + C(region_match) + C(year)"
    results = []
    for q in QUANTILES:
        try:
            mod = smf.quantreg(formula, d)
            res = mod.fit(q=q, max_iter=5000)
            save_text(RESULTS_DIR / f"Quantile_{q}_results.txt", str(res.summary()))
            for var in x_cols:
                if var in res.params.index:
                    results.append({
                        "quantile": q,
                        "variable": var,
                        "coef": res.params[var],
                        "pvalue": res.pvalues[var],
                        "significance": significance_stars(res.pvalues[var]),
                    })
        except Exception as e:
            log(f"Quantile regression q={q} failed: {e}")
            if STRICT_MODE:
                raise
    table = pd.DataFrame(results)
    table.to_csv(TABLE_DIR / "Quantile_regression_key_coefficients.csv", index=False, encoding="utf-8-sig")
    safe_to_latex(table, TABLE_DIR / "Quantile_regression_key_coefficients.tex", "Key coefficients of quantile regression", "tab:quantile")
    return table


def run_ardl(data: pd.DataFrame, x_cols: Sequence[str]) -> Tuple[Optional[object], Optional[pd.DataFrame], Optional[Dict[str, object]]]:
    d = data.copy().sort_values(["region_match", YEAR_COL])
    for v in ARDL_LAG_VARS:
        if v in d.columns:
            d[f"{v}_lag1"] = d.groupby("region_match")[v].shift(1)
            d[f"{v}_lag2"] = d.groupby("region_match")[v].shift(2)
    ardl_x_cols = list(x_cols)
    for v in ARDL_LAG_VARS:
        ardl_x_cols.extend([f"{v}_lag1", f"{v}_lag2"])
    ardl_x_cols = [c for c in dict.fromkeys(ardl_x_cols) if c in d.columns]
    return run_panelols("Panel_ARDL_style", d, ardl_x_cols, "Dynamic lagged climate relationships")


def run_random_forest(data: pd.DataFrame, x_cols: Sequence[str], with_lag: bool, name: str) -> Tuple[Optional[Pipeline], Optional[pd.DataFrame], Dict[str, object]]:
    empty_row = {
        "Model": name,
        "Purpose": "Prediction",
        "N": np.nan,
        "R2_within": np.nan,
        "R2_between": np.nan,
        "R2_overall": np.nan,
        "RMSE": np.nan,
        "MAE": np.nan,
        "CV_R2_mean": np.nan,
        "CV_R2_std": np.nan,
        "Main finding": "",
    }
    if RandomForestRegressor is None:
        log(f"{name} skipped: scikit-learn not installed.")
        return None, None, empty_row

    num_cols = [c for c in x_cols if c in data.columns]
    if not with_lag:
        num_cols = [c for c in num_cols if c != "ln_yield_lag1"]
    cat_cols = ["agro_zone"] if "agro_zone" in data.columns else []

    d = data.dropna(subset=[Y_COL] + num_cols + cat_cols + ["region_match"]).copy()
    if d.empty or len(num_cols) == 0:
        return None, None, empty_row
    X = d[num_cols + cat_cols]
    y = d[Y_COL]
    groups = d["region_match"].astype(str)

    try:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RF_RANDOM_STATE)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        preprocess = ColumnTransformer(
            transformers=[
                ("num", "passthrough", num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ],
            remainder="drop",
        )
        model = Pipeline([
            ("preprocess", preprocess),
            ("model", RandomForestRegressor(
                n_estimators=RF_N_ESTIMATORS,
                max_depth=None,
                min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                random_state=RF_RANDOM_STATE,
                n_jobs=-1,
            )),
        ])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rmse = rmse_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)

        # Group cross-validation.
        cv_mean = np.nan
        cv_std = np.nan
        try:
            n_groups = groups.nunique()
            n_splits = min(5, n_groups)
            if n_splits >= 2:
                cv = GroupKFold(n_splits=n_splits)
                scores = cross_val_score(model, X, y, cv=cv, groups=groups, scoring="r2", n_jobs=-1)
                cv_mean = float(np.nanmean(scores))
                cv_std = float(np.nanstd(scores))
                pd.DataFrame({"cv_r2": scores}).to_csv(RESULTS_DIR / f"{name}_cross_validation_scores.csv", index=False)
        except Exception as e:
            log(f"{name} cross-validation failed: {e}")

        feature_names = get_feature_names_from_pipeline(model)
        rf_reg = model.named_steps["model"]
        importance = pd.DataFrame({
            "feature": feature_names,
            "importance": rf_reg.feature_importances_,
        }).sort_values("importance", ascending=False)
        importance.to_csv(TABLE_DIR / f"{name}_feature_importance.csv", index=False, encoding="utf-8-sig")

        metrics = {
            "Model": name,
            "Purpose": "Prediction" if with_lag else "Prediction without historical production",
            "N": len(y_test),
            "R2_within": np.nan,
            "R2_between": np.nan,
            "R2_overall": r2,
            "RMSE": rmse,
            "MAE": mae,
            "CV_R2_mean": cv_mean,
            "CV_R2_std": cv_std,
            "Main finding": "Lagged production dominates" if with_lag else "Climate variables become central predictors",
        }

        # Save raw prediction metrics.
        pd.DataFrame([metrics]).to_csv(RESULTS_DIR / f"{name}_metrics.csv", index=False, encoding="utf-8-sig")
        return model, importance, metrics
    except Exception as e:
        log(f"{name} failed: {e}")
        if STRICT_MODE:
            raise
        return None, None, empty_row


def run_shap_for_rf(model: Optional[Pipeline], data: pd.DataFrame, x_cols: Sequence[str], with_lag: bool, name: str, filename: str) -> Optional[pd.DataFrame]:
    if shap is None or model is None:
        log(f"{name} SHAP skipped: shap not installed or model unavailable.")
        return None

    num_cols = [c for c in x_cols if c in data.columns]
    if not with_lag:
        num_cols = [c for c in num_cols if c != "ln_yield_lag1"]
    cat_cols = ["agro_zone"] if "agro_zone" in data.columns else []
    d = data.dropna(subset=[Y_COL] + num_cols + cat_cols).copy()
    if d.empty:
        return None
    if len(d) > SHAP_MAX_ROWS:
        d = d.sample(SHAP_MAX_ROWS, random_state=RF_RANDOM_STATE)
    X = d[num_cols + cat_cols]

    try:
        pre = model.named_steps["preprocess"]
        rf = model.named_steps["model"]
        X_trans = pre.transform(X)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        feature_names = get_feature_names_from_pipeline(model)
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_trans)

        shap_summary = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
        shap_summary.to_csv(TABLE_DIR / f"{name}_SHAP_summary.csv", index=False, encoding="utf-8-sig")

        # SHAP beeswarm summary plot.
        plt.figure(figsize=(9, 6))
        shap.summary_plot(shap_values, X_trans, feature_names=feature_names, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(FIG_DIR / filename, dpi=300, bbox_inches="tight")
        plt.close()
        log(f"Saved SHAP plot: {FIG_DIR / filename}")

        # Also save bar plot of mean |SHAP|.
        dbar = shap_summary.head(20).copy()
        dbar["feature_readable"] = dbar["feature"].map(readable_feature_name)
        dbar = dbar.sort_values("mean_abs_shap", ascending=True)
        plt.figure(figsize=(8.5, 6))
        plt.barh(dbar["feature_readable"], dbar["mean_abs_shap"])
        plt.xlabel("Mean |SHAP value|")
        plt.title(f"SHAP Feature Importance: {name}")
        save_fig(FIG_DIR / f"{Path(filename).stem}_bar.png")
        return shap_summary
    except Exception as e:
        log(f"{name} SHAP failed: {e}")
        if STRICT_MODE:
            raise
        return None


# =============================================================================
# 6. LaTeX helper files
# =============================================================================


def write_latex_snippets() -> None:
    snippet = r"""
% Put this in the LaTeX preamble:
% \graphicspath{{thesis_outputs/figures/}{thesis_outputs/tables/}}

% Descriptive statistics:
\input{thesis_outputs/tables/descriptive_statistics.tex}

% Model comparison:
\input{thesis_outputs/tables/model_comparison_summary.tex}

% Suggested figure calls:
\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{fig1_climate_data_workflow.png}
\caption{Construction of climate variables and assembly of the panel dataset}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.90\textwidth]{sample_regions_map.png}
\caption{Outline of Russia and spatial distribution of sample regions}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.90\textwidth]{mean_log_yield_map.png}
\caption{Spatial distribution of average log grain production across Russian regions}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.75\textwidth]{yearly_mean_log_yield.png}
\caption{Annual changes in regional average log grain production}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.75\textwidth]{yearly_mean_temperature.png}
\caption{Annual changes in growing-season average temperature}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.75\textwidth]{yearly_mean_precipitation.png}
\caption{Annual changes in growing-season cumulative precipitation}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.75\textwidth]{temp_quadratic_fit.png}
\caption{Nonlinear relationship between growing-season average temperature and log grain production}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.75\textwidth]{spi_quadratic_fit.png}
\caption{Nonlinear relationship between SPI and log grain production}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.75\textwidth]{spei_quadratic_fit.png}
\caption{Nonlinear relationship between SPEI and log grain production}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.75\textwidth]{quantile_temperature_coefficients.png}
\caption{Changes in growing-season average temperature coefficients across quantiles}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.75\textwidth]{quantile_spi_coefficients.png}
\caption{Changes in SPI coefficients across quantiles}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{rf_importance_with_lag.png}
\caption{Random forest variable importance: with lagged production}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{rf_importance_without_lag.png}
\caption{Random forest variable importance: without lagged production}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{shap_summary_with_lag.png}
\caption{SHAP summary plot: random forest with lagged production}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{shap_summary_without_lag.png}
\caption{SHAP summary plot: random forest without lagged production}
\end{figure}
""".strip()
    save_text(OUTPUT_ROOT / "latex_include_snippets.tex", snippet)


# =============================================================================
# 7. Main pipeline
# =============================================================================


def main() -> None:
    ensure_dirs()
    save_text(LOG_DIR / "run_config.json", json.dumps({
        "BASE_DIR": str(BASE_DIR),
        "PANEL_PATH": str(PANEL_PATH),
        "GADM_SHP_PATH": str(GADM_SHP_PATH),
        "OUTPUT_ROOT": str(OUTPUT_ROOT),
        "RF_N_ESTIMATORS": RF_N_ESTIMATORS,
        "SHAP_MAX_ROWS": SHAP_MAX_ROWS,
    }, ensure_ascii=False, indent=2))

    log("=" * 80)
    log("ONE-SHOT THESIS PIPELINE STARTED")
    log("=" * 80)
    log(f"Output root: {OUTPUT_ROOT}")

    df, original_region_col = load_panel()
    df = add_derived_variables(df, original_region_col)
    gdf = load_shapefile()
    df, gdf, shape_name_col = prepare_region_matching(df, original_region_col, gdf)

    # Save cleaned panel before model-specific dropping.
    df.to_csv(RESULTS_DIR / "cleaned_panel_with_derived_variables.csv", index=False, encoding="utf-8-sig")

    # Descriptive statistics and basic figures use the cleaned full panel.
    make_descriptive_statistics(df)
    draw_workflow()
    make_maps(df, gdf)

    yearly_line(df, Y_COL, r"Mean $\ln Y$", "Yearly Mean Log Grain Production", "yearly_mean_log_yield.png")
    yearly_line(df, "temp_grow_mean", "Temperature", "Yearly Mean Growing-season Temperature", "yearly_mean_temperature.png")
    yearly_line(df, "prec_grow_sum", "Precipitation", "Yearly Mean Growing-season Precipitation", "yearly_mean_precipitation.png")
    scatter_quad(df, "temp_grow_mean", Y_COL, "Growing-season temperature", r"$\ln Y$", "Grain Production vs Temperature", "temp_quadratic_fit.png")
    scatter_quad(df, "spi_grow_mean", Y_COL, "SPI", r"$\ln Y$", "Grain Production vs SPI", "spi_quadratic_fit.png")
    scatter_quad(df, "spei_grow_mean", Y_COL, "SPEI", r"$\ln Y$", "Grain Production vs SPEI", "spei_quadratic_fit.png")

    # Main model datasets.
    base_data, base_x_cols = build_model_data(df, BASE_X_CANDIDATES)
    nonlinear_data, nonlinear_x_cols = build_model_data(df, NONLINEAR_X_CANDIDATES)

    summary_rows: List[Dict[str, object]] = []

    # Fixed effects models.
    fe_res, fe_table, fe_row = run_panelols("FE_baseline", base_data, base_x_cols, "Average climate relationship")
    if fe_row:
        summary_rows.append(fe_row)

    nl_res, nl_table, nl_row = run_panelols("FE_nonlinear", nonlinear_data, nonlinear_x_cols, "Nonlinear climate response")
    if nl_row:
        nl_row["Main finding"] = "Tests quadratic terms of temperature, SPI and SPEI"
        summary_rows.append(nl_row)

    # Post-2014 interaction model.
    post_x_cols = [c for c in base_x_cols if c in nonlinear_data.columns]
    for v in POST2014_CLIMATE_VARS:
        col = f"post2014_x_{v}"
        if col in nonlinear_data.columns:
            post_x_cols.append(col)
    post_x_cols = list(dict.fromkeys(post_x_cols))
    post_res, post_table, post_row = run_panelols("FE_post2014_interactions", nonlinear_data, post_x_cols, "Change in climate sensitivity after 2014")
    if post_row:
        post_row["Main finding"] = "Post-2014 climate interaction specification"
        summary_rows.append(post_row)

    # Spatial lags and spatial models.
    data_for_spatial, spatial_x_cols = build_model_data(df, NONLINEAR_X_CANDIDATES)
    variables_to_lag = [Y_COL] + spatial_x_cols
    data_spatial, w, region_order = construct_spatial_lags(data_for_spatial, gdf, variables_to_lag)

    if f"W_{Y_COL}" in data_spatial.columns:
        sar_x_cols = [f"W_{Y_COL}"] + spatial_x_cols
        sar_res, sar_table, sar_row = run_panelols("SAR_style", data_spatial, sar_x_cols, "Spatial correlation in production")
        if sar_row:
            sar_row["Main finding"] = "Spatial lag of production included"
            summary_rows.append(sar_row)

        wx_cols = [f"W_{x}" for x in spatial_x_cols if f"W_{x}" in data_spatial.columns]
        sdm_x_cols = [f"W_{Y_COL}"] + spatial_x_cols + wx_cols
        sdm_res, sdm_table, sdm_row = run_panelols("SDM_style", data_spatial, sdm_x_cols, "Spatial Durbin robustness check")
        if sdm_row:
            sdm_row["Main finding"] = "Spatial lag of production and spatially lagged covariates included"
            summary_rows.append(sdm_row)
    else:
        log("SAR/SDM skipped: W_ln_yield not available.")

    # Quantile regression.
    qr_table = run_quantile_regression(data_spatial if not data_spatial.empty else nonlinear_data, nonlinear_x_cols)
    quantile_plot(qr_table, "temp_grow_mean", "Quantile Regression Coefficients: Temperature", "quantile_temperature_coefficients.png")
    quantile_plot(qr_table, "spi_grow_mean", "Quantile Regression Coefficients: SPI", "quantile_spi_coefficients.png")

    # ARDL-style model.
    ardl_source = data_spatial if not data_spatial.empty else nonlinear_data
    ardl_res, ardl_table, ardl_row = run_ardl(ardl_source, nonlinear_x_cols)
    if ardl_row:
        ardl_row["Main finding"] = "Lagged climate variables included"
        summary_rows.append(ardl_row)

    # Random forests.
    ml_source = data_spatial if not data_spatial.empty else nonlinear_data
    rf_model_lag, rf_imp_lag, rf_row_lag = run_random_forest(ml_source, nonlinear_x_cols, with_lag=True, name="RF_with_lag")
    summary_rows.append(rf_row_lag)
    plot_feature_importance(rf_imp_lag, "rf_importance_with_lag.png", "Random Forest Feature Importance (with lagged production)")

    rf_model_nolag, rf_imp_nolag, rf_row_nolag = run_random_forest(ml_source, nonlinear_x_cols, with_lag=False, name="RF_without_lag")
    summary_rows.append(rf_row_nolag)
    plot_feature_importance(rf_imp_nolag, "rf_importance_without_lag.png", "Random Forest Feature Importance (without lagged production)")

    # SHAP plots.
    run_shap_for_rf(rf_model_lag, ml_source, nonlinear_x_cols, with_lag=True, name="RF_with_lag", filename="shap_summary_with_lag.png")
    run_shap_for_rf(rf_model_nolag, ml_source, nonlinear_x_cols, with_lag=False, name="RF_without_lag", filename="shap_summary_without_lag.png")

    # Model comparison summary.
    summary = pd.DataFrame(summary_rows)
    # Standardize columns even if some rows lack CV columns.
    for c in ["Model", "Purpose", "N", "R2_within", "R2_between", "R2_overall", "RMSE", "MAE", "CV_R2_mean", "CV_R2_std", "Main finding"]:
        if c not in summary.columns:
            summary[c] = np.nan
    summary = summary[["Model", "Purpose", "N", "R2_within", "R2_between", "R2_overall", "RMSE", "MAE", "CV_R2_mean", "CV_R2_std", "Main finding"]]
    summary.to_csv(TABLE_DIR / "model_comparison_summary.csv", index=False, encoding="utf-8-sig")
    safe_to_latex(summary.round(4), TABLE_DIR / "model_comparison_summary.tex", "Model comparison summary", "tab:model_comparison")

    # Compatibility copies for old naming convention, if desired.
    alias_map = {
        "yearly_mean_log_yield.png": "01_yearly_mean_ln_yield.png",
        "yearly_mean_temperature.png": "02_yearly_mean_temp.png",
        "yearly_mean_precipitation.png": "03_yearly_mean_prec.png",
        "temp_quadratic_fit.png": "04_scatter_temp_yield_quadratic.png",
        "spi_quadratic_fit.png": "05_scatter_spi_yield_quadratic.png",
        "spei_quadratic_fit.png": "06_scatter_spei_yield_quadratic.png",
        "quantile_temperature_coefficients.png": "07_quantile_coeff_temp.png",
        "quantile_spi_coefficients.png": "08_quantile_coeff_spi.png",
        "rf_importance_with_lag.png": "09_rf_feature_importance_with_lag.png",
        "rf_importance_without_lag.png": "10_rf_feature_importance_without_lag.png",
        "sample_regions_map.png": "11_russia_outline_sample_points.png",
        "mean_log_yield_map.png": "12_russia_choropleth_mean_ln_yield.png",
    }
    for src, dst in alias_map.items():
        src_path = FIG_DIR / src
        dst_path = FIG_DIR / dst
        if src_path.exists():
            shutil.copyfile(src_path, dst_path)

    # README and LaTeX snippets.
    write_latex_snippets()
    readme = f"""
Thesis output package
=====================

Output root:
{OUTPUT_ROOT}

Folders:
- figures/: all PNG figures for the revised thesis LaTeX draft.
- tables/: descriptive statistics, model coefficients, quantile coefficients, RF importance, SHAP summary, model comparison tables.
- results/: full text outputs and intermediate CSV files.
- logs/: run configuration.

Recommended LaTeX preamble line:
\\graphicspath{{{{thesis_outputs/figures/}}{{thesis_outputs/tables/}}}}

Important files:
- figures/fig1_climate_data_workflow.png
- figures/sample_regions_map.png
- figures/mean_log_yield_map.png
- figures/yearly_mean_log_yield.png
- figures/yearly_mean_temperature.png
- figures/yearly_mean_precipitation.png
- figures/temp_quadratic_fit.png
- figures/spi_quadratic_fit.png
- figures/spei_quadratic_fit.png
- figures/quantile_temperature_coefficients.png
- figures/quantile_spi_coefficients.png
- figures/rf_importance_with_lag.png
- figures/rf_importance_without_lag.png
- figures/shap_summary_with_lag.png
- figures/shap_summary_without_lag.png
- tables/descriptive_statistics.tex
- tables/model_comparison_summary.tex
- latex_include_snippets.tex
""".strip()
    save_text(OUTPUT_ROOT / "README.txt", readme)

    zip_path = BASE_DIR / "thesis_outputs.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in OUTPUT_ROOT.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(OUTPUT_ROOT)))
    log("=" * 80)
    log("ALL DONE")
    log("=" * 80)
    log(f"Outputs saved to: {OUTPUT_ROOT}")
    log(f"ZIP package: {zip_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log("=" * 80)
        log("PIPELINE FAILED")
        log("=" * 80)
        log(str(exc))
        raise
