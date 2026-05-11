# -*- coding: utf-8 -*-
"""
Spatial Econometrics + Machine Learning Analysis
For Russia grain yield and climate panel data

Models included:
1. Two-way Fixed Effects model
2. SAR-style spatial lag model
3. SDM-style spatial Durbin model
4. Random Forest
5. XGBoost
6. SHAP feature importance
7. 2010 drought DiD
8. Quantile regression
9. Distributed lag / ARDL-like panel model

Author: Ji Yiman
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd

from libpysal.weights import Queen
from linearmodels.panel import PanelOLS

import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt


# ============================================================
# 0. Paths
# ============================================================

BASE_DIR = "/Users/littlestars/Desktop/grain_project"

PANEL_PATH = os.path.join(
    BASE_DIR,
    "data",
    "final",
    "final_panel_yield_spi_spei_subsidy.csv"
)

LARGE_FILES_DIR = os.path.join(
    BASE_DIR,
    "data",
    "large_files"
)

GADM_SHP_PATH = os.path.join(
    LARGE_FILES_DIR,
    "gadm41_RUS_shp",
    "gadm41_RUS_1.shp"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results_spatial_ml"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("PATH CHECK")
print("=" * 80)
print("Panel path:", PANEL_PATH)
print("Large files dir:", LARGE_FILES_DIR)
print("GADM shapefile path:", GADM_SHP_PATH)
print("Output dir:", OUTPUT_DIR)

if not os.path.exists(PANEL_PATH):
    raise FileNotFoundError(f"Panel CSV not found: {PANEL_PATH}")

if not os.path.exists(GADM_SHP_PATH):
    raise FileNotFoundError(f"GADM shapefile not found: {GADM_SHP_PATH}")


# ============================================================
# 1. Read data
# ============================================================

df = pd.read_csv(PANEL_PATH)
gdf = gpd.read_file(GADM_SHP_PATH)

print("\n" + "=" * 80)
print("DATA LOADED")
print("=" * 80)
print("Panel shape:", df.shape)
print("Panel columns:")
print(df.columns.tolist())

print("\nShapefile shape:", gdf.shape)
print("Shapefile columns:")
print(gdf.columns.tolist())


# ============================================================
# 2. Basic cleaning
# ============================================================

df["year"] = df["year"].astype(int)

if "region_std" not in df.columns:
    if "region" in df.columns:
        df["region_std"] = df["region"]
    else:
        raise ValueError("Neither 'region_std' nor 'region' exists in panel data.")

if "NAME_1" not in gdf.columns:
    raise ValueError("GADM shapefile does not contain 'NAME_1'. Please inspect shapefile columns.")

gdf["region_shape"] = gdf["NAME_1"]


# ============================================================
# ============================================================
# 3. Region name matching
# ============================================================
# 你的 CSV 是俄文地区名，GADM 的 NAME_1 通常是英文/拉丁转写名。
# 所以必须用俄文 -> GADM 名称映射。

manual_map = {
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

df["region_match"] = df["region_std"].replace(manual_map)

# 打印 GADM 里的真实 NAME_1，方便检查
print("\n" + "=" * 80)
print("GADM NAME_1 VALUES")
print("=" * 80)
print(sorted(gdf["region_shape"].dropna().unique()))

panel_regions = sorted(df["region_match"].dropna().unique())
shape_regions = sorted(gdf["region_shape"].dropna().unique())

unmatched_panel = sorted(set(panel_regions) - set(shape_regions))
unmatched_shape = sorted(set(shape_regions) - set(panel_regions))

print("\n" + "=" * 80)
print("REGION MATCHING CHECK")
print("=" * 80)

print("Number of panel regions:", len(panel_regions))
print("Number of shapefile regions:", len(shape_regions))

print("\nPanel regions not found in shapefile:")
for x in unmatched_panel:
    print("  -", x)

print("\nShapefile regions not found in panel, first 100:")
for x in unmatched_shape[:100]:
    print("  -", x)

matched_regions = sorted(set(panel_regions) & set(shape_regions))

df_matched = df[df["region_match"].isin(matched_regions)].copy()
gdf_matched = gdf[gdf["region_shape"].isin(matched_regions)].copy()

print("\nMatched panel shape:", df_matched.shape)
print("Matched shapefile shape:", gdf_matched.shape)
print("Matched regions:", len(matched_regions))

pd.DataFrame({"unmatched_panel_region": unmatched_panel}).to_csv(
    os.path.join(OUTPUT_DIR, "unmatched_panel_regions.csv"),
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame({"matched_region": matched_regions}).to_csv(
    os.path.join(OUTPUT_DIR, "matched_regions.csv"),
    index=False,
    encoding="utf-8-sig"
)

if len(matched_regions) < 40:
    raise ValueError(
        "Matched regions are fewer than 40. "
        "This means GADM NAME_1 uses different spelling. "
        "Check the printed GADM NAME_1 VALUES above and adjust manual_map."
    )
# ============================================================
# 4. Sort regions and construct spatial weights
# ============================================================

gdf_matched = gdf_matched.sort_values("region_shape").reset_index(drop=True)
ordered_regions = gdf_matched["region_shape"].tolist()

df_matched["region_match"] = pd.Categorical(
    df_matched["region_match"],
    categories=ordered_regions,
    ordered=True
)

df_matched = df_matched.sort_values(["year", "region_match"]).reset_index(drop=True)

w = Queen.from_dataframe(gdf_matched)
w.transform = "R"

print("\n" + "=" * 80)
print("SPATIAL WEIGHTS")
print("=" * 80)
print("Number of regions in W:", w.n)
print("Number of islands:", len(w.islands))
print("Islands:", w.islands)

neighbors_list = []

for i, neighs in w.neighbors.items():
    region_i = ordered_regions[i]
    for j in neighs:
        neighbors_list.append({
            "region": region_i,
            "neighbor": ordered_regions[j]
        })

neighbors_df = pd.DataFrame(neighbors_list)
neighbors_df.to_csv(
    os.path.join(OUTPUT_DIR, "spatial_neighbors.csv"),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 5. Variables
# ============================================================

y_col = "ln_yield"

candidate_x_cols = [
    "temp_grow_mean",
    "prec_grow_sum",
    "spi_grow_mean",
    "spei_grow_mean",
    "ln_subsidy_main",
    "ln_subsidy_per_area",
    "ln_yield_lag1",
    "temp_grow_sq",
    "spi_sq",
    "spei_sq",
    "temp_x_subsidy",
    "spi_x_subsidy",
    "spei_x_subsidy"
]

if y_col not in df_matched.columns:
    raise ValueError(f"Dependent variable {y_col} not found in panel data.")

x_cols = [c for c in candidate_x_cols if c in df_matched.columns]

print("\n" + "=" * 80)
print("MODEL VARIABLES")
print("=" * 80)
print("Dependent variable:", y_col)
print("Predictors:")
for c in x_cols:
    print("  -", c)

keep_cols = ["region_match", "year", y_col] + x_cols

if "agro_zone" in df_matched.columns:
    keep_cols.append("agro_zone")

data = df_matched[keep_cols].copy()

for c in [y_col] + x_cols:
    data[c] = pd.to_numeric(data[c], errors="coerce")

data = data.dropna(subset=[y_col] + x_cols).copy()

print("\nData after dropping missing values:", data.shape)

# ============================================================
# Fix duplicate region-year rows before spatial lag construction
# ============================================================

# 关键：先把 Categorical 转成普通字符串，避免 groupby 插入未观测类别
data["region_match"] = data["region_match"].astype(str)

dup_check = data[data.duplicated(subset=["region_match", "year"], keep=False)].copy()

if not dup_check.empty:
    print("\n" + "=" * 80)
    print("DUPLICATE REGION-YEAR ROWS FOUND")
    print("=" * 80)

    dup_count = (
        dup_check
        .groupby(["region_match", "year"], observed=True)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    print(dup_count.head(50))

    dup_check.to_csv(
        os.path.join(OUTPUT_DIR, "duplicate_region_year_rows.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    numeric_cols = [y_col] + x_cols
    other_cols = [
        c for c in data.columns
        if c not in ["region_match", "year"] + numeric_cols
    ]

    agg_dict = {}

    for col in numeric_cols:
        agg_dict[col] = "mean"

    for col in other_cols:
        agg_dict[col] = "first"

    data = (
        data
        .groupby(["region_match", "year"], as_index=False, observed=True)
        .agg(agg_dict)
    )

    print("\nAfter aggregating duplicate region-year rows:", data.shape)

else:
    print("\nNo duplicate region-year rows found.")

# 重新设定地区顺序，只保留最终数据中真实存在的地区
ordered_regions = [r for r in ordered_regions if r in set(data["region_match"])]

data["region_match"] = pd.Categorical(
    data["region_match"],
    categories=ordered_regions,
    ordered=True
)

data = data.sort_values(["year", "region_match"]).reset_index(drop=True)

print("Final data shape before spatial lag:", data.shape)
print("Final number of regions:", data["region_match"].nunique())


# ============================================================
# ============================================================
# 6. Spatial lag variables
# ============================================================

# ------------------------------------------------------------
# 6.1 Synchronize data, shapefile and spatial weights
# ------------------------------------------------------------

data["region_match"] = data["region_match"].astype(str)

valid_regions_in_data = set(data["region_match"].dropna().unique())
valid_regions_in_shape = set(gdf_matched["region_shape"].dropna().unique())

final_regions = sorted(valid_regions_in_data & valid_regions_in_shape)

print("\n" + "=" * 80)
print("SYNCHRONIZE DATA AND SHAPEFILE BEFORE SPATIAL LAG")
print("=" * 80)
print("Regions in data:", len(valid_regions_in_data))
print("Regions in shapefile:", len(valid_regions_in_shape))
print("Final common regions:", len(final_regions))

data = data[data["region_match"].isin(final_regions)].copy()
gdf_matched = gdf_matched[gdf_matched["region_shape"].isin(final_regions)].copy()

gdf_matched = gdf_matched.sort_values("region_shape").reset_index(drop=True)
ordered_regions = gdf_matched["region_shape"].tolist()

data["region_match"] = pd.Categorical(
    data["region_match"],
    categories=ordered_regions,
    ordered=True
)

data = data.sort_values(["year", "region_match"]).reset_index(drop=True)

# 重新构造空间权重矩阵
w = Queen.from_dataframe(gdf_matched)
w.transform = "R"

print("Rebuilt W regions:", w.n)
print("Data regions:", data["region_match"].nunique())
print("Number of islands:", len(w.islands))
print("Islands:", w.islands)

if w.n != data["region_match"].nunique():
    raise ValueError(
        f"W has {w.n} regions, but data has {data['region_match'].nunique()} regions."
    )


# ------------------------------------------------------------
# 6.2 Spatial lag function
# ------------------------------------------------------------

def spatial_lag_for_year(group, variable, w_obj, region_order):
    """
    Compute W * variable for one year.
    This version is robust to missing regions in a given year.
    """

    group = group.copy()
    group["region_match"] = group["region_match"].astype(str)

    temp = group.set_index("region_match").reindex(region_order)

    values = temp[variable].astype(float).values
    lag_values = np.full(len(values), np.nan, dtype=float)

    for i in range(len(values)):
        neighs = w_obj.neighbors.get(i, [])
        weights = w_obj.weights.get(i, [])

        if len(neighs) == 0:
            lag_values[i] = np.nan
            continue

        neighs = np.array(neighs, dtype=int)
        weights = np.array(weights, dtype=float)

        # 关键保护：防止空间权重编号超过当前数组长度
        valid_mask = neighs < len(values)
        neighs = neighs[valid_mask]
        weights = weights[valid_mask]

        if len(neighs) == 0:
            lag_values[i] = np.nan
            continue

        neigh_values = values[neighs]

        # 如果某些邻居在该年份没有数据，忽略这些邻居并重新标准化权重
        not_nan = ~np.isnan(neigh_values)

        if not np.any(not_nan):
            lag_values[i] = np.nan
            continue

        neigh_values = neigh_values[not_nan]
        weights = weights[not_nan]

        if weights.sum() == 0:
            lag_values[i] = np.nan
        else:
            weights = weights / weights.sum()
            lag_values[i] = np.sum(weights * neigh_values)

    result = pd.DataFrame({
        "region_match": region_order,
        "year": group["year"].iloc[0],
        f"W_{variable}": lag_values
    })

    return result


# ------------------------------------------------------------
# ------------------------------------------------------------
# 6.3 Generate spatial lags
# ------------------------------------------------------------

# 防止变量重复
variables_to_lag = list(dict.fromkeys([y_col] + x_cols))

year_lag_tables = []

for year, group in data.groupby("year", observed=True):

    year_table = pd.DataFrame({
        "region_match": ordered_regions,
        "year": year
    })

    for var in variables_to_lag:
        temp_lag = spatial_lag_for_year(group, var, w, ordered_regions)

        # 只保留当前变量的空间滞后列
        lag_col = f"W_{var}"

        year_table = year_table.merge(
            temp_lag[["region_match", "year", lag_col]],
            on=["region_match", "year"],
            how="left"
        )

    year_lag_tables.append(year_table)

spatial_lag_data = pd.concat(year_lag_tables, ignore_index=True)

# 再次去重，防止历史代码残留造成重复列
spatial_lag_data = spatial_lag_data.loc[:, ~spatial_lag_data.columns.duplicated()].copy()

data_spatial = data.merge(
    spatial_lag_data,
    on=["region_match", "year"],
    how="left"
)

print("\nBefore dropping NA after spatial lags:", data_spatial.shape)

# 不要无脑 dropna 全部列，否则 islands 会导致大量样本消失
needed_spatial_cols = [f"W_{var}" for var in variables_to_lag]
data_spatial = data_spatial.dropna(subset=[y_col] + x_cols + needed_spatial_cols).copy()

print("Data with spatial lags:", data_spatial.shape)

data_spatial.to_csv(
    os.path.join(OUTPUT_DIR, "panel_with_spatial_lags.csv"),
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# 7. Two-way FE model
# ============================================================

panel_fe = data_spatial.copy()
panel_fe = panel_fe.set_index(["region_match", "year"])

Y = panel_fe[y_col]
X = panel_fe[x_cols]
X = sm.add_constant(X)

fe_model = PanelOLS(
    Y,
    X,
    entity_effects=True,
    time_effects=True,
    drop_absorbed=True
)

fe_res = fe_model.fit(cov_type="clustered", cluster_entity=True)

print("\n" + "=" * 80)
print("TWO-WAY FE MODEL")
print("=" * 80)
print(fe_res)

with open(os.path.join(OUTPUT_DIR, "FE_results.txt"), "w", encoding="utf-8") as f:
    f.write(str(fe_res))


# ============================================================
# 8. SAR-style model
# ============================================================

sar_x_cols = [f"W_{y_col}"] + x_cols

Y_sar = panel_fe[y_col]
X_sar = panel_fe[sar_x_cols]
X_sar = sm.add_constant(X_sar)

sar_model = PanelOLS(
    Y_sar,
    X_sar,
    entity_effects=True,
    time_effects=True,
    drop_absorbed=True
)

sar_res = sar_model.fit(cov_type="clustered", cluster_entity=True)

print("\n" + "=" * 80)
print("SAR-STYLE MODEL")
print("=" * 80)
print(sar_res)

with open(os.path.join(OUTPUT_DIR, "SAR_results.txt"), "w", encoding="utf-8") as f:
    f.write(str(sar_res))


# ============================================================
# 9. SDM-style model
# ============================================================

wx_cols = [f"W_{x}" for x in x_cols if f"W_{x}" in panel_fe.columns]
sdm_x_cols = [f"W_{y_col}"] + x_cols + wx_cols

Y_sdm = panel_fe[y_col]
X_sdm = panel_fe[sdm_x_cols]
X_sdm = sm.add_constant(X_sdm)

sdm_model = PanelOLS(
    Y_sdm,
    X_sdm,
    entity_effects=True,
    time_effects=True,
    drop_absorbed=True
)

sdm_res = sdm_model.fit(cov_type="clustered", cluster_entity=True)

print("\n" + "=" * 80)
print("SDM-STYLE MODEL")
print("=" * 80)
print(sdm_res)

with open(os.path.join(OUTPUT_DIR, "SDM_results.txt"), "w", encoding="utf-8") as f:
    f.write(str(sdm_res))


model_compare = pd.DataFrame({
    "Model": ["FE", "SAR-style", "SDM-style"],
    "N": [fe_res.nobs, sar_res.nobs, sdm_res.nobs],
    "R2_within": [
        fe_res.rsquared_within,
        sar_res.rsquared_within,
        sdm_res.rsquared_within
    ],
    "R2_between": [
        fe_res.rsquared_between,
        sar_res.rsquared_between,
        sdm_res.rsquared_between
    ],
    "R2_overall": [
        fe_res.rsquared_overall,
        sar_res.rsquared_overall,
        sdm_res.rsquared_overall
    ]
})

model_compare.to_csv(
    os.path.join(OUTPUT_DIR, "model_comparison_FE_SAR_SDM.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 80)
print("FE / SAR / SDM COMPARISON")
print("=" * 80)
print(model_compare)


# ============================================================
# 10. Random Forest
# ============================================================

ml_data = data_spatial.copy()

num_cols = x_cols.copy()

if "agro_zone" in ml_data.columns:
    cat_cols = ["agro_zone"]
else:
    cat_cols = []

ml_data = ml_data.dropna(subset=[y_col] + num_cols + cat_cols).copy()

X_ml = ml_data[num_cols + cat_cols]
y_ml = ml_data[y_col]
groups = ml_data["region_match"]

gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(gss.split(X_ml, y_ml, groups=groups))

X_train = X_ml.iloc[train_idx]
X_test = X_ml.iloc[test_idx]
y_train = y_ml.iloc[train_idx]
y_test = y_ml.iloc[test_idx]

preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ],
    remainder="drop"
)

rf_model = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

try:
    rf_rmse = mean_squared_error(y_test, rf_pred, squared=False)
except TypeError:
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print("\n" + "=" * 80)
print("RANDOM FOREST")
print("=" * 80)
print("RMSE:", rf_rmse)
print("MAE:", rf_mae)
print("R2:", rf_r2)

rf_reg = rf_model.named_steps["model"]
feature_names = rf_model.named_steps["preprocess"].get_feature_names_out()

rf_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": rf_reg.feature_importances_
}).sort_values("importance", ascending=False)

rf_importance.to_csv(
    os.path.join(OUTPUT_DIR, "RF_feature_importance.csv"),
    index=False,
    encoding="utf-8-sig"
)

plt.figure(figsize=(9, 6))
plt.barh(
    rf_importance["feature"].head(20)[::-1],
    rf_importance["importance"].head(20)[::-1]
)
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "RF_feature_importance.png"), dpi=300)
plt.close()

# ============================================================
# 10B. Random Forest without lagged yield
# ============================================================
# 这一版去掉 ln_yield_lag1，用来观察气候、补贴、农业区变量的重要性。
# 论文里更适合用这一张图解释“哪些因素关键”。

ml_data_no_lag = data_spatial.copy()

num_cols_no_lag = [c for c in x_cols if c != "ln_yield_lag1"]

if "agro_zone" in ml_data_no_lag.columns:
    cat_cols_no_lag = ["agro_zone"]
else:
    cat_cols_no_lag = []

ml_data_no_lag = ml_data_no_lag.dropna(
    subset=[y_col] + num_cols_no_lag + cat_cols_no_lag
).copy()

X_ml_no_lag = ml_data_no_lag[num_cols_no_lag + cat_cols_no_lag]
y_ml_no_lag = ml_data_no_lag[y_col]
groups_no_lag = ml_data_no_lag["region_match"]

gss_no_lag = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx_no_lag, test_idx_no_lag = next(
    gss_no_lag.split(X_ml_no_lag, y_ml_no_lag, groups=groups_no_lag)
)

X_train_no_lag = X_ml_no_lag.iloc[train_idx_no_lag]
X_test_no_lag = X_ml_no_lag.iloc[test_idx_no_lag]
y_train_no_lag = y_ml_no_lag.iloc[train_idx_no_lag]
y_test_no_lag = y_ml_no_lag.iloc[test_idx_no_lag]

preprocess_no_lag = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols_no_lag),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_no_lag)
    ],
    remainder="drop"
)

rf_model_no_lag = Pipeline(
    steps=[
        ("preprocess", preprocess_no_lag),
        ("model", RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

rf_model_no_lag.fit(X_train_no_lag, y_train_no_lag)
rf_pred_no_lag = rf_model_no_lag.predict(X_test_no_lag)

try:
    rf_no_lag_rmse = mean_squared_error(y_test_no_lag, rf_pred_no_lag, squared=False)
except TypeError:
    rf_no_lag_rmse = np.sqrt(mean_squared_error(y_test_no_lag, rf_pred_no_lag))

rf_no_lag_mae = mean_absolute_error(y_test_no_lag, rf_pred_no_lag)
rf_no_lag_r2 = r2_score(y_test_no_lag, rf_pred_no_lag)

print("\n" + "=" * 80)
print("RANDOM FOREST WITHOUT LAGGED YIELD")
print("=" * 80)
print("RMSE:", rf_no_lag_rmse)
print("MAE:", rf_no_lag_mae)
print("R2:", rf_no_lag_r2)

rf_reg_no_lag = rf_model_no_lag.named_steps["model"]
feature_names_no_lag = rf_model_no_lag.named_steps["preprocess"].get_feature_names_out()

rf_importance_no_lag = pd.DataFrame({
    "feature": feature_names_no_lag,
    "importance": rf_reg_no_lag.feature_importances_
}).sort_values("importance", ascending=False)

rf_importance_no_lag.to_csv(
    os.path.join(OUTPUT_DIR, "RF_no_lag_feature_importance.csv"),
    index=False,
    encoding="utf-8-sig"
)

plt.figure(figsize=(9, 6))
plt.barh(
    rf_importance_no_lag["feature"].head(20)[::-1],
    rf_importance_no_lag["importance"].head(20)[::-1]
)
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance without Lagged Yield")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "RF_no_lag_feature_importance.png"), dpi=300)
plt.close()

# 保存预测性能
rf_no_lag_compare = pd.DataFrame({
    "Model": ["Random Forest without ln_yield_lag1"],
    "RMSE": [rf_no_lag_rmse],
    "MAE": [rf_no_lag_mae],
    "R2": [rf_no_lag_r2]
})

rf_no_lag_compare.to_csv(
    os.path.join(OUTPUT_DIR, "RF_no_lag_model_comparison.csv"),
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# 11. XGBoost
# ============================================================

xgb_rmse = np.nan
xgb_mae = np.nan
xgb_r2 = np.nan
xgb_model = None

try:
    from xgboost import XGBRegressor

    xgb_model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", XGBRegressor(
                n_estimators=600,
                learning_rate=0.03,
                max_depth=4,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1
            ))
        ]
    )

    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)

    try:
        xgb_rmse = mean_squared_error(y_test, xgb_pred, squared=False)
    except TypeError:
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))

    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    xgb_r2 = r2_score(y_test, xgb_pred)

    print("\n" + "=" * 80)
    print("XGBOOST")
    print("=" * 80)
    print("RMSE:", xgb_rmse)
    print("MAE:", xgb_mae)
    print("R2:", xgb_r2)

except Exception as e:
    print("\nXGBoost failed:")
    print(e)


ml_compare = pd.DataFrame({
    "Model": ["Random Forest", "XGBoost"],
    "RMSE": [rf_rmse, xgb_rmse],
    "MAE": [rf_mae, xgb_mae],
    "R2": [rf_r2, xgb_r2]
})

ml_compare.to_csv(
    os.path.join(OUTPUT_DIR, "ML_model_comparison.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 80)
print("ML MODEL COMPARISON")
print("=" * 80)
print(ml_compare)


# ============================================================
# 12. SHAP
# ============================================================

try:
    import shap

    if xgb_model is not None and not np.isnan(xgb_r2):
        fitted_preprocess = xgb_model.named_steps["preprocess"]
        fitted_xgb = xgb_model.named_steps["model"]

        X_test_trans = fitted_preprocess.transform(X_test)
        feature_names = fitted_preprocess.get_feature_names_out()

        if hasattr(X_test_trans, "toarray"):
            X_test_trans_dense = X_test_trans.toarray()
        else:
            X_test_trans_dense = X_test_trans

        explainer = shap.TreeExplainer(fitted_xgb)
        shap_values = explainer.shap_values(X_test_trans_dense)

        shap_summary = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0)
        }).sort_values("mean_abs_shap", ascending=False)

        shap_summary.to_csv(
            os.path.join(OUTPUT_DIR, "SHAP_summary.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        plt.figure(figsize=(9, 6))
        plt.barh(
            shap_summary["feature"].head(20)[::-1],
            shap_summary["mean_abs_shap"].head(20)[::-1]
        )
        plt.xlabel("Mean |SHAP value|")
        plt.title("SHAP Feature Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "SHAP_feature_importance.png"), dpi=300)
        plt.close()

        print("\nSHAP analysis finished.")

except Exception as e:
    print("\nSHAP failed:")
    print(e)


# ============================================================
## ============================================================
# 13. 2010 Drought DiD
# ============================================================

did_data = data_spatial.copy()

drought_var = None

if "spi_grow_mean" in did_data.columns:
    drought_var = "spi_grow_mean"
elif "spei_grow_mean" in did_data.columns:
    drought_var = "spei_grow_mean"

if drought_var is None:
    print("\nNo SPI/SPEI variable found. DiD skipped.")
else:
    drought_2010 = did_data[
        did_data["year"] == 2010
    ][["region_match", drought_var]].dropna().copy()

    threshold = drought_2010[drought_var].quantile(0.10)

    treated_regions = drought_2010.loc[
        drought_2010[drought_var] <= threshold,
        "region_match"
    ].astype(str).unique()

    did_data["region_match"] = did_data["region_match"].astype(str)
    did_data["treated"] = did_data["region_match"].isin(treated_regions).astype(int)
    did_data["post2010"] = (did_data["year"] >= 2010).astype(int)
    did_data["did"] = did_data["treated"] * did_data["post2010"]

    # 使用较短窗口，避免长期趋势干扰
    did_window = did_data[
        (did_data["year"] >= 2008) &
        (did_data["year"] <= 2012)
    ].copy()

    # DiD 不要塞太多变量，否则很容易共线
    did_control_candidates = [
        "temp_grow_mean",
        "prec_grow_sum",
        "spi_grow_mean",
        "spei_grow_mean",
        "ln_subsidy_per_area",
        "ln_yield_lag1"
    ]

    did_controls = [
        c for c in did_control_candidates
        if c in did_window.columns
    ]

    # 不要同时放 spi 和 spei，二者常常高度相关
    if "spi_grow_mean" in did_controls and "spei_grow_mean" in did_controls:
        did_controls.remove("spei_grow_mean")

    did_window = did_window.dropna(
        subset=[y_col, "did"] + did_controls
    ).copy()

    print("\n" + "=" * 80)
    print("2010 DROUGHT DID DATA CHECK")
    print("=" * 80)
    print("Drought variable:", drought_var)
    print("Threshold:", threshold)
    print("Number of treated regions:", len(treated_regions))
    print("DID window shape:", did_window.shape)
    print("DID controls:", did_controls)
    print("DID value counts:")
    print(did_window["did"].value_counts())

    if did_window["did"].nunique() < 2:
        print("\nDiD skipped: did variable has no variation.")
    else:
        did_panel = did_window.set_index(["region_match", "year"])

        Y_did = did_panel[y_col]

        # 关键：这里只放 did 和少量控制变量，不手动加 constant
        X_did = did_panel[["did"] + did_controls]

        did_model = PanelOLS(
            Y_did,
            X_did,
            entity_effects=True,
            time_effects=True,
            drop_absorbed=True,
            check_rank=False
        )

        did_res = did_model.fit(
            cov_type="clustered",
            cluster_entity=True
        )

        print("\n" + "=" * 80)
        print("2010 DROUGHT DID")
        print("=" * 80)
        print(did_res)

        with open(os.path.join(OUTPUT_DIR, "DID_2010_drought_results.txt"), "w", encoding="utf-8") as f:
            f.write("Drought variable: " + str(drought_var) + "\n")
            f.write("Threshold: " + str(threshold) + "\n")
            f.write("Number of treated regions: " + str(len(treated_regions)) + "\n")
            f.write("DID controls: " + ", ".join(did_controls) + "\n\n")
            f.write(str(did_res))

        pd.DataFrame({"treated_region": treated_regions}).to_csv(
            os.path.join(OUTPUT_DIR, "DID_treated_regions_2010.csv"),
            index=False,
            encoding="utf-8-sig"
        )
# ============================================================
# 14. Quantile regression
# ============================================================

qr_data = data_spatial.copy()
qr_data = qr_data.dropna(subset=[y_col] + x_cols).copy()

qr_formula = y_col + " ~ " + " + ".join(x_cols) + " + C(region_match) + C(year)"

quantiles = [0.25, 0.50, 0.75]
qr_results = []

for q in quantiles:
    try:
        qr_mod = smf.quantreg(qr_formula, qr_data)
        qr_res = qr_mod.fit(q=q, max_iter=5000)

        print("\n" + "=" * 80)
        print(f"QUANTILE REGRESSION q={q}")
        print("=" * 80)
        print(qr_res.summary())

        with open(os.path.join(OUTPUT_DIR, f"Quantile_{q}_results.txt"), "w", encoding="utf-8") as f:
            f.write(str(qr_res.summary()))

        params = qr_res.params

        for var in x_cols:
            if var in params.index:
                qr_results.append({
                    "quantile": q,
                    "variable": var,
                    "coef": params[var],
                    "pvalue": qr_res.pvalues[var]
                })

    except Exception as e:
        print(f"\nQuantile regression q={q} failed:")
        print(e)

qr_table = pd.DataFrame(qr_results)

qr_table.to_csv(
    os.path.join(OUTPUT_DIR, "Quantile_regression_key_coefficients.csv"),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 15. ARDL-like distributed lag panel model
# ============================================================

ardl_data = data_spatial.copy()
ardl_data = ardl_data.sort_values(["region_match", "year"])

lag_vars = [
    "temp_grow_mean",
    "spi_grow_mean",
    "spei_grow_mean",
    "prec_grow_sum"
]

lag_vars = [v for v in lag_vars if v in ardl_data.columns]

for v in lag_vars:
    ardl_data[f"{v}_lag1"] = ardl_data.groupby("region_match")[v].shift(1)
    ardl_data[f"{v}_lag2"] = ardl_data.groupby("region_match")[v].shift(2)

ardl_x_cols = x_cols.copy()

for v in lag_vars:
    ardl_x_cols += [f"{v}_lag1", f"{v}_lag2"]

ardl_x_cols = [c for c in ardl_x_cols if c in ardl_data.columns]

ardl_data = ardl_data.dropna(subset=[y_col] + ardl_x_cols).copy()

ardl_panel = ardl_data.set_index(["region_match", "year"])

Y_ardl = ardl_panel[y_col]
X_ardl = ardl_panel[ardl_x_cols]
X_ardl = sm.add_constant(X_ardl)

ardl_model = PanelOLS(
    Y_ardl,
    X_ardl,
    entity_effects=True,
    time_effects=True,
    drop_absorbed=True
)

ardl_res = ardl_model.fit(cov_type="clustered", cluster_entity=True)

print("\n" + "=" * 80)
print("ARDL-LIKE DISTRIBUTED LAG PANEL MODEL")
print("=" * 80)
print(ardl_res)

with open(os.path.join(OUTPUT_DIR, "Panel_ARDL_like_results.txt"), "w", encoding="utf-8") as f:
    f.write(str(ardl_res))


# ============================================================
# 16. Final summary
# ============================================================

summary = pd.DataFrame({
    "Model": [
        "FE",
        "SAR-style",
        "SDM-style",
        "Random Forest",
        "Random Forest without lagged yield",
        "XGBoost"
    ],
    "N": [
        fe_res.nobs,
        sar_res.nobs,
        sdm_res.nobs,
        len(y_test),
        len(y_test_no_lag),
        len(y_test)
    ],
    "Main_R2": [
        fe_res.rsquared_overall,
        sar_res.rsquared_overall,
        sdm_res.rsquared_overall,
        rf_r2,
        rf_no_lag_r2,
        xgb_r2
    ]
})

summary.to_csv(
    os.path.join(OUTPUT_DIR, "all_model_summary.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 80)
print("ALL DONE")
print("=" * 80)
print("All results saved to:")
print(OUTPUT_DIR)
print("\nMain output files:")
print("  - FE_results.txt")
print("  - SAR_results.txt")
print("  - SDM_results.txt")
print("  - model_comparison_FE_SAR_SDM.csv")
print("  - RF_feature_importance.csv")
print("  - RF_feature_importance.png")
print("  - ML_model_comparison.csv")
print("  - SHAP_summary.csv")
print("  - SHAP_feature_importance.png")
print("  - DID_2010_drought_results.txt")
print("  - Quantile_regression_key_coefficients.csv")
print("  - Panel_ARDL_like_results.txt")
print("  - all_model_summary.csv")