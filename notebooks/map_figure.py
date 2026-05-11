# -*- coding: utf-8 -*-
"""
绘制俄罗斯轮廓图，适配 panel 中俄文地区名的情况。

输出：
figures/11_russia_outline_sample_points.png
figures/12_russia_choropleth_mean_ln_yield.png
figures/region_map_match_check.csv
"""

from pathlib import Path
import difflib
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


# =========================================================
# 1. 自动确定项目根目录
# =========================================================
SCRIPT_PATH = Path(__file__).resolve()

if SCRIPT_PATH.parent.name == "notebooks":
    BASE_DIR = SCRIPT_PATH.parent.parent
else:
    BASE_DIR = SCRIPT_PATH.parent

FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("项目根目录：", BASE_DIR)
print("图片输出目录：", FIG_DIR)


# =========================================================
# 2. 基础函数
# =========================================================
def read_csv_flexible(path):
    encodings = ["utf-8", "utf-8-sig", "gbk", "cp1251", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


def cyrillic_to_latin(text):
    """
    简单俄文转写，用于地区名匹配。
    不是语言学精确转写，但足够用于匹配 GADM/Natural Earth 地区名。
    """
    if pd.isna(text):
        return ""

    text = str(text)

    table = {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d",
        "е": "e", "ё": "e", "ж": "zh", "з": "z", "и": "i",
        "й": "y", "к": "k", "л": "l", "м": "m", "н": "n",
        "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
        "у": "u", "ф": "f", "х": "kh", "ц": "ts", "ч": "ch",
        "ш": "sh", "щ": "shch", "ъ": "", "ы": "y", "ь": "",
        "э": "e", "ю": "yu", "я": "ya",
        "А": "a", "Б": "b", "В": "v", "Г": "g", "Д": "d",
        "Е": "e", "Ё": "e", "Ж": "zh", "З": "z", "И": "i",
        "Й": "y", "К": "k", "Л": "l", "М": "m", "Н": "n",
        "О": "o", "П": "p", "Р": "r", "С": "s", "Т": "t",
        "У": "u", "Ф": "f", "Х": "kh", "Ц": "ts", "Ч": "ch",
        "Ш": "sh", "Щ": "shch", "Ъ": "", "Ы": "y", "Ь": "",
        "Э": "e", "Ю": "yu", "Я": "ya",
    }

    return "".join(table.get(ch, ch) for ch in text)


def clean_region_name(x):
    """
    地区名标准化：兼容俄文、英文和拉丁转写。
    """
    if pd.isna(x):
        return ""

    s = str(x).strip()

    # 先把俄文转成拉丁
    s = cyrillic_to_latin(s)

    s = s.lower().strip()

    replacements = {
        "’": "",
        "'": "",
        "`": "",
        "´": "",
        "-": " ",
        "_": " ",
        ".": "",
        ",": "",
        "  ": " ",
    }

    for a, b in replacements.items():
        s = s.replace(a, b)

    # 英文行政后缀
    english_suffixes = [
        " oblast",
        " region",
        " republic",
        " krai",
        " autonomous okrug",
        " autonomous oblast",
        " federal city",
        " city",
        " resp",
    ]

    # 俄文行政后缀转写后的形式
    russian_suffixes = [
        " oblast",
        " oblasty",
        " oblast'",
        " respublika",
        " respublik",
        " kray",
        " krai",
        " avtonomnyy okrug",
        " avtonomnyi okrug",
        " avtonomnaya oblast",
        " avtonomnoy oblasti",
        " gorod",
    ]

    for suf in english_suffixes + russian_suffixes:
        s = s.replace(suf, "")

    # 常见转写差异手动修正
    manual = {
        "moskva": "moscow",
        "moskovskaya": "moscow",
        "moskovskaya oblast": "moscow",
        "sankt peterburg": "saint petersburg",
        "st petersburg": "saint petersburg",
        "nizhegorodskaya": "nizhny novgorod",
        "nizhnij novgorod": "nizhny novgorod",
        "nizhniy novgorod": "nizhny novgorod",
        "orlovskaya": "oryol",
        "orel": "oryol",
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
        "magadanskaya": "magadan",
        "murmanskaya": "murmansk",
        "sakhalinskaya": "sakhalin",
        "khabarovskiy": "khabarovsk",
        "krasnodarskiy": "krasnodar",
        "krasnoyarskiy": "krasnoyarsk",
        "permskiy": "perm",
        "primorskiy": "primorye",
        "stavropolskiy": "stavropol",
        "zabaykalskiy": "zabaykalsky",
        "altayskiy": "altai",
        "bashkortostan": "bashkortostan",
        "tatarstan": "tatarstan",
        "dagestan": "dagestan",
        "chechenskaya": "chechnya",
        "mordoviya": "mordovia",
        "udmurtskaya": "udmurt",
        "chuvashskaya": "chuvash",
        "mariy el": "mari el",
        "saha yakutiya": "sakha",
        "yakutiya": "sakha",
        "severnaya osetiya alaniya": "north ossetia",
        "kabardino balkarskaya": "kabardino balkar",
        "karachaevo cherkesskaya": "karachay cherkess",
    }

    s = " ".join(s.split())

    if s in manual:
        s = manual[s]

    return s


def find_col(df, candidates):
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


def fuzzy_match_one(name, candidates, cutoff=0.38):
    if not name:
        return None, 0.0

    result = difflib.get_close_matches(name, candidates, n=1, cutoff=cutoff)

    if not result:
        return None, 0.0

    best = result[0]
    score = difflib.SequenceMatcher(None, name, best).ratio()
    return best, score


# =========================================================
# 3. 自动寻找 panel CSV
# =========================================================
region_candidates = [
    "region_match",
    "matched_region",
    "region",
    "region_name",
    "region_en",
    "name",
]

value_candidates = [
    "ln_yield",
    "log_yield",
    "yield",
    "grain_yield",
    "grain_production",
    "production",
    "temp_grow_mean",
]

csv_files = [
    p for p in BASE_DIR.rglob("*.csv")
    if ".ipynb_checkpoints" not in str(p)
    and "match_check" not in p.name
    and "region_coord_match" not in p.name
    and "region_map_match" not in p.name
]

if not csv_files:
    raise FileNotFoundError("项目中没有找到任何 CSV 文件。")

best_panel = None
best_score = -1
best_info = None

for p in csv_files:
    try:
        df0 = read_csv_flexible(p)
    except Exception:
        continue

    region_col = find_col(df0, region_candidates)
    value_col = find_col(df0, value_candidates)

    score = 0
    if region_col is not None:
        score += 5
    if value_col is not None:
        score += 5
    if "ln_yield" in df0.columns:
        score += 20
    if "year" in df0.columns:
        score += 5
    if "spi" in " ".join(map(str, df0.columns)).lower():
        score += 5
    if "subsidy" in " ".join(map(str, df0.columns)).lower():
        score += 5
    if len(df0) < 50:
        score -= 10

    # 优先 final 文件夹
    if "final" in str(p).lower():
        score += 5

    if score > best_score:
        best_score = score
        best_panel = p
        best_info = (df0, region_col, value_col)

panel, panel_region_col, value_col = best_info

if panel_region_col is None or value_col is None:
    raise FileNotFoundError("没有找到合适的 panel 数据文件。")

print("\n自动选中的 panel 文件：")
print(best_panel)
print("panel_region_col =", panel_region_col)
print("value_col        =", value_col)
print("panel 行数       =", len(panel))


# =========================================================
# 4. 自动寻找 shapefile
# =========================================================
shp_files = [
    p for p in BASE_DIR.rglob("*.shp")
    if ".ipynb_checkpoints" not in str(p)
]

if not shp_files:
    raise FileNotFoundError("没有找到 shapefile。请检查 large_files 文件夹。")

# 先读取所有候选 shapefile
candidate_maps = []

for shp in shp_files:
    try:
        gdf0 = gpd.read_file(shp)
    except Exception:
        continue

    if gdf0.empty:
        continue

    shp_text = str(shp).lower()

    # 如果是全球省级边界，筛选俄罗斯
    gdf_candidate = gdf0.copy()

    for c in ["admin", "ADMIN", "geonunit", "GEONUNIT", "sov_a3", "SOV_A3", "iso_a2", "ISO_A2"]:
        if c in gdf_candidate.columns:
            if c.lower() == "sov_a3":
                sub = gdf_candidate[gdf_candidate[c].astype(str).str.contains("RUS", case=False, na=False)]
            elif c.lower() == "iso_a2":
                sub = gdf_candidate[gdf_candidate[c].astype(str).str.contains("RU", case=False, na=False)]
            else:
                sub = gdf_candidate[gdf_candidate[c].astype(str).str.contains("Russia", case=False, na=False)]
            if not sub.empty:
                gdf_candidate = sub
                break

    base_score = 0
    if "gadm" in shp_text:
        base_score += 20
    if "rus" in shp_text:
        base_score += 20
    if len(gdf_candidate) >= 40:
        base_score += 10

    candidate_maps.append((shp, gdf_candidate, base_score))

if not candidate_maps:
    raise FileNotFoundError("没有可用的俄罗斯行政区边界文件。")


# =========================================================
# 5. 自动选择最能匹配 panel 的地图名称列
# =========================================================
panel = panel.copy()
panel[panel_region_col] = panel[panel_region_col].astype(str).str.strip()
panel["_region_clean"] = panel[panel_region_col].apply(clean_region_name)
panel[value_col] = pd.to_numeric(panel[value_col], errors="coerce")

panel_regions = sorted(panel["_region_clean"].dropna().unique())

best_shp = None
best_gdf = None
best_name_col = None
best_match_count = -1
best_total_score = -1

for shp, gdf_candidate, base_score in candidate_maps:
    gdf_candidate = gdf_candidate.to_crs("EPSG:4326").copy()

    string_cols = []
    for c in gdf_candidate.columns:
        if c == "geometry":
            continue
        if gdf_candidate[c].dtype == object:
            string_cols.append(c)

    for name_col in string_cols:
        temp = gdf_candidate[[name_col, "geometry"]].copy()
        temp[name_col] = temp[name_col].astype(str).str.strip()
        temp["_region_clean"] = temp[name_col].apply(clean_region_name)

        map_regions = sorted(temp["_region_clean"].dropna().unique())

        exact_count = sum(1 for r in panel_regions if r in map_regions)

        fuzzy_count = 0
        for r in panel_regions:
            if r in map_regions:
                continue
            m, score = fuzzy_match_one(r, map_regions, cutoff=0.38)
            if m is not None:
                fuzzy_count += 1

        total_match = exact_count + fuzzy_count
        total_score = base_score + total_match

        if total_score > best_total_score:
            best_total_score = total_score
            best_match_count = total_match
            best_shp = shp
            best_gdf = gdf_candidate
            best_name_col = name_col

if best_gdf is None or best_name_col is None:
    raise ValueError("无法从 shapefile 中选择可匹配的地区名称列。")

russia_regions = best_gdf.to_crs("EPSG:4326").copy()
russia_regions[best_name_col] = russia_regions[best_name_col].astype(str).str.strip()
russia_regions["_region_clean"] = russia_regions[best_name_col].apply(clean_region_name)

print("\n自动选中的边界文件：")
print(best_shp)
print("地图地区名称列 =", best_name_col)
print("地图区域数量   =", len(russia_regions))
print("初步可匹配地区数 =", best_match_count)


# =========================================================
# 6. panel 和地图地区匹配
# =========================================================
map_regions = sorted(russia_regions["_region_clean"].dropna().unique())

map_lookup = (
    russia_regions[[best_name_col, "_region_clean"]]
    .drop_duplicates("_region_clean")
    .set_index("_region_clean")
)

match_rows = []

for r in panel_regions:
    if r in map_lookup.index:
        matched_clean = r
        score = 1.0
        method = "exact"
    else:
        matched_clean, score = fuzzy_match_one(r, map_regions, cutoff=0.38)
        method = "fuzzy" if matched_clean is not None else "unmatched"

    panel_original = panel.loc[panel["_region_clean"] == r, panel_region_col].iloc[0]

    if matched_clean is not None:
        map_original = map_lookup.loc[matched_clean, best_name_col]
    else:
        map_original = None

    match_rows.append(
        {
            "panel_region_original": panel_original,
            "panel_region_clean": r,
            "map_region_original": map_original,
            "map_region_clean": matched_clean,
            "match_method": method,
            "match_score": round(score, 3),
        }
    )

match_df = pd.DataFrame(match_rows)

match_path = FIG_DIR / "region_map_match_check.csv"
match_df.to_csv(match_path, index=False, encoding="utf-8-sig")

print("\n地区匹配检查表：")
print(match_path)
print("\n匹配情况：")
print(match_df["match_method"].value_counts(dropna=False))

unmatched = match_df[match_df["match_method"] == "unmatched"]
if not unmatched.empty:
    print("\n未匹配地区：")
    print(unmatched[["panel_region_original", "panel_region_clean"]].to_string(index=False))


# =========================================================
# 7. 合并 panel 平均值到地图
# =========================================================
panel_mean = (
    panel.dropna(subset=[value_col])
    .groupby("_region_clean", as_index=False)[value_col]
    .mean()
)

match_use = match_df.dropna(subset=["map_region_clean"])[
    ["panel_region_clean", "map_region_clean"]
].copy()

panel_mean = panel_mean.merge(
    match_use,
    left_on="_region_clean",
    right_on="panel_region_clean",
    how="left",
)

panel_mean = panel_mean.dropna(subset=["map_region_clean"])

map_data = russia_regions.merge(
    panel_mean[["map_region_clean", value_col]],
    left_on="_region_clean",
    right_on="map_region_clean",
    how="left",
)

matched_map_data = map_data.dropna(subset=[value_col]).copy()

if matched_map_data.empty:
    raise ValueError(
        "地图和 panel 仍然没有任何成功匹配。请打开 figures/region_map_match_check.csv 查看。"
    )

print("\n成功匹配到地图的地区数：", len(matched_map_data))


# =========================================================
# 8. 计算质心
# =========================================================
centroids_projected = matched_map_data.to_crs("EPSG:3857").geometry.centroid
centroids_wgs84 = gpd.GeoSeries(centroids_projected, crs="EPSG:3857").to_crs("EPSG:4326")

points_gdf = gpd.GeoDataFrame(
    matched_map_data[[best_name_col, value_col]].copy(),
    geometry=centroids_wgs84,
    crs="EPSG:4326",
)


# =========================================================
# =========================================================
# 9. 图 11：俄罗斯轮廓 + 样本地区点位
# =========================================================
fig, ax = plt.subplots(figsize=(11.5, 5.6))

russia_regions.boundary.plot(ax=ax, linewidth=0.38, color="black")

points_gdf.plot(
    ax=ax,
    markersize=14,
    alpha=0.85,
    edgecolor="black",
    linewidth=0.20,
)

minx, miny, maxx, maxy = russia_regions.total_bounds
ax.set_xlim(minx - 2, maxx + 2)
ax.set_ylim(miny - 1, maxy + 1)

ax.set_title("Sample Regions within Russia", fontsize=14)
ax.set_xlabel("Longitude", fontsize=11)
ax.set_ylabel("Latitude", fontsize=11)
ax.tick_params(axis="both", labelsize=9)
ax.grid(True, linestyle="--", alpha=0.25)

out1 = FIG_DIR / "11_russia_outline_sample_points.png"
plt.tight_layout()
plt.savefig(out1, dpi=300, bbox_inches="tight")
plt.close()

print("\n已保存：")
print(out1)

# =========================================================
# =========================================================
# 10. 图 12：俄罗斯轮廓 + 地区平均值分布图
# =========================================================
fig, ax = plt.subplots(figsize=(11.5, 5.6))

map_data.plot(
    ax=ax,
    column=value_col,
    legend=True,
    cmap="viridis",
    missing_kwds={
        "color": "lightgrey",
        "edgecolor": "white",
        "hatch": "///",
        "label": "No data",
    },
    edgecolor="black",
    linewidth=0.30,
    legend_kwds={
        "shrink": 0.65,
        "label": f"Mean {value_col}",
    },
)

ax.set_title(f"Regional Mean {value_col} within Russia", fontsize=14)
ax.set_xlabel("Longitude", fontsize=11)
ax.set_ylabel("Latitude", fontsize=11)
ax.tick_params(axis="both", labelsize=9)
ax.grid(True, linestyle="--", alpha=0.25)

minx, miny, maxx, maxy = russia_regions.total_bounds
ax.set_xlim(minx - 2, maxx + 2)
ax.set_ylim(miny - 1, maxy + 1)

out2 = FIG_DIR / "12_russia_choropleth_mean_ln_yield.png"
plt.tight_layout()
plt.savefig(out2, dpi=300, bbox_inches="tight")
plt.close()

print("\n已保存：")
print(out2)