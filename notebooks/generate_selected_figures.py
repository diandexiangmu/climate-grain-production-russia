# -*- coding: utf-8 -*-
"""
生成论文精选图表：10张左右
适用于：
/Users/littlestars/Desktop/grain_project

运行方式：
python /Users/littlestars/Desktop/grain_project/notebooks/figure.py
"""

import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 1. 项目根目录：这里已经按你的电脑路径改好
# =========================================================
PROJECT_ROOT = "/Users/littlestars/Desktop/grain_project"

# 输出图表文件夹
OUT_DIR = os.path.join(PROJECT_ROOT, "paper_figures_selected")
os.makedirs(OUT_DIR, exist_ok=True)


# =========================================================
# 2. 自动搜索文件函数
# =========================================================
def find_file(filename, root=PROJECT_ROOT):
    """
    在 root 文件夹下递归搜索 filename。
    找到后返回完整路径；找不到就报错。
    """
    for dirpath, dirnames, filenames in os.walk(root):
        if filename in filenames:
            return os.path.join(dirpath, filename)

    raise FileNotFoundError(
        f"\n没有找到文件：{filename}\n"
        f"请确认它是否在这个项目文件夹下：{root}\n"
        f"如果文件名不同，请在代码里修改对应文件名。"
    )


# =========================================================
# 3. 自动定位所需数据文件
# =========================================================
panel_path = find_file("panel_with_spatial_lags.csv")
quantile_path = find_file("Quantile_regression_key_coefficients.csv")
rf_path = find_file("RF_feature_importance.csv")
rf_no_lag_path = find_file("RF_no_lag_feature_importance.csv")

print("找到文件：")
print("panel:", panel_path)
print("quantile:", quantile_path)
print("rf:", rf_path)
print("rf_no_lag:", rf_no_lag_path)


# =========================================================
# 4. 读取数据
# =========================================================
panel = pd.read_csv(panel_path)
quant = pd.read_csv(quantile_path)
rf = pd.read_csv(rf_path)
rf_no_lag = pd.read_csv(rf_no_lag_path)


# =========================================================
# 5. 基础绘图函数
# =========================================================
def save_fig(filename):
    path = os.path.join(OUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("已保存：", path)


def readable_feature_name(name):
    mapping = {
        "num__ln_yield_lag1": r"$\ln Y_{t-1}$",
        "num__temp_grow_mean": "Growing-season temperature",
        "num__prec_grow_sum": "Growing-season precipitation",
        "num__spi_grow_mean": "SPI",
        "num__spei_grow_mean": "SPEI",
        "num__ln_subsidy_main": r"$\ln$ Subsidy",
        "num__ln_subsidy_per_area": r"$\ln$ Subsidy / Area",
        "num__temp_grow_sq": r"Temperature$^2$",
        "num__spi_sq": r"SPI$^2$",
        "num__spei_sq": r"SPEI$^2$",
        "num__temp_x_subsidy": "Temperature × Subsidy",
        "num__spi_x_subsidy": "SPI × Subsidy",
        "num__spei_x_subsidy": "SPEI × Subsidy",
        "cat__agro_zone_black_soil": "Agro zone: Black soil",
        "cat__agro_zone_volga_dry": "Agro zone: Volga dry",
        "cat__agro_zone_risky_farming": "Agro zone: Risky farming",
        "cat__agro_zone_other": "Agro zone: Other",
    }
    return mapping.get(name, name)


def yearly_line(df, col, ylabel, title, filename):
    if col not in df.columns:
        print(f"跳过 {filename}：数据中没有列 {col}")
        return

    g = df.groupby("year")[col].mean().reset_index()

    plt.figure(figsize=(8, 5))
    plt.plot(g["year"], g[col], marker="o")
    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.title(title)
    save_fig(filename)


def scatter_quad(df, xcol, ycol, xlabel, ylabel, title, filename):
    if xcol not in df.columns or ycol not in df.columns:
        print(f"跳过 {filename}：缺少列 {xcol} 或 {ycol}")
        return

    d = df[[xcol, ycol]].dropna().copy()
    x = d[xcol].values
    y = d[ycol].values

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, alpha=0.45, s=18)

    if len(d) >= 5:
        coef = np.polyfit(x, y, 2)
        xs = np.linspace(x.min(), x.max(), 200)
        ys = coef[0] * xs**2 + coef[1] * xs + coef[2]
        plt.plot(xs, ys, linewidth=2)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    save_fig(filename)


def quantile_plot(qdf, variable, title, filename):
    required = {"quantile", "variable", "coef"}
    if not required.issubset(qdf.columns):
        print(f"跳过 {filename}：分位数结果文件缺少必要列")
        return

    d = qdf[qdf["variable"] == variable].copy().sort_values("quantile")

    if d.empty:
        print(f"跳过 {filename}：没有变量 {variable}")
        return

    plt.figure(figsize=(7, 5))
    plt.plot(d["quantile"], d["coef"], marker="o")
    plt.axhline(0, linewidth=1)
    plt.xlabel("Quantile")
    plt.ylabel("Coefficient")
    plt.title(title)
    save_fig(filename)


# =========================================================
# 6. 描述性统计表
# =========================================================
desc_map = {
    "ln_yield": r"$\ln Y$",
    "ln_yield_lag1": r"$\ln Y_{t-1}$",
    "temp_grow_mean": "Temp",
    "prec_grow_sum": "Prec",
    "spi_grow_mean": "SPI",
    "spei_grow_mean": "SPEI",
    "ln_subsidy_main": r"$\ln Sub$",
    "ln_subsidy_per_area": r"$\ln Sub/A$",
}

desc_vars = [c for c in desc_map.keys() if c in panel.columns]

if desc_vars:
    desc = panel[desc_vars].describe().T

    desc_table = pd.DataFrame({
        "变量": [desc_map[c] for c in desc.index],
        "观测数": desc["count"].astype(int),
        "均值": desc["mean"].round(3),
        "标准差": desc["std"].round(3),
        "最小值": desc["min"].round(3),
        "最大值": desc["max"].round(3),
    })

    desc_csv_path = os.path.join(OUT_DIR, "descriptive_statistics.csv")
    desc_tex_path = os.path.join(OUT_DIR, "descriptive_statistics.tex")

    desc_table.to_csv(desc_csv_path, index=False, encoding="utf-8-sig")

    latex_table = desc_table.to_latex(
        index=False,
        escape=False,
        column_format="lccccc",
        caption="主要变量的描述性统计",
        label="tab:desc_stats"
    )

    with open(desc_tex_path, "w", encoding="utf-8") as f:
        f.write(latex_table)

    print("已保存描述性统计：", desc_csv_path)
    print("已保存 LaTeX 表格：", desc_tex_path)
else:
    print("没有找到可用于描述性统计的变量。")


# =========================================================
# 7. 生成精选 10 张图
# =========================================================

# 1-3 年度趋势图
yearly_line(
    panel,
    "ln_yield",
    r"Mean $\ln Y$",
    "Yearly Mean Log Grain Yield",
    "01_yearly_mean_ln_yield.png"
)

yearly_line(
    panel,
    "temp_grow_mean",
    "Temperature",
    "Yearly Mean Growing-season Temperature",
    "02_yearly_mean_temp.png"
)

yearly_line(
    panel,
    "prec_grow_sum",
    "Precipitation",
    "Yearly Mean Growing-season Precipitation",
    "03_yearly_mean_prec.png"
)


# 4-6 非线性关系图
scatter_quad(
    panel,
    "temp_grow_mean",
    "ln_yield",
    "Growing-season temperature",
    r"$\ln Y$",
    "Grain Yield vs Temperature",
    "04_scatter_temp_yield_quadratic.png"
)

scatter_quad(
    panel,
    "spi_grow_mean",
    "ln_yield",
    "SPI",
    r"$\ln Y$",
    "Grain Yield vs SPI",
    "05_scatter_spi_yield_quadratic.png"
)

scatter_quad(
    panel,
    "spei_grow_mean",
    "ln_yield",
    "SPEI",
    r"$\ln Y$",
    "Grain Yield vs SPEI",
    "06_scatter_spei_yield_quadratic.png"
)


# 7-8 分位数回归异质性
quantile_plot(
    quant,
    "temp_grow_mean",
    "Quantile Regression Coefficients: Temperature",
    "07_quantile_coeff_temp.png"
)

quantile_plot(
    quant,
    "spi_grow_mean",
    "Quantile Regression Coefficients: SPI",
    "08_quantile_coeff_spi.png"
)


# 9 随机森林：包含滞后产量
if {"feature", "importance"}.issubset(rf.columns):
    rf_use = rf.copy()
    rf_use["feature_readable"] = rf_use["feature"].map(readable_feature_name)
    rf_use = rf_use.sort_values("importance", ascending=True)

    plt.figure(figsize=(8, 6))
    plt.barh(rf_use["feature_readable"], rf_use["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance (with lagged yield)")
    save_fig("09_rf_feature_importance_with_lag.png")
else:
    print("跳过随机森林含滞后图：RF_feature_importance.csv 缺少 feature 或 importance 列")


# 10 随机森林：不包含滞后产量
if {"feature", "importance"}.issubset(rf_no_lag.columns):
    rf_nl = rf_no_lag.copy()
    rf_nl["feature_readable"] = rf_nl["feature"].map(readable_feature_name)
    rf_nl = rf_nl.sort_values("importance", ascending=True)

    plt.figure(figsize=(8, 6))
    plt.barh(rf_nl["feature_readable"], rf_nl["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance (without lagged yield)")
    save_fig("10_rf_feature_importance_without_lag.png")
else:
    print("跳过随机森林无滞后图：RF_no_lag_feature_importance.csv 缺少 feature 或 importance 列")


# =========================================================
# 8. 生成说明文件
# =========================================================
readme_path = os.path.join(OUT_DIR, "README.txt")

with open(readme_path, "w", encoding="utf-8") as f:
    f.write(
        "精选图表说明（10张）\n"
        "=================\n\n"
        "01_yearly_mean_ln_yield.png\n"
        "  用途：展示样本期内俄罗斯地区平均粮食产量对数的年度变化趋势。\n\n"
        "02_yearly_mean_temp.png\n"
        "  用途：展示生长季平均气温的年度变化趋势。\n\n"
        "03_yearly_mean_prec.png\n"
        "  用途：展示生长季累计降水的年度变化趋势。\n\n"
        "04_scatter_temp_yield_quadratic.png\n"
        "  用途：支撑温度与产量之间可能存在非线性关系。\n\n"
        "05_scatter_spi_yield_quadratic.png\n"
        "  用途：支撑 SPI 与产量之间的非线性经验关系。\n\n"
        "06_scatter_spei_yield_quadratic.png\n"
        "  用途：支撑 SPEI 与产量之间的非线性经验关系。\n\n"
        "07_quantile_coeff_temp.png\n"
        "  用途：展示温度系数在不同分位点上的异质性。\n\n"
        "08_quantile_coeff_spi.png\n"
        "  用途：展示 SPI 系数在不同分位点上的异质性。\n\n"
        "09_rf_feature_importance_with_lag.png\n"
        "  用途：展示包含滞后产量时的变量重要性排序。\n\n"
        "10_rf_feature_importance_without_lag.png\n"
        "  用途：展示剔除滞后产量后的变量重要性排序。\n"
    )

print("已保存说明文件：", readme_path)


# =========================================================
# 9. 打包为 zip
# =========================================================
zip_path = os.path.join(PROJECT_ROOT, "paper_figures_selected.zip")

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for file in os.listdir(OUT_DIR):
        full_path = os.path.join(OUT_DIR, file)
        zf.write(full_path, arcname=file)

print("\n全部完成。")
print("图表文件夹：", OUT_DIR)
print("压缩包：", zip_path)