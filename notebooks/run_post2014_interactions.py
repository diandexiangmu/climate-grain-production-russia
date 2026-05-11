# -*- coding: utf-8 -*-

import os
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

# =========================
# 1. 路径设置
# =========================

DATA_PATH = "/Users/littlestars/Desktop/grain_project/thesis_outputs/results/cleaned_panel_with_derived_variables.csv"
OUT_DIR = "/Users/littlestars/Desktop/grain_project/thesis_outputs/tables"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# 2. 读取数据
# =========================

df = pd.read_csv(DATA_PATH)

# =========================
# 3. 构造 post2014 和交互项
# =========================

df["post2014"] = (df["year"] >= 2014).astype(int)

df["post2014_x_temp"] = df["post2014"] * df["temp_grow_mean"]
df["post2014_x_spi"] = df["post2014"] * df["spi_grow_mean"]
df["post2014_x_spei"] = df["post2014"] * df["spei_grow_mean"]

# =========================
# 4. 注意：这里不要放 subsidy 变量
#    否则样本会被限制到 2016 年以后，post2014 无法识别
# =========================

y_var = "ln_yield"

x_vars = [
    "temp_grow_mean",
    "prec_grow_sum",
    "spi_grow_mean",
    "spei_grow_mean",
    "ln_area",
    "ln_yield_lag1",
    "post2014_x_temp",
    "post2014_x_spi",
    "post2014_x_spei"
]

needed_cols = ["region_std", "year", y_var] + x_vars
data = df[needed_cols].dropna().copy()

# =========================
# 5. 设置面板索引
# =========================

data = data.set_index(["region_std", "year"])

y = data[y_var]
X = data[x_vars]
X = sm.add_constant(X)

# =========================
# 6. 双向固定效应模型
# =========================

model = PanelOLS(
    y,
    X,
    entity_effects=True,
    time_effects=True,
    drop_absorbed=True,
    check_rank=False
)

res = model.fit(
    cov_type="clustered",
    cluster_entity=True
)

print(res)

# =========================
# 7. 导出结果表
# =========================

def stars(p):
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    else:
        return ""

rows = []

labels = {
    "temp_grow_mean": "Growing-season temperature",
    "prec_grow_sum": "Growing-season precipitation",
    "spi_grow_mean": "SPI",
    "spei_grow_mean": "SPEI",
    "ln_area": "$\\ln A$",
    "ln_yield_lag1": "$\\ln Y_{t-1}$",
    "post2014_x_temp": "Post-2014 $\\times$ Temperature",
    "post2014_x_spi": "Post-2014 $\\times$ SPI",
    "post2014_x_spei": "Post-2014 $\\times$ SPEI"
}

for v in x_vars:
    if v in res.params.index:
        rows.append({
            "Variable": labels.get(v, v),
            "Coefficient": res.params[v],
            "Std. error": res.std_errors[v],
            "P-value": res.pvalues[v],
            "Significance": stars(res.pvalues[v])
        })

out = pd.DataFrame(rows)

csv_path = os.path.join(OUT_DIR, "post2014_interactions_coefficients.csv")
tex_path = os.path.join(OUT_DIR, "post2014_interactions_coefficients.tex")

out.to_csv(csv_path, index=False)

with open(tex_path, "w", encoding="utf-8") as f:
    f.write("\\begin{table}[H]\n")
    f.write("\\centering\n")
    f.write("\\caption{Post-2014 climate interaction results}\n")
    f.write("\\label{tab:post2014_interactions}\n")
    f.write("\\begin{tabular}{lrrrr}\n")
    f.write("\\toprule\n")
    f.write("Variable & Coefficient & Std. error & P-value & Significance \\\\\n")
    f.write("\\midrule\n")

    for _, r in out.iterrows():
        f.write(
            f"{r['Variable']} & "
            f"{r['Coefficient']:.4f} & "
            f"{r['Std. error']:.4f} & "
            f"{r['P-value']:.4f} & "
            f"{r['Significance']} \\\\\n"
        )

    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\begin{flushleft}\n")
    f.write("\\footnotesize Notes: * $p<0.10$, ** $p<0.05$, *** $p<0.01$. Region and year fixed effects are included. Subsidy variables are excluded in this specification in order to retain both pre-2014 and post-2014 observations.\n")
    f.write("\\end{flushleft}\n")
    f.write("\\end{table}\n")

print(f"Saved: {csv_path}")
print(f"Saved: {tex_path}")
print(f"Number of observations: {int(res.nobs)}")