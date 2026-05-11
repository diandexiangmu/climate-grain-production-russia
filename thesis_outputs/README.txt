Thesis output package
=====================

Output root:
/Users/littlestars/Desktop/grain_project/thesis_outputs

Folders:
- figures/: all PNG figures for the revised thesis LaTeX draft.
- tables/: descriptive statistics, model coefficients, quantile coefficients, RF importance, SHAP summary, model comparison tables.
- results/: full text outputs and intermediate CSV files.
- logs/: run configuration.

Recommended LaTeX preamble line:
\graphicspath{{thesis_outputs/figures/}{thesis_outputs/tables/}}

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