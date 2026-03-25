# Climate Impact on Grain Production in Russia

## Overview

This project investigates the impact of climate factors (temperature and precipitation) on grain production across Russian regions.

The analysis combines panel data econometrics and spatial econometric methods to capture both temporal dynamics and spatial dependence in agricultural production.

---

## Data Sources

- **Rosstat (Russian Federal State Statistics Service)**  
  Grain production data by region and year

- **ERA5 Climate Data**  
  Temperature and precipitation

- **OpenStreetMap (Nominatim)**  
  Regional coordinates (administrative centers)

---

## Data Processing

- Raw Excel tables from Rosstat are cleaned and merged into a panel dataset
- Climate variables are matched using geographic coordinates
- Final dataset includes:
  - region
  - year
  - production
  - temperature
  - precipitation
  - latitude & longitude

---

## Methodology

The analysis is conducted in several steps:

### 1. Descriptive Analysis
- Time trends of production, temperature, and precipitation
- Scatter plots to explore relationships

### 2. Panel Data Models
- Fixed effects model (two-way)
- Nonlinear specification (quadratic terms)

### 3. Spatial Analysis
- Moran’s I test for spatial autocorrelation
- Spatial Lag Model (SAR)
- Cross-sectional OLS for comparison

---

## Key Findings

- Grain production shows a strong upward trend over time
- Temperature has a **nonlinear (U-shaped)** effect
- Precipitation exhibits an **inverted-U relationship**
- Significant **spatial dependence** exists across regions
- Spatial lag coefficient (~0.45) indicates strong spillover effects

---

## Project Structure
grain_project/
├── data/
│ ├── raw/ # original data 
│ └── processed/ # cleaned datasets
├── notebooks/
│ ├── main.ipynb # main reproducible analysis
│ └── map.ipynb
  └── analysis.ipynb
├── figures/ # generated plots
├── results/ # regression outputs
├── README.md
└── requirements.txt

---

## Reproducibility

To reproduce the results:

1. Install dependencies:
pip install -r requirements.txt

2. Run the notebook:
3.All figures and results will be generated automatically.

---

## Notes

- Large raw files (.nc, .xls) are not included in the repository
- The project focuses on reproducibility from processed datasets
- Further extensions may include full pipeline reconstruction from raw climate data

---

## Author

Ji Yiman  
Graduate Student in Mathematics & AI  
Saint Petersburg State University

