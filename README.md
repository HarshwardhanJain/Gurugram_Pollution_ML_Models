# Gurugram Pollution ML Models
Continuing IIPR Research Paper — Gurugram Air Quality Study (2020–2024)

ML analysis of air pollution in Gurugram using three datasets: CPCB ground sensors, MODIS LST satellite, and Sentinel-5P atmospheric data.

---

## Datasets

| Dataset | Source | Rows | Key Variables |
|---|---|---|---|
| CPCB Ground Sensors | Central Pollution Control Board | 7,308 | PM2.5, PM10, NO2, CO, SO2, met vars (4 sites) |
| MODIS LST | NASA MODIS Terra/Aqua | 1,827 | Land Surface Temperature (LST_C) |
| Sentinel-5P | ESA Copernicus | 1,800 | CO, NO2, O3, HCHO, AAI (aerosol index) |

After merging on date and feature engineering: **1,768 daily rows, 32 columns**.

---

## Notebooks

Run in order starting from `01`. Each notebook saves its outputs to `ML_Codes/outputs/`.

### 01 — Data Preprocessing

1. Load all three CSVs with date parsing
2. Filter CPCB to the primary site (Vikas Sadan, site 146 — most complete data across 2020–2024)
3. Compute daily averages per pollutant; forward-fill small gaps (≤3 days)
4. Merge CPCB + LST + Sentinel-5P on date using left joins
5. Drop columns that are entirely empty (CH4, SO2 from Sentinel; AT °C and PM10 from CPCB — both >99% NaN after merge)
6. Engineer features: `month`, `day_of_year`, `year`, `season` (Winter/Summer/Monsoon/Post-Monsoon), `season_enc` (integer encoding)
7. Compute lag features: `PM2.5_lag1` (previous day) and `PM2.5_lag7` (previous week)
8. Assign AQI category from PM2.5 using CPCB breakpoints: Good (0–30), Moderate (31–60), Poor (61–90), Very Poor (91–120), Severe (>120 µg/m³)
9. Save `outputs/merged_clean.csv` — 1,768 rows, 32 columns

### 02 — Exploratory Data Analysis

1. Correlation heatmap — all numeric variables
2. Correlation heatmap — CPCB pollutants only
3. Correlation heatmap — Sentinel-5P satellite variables
4. Correlation heatmap — cross-dataset (CPCB vs Sentinel)
5. Monthly PM2.5 boxplot showing seasonal variation (2020–2024)
6. AQI category distribution bar chart
7. PM2.5 time-series plot (daily, full 5-year period)
8. Monthly average PM2.5 bar chart by year
9. Scatter: Sentinel NO2 vs CPCB NO2 (satellite vs ground validation)
10. Scatter: LST vs PM2.5 (land surface temperature vs pollution)

### 03 — Regression Models (PM2.5 Prediction)

1. Define three feature sets: Linear (3 meteo vars), Ridge (6 meteo vars), Full (19 vars with satellite + lag)
2. Random 80/20 train-test split (same indices across all feature sets)
3. Median imputation + StandardScaler for each feature set independently
4. Train Linear Regression on 3-var meteo set (minimal baseline)
5. Train Ridge Regression (alpha=1.0) on 6-var meteo set (regularized baseline)
6. Train Random Forest (200 trees, max_depth=10) on full 19-var set
7. Train XGBoost (300 estimators, lr=0.05) on full 19-var set
8. Evaluate all models: MAE, RMSE, R²; save `regression_comparison.csv`
9. Plot comparison bar chart (MAE, RMSE, R² side by side)
10. Plot feature importance for Random Forest and XGBoost
11. Plot actual vs predicted scatter + time-series for best model (XGBoost)
12. Plot residual distribution and residuals vs fitted

### 04 — Classification Models (AQI Category)

1. Use same feature candidates as regression; target = AQI label (5 classes)
2. Stratified 80/20 train-test split to preserve class distribution
3. Median imputation; drop columns that are >99% NaN after imputation
4. Train Logistic Regression (baseline, max_iter=1000)
5. Train Random Forest Classifier (200 trees)
6. Train XGBoost Classifier
7. Evaluate all models: accuracy, weighted precision/recall/F1; save `classification_comparison.csv`
8. Plot confusion matrix heatmap for best model (Random Forest)
9. Plot ROC curves per AQI class (one-vs-rest)

### 05 — LSTM Time-Series Forecasting

1. Extract daily PM2.5 univariate series; resample to daily frequency; forward-fill gaps (≤7 days)
2. Normalize with MinMaxScaler (0–1 range)
3. Create sliding window sequences: 30-day lookback → predict next day
4. Time-ordered 80/20 split (1,416 train / 355 test sequences)
5. Build stacked LSTM: `LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(1)`
6. Train with Adam optimizer, EarlyStopping (patience=15), ReduceLROnPlateau
7. Inverse-transform predictions; evaluate MAE, RMSE, R²
8. Append LSTM row to `regression_comparison.csv` (same column format as notebook 03)
9. Plot training/validation loss curve
10. Plot actual vs predicted time-series and scatter on test set

### 06 — K-Means Clustering

1. Select clustering features: PM2.5, NO2, CO, WS, RH, month, season_enc (excludes NaN-heavy columns)
2. Median imputation + StandardScaler
3. Run K-Means for k=2–8; compute inertia (WCSS) and silhouette score for each k
4. Plot Elbow curve and Silhouette score to justify k=4 (matches 4 pollution seasons)
5. Fit final K-Means (k=4, n_init=20) on all 1,768 rows
6. Auto-label clusters by PM2.5 rank: Low (Monsoon), Moderate, High (Post-Monsoon), Severe (Winter)
7. Save cluster centroid table to `kmeans_centroids.csv`
8. PCA projection to 2D — scatter plot of all points coloured by cluster
9. Cluster profile heatmap (row-normalized feature centroids)
10. Stacked bar chart of cluster distribution by month (2020–2024)

---

## Results

### Regression — PM2.5 Prediction (µg/m³)

Models are trained on an 80/20 random split. Baselines use meteorological variables only; proposed models add Sentinel-5P satellite data and lag features.

| Model | Feature Set | MAE | RMSE | R² | Time (s) |
|---|---|---|---|---|---|
| Linear Regression (baseline) | Meteo (3 vars) | 43.88 | 58.71 | 0.019 | 0.0 |
| Ridge Regression (baseline) | Meteo (6 vars) | 38.65 | 53.05 | 0.199 | 0.0 |
| LSTM | PM2.5 time series | 27.91 | 38.95 | 0.402 | 45.4 |
| Random Forest | Full (19 vars) | 23.84 | 38.02 | 0.589 | 0.5 |
| **XGBoost** | **Full (19 vars)** | **23.47** | **35.88** | **0.634** | **1.0** |

XGBoost achieves the best performance (R²=0.634), a 3.3× improvement in R² over the Ridge baseline. The inclusion of Sentinel-5P satellite features and lag variables is the key driver of improvement.

---

### Classification — AQI Category (5 classes)

AQI categories derived from PM2.5 (Good / Moderate / Poor / Very Poor / Severe). Trained on an 80/20 stratified split.

| Model | Accuracy | Precision (W) | Recall (W) | F1-Score (W) |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.404 | 0.453 | 0.404 | 0.419 |
| XGBoost | 0.599 | 0.579 | 0.599 | 0.578 |
| **Random Forest** | **0.627** | **0.606** | **0.627** | **0.605** |

Random Forest achieves the best classification accuracy (62.7%), a 55% improvement over the logistic baseline. Metrics are weighted-averaged across all 5 AQI classes.

---

### Clustering — K-Means Seasonal Pollution Regimes (k=4)

K-Means applied to PM2.5, NO2, CO, wind speed, and relative humidity. Clusters map naturally to pollution seasons.

| Cluster | PM2.5 (µg/m³) | NO2 (µg/m³) | CO (mg/m³) | WS (m/s) | RH (%) |
|---|---|---|---|---|---|
| Low Pollution (Monsoon) | 48.61 | 23.09 | 1.02 | 1.46 | 66.83 |
| High Pollution (Post-Monsoon) | 70.53 | 20.88 | 1.18 | 0.69 | 68.20 |
| Moderate Pollution | 86.89 | 26.62 | 1.46 | 0.72 | 53.44 |
| Severe Pollution (Winter) | 179.99 | 41.94 | 2.45 | 0.72 | 74.80 |

Winter stagnation produces PM2.5 concentrations 3.7× higher than the monsoon cluster, driven by low wind speeds, temperature inversions, and increased CO from biomass burning.

---

## Output Files

All figures and tables are saved to `ML_Codes/outputs/`:

| File | Description |
|---|---|
| `merged_clean.csv` | Final merged dataset (1,768 rows) |
| `regression_comparison.csv` | Regression model comparison table |
| `classification_comparison.csv` | Classification model comparison table |
| `kmeans_centroids.csv` | K-Means cluster centroid values |
| `feature_importance_rf.png` | Random Forest feature importance |
| `feature_importance_xgb.png` | XGBoost feature importance |
| `xgb_actual_vs_predicted.png` | XGBoost actual vs predicted scatter |
| `lstm_forecast.png` | LSTM forecast plot |
| `kmeans_pca_scatter.png` | PCA 2D cluster visualization |
| `kmeans_cluster_heatmap.png` | Cluster profile heatmap |

---

## Dependencies

```
pip install -r ML_Codes/requirements.txt
```

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
tensorflow>=2.10.0
matplotlib>=3.6.0
seaborn>=0.12.0
openpyxl>=3.0.0
```
