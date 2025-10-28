# Predicting Track Popularity with ML

**Author:** Derya Er  

---

## Project Overview

This project predicts Spotify track popularity using machine learning. Multiple regression models were tested with different preprocessing techniques, and hyperparameter tuning was applied to improve performance.

**Goal:** Predict track popularity and analyze Spotify metrics.  

**Potential Impact:**  
The model can help music platforms, artists, and record labels understand what drives track popularity, optimize playlists, marketing strategies, and recommendation systems.  

---

## Dataset

- **Source:** [Ultimate Spotify Tracks DB on Kaggle](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db)  
- **Shape:** 232,725 rows × 18 columns  
- **Preprocessing:**
  - Checked missing values and duplicates  
  - Converted duration from milliseconds to seconds  
  - Standardized column names  

---

## Exploratory Data Analysis (EDA)

- Examined **feature distributions** using histograms and boxplots  
- Checked **feature relationships** via correlation heatmaps and scatter plots  
- Verified data quality (duplicates, missing values)  

**Example: Correlation Heatmap**  
![correlation_heatmap](images/correlation_heatmap.png)

---

## Feature Engineering & Transformation

- **Encoding:** One-hot encoding for categorical features (`genre`, `key`, `mode`, `time_signature`)  
- **Scaling:** RobustScaler, StandardScaler, MinMaxScaler  
- **Train-test split:** 80/20 split  
- Saved processed datasets using `pickle`  

**Example: Feature Distribution after Scaling**  
![feature_distribution](images/feature_distribution.png)

---

## Models Tested

| Model | Notes |
|-------|-------|
| KNN | Distance-based regression |
| Linear Regression | Baseline linear model |
| Decision Tree | Single tree regression |
| Random Forest | Ensemble of decision trees |
| XGBoost | Gradient boosting |
| LightGBM | Gradient boosting |

**Metrics Used:** R², MAE, MSE, RMSE  

**Example: R² Scores Across Models**  
![r2_scores](images/r2_scores.png)

---

## Hyperparameter Tuning

**Random Forest Tuned Parameters:**  
```python
param_dist = {
    'n_estimators': [100, 200, 300, 500, 800],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}
```

## Effect of Tuning:

Random Forest (robust-scaled): R² ↑ 0.709 → 0.718, RMSE ↓ 9.81 → 9.66

XGBoost (standard-scaled): R² ↑ 0.707 → 0.715, RMSE ↓ 9.84 → 9.71

Example: RMSE Before vs After Tuning


## Feature Importance

Top features (Random Forest, tuned):

Rank	Feature	Importance
1	genre_Movie	0.1656
2	genre_Pop	0.0918
3	genre_Children's Music	0.0824
4	genre_Rap	0.0808
5	genre_Hip-Hop	0.0541
6	genre_Rock	0.0500
7	genre_Indie	0.0441
8	genre_Folk	0.0413
9	genre_R&B	0.0366
10	genre_Opera	0.0364

Top 5 features contribute ~47.5% of importance

Top 10 features contribute ~68.3% of importance

Genre is the most predictive factor for track popularity

Example: Top 10 Feature Importances


## Results Summary

| Model                 | Dataset   | R²    | MAE   | MSE       | RMSE   |
|-----------------------|----------|-------|-------|-----------|--------|
| KNN                   | Robust   | 0.656 | 8.020 | 113.722   | 10.664 |
| Linear Regression     | Robust   | 0.626 | 8.355 | 123.799   | 11.126 |
| Decision Tree         | Robust   | 0.395 | 10.321| 200.072   | 14.145 |
| Random Forest         | Robust   | 0.709 | 7.355 | 96.257    | 9.811  |
| XGBoost               | Robust   | 0.707 | 7.385 | 96.825    | 9.840  |
| LightGBM              | Robust   | 0.706 | 7.407 | 97.380    | 9.868  |
| Random Forest (tuned) | Robust   | 0.718 | 7.258 | 93.308    | 9.660  |
| XGBoost (tuned)       | Standard | 0.715 | 7.287 | 94.332    | 9.712  |

## Key Takeaways:

Tree-based models outperform simpler models.

Hyperparameter tuning improves accuracy (~1–1.3% R² gain).

Random Forest (tuned, robust-scaled) is the best-performing model.

Example: R² Heatmap Across Models & Scalers


## Future Work

Include additional features such as artist popularity, release year, or playlist data

Test advanced models like CatBoost or stacked ensembles

Deploy the model in a real-time application to predict track popularity for new songs

## Environment & Libraries

Python 3.x

pandas, numpy, scikit-learn, xgboost, lightgbm, matplotlib, seaborn
