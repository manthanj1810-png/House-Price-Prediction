# House Price Prediction Project Report

## Introduction

House price prediction is a regression problem where the goal is to estimate the selling price of a house using its characteristics. Accurate prediction can help buyers, sellers, and real-estate businesses make informed decisions.

The objective of this project was to develop and evaluate multiple machine learning models for predicting house prices.

---

## Dataset Description

The project uses the House Prices dataset from Kaggle.

Dataset Characteristics:

* Number of observations: 1460
* Number of features: 81
* Target variable: SalePrice

The dataset contains information regarding:

* House quality
* Living area
* Garage capacity
* Basement size
* Construction year
* Bathrooms
* Lot characteristics

---

## Exploratory Data Analysis

EDA was performed to understand the dataset before model training.

### Steps Performed

1. Dataset inspection
2. Missing value analysis
3. Data type analysis
4. Target variable analysis
5. Correlation analysis

### Findings

Several columns contained significant missing values.

Strong correlations with SalePrice were observed for:

* OverallQual
* GrLivArea
* GarageCars
* GarageArea
* TotalBsmtSF
* FullBath
* YearBuilt

These features were selected for initial model development.

---

## Data Preparation

Selected Features:

* OverallQual
* GrLivArea
* GarageCars
* GarageArea
* TotalBsmtSF
* FullBath
* YearBuilt

Missing values in selected features:

* None

Dataset Split:

* Training Set: 80%
* Testing Set: 20%

Random State:

* 42

---

## Model Implementation

### Linear Regression

A baseline regression model was developed using selected features.

Results:

* MAE: 25121.62
* RMSE: 39652.79
* R²: 0.7950

The model explained approximately 79.5% of the variance in house prices.

---

### Ridge Regression

Ridge Regression was implemented to reduce overfitting through L2 regularization.

Results:

* MAE: 25115.80
* RMSE: 39651.33
* R²: 0.7950

Performance was nearly identical to Linear Regression.

---

### Lasso Regression

Lasso Regression was implemented using L1 regularization.

Results:

* MAE: 25115.80
* RMSE: 39651.33
* R²: 0.7950

No significant feature elimination occurred due to the small feature set and strong predictors.

---

### Random Forest Regression

Random Forest Regression was implemented to capture nonlinear relationships between features and house prices.

Results:

* MAE: 19534.81
* RMSE: 30077.85
* R²: 0.8821

This model achieved the best performance.

---

## Feature Importance Analysis

Random Forest feature importance:

* OverallQual: 57.92%
* GrLivArea: 18.81%
* TotalBsmtSF: 10.15%
* YearBuilt: 5.25%
* GarageArea: 4.37%
* GarageCars: 2.20%
* FullBath: 1.31%

The analysis indicates that overall house quality is the most influential factor in determining house prices.

---

## Conclusion

Four machine learning models were evaluated for house price prediction.

Among all models, Random Forest Regression achieved the highest predictive performance with an R² score of 0.8821.

The project demonstrates the complete machine learning workflow including:

* Data exploration
* Feature selection
* Model training
* Model evaluation
* Feature importance analysis
* Model persistence
* Prediction generation

