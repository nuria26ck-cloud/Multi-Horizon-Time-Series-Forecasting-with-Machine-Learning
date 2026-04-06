# Multi-Horizon-Time-Series-Forecasting-with-Machine-Learning
Predicting anonymized numeric target values across multiple time-series entities and forecast horizons using supervised machine learning, with a focus on model comparison, volatility analysis, and horizon-specific performance.

## Overview
This project predicts an anonymized continuous target variable (`y_target`) across multiple time-series entities and forecast horizons (1, 3, 10, and 25 steps ahead) using supervised machine learning.

The dataset resembles financial return data: highly noisy, centered near zero, and difficult to predict. The primary goal is to evaluate model performance across horizons and understand how data characteristics such as volatility and weak signal impact forecasting.

---

## Problem Statement
Given a large-scale, multi-entity time-series dataset:

- Predict future values of `y_target` using only past information  
- Compare model performance across multiple forecast horizons  
- Evaluate whether complex models outperform simpler baselines in a low-signal environment  

---

## Dataset

- **Training rows:** 5,337,414  
- **Test rows:** 1,447,107  
- **Features:** 86 numerical + categorical variables (`code`, `sub_code`, `sub_category`, `horizon`)  
- **Structure:** Multi-entity, time-indexed (`ts_index`)  

### Key Characteristics:
- Target distribution is **heavily concentrated near zero**
- Contains **extreme outliers (heavy tails)**
- Behaves similarly to **financial returns**
- Weak predictive signal → inherently difficult modeling task  

**Dataset not included due to size**  
Source: Hedge find -- Time-series forcasting: https://www.kaggle.com/competitions/ts-forecasting 


---

## Exploratory Data Analysis

### Target Distribution
- Sharp peak around zero (most values small)  
- Heavy tails with extreme positive and negative values  
- Boxplot confirms presence of significant outliers  

### Dataset Structure
- Forecast horizons (1, 3, 10, 25) are **relatively balanced**
- Slight decrease in observations at longer horizons  
- Time-series index shows variation in row counts, including periodic dips  

### Temporal Behavior
- Mean of `y_target` fluctuates around zero with occasional spikes  
- Standard deviation exhibits **volatility clustering**  
- Indicates periods of instability and changing variance over time  

---

## Methodology

### 1. Data Preprocessing
- Removed features with high missingness or near-zero variance  
- Applied memory optimization (downcasting)  
- Handled categorical variables via encoding where required  
- Standardized numerical features  

---

### 2. Validation Strategy
- **Time-based split**:
  - Train on earlier `ts_index`
  - Test on later `ts_index`
- Prevents data leakage  
- Reflects real-world forecasting conditions  

---

### 3. Models Evaluated

#### Linear Models (Baselines)
- Linear Regression  
- Ridge Regression  

#### Tree-Based Models
- LightGBM (efficient, scalable)  
- CatBoost (native categorical handling)  
- XGBoost (required manual encoding)  

---

### 4. Modeling Approach
- **Global model** trained across all horizons  
- **Per-horizon models** (separate LightGBM models for each horizon)

---

## Results

### Key Findings:

- Linear Regression, Ridge, LightGBM, and CatBoost all performed **similarly (errors near zero)**  
- Indicates **limited predictive signal in the dataset**  
- XGBoost performed **significantly worse**, likely due to ineffective categorical encoding  
- Model performance **decreases as forecast horizon increases**  

### Best Performing Strategy:
- Training **separate LightGBM models for each forecast horizon**
- Better captures horizon-specific patterns  

---


## Key Insights

- The target behaves like financial returns → **high noise, low signal**  
- Increasing model complexity does **not significantly improve performance**  
- Volatility varies over time → impacts predictability  
- Horizon-specific modeling is more effective than a single global model  

---

## Limitations

- Target variable is highly noisy with weak signal  
- Anonymized features limit feature engineering and interpretability  
- Dataset size (>5M rows) required memory optimization  
- XGBoost performance constrained by categorical encoding  
- Computational limitations:
  - Google Colab lacked sufficient RAM  
  - Later stages completed using Kaggle notebooks
 
## Development Environment

- Initial experimentation: Google Colab  
- Final implementation: Kaggle Notebooks (due to memory constraints)  

The notebooks folder in this repository reflects both the initial code done in Google Colab, and Kaggle Notebook. 


