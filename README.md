# House Rent Prediction ML Project 🏠

This is a beginner-friendly Machine Learning project that predicts residential house rent using regression modeling techniques. The pipeline processes a real-world housing dataset, performs feature engineering, cleans up text constraints, removes data anomalies/outliers, and builds a standardized prediction engine.

---

## 📊 Project Architecture & Workflow

The machine learning workflow follows a structured engineering process to ensure reliable model performance:

1. **Data Loading & Preprocessing**: Imports raw dataset containing attributes like size, city, furnishing, and locality.
2. **Feature Pruning**: High-cardinality and irrelevant metadata features are removed to eliminate structural noise.
3. **Feature Engineering**: Extracts numerical data out of messy categorical text strings (e.g., converting physical floor levels).
4. **Outlier Mitigation**: Trims long-tail extreme values (extreme rent prices or disproportionately large homes) that break standard linear estimators.
5. **Categorical Variable Dummy Encoding**: Maps text classes into linear binary spaces using numerical indicator arrays (`drop_first=True` applied to dodge multicollinearity).
6. **Data Partitioning**: Splits fields into distinct 80/20 standard training and evaluation subsets.
7. **Feature Normalization**: Standardizes continuous features into common geometric normal spaces via Z-score metric variance.
8. **Regression Estimation**: Fits multi-variable linear configurations across predictive datasets and scores the variance patterns.

---

## 🚀 The High-Cardinality Trap & Optimization

### The Original Problem
In the initial baseline implementation, using `pd.get_dummies()` directly on the entire dataset caused a massive issue known as the **"Curse of Dimensionality"**. Because the column `Area Locality` contained over **2,200+ unique neighborhoods**, one-hot encoding exploded the dataframe from **9 basic features to over 2,200 dynamic columns**. 

This extreme dimensional inflation caused the standard Linear Regression model to collapse into extreme numerical overfitting due to sparse feature space. This resulted in a completely broken $R^2$ evaluation score:

* ❌ **Original Model Baseline $R^2$ Score**: ~ `-5.98 × 10^25` *(Completely broken model)*

### The Optimized Fix
By analyzing the structural weights, we removed `Area Locality` since general regional price dynamics are already accurately captured by the `City` feature. Additionally, we implemented an optimized `Floor` extraction scheme and forced `drop_first=True` dummy splits to prevent multi-collinearity traps.

* ✅ **Optimized Model $R^2$ Score**: `0.6697` (~ **67% accuracy**)
* ✅ **Optimized Mean Absolute Error (MAE)**: `\u20b911,477`

---

## 🛠️ Installation & Reproduction Setup

Ensure you have Python 3.8+ installed locally. Follow these steps to set up your environment and run the project:

```bash
# 1. Clone this repository or open your working project directory
cd house-rent-prediction

# 2. Install the necessary dependencies
pip install -r requirements.txt

# 3. Run the main model engine script
python main.py