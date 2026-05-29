# 🚕 Cab Price Prediction — Machine Learning Project

> Predicting cab surge pricing using GradientBoosting, RandomForest, XGBoost & LinearRegression with **R² = 0.991**

---

## 📌 Overview

This project builds a machine learning pipeline to predict **cab surge multiplier pricing** based on real-world ride data including distance, weather conditions, demand scores, time of day, and cab type. Four regression models were trained, tuned, and compared — with GradientBoosting emerging as the best performer.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Total records | 66,367 rows |
| Features | 57 columns |
| Missing values | 0 |
| Price range | $4.5 – $76.0 |
| Cab types | CabXL, CabX, CabPool |
| Unique sources / destinations | 12 each |

**Key features used:**
- `distance` — trip distance
- `demand_score_raw` — real-time demand index
- `surge_multiplier` — **target variable**
- `name` — cab type (CabXL, CabX, CabPool)
- `source` / `destination` — pickup and drop-off locations
- Weather features: `temp`, `humidity`, `rain`, `clouds`, `wind`
- Engineered features: `weather_severity`, `temp_humidity`, `cold_intensity`, `is_morning_rush`, `rain_intensity`, `rain_squared`, `cold_late`

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-red)
![pandas](https://img.shields.io/badge/pandas-latest-150458?logo=pandas)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)

```
pandas · numpy · matplotlib · seaborn
scikit-learn · xgboost · joblib
```

---

## ⚙️ Project Pipeline

```
Raw CSV Data
    │
    ▼
Data Inspection & EDA
  └── Shape, missing values, price range, cardinality
    │
    ▼
Correlation Heatmap
  └── Top 12 features vs price identified
    │
    ▼
Feature Engineering
  └── weather_severity, temp_humidity, cold_intensity
  └── is_morning_rush, rain_intensity, rain_squared, cold_late
    │
    ▼
Preprocessing Pipeline (ColumnTransformer)
  ├── Numerical → StandardScaler
  └── Categorical → OneHotEncoder (drop='first')
    │
    ▼
Train / Test Split (80/20, random_state=42)
    │
    ▼
Model Training + RandomizedSearchCV (cv=5)
  ├── LinearRegression
  ├── RandomForest
  ├── GradientBoosting  ← Best
  └── XGBoost
    │
    ▼
Evaluation: MAE · RMSE · R² · CV R²
    │
    ▼
Feature Importance Analysis
```

---

## 📈 Results

| Model | R² | MAE | RMSE | CV R² |
|---|---|---|---|---|
| **GradientBoosting** ⭐ | **0.9907** | **0.0107** | **0.0416** | **0.9900** |
| RandomForest | 0.9906 | 0.0107 | 0.0418 | 0.9899 |
| XGBoost | 0.9892 | 0.0129 | 0.0448 | 0.9887 |
| LinearRegression | 0.8755 | 0.1153 | 0.1524 | 0.8733 |

### 🏆 Best Model: GradientBoosting
```
Best Params: n_estimators=100 | max_depth=5 | learning_rate=0.1
R²  = 0.9907
MAE = 0.0107
RMSE = 0.0416
CV R² = 0.9900
```

---

## 🔍 Key Insights

**1. Demand is everything**
`demand_score_raw` alone accounts for **74% of feature importance** — far more than distance, weather, or time. Surge pricing is almost entirely demand-driven.

**2. Cab type matters more than weather**
After demand, `cat__name_CabXL` and `cat__name_CabX` were the next most important features. Premium cab types carry inherently higher multipliers.

**3. Weather has indirect impact**
Weather features (temp, humidity, rain) show low direct importance — but they influence demand, which then drives price. This multicollinearity is visible in the correlation heatmap.

**4. Tree-based models dominate**
GradientBoosting and RandomForest both hit R² ≈ 0.99, while LinearRegression capped at 0.875 — confirming the non-linear nature of surge pricing.

---

## 📁 Project Structure

```
cab-price-prediction-ml/
│
├── cab_ai.ipynb                # Main Jupyter notebook
├── README.md                   # Project documentation
│
├── outputs/
│   ├── correlation_heatmap.png # Top 12 features heatmap
│   ├── model_comparison.png    # MAE / RMSE / R² bar charts
│   └── feature_importances.png # Top 20 feature importances
│
└── model/
    └── Best_model.pkl          # Saved GradientBoosting pipeline (joblib)
```

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/your-username/cab-price-prediction-ml.git
cd cab-price-prediction-ml
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib jupyter
```

**3. Add the dataset**
Place the CSV file in the project root and update the path in cell 2 of the notebook:
```python
df = pd.read_csv("your_dataset.csv")
```

**4. Run the notebook**
```bash
jupyter notebook cab_ai.ipynb
```

---

## 💾 Save the Best Model

To save and reload the trained pipeline:
```python
import joblib

# Save
joblib.dump(pipeline, "model/Best_model.pkl")

# Load and predict
model = joblib.load("model/Best_model.pkl")
predictions = model.predict(X_new)
```

---

## 🙋 Author

**Honey**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/honey-rana-6748b938a)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/Honey5632)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
