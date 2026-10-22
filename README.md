# Datheon Machine Learning & Data Science Internship

Welcome to my submission repository for the **Datheon Machine Learning & Data Science Internship** program. This repository contains end-to-end Jupyter Notebooks, datasets, and visualizations for the assigned internship tasks.

---

## 📌 Repository Structure
├── Task # 1.ipynb      # Multi-Class Benchmark Classification Pipeline
├── Task # 2.ipynb      # Real Estate Housing Price Prediction (Ridge & Lasso)
└── README.md           # Project Documentation

## 🚀 Tasks Overview

### **Task 1: Multi-Class Benchmark Classification Pipeline (Iris Dataset)**
- **Objective:** Build an end-to-end classification pipeline to classify Iris flower species using machine learning models.
- **Workflow & Techniques:**
  - **Exploratory Data Analysis (EDA):** Visualized feature distributions and species relationships using Seaborn pairplots.
  - **Preprocessing & Normalization:** Applied train-test split (80/20) and standardized features using `StandardScaler`.
  - **Model Training:** Trained two models: **Logistic Regression** and **Decision Tree Classifier**.
  - **Evaluation Metrics:** Evaluated models using Precision, Recall, F1-Score, Accuracy, and Confusion Matrices.
- **Deliverables:**
  - Jupyter Notebook containing complete analysis, model outputs, and confusion matrix heatmaps.

---

### **Task 2: Real Estate Housing Price Prediction (Ridge & Lasso Regression)**
- **Objective:** Predict median house values using the California Housing dataset using continuous regression models and regularization techniques.
- **Workflow & Techniques:**
  - **Data Imputation & Preprocessing:** Handled missing values using `SimpleImputer` and normalized continuous features using `StandardScaler`.
  - **Feature Engineering:** Created domain-specific interaction features (e.g., rooms per household, bedrooms per room).
  - **Model Training & Hyperparameter Tuning:** Fitted OLS Linear Regression, Ridge (L2), and Lasso (L1) models utilizing `GridSearchCV` for optimal alpha selection.
  - **Evaluation Metrics:** Evaluated model performance across RMSE, MAE, $R^2$ scores, and analyzed residual error distribution plots.
- **Deliverables:**
  - Jupyter Notebook comparing model regression metrics and residual error plots.

---

## 🛠️ Tools & Libraries Used

- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-Learn (`LinearRegression`, `Ridge`, `Lasso`, `GridSearchCV`, `SimpleImputer`, `StandardScaler`)

---

## 👤 Author
**Munawar Abbas**  
- **GitHub:** [MunnaSuggest](https://github.com/MunnaSuggest)  
- **LinkedIn:** [Munawar Abbas](https://www.linkedin.com/in/munawar-abbas-533868338)
