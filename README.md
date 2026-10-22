# Datheon Machine Learning & Data Science Internship

Welcome to my official submission repository for the **Datheon Machine Learning & Data Science Internship** program. This repository contains end-to-end Jupyter Notebooks, datasets, feature engineering pipelines, and visualizations for all 4 assigned internship tasks.

---

## 📌 Repository Structure
├── Task # 1.ipynb      # Multi-Class Benchmark Classification Pipeline
├── Task # 2.ipynb      # Real Estate Housing Price Prediction (Ridge & Lasso)
├── Task # 3.ipynb      # Predictive Customer Churn Classifier (Random Forest & XGBoost)
├── Task # 4.ipynb      # NLP Sentiment Classification on Customer Reviews (TF-IDF)
└── README.md           # Master Project Documentation

## 🚀 Tasks Overview

### **Task 1: Multi-Class Benchmark Classification Pipeline (Iris Dataset)**
- **Objective:** Build an end-to-end classification pipeline to classify Iris flower species using machine learning models.
- **Workflow & Techniques:**
  - **EDA:** Visualized feature distributions and species relationships using Seaborn pairplots.
  - **Preprocessing:** Applied stratified train-test split (80/20) and standardized continuous features using `StandardScaler`.
  - **Models:** Trained **Logistic Regression** and **Decision Tree Classifier**.
  - **Evaluation:** Precision, Recall, F1-Score, Accuracy, and Confusion Matrices.

---

### **Task 2: Real Estate Housing Price Prediction (Ridge & Lasso Regression)**
- **Objective:** Predict median house values using the California Housing dataset using continuous regression models and regularization techniques.
- **Workflow & Techniques:**
  - **Data Handling:** Imputed missing values via `SimpleImputer` and normalized features using `StandardScaler`.
  - **Feature Engineering:** Added domain-specific interaction features (rooms per household, bedrooms per room).
  - **Models:** Fitted OLS Linear Regression, Ridge (L2), and Lasso (L1) models utilizing `GridSearchCV` for hyperparameter tuning.
  - **Evaluation:** Evaluated RMSE, MAE, $R^2$ scores, and residual error distribution plots.

---

### **Task 3: Predictive Customer Churn Classifier (Random Forest & XGBoost)**
- **Objective:** Address severe class imbalance and predict telecom customer churn using ensemble learning.
- **Workflow & Techniques:**
  - **Preprocessing:** Encoded categorical features with `OneHotEncoder` and balanced target class distribution using `SMOTE`.
  - **Models:** Trained **Random Forest** and **XGBoost** classifiers with cross-validation.
  - **Evaluation:** Plotted ROC-AUC curves (>80% AUC benchmark), confusion matrices, and extracted top 10 feature importances.

---

### **Task 4: NLP Sentiment Classification on Customer Reviews (TF-IDF)**
- **Objective:** Construct an NLP text classifier to identify positive and negative sentiment from customer reviews.
- **Workflow & Techniques:**
  - **Text Preprocessing:** Lowercasing, regex cleaning (HTML/special chars), stopword removal, and WordNet lemmatization.
  - **Feature Extraction:** Converted clean corpus into numerical vectors using `TfidfVectorizer` (max 5,000 features).
  - **Models:** Benchmark comparison between **Multinomial Naive Bayes** and **Logistic Regression**.
  - **Deliverable:** Built an interactive `predict_sentiment()` helper function for live custom text inference.

---

## 🛠️ Tools & Libraries Used

- **Languages & Core:** Python, Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-Learn, XGBoost, Imbalanced-learn (SMOTE)
- **Natural Language Processing:** NLTK, Regular Expressions (`re`)

---

## 👤 Author
**Munawar Abbas**  
- **GitHub:** [MunnaSuggest](https://github.com/MunnaSuggest)  
- **LinkedIn:** [Munawar Abbas](https://www.linkedin.com/in/munawar-abbas-533868338)
