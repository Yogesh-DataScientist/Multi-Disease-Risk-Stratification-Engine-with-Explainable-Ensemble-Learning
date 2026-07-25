# 🩺 Multi-Disease Risk Stratification Engine with Explainable Ensemble Learning

> **An AI-powered Clinical Decision Support System (CDSS) for Diabetes and Heart Disease Risk Prediction using Ensemble Learning and Explainable AI (SHAP).**

---

## 🌐 Live Demo

🚀 **Streamlit Application:**
**https://multi-disease-risk-engine.streamlit.app/**

---

## 📌 Overview

The **Multi-Disease Risk Stratification Engine** is an intelligent healthcare analytics platform designed to predict the risk of **Diabetes Mellitus** and **Cardiovascular (Heart) Disease** using advanced machine learning techniques. The application combines three state-of-the-art gradient boosting algorithms—**CatBoost**, **LightGBM**, and **XGBoost**—into a **Soft Voting Ensemble Classifier** to improve predictive performance and model stability.

To improve transparency and trust, the system integrates **SHAP (SHapley Additive exPlanations)**, allowing users to understand how each clinical feature contributes to an individual prediction. The application is deployed using **Streamlit** and features an interactive dashboard with modern glassmorphism UI, Plotly visualizations, dataset exploration, benchmark analytics, and personalized health recommendations.

---

# 🎯 Key Features

## 🩸 Diabetes Risk Prediction

* Predicts diabetes risk using clinical and demographic information.
* Ensemble learning with CatBoost, LightGBM, and XGBoost.
* Interactive probability gauge.
* Individual model confidence scores.
* SHAP Waterfall Explainability.
* Personalized clinical recommendations.

---

## ❤️ Heart Disease Risk Prediction

* Predicts cardiovascular disease risk using lifestyle, medical history, and demographic factors.
* Real-time risk assessment.
* SHAP Feature Importance visualization.
* Interactive risk gauge.
* Health guidance and preventive recommendations.

---

## 🔍 Explainable AI (XAI)

The application integrates **SHAP** to make machine learning predictions transparent by providing:

* Waterfall Plot
* Feature Importance Plot
* Patient-specific explanations
* Clinical feature contribution analysis

---

## 📊 Ensemble Learning

The prediction engine combines:

* ✅ CatBoost
* ✅ LightGBM
* ✅ XGBoost

using a **Soft Voting Ensemble Classifier** to improve prediction accuracy, robustness, and reliability.

---

# 📈 Model Performance

## 🩸 Diabetes Prediction Model

| Metric    | Score     |
| --------- | --------- |
| Accuracy  | **97.2%** |
| Precision | **94.1%** |
| Recall    | **89.5%** |
| F1 Score  | **0.917** |
| ROC-AUC   | **0.988** |

---

## ❤️ Heart Disease Prediction Model

| Metric    | Score     |
| --------- | --------- |
| Accuracy  | **91.6%** |
| Precision | **88.5%** |
| Recall    | **82.4%** |
| F1 Score  | **0.853** |
| ROC-AUC   | **0.942** |

---

# 📊 Dashboard Modules

* 🏠 Home Dashboard
* 🩸 Diabetes Risk Prediction
* ❤️ Heart Disease Risk Prediction
* 📈 Ensemble Model Analytics
* 📊 Dataset Explorer
* 🔍 Explainable AI (SHAP)
* 📉 Interactive Plotly Charts
* 💡 Personalized Health Recommendations

---

# 📂 Dataset Information

The project utilizes two publicly available healthcare datasets.

## Diabetes Clinical Dataset

* Records: **100,000**
* Source:
  https://www.kaggle.com/datasets/priyamchoksi/100000-diabetes-clinical-dataset

---

## Heart Disease Dataset

* Records: **319,795**
* Source:
  https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease

---

## Combined Dataset

**Total Clinical Records Processed**

**419,795+ Patients**

---

# 🛠️ Technology Stack

### Frontend

* Streamlit
* HTML
* CSS
* Plotly

### Machine Learning

* Scikit-Learn
* CatBoost
* LightGBM
* XGBoost

### Explainable AI

* SHAP

### Data Processing

* Pandas
* NumPy
* Joblib

### Visualization

* Plotly
* Matplotlib

---

# 📁 Project Structure

```text
Multi-Disease-Risk-Engine/
│
├── app.py
├── models/
│   ├── ensemble_diabetes.pkl
│   ├── ensemble_heart.pkl
│   ├── diabetes_shap.pkl
│   └── heart_shap.pkl
│
├── datasets/
│   ├── diabetes_dataset.csv
│   └── heart_2020_cleaned.csv
│
├── assets/
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/your-username/Multi-Disease-Risk-Engine.git
```

Navigate to the project

```bash
cd Multi-Disease-Risk-Engine
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 💻 Usage

1. Open the Streamlit dashboard.
2. Select the disease prediction module.
3. Enter patient information.
4. Click **Predict Risk**.
5. View:

   * Risk Probability
   * SHAP Explainability
   * Model Confidence
   * Clinical Recommendations

---

# 💡 Future Enhancements

* PDF Medical Report Generation
* Batch Patient Prediction (CSV Upload)
* Prediction History Database
* Doctor Authentication System
* REST API Integration
* Cloud Database Support
* Electronic Health Record (EHR) Integration

---

# 👨‍💻 Author

**Yogeshwaran S**

AI & Data Science Engineer

---

# ⭐ If you found this project useful

Please consider giving the repository a **Star ⭐** to support the project.
