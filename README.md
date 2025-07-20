# 🐌 Abalone Age Prediction: Deployed Stacked Ensemble Regression Model

## 🌎 Project Overview

This repository presents an advanced machine learning solution for predicting the age of abalone specimens using physiometric attributes. It features a fully deployed, interactive web application built with **Streamlit**, integrating a custom **two-level stacked ensemble regression model**. The primary aim is to enable real-time, interpretable, and reproducible predictive analytics.

## ✨ Core Functionalities

* **Interactive User Interface:** Clean and user-friendly GUI with dynamic sliders and dropdowns for feature input.
* **Stacked Ensemble Regression:** Combines multiple models to enhance prediction accuracy.
* **Real-Time Inference:** Instant output generation based on live input updates.
* **Model Interpretability:** Visual feature importance charts highlight the impact of each input parameter.
* **Data Visualization:** Interactive plots provide insights into dataset relationships and user input positioning.
* **Reproducibility:** Seamless setup with built-in mechanisms for data retrieval and model training.

## 🔧 Model Architecture: Two-Level Stacked Ensemble

### ▶ Level 0: Base Regressors

* **Random Forest Regressor:** Reduces overfitting via ensemble decision trees.
* **Support Vector Regressor (SVR):** Effective non-linear regression for high-dimensional inputs.
* **XGBoost Regressor:** High-performance gradient boosting model.

### ▶ Level 1: Meta Regressor

* **LightGBM Regressor:** Learns optimal aggregation of base model predictions to generate final output.

This architecture enhances overall prediction robustness by exploiting the complementary strengths of diverse regression techniques.

## 📚 Technology Stack

* **Languages & Frameworks:** Python, Streamlit
* **Libraries:** Scikit-learn, XGBoost, LightGBM, Pandas, Seaborn, Matplotlib

## ⚙️ Installation & Execution

### Prerequisites

* Python >= 3.7
* pip (Python package manager)

### Step 1: Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

### Step 2: Create Virtual Environment *(Recommended)*

#### For Windows:

```bash
python -m venv venv
.\venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install streamlit pandas scikit-learn matplotlib seaborn lightgbm xgboost
```

### Step 4: Launch Application

```bash
streamlit run app.py
```

Your browser will automatically open with the deployed application interface.

## 🔍 Operational Guide

1. Use the sidebar to select the abalone's sex ('M', 'F', or 'I').
2. Adjust the sliders for other physiometric parameters.
3. View the predicted age in rings on the main display.
4. Analyze feature importances and visualize input context against dataset distributions.

## 📄 License

This project is licensed under the **MIT License**. Please refer to the [LICENSE](LICENSE) file for full details.

## 🙏 Acknowledgements

* **Dataset:** Abalone Data Set courtesy of the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/abalone).
* **Tools:** Thanks to the creators of Streamlit and the open-source contributors who made this project possible.
