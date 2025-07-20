# main_app.py

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(
    page_title="Abalone Age Predictor",
    page_icon="🐌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Model Training ---
# Use Streamlit's caching to load data and train the model only once.
@st.cache_data
def load_and_train_model():
    """
    Loads the Abalone dataset, performs feature engineering, and trains a 
    sophisticated Stacked Ensemble Regressor.
    Returns the trained model, a model for importance, and the processed dataset.
    """
    # Load the Abalone dataset from the UCI repository
    try:
        csv_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
        column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 
                        'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
        df = pd.read_csv(csv_url, header=None, names=column_names)
    except Exception as e:
        st.error(f"Failed to load dataset. Error: {e}")
        st.stop()

    # --- Feature Engineering: One-Hot Encode the 'Sex' column ---
    df_processed = pd.get_dummies(df, columns=['Sex'], prefix='Sex')

    # The 'Rings' column is our target. Age is typically Rings + 1.5
    X = df_processed.drop('Rings', axis=1)
    y = df_processed['Rings']
    
    # --- Define the Advanced Stacked Ensemble Model ---
    # Level 0: A list of diverse and powerful base models
    estimators = [
        ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
        ('svr', SVR(kernel='rbf', C=1.0)),
        ('xgb', xgb.XGBRegressor(n_estimators=100, random_state=42))
    ]
    
    # Level 1 (Meta-Model): A powerful model to combine the predictions of base models
    # Using LightGBM as the final estimator is a sophisticated choice.
    stacked_model = StackingRegressor(
        estimators=estimators,
        final_estimator=lgb.LGBMRegressor(random_state=42),
        cv=5 # Use cross-validation
    )
    
    stacked_model.fit(X, y)
    
    # For feature importance, we train a separate LightGBM model as it's more direct
    importance_model = lgb.LGBMRegressor(random_state=42)
    importance_model.fit(X, y)
    
    return stacked_model, importance_model, df_processed

# Load the trained models and dataset
model, importance_model, abalone_df = load_and_train_model()
X_full = abalone_df.drop('Rings', axis=1)
y_full = abalone_df['Rings']


# --- Sidebar for User Input ---
st.sidebar.header('Input Abalone Features')
st.sidebar.markdown("""
Enter the physical measurements of the abalone to predict its age (in rings).
""")

def user_input_features():
    """
    Creates UI elements in the sidebar for user to input features.
    Returns a DataFrame with the user's inputs, ready for the model.
    """
    sex = st.sidebar.selectbox('Sex', ('M', 'F', 'I'))
    length = st.sidebar.slider('Length (mm)', 0.0, 1.0, 0.52)
    diameter = st.sidebar.slider('Diameter (mm)', 0.0, 1.0, 0.41)
    height = st.sidebar.slider('Height (mm)', 0.0, 1.5, 0.14)
    whole_weight = st.sidebar.slider('Whole Weight (grams)', 0.0, 3.0, 0.83)
    shucked_weight = st.sidebar.slider('Shucked Weight (grams)', 0.0, 1.5, 0.36)
    viscera_weight = st.sidebar.slider('Viscera Weight (grams)', 0.0, 0.8, 0.18)
    shell_weight = st.sidebar.slider('Shell Weight (grams)', 0.0, 1.0, 0.24)
    
    # Create the feature dictionary based on user input
    data = {
        'Length': length,
        'Diameter': diameter,
        'Height': height,
        'Whole weight': whole_weight,
        'Shucked weight': shucked_weight,
        'Viscera weight': viscera_weight,
        'Shell weight': shell_weight,
        'Sex_F': 1 if sex == 'F' else 0,
        'Sex_I': 1 if sex == 'I' else 0,
        'Sex_M': 1 if sex == 'M' else 0,
    }
    
    # Ensure the columns are in the same order as the training data
    features = pd.DataFrame(data, index=[0])[X_full.columns]
    return features

input_df = user_input_features()


# --- Main Panel for Displaying Outputs ---
st.title('🐌 Abalone Age Predictor')
st.markdown("""
This application uses a **custom-built, two-level Stacked Ensemble model** to predict the age of an abalone (represented by the number of rings in its shell). This advanced technique combines multiple machine learning models to achieve higher accuracy.

**How it works:**
1.  **Base Models:** Predictions are first made by `RandomForest`, `XGBoost`, and `SVR`.
2.  **Meta-Model:** These predictions are then fed into a `LightGBM` model, which makes the final, refined prediction.
""")

# --- Prediction and Output ---
st.subheader('1. Your Input Parameters')
st.table(input_df.T)

# Make predictions
prediction = model.predict(input_df)
predicted_rings = prediction[0]

st.subheader('2. Predicted Age (in Rings)')
st.metric(label="Predicted Rings", value=f"{predicted_rings:.2f} rings")
st.write(f"The estimated age of the abalone is approximately **{predicted_rings + 1.5:.2f} years**.")


# --- Visualizations for Model Explanation ---
st.subheader('3. Model Explanation & Data Visualization')
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Feature Importance**")
    st.markdown("This chart shows which features were most important for the prediction.")
    
    feature_importance = pd.DataFrame({
        'feature': X_full.columns,
        'importance': importance_model.feature_importances_
    }).sort_values('importance', ascending=False)

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance, palette='plasma', ax=ax1)
    ax1.set_title('Feature Importance for Abalone Age')
    plt.tight_layout()
    st.pyplot(fig1)

with col2:
    st.markdown("**Shell Weight vs. Whole Weight**")
    st.markdown("This plot shows the relationship between the abalone's total weight and its shell weight. Your input is the red star.")

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    scatter = ax2.scatter(
        x=abalone_df['Whole weight'], 
        y=abalone_df['Shell weight'], 
        c=abalone_df['Rings'], 
        cmap='viridis', 
        alpha=0.5
    )
    ax2.scatter(
        input_df['Whole weight'], 
        input_df['Shell weight'], 
        marker='*', 
        s=300, 
        c='red', 
        label='Your Input',
        edgecolors='black'
    )
    
    ax2.set_xlabel('Whole Weight (grams)')
    ax2.set_ylabel('Shell Weight (grams)')
    ax2.set_title('Weight Analysis')
    ax2.legend()
    fig2.colorbar(scatter, ax=ax2, label='Rings (Age)')
    
    plt.tight_layout()
    st.pyplot(fig2)

st.info("""
**How to Run This App:**
1.  Save the code as `app.py`.
2.  Install necessary libraries: `pip install streamlit pandas scikit-learn matplotlib seaborn lightgbm xgboost`
3.  Run from your terminal: `streamlit run app.py`
""")
