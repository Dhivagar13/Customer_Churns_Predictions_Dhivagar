import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.inspection import permutation_importance
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

# Set wide layout & page icon
st.set_page_config(page_title="Telecom Churn Prediction Dashboard", page_icon="📞", layout="wide")

# Custom CSS for dark background and styling
st.markdown("""
    <style>
    /* Background color */
    .stApp {
        background-color: #121212;
        color: #e0e0e0;
    }

    /* Sidebar styles */
    .css-1d391kg {
        background-color: #1e1e1e;
        color: #e0e0e0;
    }
    /* Sidebar header */
    .css-1v3fvcr {
        color: #ffffff;
        font-weight: 700;
        font-size: 1.3rem;
    }

    /* Main title */
    .css-1d391kg h1 {
        color: #00ffcc;
        text-shadow: 1px 1px 2px #000;
    }

    /* Dataframe styling */
    .stDataFrame>div>div>div {
        background-color: #222 !important;
        color: #eee !important;
    }

    /* Button style */
    div.stButton > button:first-child {
        background-color: #00bfa5;
        color: white;
        border-radius: 10px;
        border: none;
        font-weight: bold;
        padding: 10px 24px;
        cursor: pointer;
    }
    div.stButton > button:hover {
        background-color: #00796b;
        transition: background-color 0.3s ease;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Dataset Overview", "Churn Analysis", "Model Training & Evaluation", "Feature Importance"])

# Upload CSV file (placed outside menu for global use)
uploaded_file = st.sidebar.file_uploader("Upload your CSV data file", type=["csv"])

if uploaded_file is not None:
    # Load data once for all menu items
    data = pd.read_csv(uploaded_file)

    # Create churn column for analysis
    if 'total_rech_amt_9' in data.columns:
        data['Churn'] = data['total_rech_amt_9'].apply(lambda x: 1 if x == 0 else 0)
    else:
        st.sidebar.error("Column 'total_rech_amt_9' missing; 'Churn' cannot be created.")
        st.stop()

    # Common preprocessing for modeling
    cols_to_drop = []
    if 'mobile_number' in data.columns:
        cols_to_drop.append('mobile_number')
    if 'Churn' in data.columns:
        cols_to_drop.append('Churn')
    if cols_to_drop:
        features = data.drop(columns=cols_to_drop)
    else:
        features = data.copy()
    target = data['Churn'] if 'Churn' in data.columns else None
    if target is None:
        st.sidebar.error("Churn column missing; cannot proceed with modeling.")
        st.stop()
    features = features.select_dtypes(include=[np.number]).fillna(0)
    le = LabelEncoder()
    target_encoded = le.fit_transform(target)
    X_train, X_test, y_train, y_test = train_test_split(features, target_encoded, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if menu == "Dataset Overview":
        st.header("📋 Dataset Overview")
        st.markdown(f"*Shape:* {data.shape}")
        st.markdown("*Data Types:*")
        dtype_df = pd.DataFrame(data.dtypes, columns=['Data Type']).reset_index()
        dtype_df.columns = ['Feature', 'Data Type']
        st.dataframe(dtype_df)
        st.markdown("*Sample Data:*")
        st.dataframe(data.head())

        st.markdown("*Missing Values:*")
        missing_df = data.isnull().sum()
        missing_df = missing_df[missing_df > 0]
        if not missing_df.empty:
            st.dataframe(missing_df)
        else:
            st.success("No missing values detected.")

    elif menu == "Churn Analysis":
        st.header("📊 Churn Analysis and Visualization")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Churn Distribution")
            churn_counts = data['Churn'].value_counts()
            fig1, ax1 = plt.subplots()
            sns.barplot(x=churn_counts.index, y=churn_counts.values, palette='Set2', ax=ax1)
            ax1.set_xticklabels(['Not Churned', 'Churned'])
            ax1.set_ylabel("Count")
            ax1.set_title("Churn Distribution")
            st.pyplot(fig1)

        with col2:
            st.subheader("Churn Rate & Average Recharge")
            churn_rate = data['Churn'].value_counts(normalize=True).mul(100).rename({0: "Not Churned (%)", 1: "Churned (%)"})
            st.write(churn_rate)

            numeric_cols = ['total_rech_amt_6', 'total_rech_amt_7', 'total_rech_amt_8', 'total_rech_amt_9']
            existing_numeric_cols = [col for col in numeric_cols if col in data.columns]
            if existing_numeric_cols:
                st.write("Average Recharge Amounts (Months 6-9):")
                st.write(data[existing_numeric_cols].mean())
            else:
                st.warning("Recharge amount columns for months 6-9 not found.")

    elif menu == "Model Training & Evaluation":
        st.header("🔍 Model Training & Evaluation")
        models = {
            "Logistic Regression": LogisticRegression(max_iter=500),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
        }

        metrics = {}
        for model_name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

            accuracy = model.score(X_test_scaled, y_test)
            f1 = classification_report(y_test, y_pred, output_dict=True)['1']['f1-score']
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            metrics[model_name] = {"Accuracy": accuracy, "F1 Score": f1, "RMSE": rmse, "ROC AUC": roc_auc}

            st.markdown(f"### {model_name} Evaluation")

            cols = st.columns([1, 1.5])
            with cols[0]:
                st.text(classification_report(y_test, y_pred, target_names=["Not Churned", "Churned"]))
                cm = confusion_matrix(y_test, y_pred)
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=["Not Churned", "Churned"], yticklabels=["Not Churned", "Churned"], ax=ax_cm)
                ax_cm.set_xlabel('Predicted')
                ax_cm.set_ylabel('Actual')
                ax_cm.set_title(f'Confusion Matrix - {model_name}')
                st.pyplot(fig_cm)

            with cols[1]:
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                fig_roc, ax_roc = plt.subplots()
                ax_roc.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})', color='#00FFAA')
                ax_roc.plot([0, 1], [0, 1], 'k--')
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.set_title(f'ROC Curve - {model_name}')
                ax_roc.legend(loc='lower right')
                st.pyplot(fig_roc)

        st.subheader("Summary of Model Metrics")
        metrics_df = pd.DataFrame(metrics).T
        st.dataframe(metrics_df.style.format({
            "Accuracy": "{:.3f}",
            "F1 Score": "{:.3f}",
            "RMSE": "{:.3f}",
            "ROC AUC": "{:.3f}"
        }))

    elif menu == "Feature Importance":
        st.header("🌟 Random Forest Feature Importance")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        y_pred_rf = rf_model.predict(X_test_scaled)

        st.subheader("Random Forest Model Evaluation")
        st.text(classification_report(y_test, y_pred_rf, target_names=["Not Churned", "Churned"]))

        col1, col2 = st.columns(2)

        with col1:
            cm_rf = confusion_matrix(y_test, y_pred_rf)
            fig_cm_rf, ax_cm_rf = plt.subplots()
            sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
                        xticklabels=["Not Churned", "Churned"], yticklabels=["Not Churned", "Churned"], ax=ax_cm_rf)
            ax_cm_rf.set_xlabel('Predicted')
            ax_cm_rf.set_ylabel('Actual')
            ax_cm_rf.set_title('Confusion Matrix - Random Forest')
            st.pyplot(fig_cm_rf)

        with col2:
            feature_importance = rf_model.feature_importances_
            feature_names = features.columns
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance}).sort_values(by='Importance', ascending=False)

            fig_imp, ax_imp = plt.subplots(figsize=(8,6))
            sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis', ax=ax_imp)
            ax_imp.set_title("Feature Importance")
            st.pyplot(fig_imp)

        result = permutation_importance(rf_model, X_test_scaled, y_test, n_repeats=10, random_state=42)
        perm_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': result.importances_mean}).sort_values(by='Importance', ascending=False)

        st.subheader("Permutation Feature Importance")
        fig_perm, ax_perm = plt.subplots(figsize=(8,6))
        sns.barplot(data=perm_importance_df, x='Importance', y='Feature', palette='plasma', ax=ax_perm)
        ax_perm.set_title("Permutation Feature Importance")
        st.pyplot(fig_perm)

        st.dataframe(importance_df.reset_index(drop=True))
        st.dataframe(perm_importance_df.reset_index(drop=True))

else:
    st.info("Please upload a CSV file to get started.")
