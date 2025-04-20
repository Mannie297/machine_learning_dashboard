"""
Machine Learning Dashboard
A comprehensive dashboard for visualizing machine learning model performance,
training metrics, and predictions. Built with Streamlit and scikit-learn.
"""

# Import required libraries
import streamlit as st  # For creating the web interface
import plotly.express as px  # For creating interactive plots
import plotly.graph_objects as go  # For creating custom plots
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve  # For model evaluation
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # For ensemble models
from sklearn.linear_model import LogisticRegression  # For logistic regression
from sklearn.svm import SVC  # For support vector machines
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc  # For model metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder  # For data preprocessing
import joblib  # For model saving/loading
import os  # For file operations
from datetime import datetime  # For timestamp generation
import seaborn as sns  # For statistical visualizations
import matplotlib.pyplot as plt  # For plotting

# Configure the Streamlit page
st.set_page_config(
    page_title="ML Model Dashboard",  # Browser tab title
    page_icon="📊",  # Browser tab icon
    layout="wide"  # Use full page width
)

# Display title and description
st.title("Emmanuel.O Machine Learning Model Dashboard")
st.markdown("""
This dashboard provides interactive visualization of machine learning model performance,
training metrics, and predictions. Upload your dataset or use the sample data to get started.
""")

# Display dataset requirements in sidebar
st.sidebar.header("Dataset Requirements")
st.sidebar.markdown("""
**Your dataset should:**
- Be in CSV format
- Contain numerical features
- Have at least 2 classes in the target variable
- Have no missing values (or enable 'Handle Missing Values')
""")

# File upload section
st.sidebar.header("Model Configuration")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])

# Function to load sample Iris dataset
@st.cache_data
def load_sample_data():
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                     columns=iris['feature_names'] + ['target'])
    return df

# Load and validate dataset
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        # Validate dataset structure
        if len(df.columns) < 2:
            st.error("Dataset must have at least 2 columns (features + target)")
            st.stop()
        
        # Check for non-numeric columns
        non_numeric = df.select_dtypes(exclude=['number']).columns
        if len(non_numeric) > 0:
            st.warning(f"Non-numeric columns detected: {', '.join(non_numeric)}. These will be excluded from analysis.")
            df = df.select_dtypes(include=['number'])
        
        # Check for missing values
        if df.isnull().any().any():
            st.warning("Dataset contains missing values. Enable 'Handle Missing Values' to process them.")
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.stop()
else:
    df = load_sample_data()
    st.info("Using sample Iris dataset. Upload your own data to analyze.")

# Display dataset information
st.subheader("Dataset Information")
col1, col2 = st.columns(2)
with col1:
    st.write("Shape:", df.shape)  # Display dataset dimensions
    st.write("Features:", len(df.columns) - 1)  # Number of features
with col2:
    st.write("Samples:", len(df))  # Number of samples
    if 'target' in df.columns:
        st.write("Classes:", len(df['target'].unique()))  # Number of classes

# Display dataset preview
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Display data statistics
st.subheader("Data Statistics")
st.dataframe(df.describe())

# Data preprocessing options
st.sidebar.subheader("Data Preprocessing")
normalize_data = st.sidebar.checkbox("Normalize Features", value=True)
handle_missing = st.sidebar.checkbox("Handle Missing Values", value=True)

# Handle missing values if enabled
if handle_missing:
    df = df.fillna(df.mean())

# Feature selection
st.sidebar.subheader("Feature Selection")
features = st.sidebar.multiselect(
    "Select features for training",
    options=df.columns[:-1],
    default=df.columns[:-1].tolist()
)

# Target variable selection
target = st.sidebar.selectbox(
    "Select target variable",
    options=df.columns,
    index=len(df.columns)-1
)

# Validate target variable
if len(df[target].unique()) < 2:
    st.error("Target variable must have at least 2 classes")
    st.stop()

# Model selection
st.sidebar.subheader("Model Selection")
model_type = st.sidebar.selectbox(
    "Select Model",
    ["Random Forest", "Gradient Boosting", "Logistic Regression", "SVM"]
)

# Model parameter configuration based on selected model
st.sidebar.subheader("Model Parameters")
if model_type == "Random Forest":
    n_estimators = st.sidebar.slider("Number of trees", 10, 200, 100)
    max_depth = st.sidebar.slider("Max depth", 2, 20, 5)
elif model_type == "Gradient Boosting":
    n_estimators = st.sidebar.slider("Number of trees", 10, 200, 100)
    learning_rate = st.sidebar.slider("Learning rate", 0.01, 0.5, 0.1)
elif model_type == "Logistic Regression":
    C = st.sidebar.slider("Regularization strength", 0.01, 10.0, 1.0)
elif model_type == "SVM":
    C = st.sidebar.slider("Regularization strength", 0.01, 10.0, 1.0)
    kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly"])

# Cross-validation configuration
cv_folds = st.sidebar.slider("Cross-validation folds", 2, 10, 5)

# Train-test split configuration
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2)

# Prepare data for training
X = df[features]
y = df[target]

# Normalize features if enabled
if normalize_data:
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Function to train selected model
@st.cache_resource  # Cache the trained model
def train_model(model_type, **params):
    if model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=params.get('n_estimators', 100),
                                     max_depth=params.get('max_depth', 5),
                                     random_state=42)
    elif model_type == "Gradient Boosting":
        model = GradientBoostingClassifier(n_estimators=params.get('n_estimators', 100),
                                         learning_rate=params.get('learning_rate', 0.1),
                                         random_state=42)
    elif model_type == "Logistic Regression":
        model = LogisticRegression(C=params.get('C', 1.0), random_state=42)
    elif model_type == "SVM":
        model = SVC(C=params.get('C', 1.0),
                   kernel=params.get('kernel', 'rbf'),
                   probability=True,
                   random_state=42)
    
    model.fit(X_train, y_train)
    return model

# Prepare model parameters
model_params = {
    'n_estimators': n_estimators if model_type in ["Random Forest", "Gradient Boosting"] else None,
    'max_depth': max_depth if model_type == "Random Forest" else None,
    'learning_rate': learning_rate if model_type == "Gradient Boosting" else None,
    'C': C if model_type in ["Logistic Regression", "SVM"] else None,
    'kernel': kernel if model_type == "SVM" else None
}

# Train model with error handling
try:
    model = train_model(model_type, **model_params)
except Exception as e:
    st.error(f"Error training model: {str(e)}")
    st.stop()

# Cross-validation with error handling
try:
    cv_scores = cross_val_score(model, X, y, cv=cv_folds)
    st.subheader("Cross-validation Results")
    st.write(f"Mean CV Score: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
except Exception as e:
    st.warning(f"Cross-validation failed: {str(e)}")

# Learning curve with error handling
try:
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv_folds, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_mean,
        name='Training Score',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=train_sizes, y=test_mean,
        name='Cross-validation Score',
        line=dict(color='red')
    ))
    fig.update_layout(
        title='Learning Curve',
        xaxis_title='Training Examples',
        yaxis_title='Score',
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.warning(f"Learning curve generation failed: {str(e)}")

# Model performance visualization
st.subheader("Model Performance")
col1, col2 = st.columns(2)

# Feature importance plot
with col1:
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
                     title='Feature Importance')
        st.plotly_chart(fig, use_container_width=True)

# Confusion matrix
with col2:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted ' + str(i) for i in range(len(np.unique(y)))],
        y=['Actual ' + str(i) for i in range(len(np.unique(y)))],
        colorscale='Blues'
    ))
    fig.update_layout(title='Confusion Matrix')
    st.plotly_chart(fig, use_container_width=True)

# Model metrics
st.subheader("Model Metrics")
accuracy = accuracy_score(y_test, y_pred)
st.metric("Accuracy", f"{accuracy:.2%}")

# Classification report
st.subheader("Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)

# ROC curve for binary classification
if len(np.unique(y)) == 2:
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr,
                                mode='lines',
                                name=f'ROC curve (AUC = {roc_auc:.2f})'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                                mode='lines',
                                name='Random',
                                line=dict(dash='dash')))
        fig.update_layout(title='ROC Curve',
                         xaxis_title='False Positive Rate',
                         yaxis_title='True Positive Rate')
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"ROC curve generation failed: {str(e)}")

# Prediction interface
st.subheader("Make Predictions")
st.write("Enter feature values to make predictions:")

# Create input fields for each feature
input_data = {}
cols = st.columns(len(features))
for i, feature in enumerate(features):
    with cols[i]:
        input_data[feature] = st.number_input(feature, value=float(df[feature].mean()))

# Make prediction with error handling
if st.button("Predict"):
    try:
        input_df = pd.DataFrame([input_data])
        if normalize_data:
            input_df = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
        
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        
        st.write("Prediction:", prediction)
        st.write("Class Probabilities:")
        for i, prob in enumerate(probability):
            st.write(f"Class {i}: {prob:.2%}")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Model saving with error handling
if st.sidebar.button("Save Model"):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f'model_{timestamp}.joblib'
        joblib.dump(model, model_path)
        st.sidebar.success(f"Model saved successfully as {model_path}!")
    except Exception as e:
        st.sidebar.error(f"Failed to save model: {str(e)}")

# Export results with error handling
if st.sidebar.button("Export Results"):
    try:
        # Create a properly structured dictionary for the results
        results = {
            'Model Type': [model_type],
            'Parameters': [str(model_params)],
            'Accuracy': [accuracy],
            'Mean CV Score': [cv_scores.mean()],
            'CV Score Std': [cv_scores.std()],
            'Classification Report': [str(report)]
        }
        
        # Create DataFrame with proper structure
        results_df = pd.DataFrame(results)
        
        # Save to CSV with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f'results_{timestamp}.csv'
        results_df.to_csv(results_path, index=False)
        st.sidebar.success(f"Results exported to {results_path}!")
    except Exception as e:
        st.sidebar.error(f"Failed to export results: {str(e)}") 