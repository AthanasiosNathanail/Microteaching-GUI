import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import export_text, plot_tree
import io

# -------------------------
# Sample Data (Synthetic)
# -------------------------
def load_sample_data():
    np.random.seed(42)  # For reproducibility
    N = 100
    # Create a small synthetic dataset with plausible columns
    sample_df = pd.DataFrame({
        "GammaRay": np.random.uniform(50, 150, N),
        "Porosity": np.random.uniform(0.05, 0.3, N),
        "Permeability": np.random.uniform(10, 1000, N),
        "Resistivity": np.random.uniform(0.5, 50, N),
        "Depth": np.linspace(1000, 2000, N),  # from 1000m to 2000m
        "Seal Integrity": np.random.randint(0, 2, N),
        # Binary target (0 = Non-Reservoir, 1 = Reservoir)
        "Reservoir Flag": np.random.randint(0, 2, N)
    })
    return sample_df

# -------------------------
# Streamlit UI
# -------------------------
st.title("Reservoir Classification using Decision Trees")
st.write("Upload geophysical data or use sample data to classify reservoir and non-reservoir zones using a Random Forest model.")

# Radio button to select data source
data_source = st.radio(
    "Select Data Source",
    ("Use sample data", "Upload your own CSV")
)

# If user chooses to upload a CSV
if data_source == "Upload your own CSV":
    data_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if data_file is not None:
        df = pd.read_csv(data_file)
        st.write("### Dataset Preview")
        st.write(df.head())
    else:
        st.warning("Please upload a CSV file to proceed.")
        st.stop()  # Stop the script until a file is uploaded

# If user chooses sample data
else:
    df = load_sample_data()
    st.write("### Using Sample Data")
    st.write(df.head())

# Selecting features and target
st.write("### Select Features and Target Variable")
all_columns = df.columns.tolist()

# You can adjust the default features here if you wish
features = st.multiselect(
    "Select geophysical features", 
    all_columns, 
    default=["GammaRay", "Porosity", "Permeability", "Resistivity", "Depth", "Seal Integrity"]
)

# By default, assume the target is the last column (or specify your known target)
target = st.selectbox(
    "Select Target Variable (Reservoir/Non-Reservoir)", 
    all_columns, 
    index=len(all_columns)-1
)

if features and target:
    X = df[features]
    y = df[target]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Model hyperparameters
    st.sidebar.header("Model Hyperparameters")
    n_estimators = st.sidebar.slider(
        "Number of Trees", 
        min_value=1, 
        max_value=200, 
        value=100, 
        step=10
    )
    max_depth = st.sidebar.slider(
        "Max Depth", 
        min_value=2, 
        max_value=20, 
        value=5, 
        step=1
    )
    
    # Train Model
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    st.write("### Model Performance")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write("Confusion Matrix:")
    st.write(pd.DataFrame(
        confusion_matrix(y_test, y_pred), 
        columns=["Predicted Non-Reservoir", "Predicted Reservoir"], 
        index=["Actual Non-Reservoir", "Actual Reservoir"]
    ))
    
    # st.write("### Classification Report")
    # st.text(classification_report(y_test, y_pred))
    
    # Feature Importance
    st.write("### Feature Importance")
    feature_importance = pd.DataFrame({
        "Feature": features, 
        "Importance": model.feature_importances_
    })
    feature_importance = feature_importance.sort_values(by="Importance", ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x=feature_importance["Importance"], y=feature_importance["Feature"], ax=ax)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Importance in Reservoir Classification")
    st.pyplot(fig)
    
    # Structured Well Log Data Visualization
    st.write("### Well Log Data Visualization")
    toggle_highlight = st.checkbox("Highlight Predicted Reservoir Zones")
    fig, axes = plt.subplots(1, len(features), figsize=(15, 8), sharey=True)
    
    # If there's only one feature, axes is a single object, not a list
    if len(features) == 1:
        axes = [axes]
    
    for i, feature in enumerate(features):
        axes[i].plot(df[feature], df["Depth"], label=feature, color=np.random.rand(3,))
        axes[i].set_xlabel(feature)
        axes[i].invert_yaxis()  # Ensure depth goes from shallow to deep
        axes[i].legend()
        if toggle_highlight:
            # Assuming 1 = reservoir, 0 = non-reservoir
            reservoir_zones = df[df[target] == 1]
            axes[i].fill_betweenx(
                reservoir_zones["Depth"], 
                df[feature].min(), 
                df[feature].max(), 
                color='yellow', 
                alpha=0.6, 
                zorder=2  # Bring highlight to front
            )
    st.pyplot(fig)
    
    # Make Predictions on New Data
    st.write("### Make Predictions on New Data")
    input_data = {
        feature: st.number_input(
            f"{feature}", 
            float(df[feature].min()), 
            float(df[feature].max()), 
            float(df[feature].mean())
        ) 
        for feature in features
    }
    
    if st.button("Predict Reservoir Potential"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        st.write(f"#### Predicted Class: **{prediction}**")
        st.write(f"(1 = reservoir, 0 = non-reservoir)")
        
        # Export Predictions
        output_df = input_df.copy()
        output_df["Predicted Class"] = prediction
        csv = output_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Prediction Results", 
            data=csv, 
            file_name="predictions.csv", 
            mime="text/csv"
        )
    
    # Interactive Decision Tree Visualization
    st.write("### Decision Tree Visualization")
    if st.checkbox("Show Tree Structure"):
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_tree(
            model.estimators_[0], 
            feature_names=features, 
            class_names=["Non-Reservoir", "Reservoir"], 
            filled=True, 
            ax=ax
        )
        st.pyplot(fig)
    
    if st.checkbox("Show Text Representation"):
        tree_rules = export_text(model.estimators_[0], feature_names=features)
        st.text(tree_rules)
