import streamlit as st
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Page Configuration
st.set_page_config(
    page_title="💳 Credit Card Customer Segmentation",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #2c3e50;
            margin-top: 1rem;
            margin-bottom: 1rem;
            border-bottom: 2px solid #1f77b4;
            padding-bottom: 0.5rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
        }
        .info-box {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #1f77b4;
            margin: 1rem 0;
        }
        .stAlert {
            border-radius: 10px;
        }
        div[data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: bold;
        }
        .css-1r6slb0 {
            background-color: #f8f9fa;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: 4rem;
            padding: 0 2rem;
            background-color: #f0f2f6;
            border-radius: 10px 10px 0 0;
            font-weight: bold;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1f77b4;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Title Section
st.markdown('<p class="main-header">💳 Credit Card Customer Segmentation</p>', unsafe_allow_html=True)
st.markdown("---")

# File Upload Section
uploaded_file = st.file_uploader(
    "📁 Upload Customer Data CSV File",
    type=["csv"],
    help="Upload a CSV file containing customer credit card data"
)

if uploaded_file is not None:
    # Load Data
    df = pd.read_csv(uploaded_file)
    
    # Success Message
    st.success("✅ Dataset Uploaded Successfully!")
    st.balloons()
    
    # Create Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dataset Overview",
        "🔧 Preprocessing",
        "🎯 Credit Prediction",
        "🔮 Clustering Analysis",
        "📈 Visualizations"
    ])

    # ==========================
    # TAB 1 - DATASET OVERVIEW
    # ==========================
    with tab1:
        st.markdown('<p class="sub-header">📊 Dataset Overview</p>', unsafe_allow_html=True)
        
        # Dataset Info Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📋 Total Rows", df.shape[0])
        with col2:
            st.metric("📊 Total Columns", df.shape[1])
        with col3:
            st.metric("🔢 Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
        with col4:
            st.metric("📝 Categorical Columns", len(df.select_dtypes(include=['object']).columns))
        
        st.markdown("---")
        
        # Data Preview
        st.markdown("#### 📄 Data Preview (First 10 Rows)")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Column Information
        with st.expander("ℹ️ Column Information", expanded=False):
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.values,
                'Non-Null Count': df.notna().sum().values,
                'Null Count': df.isnull().sum().values,
                'Null %': (df.isnull().sum() / len(df) * 100).round(2).values
            })
            st.dataframe(col_info, use_container_width=True)
        
        # Missing Values Visualization
        st.markdown("#### 🔍 Missing Values Analysis")
        missing_df = pd.DataFrame(df.isnull().sum(), columns=["Missing Values"])
        missing_df = missing_df[missing_df["Missing Values"] > 0]
        
        if not missing_df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            missing_df.plot(kind='barh', ax=ax, color='#ff6b6b')
            ax.set_title('Missing Values by Column', fontsize=16, fontweight='bold')
            ax.set_xlabel('Number of Missing Values')
            st.pyplot(fig)
        else:
            st.success("🎉 No missing values found in the dataset!")

    # ==========================
    # TAB 2 - PREPROCESSING
    # ==========================
    with tab2:
        st.markdown('<p class="sub-header">🔧 Data Preprocessing</p>', unsafe_allow_html=True)
        
        data = df.copy()
        
        # Remove CUST_ID if exists
        if "CUST_ID" in data.columns:
            data = data.drop("CUST_ID", axis=1)
            st.info("🗑️ Removed 'CUST_ID' column (identifier column)")
        
        # Handle Missing Values
        with st.spinner("Processing missing values..."):
            imputer = SimpleImputer(strategy="mean")
            data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
            st.success("✅ Missing values handled using mean imputation")
        
        # Standardize Data
        with st.spinner("Standardizing data..."):
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data_imputed)
            st.success("✅ Data standardized successfully")
        
        # Display Preprocessing Results
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📊 Before Standardization")
            st.dataframe(data.head(), use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 After Standardization")
            st.dataframe(pd.DataFrame(scaled_data, columns=data.columns).head(), use_container_width=True)
        
        # Summary Statistics
        with st.expander("📈 Summary Statistics After Preprocessing", expanded=False):
            st.dataframe(pd.DataFrame(scaled_data, columns=data.columns).describe(), use_container_width=True)

    # ==========================
    # TAB 3 - CREDIT PREDICTION
    # ==========================
    with tab3:
        st.markdown('<p class="sub-header">🎯 Credit Limit Prediction</p>', unsafe_allow_html=True)
        
        if "CREDIT_LIMIT" not in df.columns:
            st.error("❌ CREDIT_LIMIT column not found in the dataset!")
        else:
            st.info("💡 Predicting whether customers have above or below median credit limit")
            
            # Prepare Data
            X_pred = df.drop(columns=["CUST_ID", "CREDIT_LIMIT"], errors="ignore")
            y_pred = df["CREDIT_LIMIT"]
            
            # Handle Missing Values
            imputer = SimpleImputer(strategy="median")
            X_pred = pd.DataFrame(imputer.fit_transform(X_pred), columns=X_pred.columns)
            
            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(
                X_pred, y_pred, test_size=0.2, random_state=42
            )
            
            # Scale Data
            scaler_pred = StandardScaler()
            X_train_scaled = scaler_pred.fit_transform(X_train)
            X_test_scaled = scaler_pred.transform(X_test)
            
            # Binary Classification Target
            median_limit = y_train.median()
            y_train_binary = (y_train > median_limit).astype(int)
            y_test_binary = (y_test > median_limit).astype(int)
            
            # Train Model
            with st.spinner("Training Logistic Regression model..."):
                model = LogisticRegression(max_iter=2000, random_state=42)
                model.fit(X_train_scaled, y_train_binary)
                y_pred_binary = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_test_binary, y_pred_binary)
            roc_auc = roc_auc_score(y_test_binary, y_pred_proba)
            
            # Display Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎯 Accuracy", f"{accuracy:.2%}", delta="Model Performance")
            with col2:
                st.metric("📊 ROC AUC", f"{roc_auc:.2%}", delta="Discrimination Power")
            with col3:
                st.metric("📈 Median Limit", f"${median_limit:,.2f}", delta="Threshold")
            
            st.markdown("---")
            
            # Confusion Matrix
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🔢 Confusion Matrix")
                cm = confusion_matrix(y_test_binary, y_pred_binary)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, 
                           xticklabels=['Below Median', 'Above Median'],
                           yticklabels=['Below Median', 'Above Median'])
                ax.set_xlabel('Predicted', fontsize=12)
                ax.set_ylabel('Actual', fontsize=12)
                ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
                st.pyplot(fig)
            
            with col2:
                st.markdown("#### 📊 Classification Report")
                report = classification_report(y_test_binary, y_pred_binary, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.background_gradient(cmap='Blues'), use_container_width=True)
            
            # ROC Curve
            st.markdown("#### 📈 ROC Curve")
            fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

    # ==========================
    # TAB 4 - CLUSTERING ANALYSIS
    # ==========================
    with tab4:
        st.markdown('<p class="sub-header">🔮 K-Means Clustering Analysis</p>', unsafe_allow_html=True)
        
        # Cluster Selection
        col1, col2 = st.columns([1, 2])
        with col1:
            n_clusters = st.slider(
                "🎯 Number of Clusters",
                min_value=2,
                max_value=10,
                value=4,
                help="Select the number of customer segments"
            )
        
        # Perform Clustering
        with st.spinner(f"Performing K-Means clustering with {n_clusters} clusters..."):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(scaled_data)
            result = df.copy()
            result["Cluster"] = clusters
        
        # Silhouette Score
        silhouette_avg = silhouette_score(scaled_data, clusters)
        st.metric("📊 Silhouette Score", f"{silhouette_avg:.3f}", 
                 help="Measures how similar points are to their own cluster compared to other clusters")
        
        st.markdown("---")
        
        # Cluster Distribution
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📊 Cluster Distribution")
            cluster_counts = result["Cluster"].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(cluster_counts.index, cluster_counts.values, color=plt.cm.viridis(np.linspace(0, 1, len(cluster_counts))))
            ax.set_xlabel('Cluster', fontsize=12)
            ax.set_ylabel('Number of Customers', fontsize=12)
            ax.set_title('Customer Distribution by Cluster', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            # Add value labels on bars
            for bar, val in zip(bars, cluster_counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(val), 
                       ha='center', va='bottom', fontweight='bold')
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### 📋 Cluster Summary Statistics")
            cluster_summary = result.groupby("Cluster").mean(numeric_only=True)
            st.dataframe(cluster_summary.style.background_gradient(cmap='Blues'), use_container_width=True)
        
        # Cluster Details
        st.markdown("#### 📋 Customer Data with Cluster Labels")
        st.dataframe(result.head(10), use_container_width=True)
        
        # Download Clustered Data
        csv = result.to_csv(index=False)
        st.download_button(
            label="📥 Download Clustered Data",
            data=csv,
            file_name=f"clustered_customers_{n_clusters}_clusters.csv",
            mime="text/csv",
            use_container_width=True
        )

    # ==========================
    # TAB 5 - VISUALIZATIONS
    # ==========================
    with tab5:
        st.markdown('<p class="sub-header">📈 Interactive Visualizations</p>', unsafe_allow_html=True)
        
        # PCA Visualization
        st.markdown("#### 🎯 PCA - Customer Segmentation Visualization")
        
        with st.spinner("Computing PCA..."):
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(scaled_data)
            pca_df = pd.DataFrame(pca_data, columns=["PCA1", "PCA2"])
            
            model = KMeans(n_clusters=4, random_state=42, n_init=10)
            pca_df["Cluster"] = model.fit_predict(scaled_data)
            pca_df["Cluster"] = pca_df["Cluster"].astype(str)
        
        # Explained Variance
        explained_variance = pca.explained_variance_ratio_
        st.info(f"📊 PCA1 explains {explained_variance[0]:.2%} variance, PCA2 explains {explained_variance[1]:.2%} variance")
        
        # Get column names for hover data (excluding CUST_ID if it exists)
        hover_cols = []
        for col in df.columns[:5]:
            if col != 'CUST_ID' and col in df.columns:
                hover_cols.append(col)
        
        # If no columns left, use the first available columns
        if not hover_cols:
            hover_cols = list(df.columns[:3])
        
        # Interactive Scatter Plot
        fig = px.scatter(
            pca_df,
            x="PCA1",
            y="PCA2",
            color="Cluster",
            title="<b>Customer Segments Visualization</b>",
            labels={
                "PCA1": f"Principal Component 1 ({explained_variance[0]:.2%})", 
                "PCA2": f"Principal Component 2 ({explained_variance[1]:.2%})"
            },
            color_discrete_sequence=px.colors.qualitative.Set2,
            hover_data=pca_df.columns  # Use only columns from pca_df
        )
        fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
        fig.update_layout(
            height=600,
            showlegend=True,
            legend_title_text='Cluster',
            hovermode='closest'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Correlation Heatmap
        st.markdown("#### 🔥 Correlation Heatmap")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(
                data_imputed.corr(),
                cmap="coolwarm",
                ax=ax,
                annot=True,
                fmt=".2f",
                linewidths=0.5,
                cbar_kws={"shrink": 0.8}
            )
            ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### 📊 Variable Distribution")
            # Select variable for distribution
            selected_col = st.selectbox("Select variable to visualize", data_imputed.columns)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(data_imputed[selected_col], bins=30, edgecolor='black', alpha=0.7, color='#1f77b4')
            ax.axvline(data_imputed[selected_col].mean(), color='red', linestyle='dashed', linewidth=2, label='Mean')
            ax.axvline(data_imputed[selected_col].median(), color='green', linestyle='dashed', linewidth=2, label='Median')
            ax.set_xlabel(selected_col, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'Distribution of {selected_col}', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

else:
    # Welcome Message
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; margin: 2rem 0;">
        <h2 style="color: white; font-size: 2.5rem;">🚀 Welcome to Credit Card Customer Segmentation</h2>
        <p style="color: white; font-size: 1.2rem; margin-top: 1rem;">
            Upload your customer data to unlock powerful insights through machine learning
        </p>
        <p style="color: white; opacity: 0.8; margin-top: 0.5rem;">
            📊 Analyze customer segments | 🎯 Predict credit limits | 📈 Visualize patterns
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features Grid
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background: #f0f2f6; padding: 1.5rem; border-radius: 10px; text-align: center; height: 200px;">
            <h3 style="font-size: 2rem;">📊</h3>
            <h4>Dataset Overview</h4>
            <p style="color: #666;">View and analyze your customer dataset structure</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #f0f2f6; padding: 1.5rem; border-radius: 10px; text-align: center; height: 200px;">
            <h3 style="font-size: 2rem;">🎯</h3>
            <h4>Credit Prediction</h4>
            <p style="color: #666;">Predict customer credit limit categories</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: #f0f2f6; padding: 1.5rem; border-radius: 10px; text-align: center; height: 200px;">
            <h3 style="font-size: 2rem;">🔮</h3>
            <h4>Clustering Analysis</h4>
            <p style="color: #666;">Segment customers using K-Means clustering</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>💡 <strong>Supported Features:</strong> Missing value imputation, Data standardization, Logistic Regression, K-Means Clustering, PCA Visualization</p>
        <p style="font-size: 0.9rem;">📁 Upload your CSV file to get started</p>
    </div>
    """, unsafe_allow_html=True)