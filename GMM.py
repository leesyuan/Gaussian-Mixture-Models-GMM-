import streamlit as st
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Function to handle GMM and visualizations
def run_gmm(df, n_components, covariance_type, max_iter, random_state, use_bic):
    sales_columns = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
    
    # Preprocessing: Fill missing values and add derived features
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Year'].fillna(df['Year'].median(), inplace=True)
    df['Years_Since_Release'] = 2024 - df['Year']
    
    df['NA_Sales_Ratio'] = df['NA_Sales'] / df['Global_Sales']
    df['EU_Sales_Ratio'] = df['EU_Sales'] / df['Global_Sales']
    df['JP_Sales_Ratio'] = df['JP_Sales'] / df['Global_Sales']
    df['Other_Sales_Ratio'] = df['Other_Sales'] / df['Global_Sales']
    
    platform_sales_avg = df.groupby('Platform')['Global_Sales'].mean().to_dict()
    df['Platform_Avg_Sales'] = df['Platform'].map(platform_sales_avg)
    
    genre_sales_avg = df.groupby('Genre')['Global_Sales'].mean().to_dict()
    df['Genre_Avg_Sales'] = df['Genre'].map(genre_sales_avg)
    
    # Select features for clustering
    features = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales',
                'Years_Since_Release', 'NA_Sales_Ratio', 'EU_Sales_Ratio', 'JP_Sales_Ratio',
                'Platform_Avg_Sales', 'Genre_Avg_Sales']
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features].dropna())
    
    # GMM Model selection
    if use_bic:
        st.write("Optimizing GMM using BIC...")
    else:
        st.write("Optimizing GMM using AIC...")

    # Use BIC or AIC to find the best number of components
    best_score = np.inf
    best_gmm = None
    for n in range(1, n_components + 1):
        gmm = GaussianMixture(n_components=n, covariance_type=covariance_type, max_iter=max_iter, random_state=random_state)
        gmm.fit(X_scaled)
        score = gmm.bic(X_scaled) if use_bic else gmm.aic(X_scaled)
        if score < best_score:
            best_score = score
            best_gmm = gmm
    
    st.write(f'Best number of components based on {"BIC" if use_bic else "AIC"}: {best_gmm.n_components}')
    
    # Fit the best GMM model and predict clusters
    clusters = best_gmm.predict(X_scaled)
    df['Cluster'] = clusters
    
    # Analyze the mean of each feature by cluster
    cluster_summary = df.groupby('Cluster').mean(numeric_only=True)
    st.write("Cluster Summary:")
    st.write(cluster_summary)
    
    # Evaluate clustering performance
    silhouette_avg = silhouette_score(X_scaled, clusters)
    ch_score = calinski_harabasz_score(X_scaled, clusters)
    db_score = davies_bouldin_score(X_scaled, clusters)
    
    st.write(f'Silhouette Score: {silhouette_avg}')
    st.write(f'Calinski-Harabasz Index: {ch_score}')
    st.write(f'Davies-Bouldin Index: {db_score}')
    
    # PCA for 2D visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    st.write("PCA - GMM Clustering Results:")
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50)
    plt.colorbar(scatter, label='Cluster')
    plt.title('GMM Clusters Visualization (PCA)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    st.pyplot(fig)
    
    # t-SNE for 2D visualization
    tsne = TSNE(n_components=2, random_state=random_state)
    X_tsne = tsne.fit_transform(X_scaled)
    
    st.write("t-SNE - GMM Clustering Results:")
    fig, ax = plt.subplots()
    scatter_tsne = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters, cmap='viridis', s=50)
    plt.colorbar(scatter_tsne, label='Cluster')
    plt.title('GMM Clusters Visualization (t-SNE)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    st.pyplot(fig)

# Streamlit app layout
st.title('Gaussian Mixture Models (GMM) Clustering Analysis')

# Add a file uploader to allow users to upload CSV files
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Display the first few rows of the DataFrame
    st.write("Here are the first few rows of your file:")
    st.write(df.head())

    # User inputs for GMM parameters
    n_components = st.slider('Select the maximum number of components (clusters):', min_value=1, max_value=20, value=5)
    covariance_type = st.selectbox('Select covariance type:', ('full', 'tied', 'diag', 'spherical'))
    max_iter = st.slider('Maximum iterations for GMM:', min_value=100, max_value=500, value=200, step=50)
    random_state = st.number_input('Random state (for reproducibility):', min_value=0, value=42)
    use_bic = st.radio('Optimize using:', ('BIC', 'AIC')) == 'BIC'

    # Run GMM clustering and analysis with user-selected parameters
    run_gmm(df, n_components, covariance_type, max_iter, random_state, use_bic)
else:
    st.write("Please upload a CSV file to proceed.")
