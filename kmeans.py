# kmeans.py

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def run_kmeans():
    df = pd.read_csv('Mall_Customers.csv')
    df.rename(columns={'Annual Income (k$)': 'Income', 'Spending Score (1-100)': 'Score'}, inplace=True)
    
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})  # Convert gender to numeric
    features = df[['Gender', 'Age', 'Income', 'Score']]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    return df, kmeans, scaler
