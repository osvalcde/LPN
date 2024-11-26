import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk

# Download the stop words for Spanish from NLTK
nltk.download('stopwords')

# Get the list of stop words in Spanish
spanish_stop_words = stopwords.words('spanish')

# Path to the folder
folder_path = 'data_stopwords'

# Read and combine files
all_files = [f for f in os.listdir(folder_path) if f.startswith('cleaned_no_stopwords_') and f.endswith('.csv')]
df_combined = pd.concat([pd.read_csv(os.path.join(folder_path, file)) for file in all_files], ignore_index=True)

# Ensure we work with the 'text' column
df_combined = df_combined[['text']]

# Handle invalid data
df_combined = df_combined.dropna(subset=['text'])  # Drop null values
df_combined['text'] = df_combined['text'].astype(str).str.strip()  # Ensure valid strings
df_combined = df_combined[df_combined['text'] != '']  # Drop empty texts

# Vectorize text using TF-IDF and NLTK stop words
vectorizer = TfidfVectorizer(max_features=5000, stop_words=spanish_stop_words)
X = vectorizer.fit_transform(df_combined['text'])

# Apply K-means to cluster tweets
num_clusters = 5  # Number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# Assign cluster labels to the original data
df_combined['cluster'] = kmeans.labels_

# Reduce dimensionality for visualization (PCA to 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

# Add PCA coordinates to the DataFrame
df_combined['x'] = X_pca[:, 0]
df_combined['y'] = X_pca[:, 1]

# Count the size of each cluster for bubble sizes
cluster_sizes = df_combined['cluster'].value_counts().sort_index()

# Bubble Chart
plt.figure(figsize=(12, 8))
for cluster in range(num_clusters):
    cluster_points = df_combined[df_combined['cluster'] == cluster]
    plt.scatter(
        cluster_points['x'],
        cluster_points['y'],
        s=cluster_sizes[cluster] * 10,  # Bubble size based on cluster size
        alpha=0.5,
        label=f'Cluster {cluster}'
    )

plt.title('Bubble Chart of Clusters (K-means)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()
