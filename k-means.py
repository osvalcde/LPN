import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Lista personalizada de stop words en español
spanish_stop_words = [
    'yo', 'tú', 'él', 'ella', 'nosotros', 'nosotras', 'vosotros', 'vosotras', 'ellos', 'ellas',
    'mi', 'mis', 'tu', 'tus', 'su', 'sus', 'nuestro', 'nuestra', 'nuestros', 'nuestras',
    'vuestro', 'vuestra', 'vuestros', 'vuestras', 'es', 'soy', 'eres', 'somos', 'son',
    'un', 'una', 'unos', 'unas', 'y', 'o', 'pero', 'porque', 'aunque', 'que', 'como', 'donde',
    'cuando', 'cuanto', 'cual', 'quien', 'de', 'del', 'al', 'lo', 'la', 'los', 'las', 'a', 'e', 'i', 'u',
    'por', 'para', 'con', 'sin', 'sobre', 'entre', 'hasta', 'si', 'no', 'ni', 'me', 'te', 'se', 'le', 'les',
    'esto', 'eso', 'aquello'
]

# Ruta de la carpeta
folder_path = 'data_stopwords'

# Leer y combinar archivos
all_files = [f for f in os.listdir(folder_path) if f.startswith('cleaned_no_stopwords_') and f.endswith('.csv')]
df_combined = pd.concat([pd.read_csv(os.path.join(folder_path, file)) for file in all_files], ignore_index=True)

# Asegurarnos de trabajar con la columna "text"
df_combined = df_combined[['text']]

# Manejo de datos no válidos
df_combined = df_combined.dropna(subset=['text'])  # Eliminar valores nulos
df_combined['text'] = df_combined['text'].astype(str).str.strip()  # Asegurar cadenas válidas
df_combined = df_combined[df_combined['text'] != '']  # Eliminar textos vacíos

# Vectorizar texto con TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words=spanish_stop_words)
X = vectorizer.fit_transform(df_combined['text'])

# Aplicar K-means para agrupar tweets
num_clusters = 5  # Número de clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# Asignar etiquetas a los datos originales
df_combined['cluster'] = kmeans.labels_

# Reducir dimensionalidad para visualización (PCA a 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

# Añadir las coordenadas PCA al DataFrame
df_combined['x'] = X_pca[:, 0]
df_combined['y'] = X_pca[:, 1]

# Visualizar los clusters
plt.figure(figsize=(10, 7))
for cluster in range(num_clusters):
    cluster_points = df_combined[df_combined['cluster'] == cluster]
    plt.scatter(cluster_points['x'], cluster_points['y'], label=f'Cluster {cluster}')

plt.title('Clusters de Tweets (K-means)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend()
plt.show()

