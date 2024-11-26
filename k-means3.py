import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
import seaborn as sns

# Descargar la lista de stop words en español de NLTK (si no lo has hecho ya)
nltk.download('stopwords')

# Obtener lista de palabras vacías en español
spanish_stop_words = stopwords.words('spanish')

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

# Vectorizar texto con TF-IDF usando stop words de NLTK
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

# Crear un mapa de calor para visualizar los clusters
plt.figure(figsize=(10, 8))
heatmap_data = pd.DataFrame({'x': X_pca[:, 0], 'y': X_pca[:, 1], 'cluster': kmeans.labels_})
heatmap_pivot = heatmap_data.pivot_table(index='y', columns='x', values='cluster', aggfunc='mean')

sns.heatmap(heatmap_pivot, cmap='coolwarm', cbar=True)
plt.title('Mapa de Calor de Clusters (K-means)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()
