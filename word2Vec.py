import pandas as pd
from gensim.models import FastText
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Leer el archivo cargado
ruta_csv = "/mnt/data/cleaned_no_stopwords_00.csv"
df = pd.read_csv(ruta_csv)

# Asegurarse de que la columna 'text' exista
columna_texto = 'text'
if columna_texto not in df.columns:
    raise ValueError(f"La columna '{columna_texto}' no existe en el archivo.")

# Tokenización: Dividir los textos en listas de palabras
df['tokens'] = df[columna_texto].dropna().apply(lambda x: x.split())

# Entrenar el modelo FastText
model = FastText(sentences=df['tokens'], vector_size=100, window=5, min_count=2, workers=4, sg=1)

# Seleccionar palabras más frecuentes para visualizar
palabras_frecuentes = list(model.wv.index_to_key[:100])  # Top 100 palabras más frecuentes
vectores = [model.wv[word] for word in palabras_frecuentes]

# Reducir la dimensionalidad con t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
vectores_tsne = tsne.fit_transform(vectores)

# Graficar las palabras
plt.figure(figsize=(12, 8))
for i, palabra in enumerate(palabras_frecuentes):
    x, y = vectores_tsne[i, :]
    plt.scatter(x, y, marker='o', color='blue')
    plt.text(x + 0.02, y + 0.02, palabra, fontsize=9)
plt.title("Visualización t-SNE de Representaciones FastText", fontsize=16)
plt.show()

