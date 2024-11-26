import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Leer la matriz TF-IDF generada
ruta_tfidf_csv = "data_stopwords/tfidf_matrix.csv"
tfidf_df = pd.read_csv(ruta_tfidf_csv)

# Filtrar palabras con más de 2 caracteres
# El índice de tfidf_df.columns contiene las palabras
vocabulario = [word for word in tfidf_df.columns if len(word) >= 3]

# Crear un nuevo DataFrame con las palabras filtradas
tfidf_df_filtrado = tfidf_df[vocabulario]

# Calcular la importancia promedio de cada palabra
importancia_promedio = tfidf_df_filtrado.mean(axis=0).sort_values(ascending=False)

# Mostrar las 10 palabras más importantes en un gráfico de barras
plt.figure(figsize=(10, 6))
importancia_promedio.head(10).plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Top 10 Palabras Más Importantes (TF-IDF)", fontsize=16)
plt.xlabel("Palabras", fontsize=12)
plt.ylabel("Importancia Promedio", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Crear una nube de palabras con las palabras más importantes
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    colormap='viridis',
).generate_from_frequencies(importancia_promedio)

# Mostrar la nube de palabras
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Nube de Palabras Basada en TF-IDF", fontsize=16)
plt.tight_layout()
plt.show()
