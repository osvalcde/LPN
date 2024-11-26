import dask.dataframe as dd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Leer los archivos de la carpeta 'data_stopwords'
ruta_archivos_csv = "data_stopwords/cleaned_no_stopwords_*.csv"
df = dd.read_csv(ruta_archivos_csv, assume_missing=True)

# Combinar todos los textos de la columna 'text' en una sola cadena
columna_texto = 'text'

if columna_texto in df.columns:
    # Recopilar el texto completo
    texto_completo = " ".join(df[columna_texto].dropna().compute())

    # Crear la nube de palabras
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white', 
        colormap='viridis', 
        max_words=200
    ).generate(texto_completo)

    # Mostrar la nube de palabras
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Nube de Palabras", fontsize=20)
    plt.show()
else:
    print(f"La columna '{columna_texto}' no existe en los archivos de 'data_stopwords'.")
