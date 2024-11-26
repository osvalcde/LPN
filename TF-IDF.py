import dask.dataframe as dd
from sklearn.feature_extraction.text import TfidfVectorizer

# Ruta de los archivos .csv en la carpeta 'data_stopwords'
ruta_archivos_csv = "data_stopwords/cleaned_no_stopwords_*.csv"

# Leer los archivos .csv con Dask
df = dd.read_csv(ruta_archivos_csv, assume_missing=True)

# Nombre de la columna de texto
columna_texto = 'text'

# Verificar si la columna existe
if columna_texto in df.columns:
    # Obtener todo el texto de la columna como una lista (convierte Dask a pandas para TF-IDF)
    textos = df[columna_texto].dropna().compute().tolist()

    # Inicializar el vectorizador TF-IDF
    vectorizer = TfidfVectorizer(max_features=500)  # Limita a 500 características más importantes

    # Transformar el texto en vectores TF-IDF
    tfidf_matrix = vectorizer.fit_transform(textos)

    # Obtener las palabras más importantes (vocabulario)
    palabras = vectorizer.get_feature_names_out()

    # Mostrar información básica
    print("Matriz TF-IDF:")
    print(tfidf_matrix.toarray())  # Imprime la matriz como un array
    print("\nPalabras importantes (vocabulario):")
    print(palabras)

    # Guardar la matriz TF-IDF como archivo CSV
    import pandas as pd
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=palabras)
    tfidf_df.to_csv("data_stopwords/tfidf_matrix.csv", index=False)
    print("Matriz TF-IDF guardada en 'data_stopwords/tfidf_matrix.csv'.")
else:
    print(f"La columna '{columna_texto}' no existe en los archivos de la carpeta 'data_stopwords'.")
