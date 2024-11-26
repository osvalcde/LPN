import dask.dataframe as dd
from pysentimiento import create_analyzer
import matplotlib.pyplot as plt

# Inicializar el analizador de sentimientos
sentiment_analyzer = create_analyzer(task="sentiment", lang="es")

# Ruta de los archivos .csv en la carpeta 'data_stopwords'
ruta_archivos_csv = "data_stopwords/cleaned_no_stopwords_*.csv"

# Leer los archivos .csv con Dask
df = dd.read_csv(ruta_archivos_csv, assume_missing=True)

# Nombre de la columna de texto
columna_texto = 'text'

# Función para identificar el sentimiento
def identificar_sentimiento(texto):
    if isinstance(texto, str) and texto.strip():  # Verificar que no esté vacío
        resultado = sentiment_analyzer.predict(texto)
        return resultado.output  # Devuelve 'POS', 'NEG' o 'NEU'
    return "NEU"  # Asumir neutral si el texto está vacío o no es válido

# Verificar si la columna existe
if columna_texto in df.columns:
    # Aplicar la función a la columna de texto
    df['sentimiento'] = df[columna_texto].map_partitions(
        lambda partition: partition.dropna().apply(identificar_sentimiento)
    )
    
    # Computar los resultados para análisis
    resultados = df['sentimiento'].compute()

    # Calcular los porcentajes de cada tipo de sentimiento
    porcentajes = resultados.value_counts(normalize=True) * 100

    # Crear el gráfico de barras
    plt.figure(figsize=(8, 6))
    porcentajes.plot(kind='bar', color=['green', 'red', 'blue'], edgecolor='black')
    plt.title("Distribución de Opiniones en los Tweets", fontsize=16)
    plt.xlabel("Opinión", fontsize=12)
    plt.ylabel("Porcentaje (%)", fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
else:
    print(f"La columna '{columna_texto}' no existe en los archivos de la carpeta 'data_stopwords'.")
