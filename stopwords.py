import dask.dataframe as dd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import os

# Descargar recursos necesarios de NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Definir stopwords en español
stop_words = set(stopwords.words('spanish'))

# Crear carpeta de salida
os.makedirs("data_stopwords", exist_ok=True)

# Ruta de los archivos tokenizados
ruta_archivos_csv = "data_tokenizado/tokenized_train_*.csv"

# Leer los archivos .csv con Dask
df = dd.read_csv(ruta_archivos_csv, assume_missing=True)

# Nombre de la columna de texto
columna_texto = 'text'

# Función para eliminar stopwords
def eliminar_stopwords(texto):
    if isinstance(texto, str):
        # Tokenizar texto
        tokens = word_tokenize(texto.lower())
        # Filtrar palabras que no son stopwords
        tokens_filtrados = [word for word in tokens if word not in stop_words]
        # Reconstruir texto limpio
        return " ".join(tokens_filtrados)
    return ""

# Verificar si la columna existe
if columna_texto in df.columns:
    # Aplicar la eliminación de stopwords
    df[columna_texto] = df[columna_texto].map_partitions(
        lambda partition: partition.dropna().apply(eliminar_stopwords)
    )
    
    # Guardar los datos procesados como CSV en la carpeta 'data_stopwords'
    df.to_csv("data_stopwords/cleaned_no_stopwords_*.csv", index=False, single_file=False)
    print("Stopwords eliminadas y archivos guardados en la carpeta 'data_stopwords/'.")
else:
    print(f"La columna '{columna_texto}' no existe en los archivos de la carpeta 'data_tokenizado'.")
