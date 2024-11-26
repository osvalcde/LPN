import dask.dataframe as dd
from nltk.tokenize import word_tokenize
import os
import nltk

# Descargar los recursos necesarios de NLTK
nltk.download('punkt')

# Ruta de los archivos .csv en la carpeta 'data_limpio'
ruta_archivos_csv = "data_limpio/*.csv"

# Crear carpeta de salida para los archivos tokenizados
os.makedirs("data_tokenizado", exist_ok=True)

# Leer los archivos .csv
df = dd.read_csv(ruta_archivos_csv, assume_missing=True)

# Nombre de la columna de texto
columna_texto = 'text'

# Función para tokenizar el texto
def tokenizar_texto(texto):
    if isinstance(texto, str):
        return " ".join(word_tokenize(texto))  # Concatenar tokens con espacio
    return ""

# Verificar si la columna existe
if columna_texto in df.columns:
    # Aplicar la tokenización a cada partición
    df[columna_texto] = df[columna_texto].map_partitions(
        lambda partition: partition.dropna().apply(tokenizar_texto)
    )

    # Guardar los datos tokenizados como archivos CSV
    df.to_csv("data_tokenizado/tokenized_train_*.csv", index=False, single_file=False)
    print("Tokenización completada y archivos guardados en la carpeta 'data_tokenizado/'.")
else:
    print(f"La columna '{columna_texto}' no existe en los archivos de la carpeta 'data_limpio'.")
