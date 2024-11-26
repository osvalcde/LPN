import dask.dataframe as dd
import re
import os

# Crear el directorio de salida 'data_limpio' si no existe
os.makedirs("data_limpio", exist_ok=True)

# Función para limpiar texto
def limpiar_texto(texto):
    if isinstance(texto, str):
        texto_limpio = re.sub(r"http\S+|@\S+|#\S+|[^a-zA-ZáéíóúñÁÉÍÓÚÑ ]", "", texto)
        texto_limpio = re.sub(r"\s+", " ", texto_limpio).strip()
        return texto_limpio
    return texto

# Leer archivos CSV y manejar errores
try:
    df = dd.read_csv(
        "output/*.csv",
        assume_missing=True,
        on_bad_lines="skip",  # Ignorar filas problemáticas
        engine="python",  # Motor tolerante
    )
except Exception as e:
    print(f"Error al leer los archivos CSV: {e}")
    exit()

# Nombre de la columna a trabajar
columna_texto = "text"  # Cambia si es necesario

# Verificar si la columna existe
if columna_texto in df.columns:
    # Aplicar la limpieza
    df[columna_texto] = df[columna_texto].map_partitions(
        lambda partition: partition.dropna().apply(limpiar_texto) if not partition.empty else partition
    )

    # Guardar el dataset limpio como CSV
    df.to_csv("data_limpio/cleaned_train_*.csv", index=False, single_file=False)
    print("Limpieza completada y archivos guardados en 'data_limpio/'.")
else:
    print(f"La columna '{columna_texto}' no existe en el dataset.")

