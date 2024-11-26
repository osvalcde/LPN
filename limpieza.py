import dask.dataframe as dd
import re
import os

# Crear el directorio de salida si no existe
os.makedirs("output", exist_ok=True)

# Función de limpieza
def limpiar_texto(texto):
    if isinstance(texto, str):
        texto_limpio = re.sub(r"http\S+|@\S+|#\S+|[^a-zA-ZáéíóúñÁÉÍÓÚÑ ]", "", texto)
        texto_limpio = re.sub(r"\s+", " ", texto_limpio).strip()
        return texto_limpio
    return texto

# Leer el dataset
df = dd.read_csv("output/*.csv", assume_missing=True)

# Nombre de la columna de texto
columna_texto = 'text'  # Cambia por el nombre correcto

# Verificar si la columna existe
if columna_texto in df.columns:
    # Aplicar la limpieza
    df[columna_texto] = df[columna_texto].map_partitions(
        lambda partition: partition.dropna().apply(limpiar_texto) if not partition.empty else partition
    )

    # Guardar el dataset limpio
    df.to_parquet("output/cleaned_train.parquet", index=False)
    print("Limpieza completada y dataset guardado.")
else:
    print(f"La columna '{columna_texto}' no existe en el dataset.")

