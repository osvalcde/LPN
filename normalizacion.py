import dask.dataframe as dd
from textblob import TextBlob  # Para corrección ortográfica (opcional)

# Función para normalizar texto
def normalizar_texto(texto, corregir_ortografia=False):
    if isinstance(texto, str):
        # Convertir a minúsculas
        texto = texto.lower()
        # Opcional: corrección ortográfica
        if corregir_ortografia:
            texto = str(TextBlob(texto).correct())
        return texto
    return texto

# Ruta de los archivos .parquet
ruta_archivos_parquet = "output/cleaned_train*.parquet"

# Leer múltiples archivos .parquet con Dask
df = dd.read_parquet(ruta_archivos_parquet)

# Nombre de la columna a trabajar
columna_texto = 'texto'

# Verificar si la columna existe
if columna_texto in df.columns:
    # Normalizar el texto (ajustar corregir_ortografia a True si lo deseas)
    corregir_ortografia = False  # Cambia a True para habilitar corrección ortográfica
    df[columna_texto] = df[columna_texto].map_partitions(
        lambda partition: partition.dropna().apply(normalizar_texto, args=(corregir_ortografia,))
    )

    # Guardar los archivos normalizados
    df.to_parquet("output/normalized_train_*.parquet", index=False)
    print("Normalización completada y archivos guardados en 'output/normalized_train_*.parquet'.")
else:
    print(f"La columna '{columna_texto}' no existe en los archivos en la ruta {ruta_archivos_parquet}.")
