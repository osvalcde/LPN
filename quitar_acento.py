import pandas as pd
import unicodedata

# Función para eliminar acentos
def eliminar_acentos(texto):
    """
    Elimina los acentos de un texto dado.
    """
    if isinstance(texto, str):
        texto = unicodedata.normalize('NFD', texto)
        texto = texto.encode('ascii', 'ignore').decode('utf-8')
        return texto
    return texto

# Cargar el archivo CSV
ruta_archivo = 'data_stopwords/cleaned_no_stopwords_01.csv'  # Cambiar por la ruta del archivo
df = pd.read_csv(ruta_archivo)

# Verificar que exista la columna 'text'
if 'text' in df.columns:
    # Limpiar la columna 'text'
    df['text'] = df['text'].apply(eliminar_acentos)

    # Guardar el archivo limpio
    ruta_guardado = 'data_stopwords/cleaned_no_stopwords_00_cleaned.csv'
    df.to_csv(ruta_guardado, index=False)
    print(f"Archivo procesado y guardado en: {ruta_guardado}")
else:
    print("La columna 'text' no existe en el archivo.")
