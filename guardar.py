import dask.dataframe as dd

# Definir rutas de entrada y salida
splits = {'train': 'data/train-*-of-*.parquet', 'test': 'data/test-*-of-*.parquet'}
dataset_path = "hf://datasets/pysentimiento/spanish-tweets-small/" + splits["test"]

# Leer el dataset Parquet usando Dask
df = dd.read_parquet(dataset_path)

# Guardar el dataset en formato CSV local
output_csv_path = "output/train_data_*.csv"  # Usa un patrón para múltiples archivos
df.to_csv(output_csv_path, index=False, single_file=False)

print(f"Dataset guardado como CSV en: {output_csv_path}")