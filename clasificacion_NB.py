import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Función para verificar si un valor es convertible a float (opcional en caso de duda)
def es_convertible(valor):
    try:
        float(valor)
        return True
    except ValueError:
        return False

# Ruta de la carpeta
folder_path = 'data_stopwords'

# Leer y combinar archivos
all_files = [f for f in os.listdir(folder_path) if f.startswith('cleaned_no_stopwords_') and f.endswith('.csv')]
df_combined = pd.concat([pd.read_csv(os.path.join(folder_path, file)) for file in all_files], ignore_index=True)

# Asegurarnos de trabajar con las columnas relevantes
df_combined = df_combined[['text']]

# Manejo de datos no válidos
# Eliminar valores nulos en la columna "text"
df_combined = df_combined.dropna(subset=['text'])

# Convertir todo a cadenas y eliminar filas vacías
df_combined['text'] = df_combined['text'].astype(str).str.strip()
df_combined = df_combined[df_combined['text'] != '']

# Crear etiquetas de ejemplo (simulación manual para este caso)
import random
categories = ['deporte', 'política', 'entretenimiento', 'tecnología', 'salud']
df_combined['label'] = [random.choice(categories) for _ in range(len(df_combined))]

# Separar texto y etiquetas
X = df_combined['text']
y = df_combined['label']

# Verificar e ignorar filas no válidas (opcional, si hay dudas de valores problemáticos)
X = X[X.apply(lambda texto: isinstance(texto, str))]

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizar texto
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Entrenar un modelo simple (Naive Bayes)
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predicciones
y_pred = model.predict(X_test_vec)

# Evaluar el modelo
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred))

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred, labels=categories)
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=categories, yticklabels=categories, cmap='Blues')
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Verdad")
plt.show()

# Distribución de categorías
category_counts = df_combined['label'].value_counts()
plt.figure(figsize=(8, 5))
category_counts.plot(kind='bar', color='skyblue')
plt.title("Distribución de Categorías")
plt.xlabel("Categorías")
plt.ylabel("Frecuencia")
plt.xticks(rotation=45)
plt.show()
