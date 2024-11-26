from datasets import load_dataset, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import pandas as pd
import os

# Cargar los tweets
folder_path = 'data_stopwords'

# Leer y combinar archivos
all_files = [f for f in os.listdir(folder_path) if f.startswith('cleaned_no_stopwords_') and f.endswith('.csv')]
df_combined = pd.concat([pd.read_csv(os.path.join(folder_path, file)) for file in all_files], ignore_index=True)

# Asegurarnos de trabajar con la columna "text"
df_combined = df_combined[['text']]

# Manejar datos no válidos
df_combined = df_combined.dropna(subset=['text'])  # Eliminar valores nulos
df_combined['text'] = df_combined['text'].astype(str).str.strip()  # Convertir a texto
df_combined = df_combined[df_combined['text'] != '']  # Eliminar textos vacíos

# Preparar datos para Hugging Face
tweets = df_combined['text'].tolist()
dataset = Dataset.from_dict({"text": tweets})

# Tokenizador y modelo GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Asignar el eos_token como pad_token
tokenizer.pad_token = tokenizer.eos_token

# Tokenización de los tweets
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",  # Desactivar evaluación
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=1000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=500
)

# Entrenador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# Entrenar el modelo
trainer.train()

# Guardar el modelo afinado
model.save_pretrained("./tweet_generator_model")
tokenizer.save_pretrained("./tweet_generator_model")
