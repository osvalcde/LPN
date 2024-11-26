#import pysentimiento
#print(pysentimiento.__version__)

from pysentimiento import create_analyzer

# Inicializar el analizador de emociones
emotion_analyzer = create_analyzer(task="emotion", lang="es")

# Probar con un texto de ejemplo
texto = "Estoy muy feliz hoy, es un gran día"
resultado = emotion_analyzer.predict(texto)

print("Texto:", texto)
print("Emoción Predicha:", resultado.output)
print("Probabilidades:", resultado.probas)


