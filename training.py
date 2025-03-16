import nltk
from nltk import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.optimizers.schedules import ExponentialDecay
import random

# Cargar archivo JSON
data_file = open('intents.json', 'r', encoding='utf-8').read()
intents = json.loads(data_file)
lemmatizer = WordNetLemmatizer()

# Inicializar listas para almacenar palabras, clases y documentos
words = []
classes = []
documents = []
ignore_words = ['?', '!']

# Procesar patrones y clases desde el archivo JSON
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)  # Divide cada frase en palabras
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Normalizar palabras (convertirlas a minúsculas y lematizarlas)
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))  # Eliminar duplicados y ordenar

# Guardar las palabras y clases en archivos .pkl
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Crear bolsas de palabras y salidas
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1  # Set the correct output
    training.append([bag, output_row])

# Mezclar los datos de entrenamiento
random.shuffle(training)

# Separar las características y las etiquetas
train_x = [row[0] for row in training]
train_y = [row[1] for row in training]

train_x = np.array(train_x)
train_y = np.array(train_y)

# Crear el modelo de la red neuronal
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Crear el schedule para la tasa de aprendizaje
lr_schedule = ExponentialDecay(
    initial_learning_rate=0.01,  # tasa de aprendizaje inicial
    decay_steps=10000,  # pasos para disminuir la tasa de aprendizaje
    decay_rate=0.9  # la tasa de disminución
)

# Crear el optimizador con el schedule
sgd = SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)

# Compilar el modelo con el optimizador
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Entrenar el modelo
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=8, verbose=1)

# Guardar el modelo entrenado
model.save('chatbot_model.h5')

print("Modelo creado correctamente")
