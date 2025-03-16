import random
import json
import pickle
import numpy as np
import nltk
import sqlite3
from difflib import get_close_matches
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Inicialización de lematizador
lemmatizer = WordNetLemmatizer()

# Cargar los datos de intentos y vocabulario
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Conectar a la base de datos SQLite
db = sqlite3.connect('chatbot_knowledge.db')
cursor = db.cursor()

# Crear tabla si no existe
cursor.execute("""
CREATE TABLE IF NOT EXISTS knowledge (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT UNIQUE,
    answer TEXT
)
""")
db.commit()

# Funciones del chatbot (predicción de intención)
def clean_up_sentence(sentence):
    """Convierte la oración a una lista de palabras lematizadas"""
    sentence_words = nltk.word_tokenize(sentence)  # Tokeniza la oración
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]  # Lematiza las palabras
    return sentence_words

def bag_of_words(sentence):
    """Convierte la oración en una bolsa de palabras binaria (1 o 0 para cada palabra en el vocabulario)"""
    sentence_words = clean_up_sentence(sentence)  # Limpia la oración
    bag = [0] * len(words)  # Inicializa la bolsa de palabras con ceros
    for w in sentence_words:  # Recorre las palabras de la oración
        for i, word in enumerate(words):  # Compara con el vocabulario
            if word == w:
                bag[i] = 1  # Marca 1 si la palabra está presente
    return np.array(bag)  # Devuelve la bolsa de palabras como un arreglo numpy

def predict_class(sentence):
    """Predice la clase de la intención de la oración dada"""
    bow = bag_of_words(sentence)  # Crea una bolsa de palabras a partir de la entrada del usuario
    res = model.predict(np.array([bow]))[0]  # Predice la clase de la entrada del usuario
    ERROR_THRESHOLD = 0.75  # Umbral de error (25%)
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]  # Filtra los resultados por el umbral
    results.sort(key=lambda x: x[1], reverse=True)  # Ordena los resultados por la probabilidad
    return_list = []  # Lista vacía para almacenar los resultados
    for r in results:  # Recorre los resultados
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})  # Añade la intención y probabilidad
    return return_list  # Devuelve la lista de intenciones y probabilidades

def get_response(intents_list):
    """Obtiene una respuesta basada en la intención predicha"""
    tag = intents_list[0]['intent']  # Obtiene la intención predicha
    list_of_intents = intents['intents']  # Obtiene la lista de intenciones definidas
    for i in list_of_intents:  # Busca la intención correspondiente
        if i['tag'] == tag:
            result = random.choice(i['responses'])  # Escoge una respuesta aleatoria de las opciones
            break
    return result  # Devuelve la respuesta

def learn_new_knowledge(user_input):
    """Permite que el chatbot aprenda nuevas respuestas"""
    cursor.execute("SELECT answer FROM knowledge WHERE question = ?", (user_input,))
    row = cursor.fetchone()
    if row:
        return f"Ya conozco la respuesta a '{user_input}': {row[0]}"
    
    new_answer = input(f"No sé cómo responder a '{user_input}'. ¿Qué debería decir en este caso?\n")
    cursor.execute("INSERT INTO knowledge (question, answer) VALUES (?, ?)", (user_input, new_answer))
    db.commit()
    return "¡Gracias! Ahora lo recordaré para la próxima."

def find_best_match(user_input):
    """Busca la mejor coincidencia de una pregunta similar en la base de datos"""
    all_questions = {row[0]: row[1] for row in cursor.execute("SELECT question, answer FROM knowledge")}
    best_match = get_close_matches(user_input, all_questions.keys(), n=1, cutoff=0.7)
    if best_match:
        return all_questions[best_match[0]]
    return None

def chatbot_response(user_input):
    """Función principal para obtener una respuesta del chatbot"""
    # Primero, tratamos de predecir la intención con el modelo de deep learning
    intents_list = predict_class(user_input)
    if intents_list:
        response = get_response(intents_list)  # Si el modelo tiene una respuesta, la devuelve
    else:
        # Si no se puede predecir la intención, buscamos en la base de datos
        response = find_best_match(user_input)
        if not response:
            response = learn_new_knowledge(user_input)  # Si no hay respuesta en la base de datos, aprender
    return response  # Devuelve la respuesta final

# Función para interactuar con el chatbot
def chat():
    print("Chatbot: ¡Hola! ¿En qué puedo ayudarte?")
    while True:
        user_input = input("Tú: ")
        if user_input.lower() in ["salir", "adiós", "bye"]:
            print("Chatbot: ¡Hasta luego!")
            break
        response = chatbot_response(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chat()
    db.close()  # Cerrar la conexión con la base de datos
