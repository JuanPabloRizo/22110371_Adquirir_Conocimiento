import streamlit as st
from chatbot import chat,chatbot_response, intents

# Título de la aplicación
st.title("Asistente Virtual")

# Inicialización de variables de sesión
if "messages" not in st.session_state:
    st.session_state.messages = []  # Lista para almacenar los mensajes del usuario y del asistente

if "first_message" not in st.session_state:
    st.session_state.first_message = True  # Indicador para manejar el primer mensaje del asistente

# Mostrar todos los mensajes previos
for message in st.session_state.messages:
    with st.chat_message(message["role"]):  # Mostrar mensajes del usuario y asistente
        st.markdown(message["content"])

# Manejar el primer mensaje del asistente
if st.session_state.first_message:
    with st.chat_message("assistant"):
        st.markdown("Hola, ¿cómo puedo ayudarte?")  # Mensaje inicial del asistente
    # Guardar este mensaje en la lista de mensajes
    st.session_state.messages.append({"role": "assistant", "content": "Hola, ¿cómo puedo ayudarte?"})
    st.session_state.first_message = False

# Capturar entrada del usuario
prompt = st.chat_input("Escribe tu mensaje aquí...")  # Captura la entrada del usuario
if prompt:
    # Mostrar el mensaje del usuario en la interfaz
    with st.chat_message("user"):
        st.markdown(prompt)
    # Guardar el mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Intentar predecir la intención del usuario y obtener la respuesta
    try:
        res = chatbot_response(prompt)
    except Exception as e:
        res = "Ups, ocurrió un error. Por favor, intenta de nuevo."  # Manejo de errores

    # Respuesta automática del asistente
    with st.chat_message("assistant"):
        st.markdown(res)
    # Guardar el mensaje del asistente
    st.session_state.messages.append({"role": "assistant", "content": res})
