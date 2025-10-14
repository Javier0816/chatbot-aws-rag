import os
import json
from dotenv import load_dotenv
from typing import Dict
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# --- Imports RAG ---
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# --- Imports Telegram (MODERNIZADOS) ---
# CAMBIO CRÍTICO: Usaremos solo 'Bot' y el resto de telegram.ext se manejará internamente
from telegram import Bot
# La nueva forma de importar manejadores y el ApplicationBuilder:
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters 

# --- Imports para Servir HTML ---
from fastapi.staticfiles import StaticFiles 
from fastapi.responses import HTMLResponse


# --- 1. CONFIGURACIÓN INICIAL Y CLAVES ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
telegram_token = os.getenv("TELEGRAM_TOKEN")

# Asegúrate de que todas las claves de AWS y OpenAI estén en Render
CHROMA_DB_PATH = "./chroma_db" 
qa_chain = None 
telegram_app = None # Usaremos una instancia de Application para manejar el bot

print("Inicializando el chatbot...")

# --- 2. LÓGICA RAG: CARGA DE CHROMA DB ---
try:
    if os.path.exists(CHROMA_DB_PATH):
        print("Base de datos persistida encontrada. Cargando...")
        # NOTA: Asegúrate de que las variables de AWS S3 si aún las tienes, no causen problemas aquí
        embeddings = OpenAIEmbeddings(api_key=api_key)
        vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
        
        # CONFIGURACIÓN DEL LLM Y LA CADENA RAG
        llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key, temperature=0.5)
        retriever = vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
        print("Cadena RAG inicializada.")
    
    else:
        print("ADVERTENCIA: Base de datos no encontrada. El chatbot no funcionará.")
        qa_chain = None

except Exception as e:
    print(f"Error fatal durante la inicialización de RAG: {e}")
    qa_chain = None 


# --- 3. CONFIGURACIÓN DE LA API CON FASTAPI ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 4. CONFIGURACIÓN DE TELEGRAM (WEBHOOKS) ---

# Los handlers de Telegram ahora reciben un argumento 'update' y 'context'
async def start_handler(update: object, context: object):
    """Maneja el comando /start"""
    if update.message:
        await update.message.reply_text("¡Hola! Soy tu Chatbot RAG empresarial. Envíame una pregunta para comenzar.")

async def message_handler(update: object, context: object):
    """Maneja los mensajes de texto y usa el RAG para responder"""
    if not qa_chain:
        await update.message.reply_text("Error: El sistema RAG no está inicializado.")
        return

    if update.message and update.message.text:
        query = update.message.text
        # Señal de "escribiendo..."
        # Usamos update.effective_chat para garantizar el ID del chat
        if update.effective_chat:
             await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
        
        print(f"Pregunta de Telegram recibida: {query}")
        
        try:
            response = qa_chain.invoke({'query': query})
            final_response = response.get('result', 'Lo siento, no pude encontrar una respuesta relevante.')
            await update.message.reply_text(final_response)
            
        except Exception as e:
            print(f"Error al procesar la consulta RAG: {e}")
            await update.message.reply_text("Lo siento, hubo un error interno al buscar la información.")


@app.on_event("startup")
async def startup_event():
    """Inicializa el bot y el application al iniciar FastAPI"""
    global telegram_app
    if not telegram_token:
        print("ERROR: TELEGRAM_TOKEN no está configurado. La funcionalidad de Telegram estará deshabilitada.")
        return

    print("Inicializando Telegram Bot Application...")
    
    # NUEVA SINTAXIS: Usamos ApplicationBuilder para crear la instancia del bot.
    telegram_app = ApplicationBuilder().token(telegram_token).build()
    
    # AÑADIR HANDLERS
    telegram_app.add_handler(CommandHandler("start", start_handler))
    telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))
    
    # Configura el webhook de Telegram
    webhook_url = os.getenv("RENDER_EXTERNAL_URL") or "https://chatbot-aws-rag-2.onrender.com/"
    
    full_webhook_url = f"{webhook_url}telegram"
    # Esto es crucial: asegura que la URL de Render sea la que recibe los mensajes
    await telegram_app.bot.set_webhook(full_webhook_url) 
    print(f"Webhook de Telegram configurado en: {full_webhook_url}")
    


@app.post("/telegram")
async def telegram_webhook(request: Request):
    """Endpoint que recibe todas las actualizaciones de Telegram"""
    if not telegram_app:
        return {"status": "error", "message": "Telegram Application no inicializada"}

    body = await request.json()
    
    # Procesar la actualización con el Application
    update = telegram_app.update_class.de_json(body, telegram_app.bot)
    await telegram_app.process_update(update)
    
    # Responde 200 OK inmediatamente
    return {"status": "ok"}


# --- 5. ENDPOINT PARA LA INTERFAZ WEB (HTML) ---
class Message(BaseModel):
    query: str

@app.post("/chat")
async def get_chatbot_response(message: Message):
    """Maneja las peticiones POST de la interfaz web HTML."""
    if not qa_chain:
        return {"response": "Error interno del servidor: El sistema RAG no se pudo inicializar."}
    
    try:
        response = qa_chain.invoke({'query': message.query})
        return {"response": response['result']}
    except Exception as e:
        print(f"Error al procesar la consulta: {e}")
        return {"response": "Lo siento, hubo un error al procesar tu solicitud."}


# --- 6. SERVIR ARCHIVOS ESTÁTICOS (DEBE IR AL FINAL) ---
if not os.path.exists("static"):
    os.makedirs("static")
    
app.mount("/", StaticFiles(directory="static", html=True), name="static")