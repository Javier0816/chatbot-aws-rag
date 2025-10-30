import os
import json
from dotenv import load_dotenv
from typing import Dict
from fastapi import FastAPI, Request, BackgroundTasks
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# --- Imports RAG ---
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# --- Imports Telegram ---
from telegram import Bot
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, Application

# --- Imports para Servir HTML ---
from fastapi.staticfiles import StaticFiles 
from fastapi.responses import HTMLResponse

# --- 1. CONFIGURACIÓN INICIAL Y CLAVES ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
telegram_token = os.getenv("TELEGRAM_TOKEN")

CHROMA_DB_PATH = "./chroma_db" 
qa_chain = None 
telegram_app: Application = None # Definimos el tipo como Application
global_bot: Bot = None # Almacenamos el objeto Bot para la activación

print("Inicializando el chatbot...")

# --- 2. LÓGICA RAG: CARGA DE CHROMA DB ---
try:
    if os.path.exists(CHROMA_DB_PATH):
        print("Base de datos persistida encontrada. Cargando...")
        embeddings = OpenAIEmbeddings(api_key=api_key)
        vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
        
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


# --- 4. CONFIGURACIÓN DE TELEGRAM ---

# Handlers (Sin cambios en la lógica, solo en la inicialización)
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
        if update.effective_chat:
             # Señal de "escribiendo..."
             await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
        
        print(f"Pregunta de Telegram recibida: {query}")
        
        try:
            response = qa_chain.invoke({'query': query})
            final_response = response.get('result', 'Lo siento, no pude encontrar una respuesta relevante.')
            await update.message.reply_text(final_response)
            
        except Exception as e:
            print(f"Error al procesar la consulta RAG: {e}")
            await update.message.reply_text("Lo siento, hubo un error interno al buscar la información.")


# --- Funciones y Endpoints de Telegram ---

async def handle_telegram_update(body: Dict):
    """
    Función que maneja la actualización de Telegram en segundo plano.
    Incluye toda la inicialización dentro de la tarea de fondo para asegurar
    que la Application tenga un estado fresco y válido, resolviendo el error.
    """
    if not telegram_token:
        print("Error: TELEGRAM_TOKEN no está disponible.")
        return

    # 1. INICIALIZACIÓN COMPLETA DENTRO DEL BACKGROUND TASK
    # Esto asegura que la App está activa cuando se llama a process_update
    try:
        app_builder = ApplicationBuilder().token(telegram_token)
        local_telegram_app = app_builder.build()
        
        # Agregar handlers
        local_telegram_app.add_handler(CommandHandler("start", start_handler))
        local_telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))

        # 2. Inicialización Asíncrona
        await local_telegram_app.initialize()

        # 3. Procesar la actualización
        await local_telegram_app.process_update(body)

    except Exception as e:
        print(f"CRITICAL ERROR - Fallo irrecuperable en background task: {e}")


@app.on_event("startup")
async def startup_event():
    """Inicializa la ApplicationBuilder y el objeto Bot."""
    global global_bot, telegram_app
    if not telegram_token:
        print("ERROR: TELEGRAM_TOKEN no está configurado.")
        return
    
    # Solo necesitamos construir el objeto Application una vez para fines de activación
    telegram_app = ApplicationBuilder().token(telegram_token).build()
    global_bot = telegram_app.bot
    
    print("Telegram Bot Application inicializada para activación.")


@app.post("/telegram")
async def telegram_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Endpoint que recibe todas las actualizaciones de Telegram.
    Usa BackgroundTasks para responder 200 OK inmediatamente.
    """
    # 1. Leer el cuerpo de la petición (el JSON de Telegram)
    body = await request.json()
    
    # 2. Agregar el procesamiento a las tareas de fondo.
    # El procesamiento completo del bot ocurre en la tarea de fondo
    background_tasks.add_task(handle_telegram_update, body)
    
    # 3. Respuesta 200 OK inmediata
    return {"status": "ok"}


@app.get("/set_telegram_webhook")
async def set_webhook():
    """Endpoint para configurar manualmente el webhook después de que el servidor esté LIVE."""
    if not telegram_token or not global_bot:
        return {"status": "error", "message": "Falta el token de Telegram o el bot no está inicializado."}
        
    try:
        webhook_url = os.getenv("RENDER_EXTERNAL_URL") or "https://chatbot-aws-rag-2.onrender.com/"
        full_webhook_url = f"{webhook_url}telegram"
        
        # Usamos el objeto global_bot para hacer la activación
        await global_bot.set_webhook(full_webhook_url, read_timeout=10, write_timeout=10) 
        
        return {
            "status": "success",
            "message": "Webhook de Telegram configurado exitosamente.",
            "url_configurada": full_webhook_url
        }
    except Exception as e:
        return {"status": "error", "message": f"Fallo al configurar el webhook: {e}"}


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