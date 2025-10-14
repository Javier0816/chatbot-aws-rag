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

# --- Imports Telegram (NUEVOS) ---
from telegram import Bot, Update
from telegram.ext import Dispatcher, MessageHandler, filters

# --- Imports para Servir HTML ---
from fastapi.staticfiles import StaticFiles 
from fastapi.responses import HTMLResponse


# --- 1. CONFIGURACIÓN INICIAL Y CLAVES ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
telegram_token = os.getenv("TELEGRAM_TOKEN") # ¡Nueva variable!

# Asegúrate de que todas las claves de AWS y OpenAI estén en Render
CHROMA_DB_PATH = "./chroma_db" 
qa_chain = None 
bot = None
dispatcher = None

print("Inicializando el chatbot...")

# --- 2. LÓGICA RAG: CARGA DE CHROMA DB ---
try:
    if os.path.exists(CHROMA_DB_PATH):
        print("Base de datos persistida encontrada. Cargando...")
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

async def start_handler(update, context):
    """Maneja el comando /start"""
    if update.message:
        await update.message.reply_text("¡Hola! Soy tu Chatbot RAG empresarial. Envíame una pregunta para comenzar.")

async def message_handler(update, context):
    """Maneja los mensajes de texto y usa el RAG para responder"""
    if not qa_chain:
        await update.message.reply_text("Error: El sistema RAG no está inicializado.")
        return

    if update.message and update.message.text:
        query = update.message.text
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


@app.on_event("startup")
async def startup_event():
    """Inicializa el bot y el dispatcher al iniciar FastAPI"""
    global bot, dispatcher
    if not telegram_token:
        print("ERROR: TELEGRAM_TOKEN no está configurado.")
        return

    print("Inicializando Telegram Bot...")
    bot = Bot(telegram_token)
    
    dispatcher = Dispatcher(bot, update_queue=None, use_context=True)
    dispatcher.add_handler(MessageHandler(filters.COMMAND, start_handler))
    dispatcher.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))
    
    # Configura el webhook de Telegram
    webhook_url = os.getenv("RENDER_EXTERNAL_URL")
    if webhook_url:
        full_webhook_url = f"{webhook_url}telegram"
        # Esto es crucial: asegura que la URL de Render sea la que recibe los mensajes
        await bot.set_webhook(full_webhook_url) 
        print(f"Webhook de Telegram configurado en: {full_webhook_url}")
    else:
        print("ADVERTENCIA: RENDER_EXTERNAL_URL no está disponible. El webhook NO se configuró.")


@app.post("/telegram")
async def telegram_webhook(request: Request):
    """Endpoint que recibe todas las actualizaciones de Telegram"""
    if not dispatcher:
        return {"status": "error", "message": "Dispatcher no inicializado"}

    body = await request.json()
    update = Update.de_json(body, bot)
    dispatcher.process_update(update)
    
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
# Montamos la carpeta 'static' en la raíz (/). 
# Solo se usará si el servidor no encuentra las rutas /chat o /telegram.
if not os.path.exists("static"):
    os.makedirs("static")
    
app.mount("/", StaticFiles(directory="static", html=True), name="static")
