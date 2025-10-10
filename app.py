import os
from dotenv import load_dotenv
import json
import boto3

from langchain_community.document_loaders.s3_file import S3FileLoader

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
# Imports necesarios para servir archivos web
from fastapi.staticfiles import StaticFiles 
from fastapi.responses import HTMLResponse


# --- 1. CONFIGURACIÓN INICIAL Y CARGA DE CLAVES ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_bucket_name = os.getenv("AWS_BUCKET_NAME")

print("Inicializando el chatbot...")

# Ruta donde se guardará/cargará la base de datos vectorial
CHROMA_DB_PATH = "./chroma_db" 

# --- 2. LÓGICA RAG: CARGA O CREACIÓN DE CHROMA DB ---
try:
    if os.path.exists(CHROMA_DB_PATH):
        # A) CARGAR (Solución al error de memoria de Render)
        print("Base de datos persistida encontrada. Cargando...")
        embeddings = OpenAIEmbeddings(api_key=api_key)
        vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
        print("Base de datos cargada desde el disco.")
    
    else:
        # B) CREAR Y PERSISTIR (Solo se ejecuta si la carpeta chroma_db se pierde)
        print("Base de datos no encontrada. El servidor está configurado para cargar desde disco.")
        # Como no se espera que esto se ejecute en Render, asumimos None si falla la carga.
        vectorstore = None 

except Exception as e:
    print(f"Error fatal durante la inicialización de RAG/S3: {e}")
    vectorstore = None 

# --- 3. CONFIGURACIÓN DEL LLM Y LA CADENA RAG ---

if vectorstore:
    print("Configurando el LLM y la cadena RAG...")
    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key, temperature=0.5)
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    print("Chatbot inicializado. ¡Listo para recibir peticiones!")
else:
    qa_chain = None
    print("¡ADVERTENCIA! La cadena RAG no se pudo inicializar. Las peticiones fallarán.")


# --- 4. CONFIGURACIÓN DE LA API CON FASTAPI ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    query: str

# 4.1 ENDPOINT PARA EL CHAT (La funcionalidad principal)
@app.post("/chat")
async def get_chatbot_response(message: Message):
    if not qa_chain:
        return {"response": "Error interno del servidor: El sistema RAG no se pudo inicializar."}
    
    try:
        response = qa_chain.invoke({'query': message.query})
        return {"response": response['result']}
    except Exception as e:
        print(f"Error al procesar la consulta: {e}")
        return {"response": "Lo siento, hubo un error al procesar tu solicitud."}


# 4.2 SERVIR ARCHIVOS ESTÁTICOS (LA INTERFAZ WEB)
# Creamos el directorio 'static' si no existe.
# Es buena práctica poner los archivos web en una carpeta 'static'.
if not os.path.exists("static"):
    os.makedirs("static")

# Montamos la carpeta 'static' en la raíz. 
# Esto le dice a FastAPI que si alguien accede a la URL raíz (/), 
# debe buscar el archivo index.html dentro de 'static'.
app.mount("/", StaticFiles(directory="static", html=True), name="static")