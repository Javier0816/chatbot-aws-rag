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
    # Verificamos si la base de datos ya existe
    if os.path.exists(CHROMA_DB_PATH):
        # A) CARGAR (Solución al error de memoria de Render)
        print("Base de datos persistida encontrada. Cargando...")
        embeddings = OpenAIEmbeddings(api_key=api_key)
        # Cargamos la base de datos directamente del disco (rápido y usa poca RAM)
        vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
        print("Base de datos cargada desde el disco.")
    
    else:
        # B) CREAR Y PERSISTIR (Solo se ejecuta la primera vez o si no encuentra la carpeta)
        print("Base de datos no encontrada. Creando y persistiendo el vector store...")
        
        # 2.1. Carga de documentos desde S3
        documents = []
        s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
        objects = s3.list_objects_v2(Bucket=aws_bucket_name)

        print("Cargando documentos de S3...")
        for obj in objects.get('Contents', []):
            file_key = obj['Key']
            print(f"  -> Cargando archivo: {file_key}")
            loader = S3FileLoader(aws_bucket_name, file_key)
            docs = loader.load()
            documents.extend(docs)

        # 2.2. Dividir texto (Chunking)
        print(f"Dividiendo {len(documents)} documentos en chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        # 2.3. Generar embeddings y persistir
        print(f"Generando embeddings de {len(texts)} chunks y guardando en disco...")
        embeddings = OpenAIEmbeddings(api_key=api_key)
        
        # Creamos la base de datos y la guardamos en CHROMA_DB_PATH
        vectorstore = Chroma.from_documents(
            texts, 
            embeddings, 
            persist_directory=CHROMA_DB_PATH
        )
        vectorstore.persist() # Guarda todos los archivos del índice en la carpeta
        print("Base de datos creada y guardada en disco.")

except Exception as e:
    print(f"Error fatal durante la inicialización de RAG/S3: {e}")
    # En un entorno de producción, puedes lanzar una excepción o fallback aquí
    vectorstore = None # Asegurarse de que vectorstore no esté definido si hay un error

# --- 3. CONFIGURACIÓN DEL LLM Y LA CADENA RAG ---

if vectorstore:
    print("Configurando el LLM y la cadena RAG...")
    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key, temperature=0.5)

    retriever = vectorstore.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    print("Chatbot inicializado. ¡Listo para recibir peticiones!")
else:
    # Si la inicialización falló, definimos una cadena dummy para evitar errores de referencia
    qa_chain = None
    print("¡ADVERTENCIA! La cadena RAG no se pudo inicializar. Las peticiones fallarán.")


# --- 4. CONFIGURACIÓN DE LA API CON FASTAPI ---
app = FastAPI()

# Permite que la interfaz HTML acceda a esta API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"Hola": "Bienvenido al Chatbot API. Para chatear, usa el endpoint /chat con POST."}

@app.post("/chat")
async def get_chatbot_response(message: Message):
    if not qa_chain:
        return {"response": "Error interno del servidor: La base de datos no se pudo cargar durante la inicialización."}
    
    try:
        response = qa_chain.invoke({'query': message.query})
        return {"response": response['result']}
    except Exception as e:
        print(f"Error al procesar la consulta: {e}")
        return {"response": "Lo siento, hubo un error al procesar tu solicitud."}