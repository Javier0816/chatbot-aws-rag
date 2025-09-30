# app.py

import os
from dotenv import load_dotenv

from langchain_community.document_loaders.s3_file import S3FileLoader

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# --- 1. Inicialización de la cadena RAG ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_bucket_name = os.getenv("AWS_BUCKET_NAME")

print("Inicializando el chatbot...")

# Cargar documentos desde S3
print("Cargando los documentos de S3...")

documents = []
# Lista los archivos en tu bucket de S3
# Nota: boto3 debe estar configurado para tu región de S3, o puedes especificarla
import boto3
s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
objects = s3.list_objects_v2(Bucket=aws_bucket_name)

# Itera sobre los archivos y carga cada uno con el S3FileLoader
for obj in objects.get('Contents', []):
    file_key = obj['Key']
    print(f"Cargando archivo: {file_key}")
    loader = S3FileLoader(aws_bucket_name, file_key)
    docs = loader.load()
    documents.extend(docs)

# ... (El resto del código sigue igual)

print(f"Dividiendo {len(documents)} documentos en chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

print(f"Generando embeddings de {len(texts)} chunks y creando la base de datos vectorial...")
embeddings = OpenAIEmbeddings(api_key=api_key)
vectorstore = Chroma.from_documents(texts, embeddings)

print("Configurando el LLM...")
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key, temperature=0.5)

print("Creando el retriever...")
retriever = vectorstore.as_retriever()

print("Construyendo la cadena RAG...")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

print("Chatbot inicializado. ¡Listo para recibir peticiones!")

# --- 3. Configuración de la API con FastAPI ---
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

@app.get("/")
def read_root():
    return {"Hola": "Bienvenido al Chatbot API"}

@app.post("/chat")
async def get_chatbot_response(message: Message):
    response = qa_chain.invoke({'query': message.query})
    return {"response": response['result']}