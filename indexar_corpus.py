import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

CARPETA_DATOS = "datos"
DB_DIR = "chroma_db_laboratorio"
MAX_PAGINAS = 15  # Límite solo para PDFs pesados

print(f"🌍 Cargando corpus (PDFs acotados a {MAX_PAGINAS} pgs y TXT manuscrito completo)...")
documentos = []

for archivo in os.listdir(CARPETA_DATOS):
    ruta_completa = os.path.join(CARPETA_DATOS, archivo)
    
    # Omitir el PDF escaneado (ya tenemos el .txt transcrito)
    if archivo == "4m_cuaderno_manuscrito.pdf":
        print(f"⏩ Omitiendo '{archivo}' (se usa la versión .txt transcrita).")
        continue

    # Cargar PDFs acotados
    if archivo.endswith(".pdf"):
        try:
            loader = PyPDFLoader(ruta_completa)
            docs = loader.load()
            docs_limitados = docs[:MAX_PAGINAS]
            for d in docs_limitados:
                d.metadata["source"] = archivo
            documentos.extend(docs_limitados)
            print(f"✓ Cargado PDF: {archivo} ({len(docs_limitados)} pgs)")
        except Exception as e:
            print(f"❌ Error al cargar {archivo}: {e}")

    # Cargar TXT COMPLETO (Bitácora manuscrita)
    elif archivo.endswith(".txt") and archivo != "requirements.txt":
        try:
            loader = TextLoader(ruta_completa, encoding="utf-8")
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = archivo
            documentos.extend(docs)
            print(f"✓ Cargado TXT: {archivo} ({len(docs)} doc)")
        except Exception as e:
            print(f"❌ Error al cargar {archivo}: {e}")

print(f"\n📦 Total de documentos/páginas cargados: {len(documentos)}")

# Dividir en fragmentos
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
chunks = text_splitter.split_documents(documentos)
print(f"✂️ Total de fragmentos generados: {len(chunks)}")

# Embeddings e indexación
print("🧠 Generando embeddings con 'bge-m3'...")
embeddings = OllamaEmbeddings(model="bge-m3")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_DIR
)

print(f"✅ ¡Base de datos vectorial lista en '{DB_DIR}'!")