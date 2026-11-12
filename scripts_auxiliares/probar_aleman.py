from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="bge-m3")
vectorstore = Chroma(
    persist_directory="chroma_db_laboratorio",
    embedding_function=embeddings,
    collection_name="rag_trilingue_lab"
)

# 1. Verificar si existen fragmentos guardados de 3a_.pdf
datos = vectorstore.get()
docs_3a = [
    (text, meta) for text, meta in zip(datos["documents"], datos["metadatas"])
    if "3a" in str(meta.get("fuente", ""))
]

print("=== DIAGNÓSTICO DE 3a_.pdf ===")
print(f"Total de fragmentos de 3a_.pdf en ChromaDB: {len(docs_3a)}")

if len(docs_3a) == 0:
    print("\n❌ CAUSA ENCONTRADA: 3a_.pdf es un PDF escaneado (imagen) sin texto digital.")
    print("PyPDFLoader no pudo extraer caracteres de este archivo.")
else:
    print("\n✓ Texto extraído correctamente. Muestra del primer fragmento:")
    print("-" * 60)
    print(docs_3a[0][0][:300])
    print("-" * 60)
    
    print("\nProbando búsqueda con k=10 para ver posiciones:")
    resultados = vectorstore.similarity_search_with_score("temperatura", k=10)
    for i, (doc, score) in enumerate(resultados, 1):
        print(f"[{i}] Fuente: {doc.metadata.get('fuente')} (Pág {doc.metadata.get('pagina')}) - Distancia: {score:.4f}")