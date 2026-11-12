from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Conectar al índice generado
embeddings = OllamaEmbeddings(model="bge-m3")

vectorstore = Chroma(
    persist_directory="chroma_db_laboratorio",
    embedding_function=embeddings,
    collection_name="rag_trilingue_lab"
)

pregunta = "¿Qué resistencia se midió en el ensayo de Zn 135?"

# Recuperar fragmentos junto con la métrica de distancia
hallazgos = vectorstore.similarity_search_with_score(query=pregunta, k=3)

for i, (doc, distancia) in enumerate(hallazgos, start=1):
    fuente = doc.metadata.get("fuente", "Desconocida")
    pagina = doc.metadata.get("pagina", "N/A")
    print(f"[{i}] Fuente: {fuente} (Pág. {pagina}) — Distancia: {distancia:.4f}")
    print(f"    Muestra: {doc.page_content[:150]}...\n")