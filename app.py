import gradio as gr
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from utils import formatear_contexto_prompt, generar_html_evidencia


DB_DIR = "chroma_db_laboratorio"

# 1. Cargar embeddings y base de datos vectorial
embeddings = OllamaEmbeddings(model="bge-m3")
vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

# Retriever estándar por similitud (k=3) para evitar bloqueos por threshold
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 2. Cargar el LLM de Ollama
llm = ChatOllama(model="granite4", temperature=0.1)

# 2. Cargar el LLM de Ollama (ajustado a temperature=0.0 para evitar variaciones)
llm = ChatOllama(model="granite4", temperature=0.0)

def responder_consulta(pregunta: str):
    if not pregunta.strip():
        return (
            "Por favor, ingresá una consulta válida.", 
            "<p style='color: #757575;'>Esperando consulta...</p>"
        )

    # Recuperar documentos por similitud
    docs = retriever.invoke(pregunta)

    if not docs:
        return (
            "No encuentro esa información en los documentos de laboratorio.", 
            "<p style='color: #757575;'>No se encontraron fuentes de evidencia asociadas.</p>"
        )

    # 1. Usamos la función de utils.py para construir el texto de contexto
    contexto_texto = formatear_contexto_prompt(docs)

    prompt = f"""Eres un asistente técnico de laboratorio. Responde a la siguiente consulta utilizando ÚNICAMENTE la información provista en el contexto. 
Si la respuesta incluye datos concretos, cita explícitamente la fuente entre corchetes (ej: [IR1.pdf, pág. 2] o [4m_cuaderno_manuscrito.txt]).

Contexto:
{contexto_texto}

Consulta del usuario: {pregunta}
Respuesta:"""

    respuesta_llm = llm.invoke(prompt).content

    # 2. Usamos la función de utils.py para construir el Panel de Evidencia HTML
    html_evidencia = generar_html_evidencia(docs)

    return respuesta_llm, html_evidencia

# 3. Interfaz de usuario con Gradio
with gr.Blocks(title="Sistema RAG — Documentación de Laboratorio") as demo:
    gr.Markdown("# 🔬 Sistema RAG — Documentación de Laboratorio")
    gr.Markdown("*Consulta Inteligente y Trazabilidad de Evidencia Técnica*")
    
    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("### 💬 CONSULTA TÉCNICA DEL USUARIO")
            with gr.Group():
                txt_pregunta = gr.Textbox(
                    placeholder="Escribí tu pregunta sobre las celdas, ensayos o informes...", 
                    lines=3, 
                    show_label=False
                )
                btn_enviar = gr.Button("🚀 Enviar Consulta", variant="primary")
            
            gr.Markdown("### 🤖 RESPUESTA GENERADA (CON CITAS)")
            with gr.Group():
                txt_respuesta = gr.Textbox(
                    lines=10, 
                    show_label=False, 
                    interactive=False
                )
            
        with gr.Column(scale=5):
            gr.Markdown("### 📑 Panel de Evidencia y Trazabilidad")
            html_panel_evidencia = gr.HTML()

    btn_enviar.click(
        fn=responder_consulta,
        inputs=[txt_pregunta],
        outputs=[txt_respuesta, html_panel_evidencia]
    )

if __name__ == "__main__":
    demo.launch()