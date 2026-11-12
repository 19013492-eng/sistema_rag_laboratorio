"""Módulo de utilidades y formateo para el sistema RAG de laboratorio.

Este módulo concentra las funciones auxiliares de preparación de contexto,
formateo HTML para el panel de evidencia en Gradio y funciones de visualización.
"""

from collections.abc import Sequence
from typing import Any
import matplotlib.pyplot as plt

# Paleta de colores institucional para gráficos
AZUL = "#1e88e5"
NARANJA = "#eb6834"
GRIS = "#8b8b86"
TINTA_SUAVE = "#37474f"
LINEA = "#c3c2b7"


def formatear_contexto_prompt(docs: Sequence[Any]) -> str:
    """Prepara el bloque de texto con metadatos para inyectar en el prompt del LLM.

    Args:
        docs: Lista de fragmentos (Document) recuperados por el retriever.

    Returns:
        Cadena de texto estructurada con etiquetas de fuente y página.
    """
    contexto_texto = ""
    for idx, doc in enumerate(docs, 1):
        origen = doc.metadata.get("source", "Desconocido")
        if origen.endswith(".pdf"):
            pagina = f"Pág. {doc.metadata.get('page', 0) + 1}"
        else:
            pagina = "Bitácora / Documento de Texto"
            
        contexto_texto += f"[Fuente {idx}: {origen}, {pagina}]\n{doc.page_content}\n\n"
    return contexto_texto


def generar_html_evidencia(docs: Sequence[Any]) -> str:
    """Construye las tarjetas HTML estilizadas para el panel derecho de Gradio.

    Args:
        docs: Lista de fragmentos (Document) recuperados por el retriever.

    Returns:
        Código HTML con bordes, etiquetas y formato responsivo.
    """
    html_evidencia = ""
    for idx, doc in enumerate(docs, 1):
        origen = doc.metadata.get("source", "Desconocido")
        
        if origen.endswith(".pdf"):
            pag_str = f"<b>Página:</b> {doc.metadata.get('page', 0) + 1}"
        else:
            pag_str = "<b>Tipo:</b> Bitácora / Documento de Texto"
            
        texto_snippet = doc.page_content.replace("\n", "<br>")
        
        html_evidencia += f"""
        <div style="background-color: #eef6ff; border: 1px solid #90caf9; border-left: 5px solid #1e88e5; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
            <p style="margin: 0 0 6px 0; color: #0d47a1; font-weight: bold;">📌 Evidencia {idx}</p>
            <p style="margin: 0 0 8px 0; font-size: 0.9em; color: #37474f;"><b>Archivo:</b> <code>{origen}</code> | {pag_str}</p>
            <div style="background-color: #ffffff; border: 1px solid #e0e0e0; padding: 10px; border-radius: 5px; font-size: 0.85em; max-height: 120px; overflow-y: auto; color: #212121;">
                {texto_snippet}
            </div>
        </div>
        """
    return html_evidencia


def dibujar_barras_rendimiento(
    etiquetas: Sequence[str],
    valores: Sequence[float],
    titulo: str,
    etiqueta_eje_x: str,
    formato: str = "{:.2f} s",
) -> None:
    """Dibuja una gráfica horizontal para comparar tiempos de respuesta o métricas.

    Args:
        etiquetas: Nombre de cada prueba o modelo.
        valores: Mediciones numéricas.
        titulo: Título principal de la gráfica.
        etiqueta_eje_x: Nombre de la métrica y unidad.
        formato: Formato numérico para el texto de cada barra.
    """
    if len(etiquetas) != len(valores):
        raise ValueError("etiquetas y valores deben tener el mismo largo.")

    if len(etiquetas) == 0:
        raise ValueError("Hace falta al menos un valor para dibujar.")

    figura, ejes = plt.subplots(figsize=(8, 0.65 * len(etiquetas) + 1.4))
    posiciones = list(range(len(etiquetas)))
    ejes.barh(posiciones, valores, color=AZUL, height=0.6)
    ejes.set_yticks(posiciones, etiquetas)
    ejes.invert_yaxis()
    ejes.set_title(titulo, loc="left", fontweight="bold")
    ejes.set_xlabel(etiqueta_eje_x)
    ejes.axvline(0, color=LINEA, linewidth=1)
    ejes.spines["top"].set_visible(False)
    ejes.spines["right"].set_visible(False)
    ejes.spines["left"].set_visible(False)
    ejes.tick_params(axis="y", length=0)

    mayor_valor = max(abs(valor) for valor in valores) if valores else 1.0
    margen = max(mayor_valor * 0.03, 0.01)

    for posicion in posiciones:
        valor = valores[posicion]
        ejes.text(
            valor + margen,
            posicion,
            formato.format(valor),
            va="center",
            color=TINTA_SUAVE,
        )

    plt.tight_layout()
    plt.show()