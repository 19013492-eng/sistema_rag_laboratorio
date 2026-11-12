
# Sistema RAG de Documentación de Laboratorio, Metrología y Bitácoras de Ensayo

Este proyecto implementa un sistema de **Generación Aumentada por Recuperación (RAG)** totalmente local, transparente y reproducible para la consulta, trazabilidad y verificación de documentación técnica de laboratorio, integrando manuales de instrumentación en inglés (`1e_.pdf`), bitácoras experimentales manuscritas estructuradas (`4m_cuaderno_manuscrito.txt`) e informes metrológicos (`IR1.pdf`, `IR2.pdf`, `IR3.pdf`).

---

## 1. Problema y Destinatario

### Delimitación del problema
En laboratorios de metrología térmica y calibración, el personal técnico debe consultar frecuentemente documentación heterogénea: manuales de especificación en inglés, informes técnicos en español y cuadernos manuscritos de registro diario. La búsqueda manual entre estos formatos es lenta y propensa a la omisión de parámetros críticos, cotas de montaje o mediciones de extremo detalle.

### Alcance de consultas
* **Qué preguntas PUEDE resolver:** Consultas técnicas sobre especificaciones operativas de instrumentos en inglés, mediciones experimentales y estabilidad de celdas en bitácoras manuscritas (celda 2PTB83, sensor TRP 30, ciclos de medición y cotas dimensionales), y rangos de calibración o mantenimiento predictivo por termografía en español (-60 °C a 1100 °C).
* **Qué preguntas NO PUEDE resolver:** Consultas sobre procedimientos o instrumentos ausentes en el corpus (como calibración de termómetros de mercurio de vidrio), datos contables/financieros, o cualquier información externa no respaldada explícitamente por los documentos.

### Destinatario
Metrólogos, técnicos de laboratorio e ingenieros de mantenimiento que necesitan realizar preguntas en lenguaje natural (español) y obtener respuestas precisas, con citas explícitas de página/archivo y acceso directo a la evidencia original.

---

## 2. Corpus Documental

El corpus se encuentra en la carpeta `datos/` y está compuesto por documentos técnicos reales y complementarios:

| Archivo en `datos/` | Origen / Autoría | Fecha / Versión | Permiso / Condición de Uso | Idioma / Formato |
| :--- | :--- | :--- | :--- | :--- |
| `1e_.pdf` | Manual / Ficha técnica de instrumentación de laboratorio | 2000 | Documentación técnica de referencia | Inglés (PDF nativo) |
| `4m_cuaderno_manuscrito.txt` | Bitácora experimental de laboratorio de metrología térmica | Agosto–Septiembre 2000 | Registro empírico de laboratorio | Español / Tablas y cotas (Texto) |
| `IR1.pdf` | Instituto Nacional de Técnica Aeroespacial (INTA) - Área Infrarrojo | 2016 | Documento técnico de divulgación pública | Español (PDF) |
| `IR2.pdf` | Lidia Neita Duarte & Elkin Peña Rodríguez (UPB / SENA) | 2011 | Monografía académica de uso educativo | Español (PDF) |
| `IR3.pdf` | Eduardo Miquel Hernández (Universidad de La Laguna) | 2020 | Trabajo de Fin de Grado (TFG) de acceso abierto | Español (PDF) |

> **Tratamiento del Manuscrito:** La bitácora original escaneada (`4m_cuaderno_manuscrito.pdf`, ~33 MB) se omite intencionalmente durante la indexación automática en `indexar_corpus.py` para procesar en su lugar `4m_cuaderno_manuscrito.txt`. Esta versión estructurada preserva las anotaciones cronológicas, las cotas espaciales de los sensores y las tablas de mediciones de puente de impedancia, garantizando una segmentación limpia y libre de ruido de escaneo.

---

## 3. Arquitectura del Pipeline RAG

```text
[Corpus en datos/ (1e_.pdf, 4m_cuaderno_manuscrito.txt, IR1.pdf, etc.)]
       │
       ▼
[Carga y Filtrado] (PyPDFLoader limitado a 15 pgs + TextLoader completo UTF-8)
       │
       ▼
[Segmentación Semántica] (RecursiveCharacterTextSplitter: chunk_size=800, chunk_overlap=150)
       │
       ▼
[Embeddings Multilingües] (bge-m3 vía Ollama) ──> Representación vectorial común EN / ES / Tablas
       │
       ▼
[Vectorstore Persistente] (ChromaDB en chroma_db_laboratorio/)
       │
       ▼
[Retriever (K=3)] <── [Consulta del Usuario]
       │
       ▼
[LLM Local] (granite4 vía Ollama | temperature=0.1) ──> [Prompt con Reglas de Citas y Abstención]
       │
       ▼
[Interfaz Gradio en app.py] (Panel Izquierdo: Respuesta con Citas | Panel Derecho: Evidencia Cruda y Trazabilidad)
```
### Componentes y Parámetros del Código

* **Carga (`indexar_corpus.py`):** Los PDFs extensos se acotan a las primeras 15 páginas (`MAX_PAGINAS = 15`) para optimizar el rendimiento y pertinencia técnica, mientras que el cuaderno manuscrito (`4m_cuaderno_manuscrito.txt`) se carga completo en UTF-8.
* **Segmentación:** `chunk_size=800` caracteres y `chunk_overlap=150` para preservar la continuidad entre filas de tablas y párrafos técnicos sin desbordar el contexto.
* **Embeddings:** `bge-m3` ejecutado de forma local vía Ollama.
* **Vectorstore:** Chroma persistente en el directorio `chroma_db_laboratorio/`.
* **Inferencia y LLM (`app.py`):** `ChatOllama(model="granite4", temperature=0.1)` acoplado a un retriever por similitud con $k=3$.
* **Interfaz de Usuario:** Gradio (`gr.Blocks`) de doble panel ergonómico con renderizado de fragmentos en tarjetas estilizadas HTML.

---

## 4. Instalación y Configuración Local

### Requisitos previos
* Python 3.12 (o superior).
* Ollama instalado y ejecutándose localmente.

### Paso 1: Descargar los modelos en Ollama
Ejecutar en la terminal:
```bash
ollama pull bge-m3
ollama pull granite4
```
---

### Paso 2: Instalación de dependencias de Python
Crear el entorno virtual con `uv` e instalar las dependencias desde la carpeta del proyecto:
```bash
uv venv --python 3.12
# En Windows activar con: .venv\Scripts\activate

uv pip install -r requirements.txt 
(Contenido del requirements.txt: langchain-core, langchain-community, langchain-chroma, langchain-ollama, langchain-text-splitters, chromadb, pypdf, gradio).
```

### Paso 3: Iniciar la interfaz interactiva
```bash
python app.py
```
Abrir en el navegador la dirección local indicada por Gradio (habitualmente http://127.0.0.1:7860).


## 5. Reindexación del Corpus

Si se modifican los archivos en la carpeta `datos/`, el tamaño de fragmentación o el modelo de embeddings, el índice vectorial local debe reindexarse:

1. Eliminar la carpeta local del índice vectorial (si existe):
   ```powershell
   Remove-Item -Recurse -Force chroma_db_laboratorio
   ```
2. Ejecutar el script de ingesta e indexación:
   ```bash
   python indexar_corpus.py
   ```

> **Nota de Seguridad y Exclusión:** La carpeta del entorno virtual (`.venv/`), los archivos de variables de entorno (`.env`) y la base de datos vectorial generada (`chroma_db_laboratorio/`) están expresamente excluidos del repositorio de GitHub mediante `.gitignore` para garantizar la reproducibilidad limpia del sistema desde los archivos fuentes en `datos/`.

---

## 6. Pruebas y Evaluación del Sistema

Evaluación del sistema mediante 5 casos de prueba prefijados (3 consultas respondibles distribuidas entre los tres informes y 2 consultas no respondibles para verificar la abstención estricta):

| # | Tipo de Consulta | Pregunta del Usuario | Comportamiento Esperado | Resultado / Diagnóstico |
| :---: | :--- | :--- | :--- | :--- |
| **1** | Respondible <br> ( / `1e_.pdf`) | ¿A qué se denomina responsividad espectral? | Recupera el fragmento en inglés de `1e_.pdf`, traduce y sintetiza la respuesta técnica en español citando la página del PDF. | **Éxito:** Recuperación multilingüe precisa sin requerir pretraducción. |
| **2** | Respondible <br> (Bitácora / `4m_cuaderno_manuscrito.txt`) | ¿Qué mediciones y cotas de sensor se registraron para la celda 2PTB83 en la bitácora de agosto? | Recupera los valores de puente, ciclos y distancias del sensor TRP 30 citando explícitamente `[4m_cuaderno_manuscrito.txt]`. | **Éxito:** Extracción de datos tabulares y notas de cuaderno. |
| **3** | Respondible <br> (`IR1.pdf`) | ¿Cuál es el rango de temperatura para calibración de fuentes de radiación infrarroja del laboratorio INTA? | Responde en español citando `[IR1.pdf, pág. 1]` indicando el rango de -60 °C a 1100 °C. | **Éxito:** Precisión metrológica en informes oficiales. |
| **4** | No Respondible <br> (Abstención 1) | ¿Cuál es el protocolo de calibración para termómetros de mercurio de vidrio del año 2024? | Responde declarando la ausencia de datos en los documentos de laboratorio. | **Éxito:** El sistema no inventa información y activa la abstención. |
| **5** | No Respondible <br> (Abstención 2) | ¿Cuál es el costo de un termómetro IR? | Responde declarando la ausencia de registros presupuestarios en el corpus. | **Éxito:** Mitigación total de alucinaciones ante consultas fuera de dominio. |


> **Limitación observada en las pruebas:** Para dibujos a mano alzada y figuras técnicas en el cuaderno manuscrito (como cotas marginales o niveles de hielo/agua), el pipeline de texto plano depende de una transcripción o estructuración previa en .txt. En un entorno productivo a gran escala, esta limitación se supera incorporando un modelo de visión multimodal (VLM / Document AI) para el etiquetado y descripción automática de regiones gráficas.

---

## 7. Decisiones de Diseño y Límites

* **Embeddings Multilingües (`bge-m3`):** Se eligió por su capacidad de alinear semánticamente consultas formuladas en español con evidencia técnica redactada en inglés (`1e_.pdf`) o bitácoras numéricas, sin necesidad de pasos intermedios de traducción.
* **Ajuste de Fragmentación (`chunk_size=800`, `overlap=150`):** Establecido tras validar que 800 caracteres garantizan que una tabla horaria de mediciones o una sección de especificación queden contenidas en un único bloque de contexto.
* **Modelo Generador (`granite4` a `temperature=0.1`):** La temperatura casi nula elimina la variabilidad estocástica y obliga al modelo a ceñirse exclusivamente al contexto provisto.
* **Transparencia y Trazabilidad:** La interfaz de Gradio implementa un panel derecho dedicado a mostrar el texto crudo exacto y los metadatos recuperados (`source` y `page`), permitiendo la auditoría inmediata de la respuesta generada.

---

## 8. Declaración de Uso de Inteligencia Artificial

En cumplimiento con las normas de transparencia del trabajo integrador:
1. **Sugerencias de código:** Asistencia en la orquestación de componentes LangChain, integración de ChromaDB y diseño del layout visual en Gradio.
2. **Estructura de documentación:** Apoyo en el formato y redacción técnica del `README.md` conforme a las consignas de evaluación de la cátedra.
3. **Verificación personal:** Toda la configuración del entorno, la descarga y ejecución local de los modelos en Ollama (`bge-m3` y `granite4`), la indexación del corpus, la transcripción del cuaderno manuscrito y las 5 pruebas de validación fueron ejecutadas, testeadas y auditadas manualmente en el entorno local.