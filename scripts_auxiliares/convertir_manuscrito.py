import pypdfium2 as pdfium
import ollama
import os

pdf_path = "datos/4m_cuaderno_manuscrito.pdf"
txt_output = "datos/4m_cuaderno_manuscrito.txt"

print("🔍 Cargando el PDF del cuaderno manuscrito...")
pdf = pdfium.PdfDocument(pdf_path)

texto_completo = ""

for i, page in enumerate(pdf, 1):
    print(f"🤖 Transcribiendo página {i}/7 con Modelo de Visión...")
    
    # Renderizar la página como imagen PNG temporal
    img_path = f"temp_page_{i}.png"
    image = page.render(scale=2).to_pil()
    image.save(img_path)
    
    # Le enviamos la imagen directamente al modelo de visión en Ollama
    response = ollama.chat(
        model='llava', # o 'qwen2-vl'
        messages=[{
            'role': 'user',
            'content': 'Transcribe fielmente todo el texto, números y anotaciones de esta página de bitácora de laboratorio manuscrita. Devuelve únicamente el texto transcrito en español sin comentarios extras.',
            'images': [img_path]
        }]
    )
    
    texto_pagina = response['message']['content']
    texto_completo += f"\n--- Bitácora Manuscrita (Página {i}) ---\n{texto_pagina}\n"
    
    if os.path.exists(img_path):
        os.remove(img_path)

with open(txt_output, "w", encoding="utf-8") as f:
    f.write(texto_completo)

print(f"✅ Transcripción inteligente completada: '{txt_output}'")