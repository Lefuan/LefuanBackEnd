from flask import Flask, jsonify, request
from flask_cors import CORS
import subprocess
import urllib.request
import os
import sys
import re
import base64

app = Flask(__name__)
CORS(app)

# Mapeo de nombres amigables a archivos reales
SCRIPTS = {
    # Hito 1
    'hito1': 'KuramotoPRO.py',

    # Hito 2
    'hito2': 'kuramoto_explorer.py',

    # Hito 3
    'hito3': 'Kuramoto7_1.py',

    # Hito 4
    'hito4': 'modelo_efectivo_46_capas.py',

    # Hito 5
    'hito5': 'Kuramoto7_4.py',

    # Hito 6
    'hito6': 'KuramotoPODB.py',

    # Hito 7
    'hito7': 'MacroPODB.py'
}

@app.route('/')
def home():
    return jsonify({
        "status": "ok",
        "mensaje": "Backend Kuramoto - Framework P-O-D-B",
        "scripts_disponibles": list(SCRIPTS.keys()),
        "descripcion": {
            "hito1": "🌀 Hito 1: KuramotoPRO (base)",
            "hito2": "📊 Hito 2: kuramoto_explorer (2 capas)",
            "hito3": "⚙️ Hito 3: Kuramoto7_1 (3 niveles)",
            "hito4": "📈 Hito 4: modelo_efectivo_46_capas",
            "hito5": "🔄 Hito 5: Kuramoto7_4 (7 topologías)",
            "hito6": "🔬 Hito 6: KuramotoPODB (PODB por conexión)",
            "hito7": "🚀 Hito 7: MacroPODB (unificado)"
        }
    })

@app.route('/ejecutar/<hito>')
def ejecutar_hito(hito):
    if hito not in SCRIPTS:
        return jsonify({'success': False, 'error': f'Hito "{hito}" no encontrado'})

    try:
        script_file = SCRIPTS[hito]
        url = f"https://lefuan.neocities.org/Framework/{script_file}"
        local_path = f"/tmp/{script_file}"

        # Descargar script
        urllib.request.urlretrieve(url, local_path)

        # Ejecutar con timeout y capturar todo
        result = subprocess.run(
            [sys.executable, local_path],
            capture_output=True,
            text=True,
            timeout=60
        )

        # Buscar imágenes base64 en la salida
        output_text = result.stdout
        imagenes = []

        # Patrón para encontrar imágenes en base64
        patron = r'IMAGEN_BASE64:([A-Za-z0-9+/=]+)'
        for match in re.finditer(patron, output_text):
            imagenes.append(match.group(1))
            # Quitar la línea de la imagen del texto
            output_text = output_text.replace(match.group(0), '')

        # Si no hay imágenes explícitas, buscar figuras guardadas
        if not imagenes:
            # Buscar archivos PNG en /tmp
            for f in os.listdir('/tmp'):
                if f.endswith('.png'):
                    with open(f'/tmp/{f}', 'rb') as img_file:
                        img_data = base64.b64encode(img_file.read()).decode('utf-8')
                        imagenes.append(img_data)
                    os.remove(f'/tmp/{f}')

        return jsonify({
            'success': True,
            'hito': hito,
            'script': script_file,
            'output': output_text,
            'error': result.stderr,
            'imagenes': imagenes
        })

    except subprocess.TimeoutExpired:
        return jsonify({'success': False, 'error': 'Timeout: el script tardó más de 60 segundos'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
