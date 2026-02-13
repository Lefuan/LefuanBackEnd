from flask import Flask, jsonify
from flask_cors import CORS
import subprocess
import urllib.request
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({"status": "ok", "mensaje": "Backend Kuramoto funcionando"})

@app.route('/ejecutar/<script>')
def ejecutar(script):
    try:
        # Descargar el script desde Neocities
        url = f"https://lefuan.neocities.org/Framework/{script}.py"
        urllib.request.urlretrieve(url, f"{script}.py")

        # Ejecutarlo
        result = subprocess.run(['python', f"{script}.py"],
                               capture_output=True, text=True, timeout=30)

        return jsonify({
            'success': True,
            'output': result.stdout,
            'error': result.stderr
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
