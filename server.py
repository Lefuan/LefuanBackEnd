from flask import Flask, request, jsonify
from flask_cors import CORS

# --- CONFIGURACIÓN DE LA APLICACIÓN ---
app = Flask(__name__)
# Habilitar CORS para permitir la comunicación entre el frontend (index.html) y la API
CORS(app)

# --- LÓGICA DEL MODELO RE²M ---

def simular_re2m(pasos_de_tiempo, nodos, enlaces, nodos_fijos):
    """
    Función que simula la dinámica del modelo RE²M (un modelo de colapso de información).

    Esta es una implementación simplificada.
    """

    # 1. Inicialización de la red
    # Crea un diccionario para el estado actual de los nodos: {id: estado, ...}
    current_states = {n['id']: n['estado_inicial'] for n in nodos}
    # Crea un diccionario para la energía actual de los nodos: {id: energia, ...}
    current_energies = {n['id']: 0.0 for n in nodos}
    # Obtiene los umbrales NIR
    nir_thresholds = {n['id']: float(n['nir']) for n in nodos}

    # Prepara el historial
    historial_estados = [current_states.copy()]
    historial_energies = [current_energies.copy()]

    # 2. Información de Nodos (para el encabezado de la tabla del frontend)
    nodos_info = {n['id']: n['nombre'] for n in nodos}

    # 3. Bucle de Simulación
    for t in range(1, pasos_de_tiempo + 1):
        next_states = current_states.copy()
        next_energies = current_energies.copy()

        # Calcular la energía entrante (E_in) para cada nodo
        incoming_energy = {n['id']: 0.0 for n in nodos}

        for enlace in enlaces:
            origen = enlace['origen_id']
            destino = enlace['destino_id']
            estado_relacional = enlace['estado_relacional']
            energia_enlace = enlace['energia_enlace']

            # Solo los enlaces con estado P, O o D transportan energía
            if estado_relacional in ['P', 'O', 'D'] and current_states[origen] in ['P', 'O', 'D']:
                # Simplificación: la energía que llega se suma.
                incoming_energy[destino] += energia_enlace

        # Aplicar reglas de colapso a los nodos no fijos
        for node_id, nir in nir_thresholds.items():
            if node_id not in nodos_fijos:
                e_in = incoming_energy[node_id]

                # Regla de Colapso Simplificada: Si la energía entrante supera el NIR,
                # el nodo pasa a Partícula (P), a menos que ya esté Borrado (B).
                if e_in >= nir and current_states[node_id] != 'B':
                    next_states[node_id] = 'P'
                elif current_states[node_id] == 'P' and e_in == 0:
                    # Ejemplo: Si colapsó pero ya no recibe energía, vuelve a Onda.
                    next_states[node_id] = 'O'

                # Actualizar la energía (como una medida de entropía o tensión)
                next_energies[node_id] = round(e_in, 2)
            else:
                # Los nodos fijos mantienen su estado y energía (o la energía entrante si se desea)
                next_energies[node_id] = round(incoming_energy[node_id], 2)

        current_states = next_states
        current_energies = next_energies

        historial_estados.append(current_states.copy())
        historial_energies.append(current_energies.copy())

    return historial_estados, historial_energies, nodos_info

# --- RUTA DE LA API FLASK ---

@app.route('/simular', methods=['POST'])
def simular():
    """Recibe la configuración del modelo desde el frontend y devuelve los resultados."""
    try:
        data = request.get_json()

        pasos_de_tiempo = data.get('pasos_de_tiempo', 2)
        nodos = data.get('nodos', [])
        enlaces = data.get('enlaces', [])
        nodos_fijos = data.get('nodos_fijos', [])

        if not nodos:
            return jsonify({'success': False, 'error': 'No se proporcionaron nodos para la simulación.'}), 400

        # Ejecutar la simulación
        historial_estados, historial_energies, nodos_info = simular_re2m(
            pasos_de_tiempo, nodos, enlaces, nodos_fijos
        )

        # Devolver el resultado en el formato EXACTO esperado por el JavaScript
        response = {
            'success': True,
            'message': f"Simulación completada en {pasos_de_tiempo} pasos.",
            'historial_estados': historial_estados,
            'historial_energies': historial_energies,
            'nodos_info': nodos_info
        }

        return jsonify(response), 200

    except Exception as e:
        # En caso de error de servidor, imprime el error para debug y lo envía al log del frontend
        print(f"Error en la simulación: {e}")
        return jsonify({'success': False, 'error': f'Error interno del servidor: {e}'}), 500

if __name__ == '__main__':
    # Ejecutar en el puerto 5000, accesible desde 127.0.0.1
    app.run(host='127.0.0.1', port=5000, debug=True)
