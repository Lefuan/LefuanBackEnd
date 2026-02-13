from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Obligatorio para Koyeb
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

app = Flask(__name__)
CORS(app)

# ============================================
# FUNCIÓN PARA KURAMOTO1 (Hito 1)
# ============================================
def ejecutar_kuramoto1():
    """Código de kuramoto1.py - Simulación básica 3 niveles"""
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    
    # ============================================
    # PARÁMETROS DEL SISTEMA
    # ============================================
    N1 = 5      # metrónomos en plataforma raíz
    N2 = 4      # metrónomos en cada plataforma de nivel 2
    N3 = 3      # metrónomos en cada plataforma de nivel 3

    # Frecuencias naturales
    np.random.seed(42)
    omega1 = np.random.normal(1.0, 0.1, N1)
    omega2 = np.random.normal(1.0, 0.15, 3 * N2)
    omega3 = np.random.normal(1.0, 0.2, 9 * N3)

    # Fuerzas de acoplamiento
    K_intra = 2.0
    K_inter = 1.5

    # Tiempo de simulación
    t_span = (0, 50)
    t_eval = np.linspace(0, 50, 1000)

    # ============================================
    # CONSTRUCCIÓN DEL VECTOR DE ESTADO
    # ============================================
    def indices():
        idx1 = slice(0, N1)
        idx2 = slice(N1, N1 + 3*N2)
        idx3 = slice(N1 + 3*N2, N1 + 3*N2 + 9*N3)
        return idx1, idx2, idx3

    idx1, idx2, idx3 = indices()

    # ============================================
    # ECUACIONES DIFERENCIALES
    # ============================================
    def kuramoto_hierarchical(t, theta):
        dtheta = np.zeros_like(theta)

        # NIVEL 1
        theta1 = theta[idx1]
        for i in range(N1):
            dtheta[i] = omega1[i] + (K_intra/N1) * np.sum(np.sin(theta1 - theta1[i]))

        # NIVEL 2
        theta2 = theta[idx2].reshape(3, N2)
        dtheta2 = np.zeros_like(theta2)
        r1 = np.mean(np.exp(1j * theta1))
        phi1 = np.angle(r1)

        for p in range(3):
            for i in range(N2):
                intra = (K_intra/N2) * np.sum(np.sin(theta2[p] - theta2[p, i]))
                inter = K_inter * np.sin(phi1 - theta2[p, i])
                dtheta2[p, i] = omega2[p*N2 + i] + intra + inter

        dtheta[idx2] = dtheta2.flatten()

        # NIVEL 3
        theta3 = theta[idx3].reshape(9, N3)
        dtheta3 = np.zeros_like(theta3)
        r2 = np.mean(np.exp(1j * theta2), axis=1)
        phi2 = np.angle(r2)

        for p in range(9):
            madre = p // 3
            for i in range(N3):
                intra = (K_intra/N3) * np.sum(np.sin(theta3[p] - theta3[p, i]))
                inter = K_inter * np.sin(phi2[madre] - theta3[p, i])
                dtheta3[p, i] = omega3[p*N3 + i] + intra + inter

        dtheta[idx3] = dtheta3.flatten()
        return dtheta

    # ============================================
    # SIMULACIÓN
    # ============================================
    theta0 = np.concatenate([
        np.random.uniform(-np.pi, np.pi, N1),
        np.random.uniform(-np.pi, np.pi, 3*N2),
        np.random.uniform(-np.pi, np.pi, 9*N3)
    ])

    print("Simulando sistema jerárquico de 44 osciladores...")
    sol = solve_ivp(kuramoto_hierarchical, t_span, theta0, t_eval=t_eval, method='RK45')

    # ============================================
    # CÁLCULO DE PARÁMETROS DE ORDEN
    # ============================================
    r1_t = np.abs(np.mean(np.exp(1j * sol.y[idx1]), axis=0))
    theta2_t = sol.y[idx2].reshape(3, N2, -1)
    r2_plataformas = np.abs(np.mean(np.exp(1j * theta2_t), axis=1))
    theta3_t = sol.y[idx3].reshape(9, N3, -1)
    r3_plataformas = np.abs(np.mean(np.exp(1j * theta3_t), axis=1))

    # ============================================
    # VISUALIZACIÓN
    # ============================================
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(sol.t, r1_t, 'b-', linewidth=2)
    axes[0].set_ylabel('r1 (Nivel 1)')
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Sincronización en Plataforma Raíz (Nivel 1)')

    for p in range(3):
        axes[1].plot(sol.t, r2_plataformas[p], label=f'Plataforma {p+1}', linewidth=1.5)
    axes[1].set_ylabel('r2 (Nivel 2)')
    axes[1].set_ylim(0, 1.1)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_title('Sincronización en 3 Plataformas del Nivel 2')

    for p in range(9):
        axes[2].plot(sol.t, r3_plataformas[p], label=f'P{p+1}', linewidth=1, alpha=0.7)
    axes[2].set_xlabel('Tiempo')
    axes[2].set_ylabel('r3 (Nivel 3)')
    axes[2].set_ylim(0, 1.1)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Sincronización en 9 Plataformas del Nivel 3')

    plt.tight_layout()

    # Capturar gráfica
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close('all')

    # ANÁLISIS
    umbral = 0.9
    output = []
    output.append("\n=== TIEMPOS DE SINCRONIZACIÓN (r > 0.9) ===")
    
    t_sync1 = sol.t[np.where(r1_t > umbral)[0][0]] if np.any(r1_t > umbral) else np.inf
    output.append(f"Nivel 1: t = {t_sync1:.2f}")
    output.append("\nNivel 2:")
    for p in range(3):
        idx = np.where(r2_plataformas[p] > umbral)[0]
        if len(idx) > 0:
            output.append(f"  Plataforma {p+1}: t = {sol.t[idx[0]]:.2f}")
        else:
            output.append(f"  Plataforma {p+1}: NO sincroniza")
    
    output.append("\nNivel 3:")
    for p in range(9):
        idx = np.where(r3_plataformas[p] > umbral)[0]
        if len(idx) > 0:
            output.append(f"  Plataforma {p+1}: t = {sol.t[idx[0]]:.2f}")
        else:
            output.append(f"  Plataforma {p+1}: NO sincroniza")
    
    return "\n".join(output), img_base64

# ============================================
# FUNCIÓN PARA KURAMOTO2 (Hito 2 - Explorer)
# ============================================
def ejecutar_kuramoto2():
    """Código de kuramoto2.py - Simulación con parámetros ajustables"""
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp

    def simular_kuramoto(N1=5, N2=4, N3=3, K_intra=2.0, K_inter=1.5,
                         T=25, puntos=150, semilla=42):
        """Simula el sistema jerárquico de Kuramoto con parámetros ajustables"""
        np.random.seed(semilla)
        omega1 = np.random.normal(1.0, 0.1, N1)
        omega2 = np.random.normal(1.0, 0.15, 3 * N2)
        omega3 = np.random.normal(1.0, 0.2, 9 * N3)

        t_span = (0, T)
        t_eval = np.linspace(0, T, puntos)

        def kuramoto_hierarchical(t, theta):
            dtheta = np.zeros_like(theta)
            idx1 = slice(0, N1)
            idx2 = slice(N1, N1 + 3*N2)
            idx3 = slice(N1 + 3*N2, N1 + 3*N2 + 9*N3)

            # Nivel 1
            theta1 = theta[idx1]
            for i in range(N1):
                dtheta[i] = omega1[i] + (K_intra/N1) * np.sum(np.sin(theta1 - theta1[i]))

            # Nivel 2
            theta2 = theta[idx2].reshape(3, N2)
            dtheta2 = np.zeros_like(theta2)
            r1 = np.mean(np.exp(1j * theta1))
            phi1 = np.angle(r1)
            for p in range(3):
                for i in range(N2):
                    intra = (K_intra/N2) * np.sum(np.sin(theta2[p] - theta2[p, i]))
                    inter = K_inter * np.sin(phi1 - theta2[p, i])
                    dtheta2[p, i] = omega2[p*N2 + i] + intra + inter
            dtheta[idx2] = dtheta2.flatten()

            # Nivel 3
            theta3 = theta[idx3].reshape(9, N3)
            dtheta3 = np.zeros_like(theta3)
            theta2_flat = theta[idx2].reshape(3, N2)
            r2 = np.mean(np.exp(1j * theta2_flat), axis=1)
            phi2 = np.angle(r2)
            for p in range(9):
                madre = p // 3
                for i in range(N3):
                    intra = (K_intra/N3) * np.sum(np.sin(theta3[p] - theta3[p, i]))
                    inter = K_inter * np.sin(phi2[madre] - theta3[p, i])
                    dtheta3[p, i] = omega3[p*N3 + i] + intra + inter
            dtheta[idx3] = dtheta3.flatten()
            return dtheta

        theta0 = np.random.uniform(-np.pi, np.pi, N1 + 3*N2 + 9*N3)
        sol = solve_ivp(kuramoto_hierarchical, t_span, theta0,
                        t_eval=t_eval, method='RK45')

        # Calcular sincronización
        idx1 = slice(0, N1)
        idx2 = slice(N1, N1 + 3*N2)
        idx3 = slice(N1 + 3*N2, N1 + 3*N2 + 9*N3)

        r1 = np.abs(np.mean(np.exp(1j * sol.y[idx1]), axis=0))
        theta2_hist = sol.y[idx2].reshape(3, N2, -1)
        r2 = np.abs(np.mean(np.exp(1j * theta2_hist), axis=1))
        theta3_hist = sol.y[idx3].reshape(9, N3, -1)
        r3 = np.abs(np.mean(np.exp(1j * theta3_hist), axis=1))

        return sol.t, r1, r2, r3

    # Ejecutar los tres casos
    output_lines = []
    
    # Caso 1: Configuración original
    output_lines.append(f"\n{'='*50}")
    output_lines.append(f"Simulando: 5 + 3×4 + 9×3 = 44 osciladores")
    output_lines.append(f"K_intra=2.0, K_inter=1.5, T=25s")
    output_lines.append('='*50)
    
    t, r1, r2, r3 = simular_kuramoto(N1=5, N2=4, N3=3, K_intra=2.0, K_inter=1.5, T=25)
    
    output_lines.append("\n=== ANÁLISIS DE SINCRONIZACIÓN ===")
    output_lines.append(f"Nivel 1 - Final: r={r1[-1]:.3f}")
    output_lines.append(f"Nivel 2 - Promedio final: r={np.mean(r2[:,-1]):.3f}")
    output_lines.append(f"Nivel 3 - Promedio final: r={np.mean(r3[:,-1]):.3f}")

    # Caso 2: Acoplamiento débil
    output_lines.append(f"\n{'='*50}")
    output_lines.append(f"Simulando: 5 + 3×4 + 9×3 = 44 osciladores")
    output_lines.append(f"K_intra=2.0, K_inter=0.5, T=25s")
    output_lines.append('='*50)
    
    t, r1, r2, r3 = simular_kuramoto(N1=5, N2=4, N3=3, K_intra=2.0, K_inter=0.5, T=25)
    
    output_lines.append("\n=== ANÁLISIS DE SINCRONIZACIÓN ===")
    output_lines.append(f"Nivel 1 - Final: r={r1[-1]:.3f}")
    output_lines.append(f"Nivel 2 - Promedio final: r={np.mean(r2[:,-1]):.3f}")
    output_lines.append(f"Nivel 3 - Promedio final: r={np.mean(r3[:,-1]):.3f}")

    # Caso 3: Acoplamiento fuerte
    output_lines.append(f"\n{'='*50}")
    output_lines.append(f"Simulando: 5 + 3×4 + 9×3 = 44 osciladores")
    output_lines.append(f"K_intra=2.0, K_inter=3.0, T=25s")
    output_lines.append('='*50)
    
    t, r1, r2, r3 = simular_kuramoto(N1=5, N2=4, N3=3, K_intra=2.0, K_inter=3.0, T=25)
    
    output_lines.append("\n=== ANÁLISIS DE SINCRONIZACIÓN ===")
    output_lines.append(f"Nivel 1 - Final: r={r1[-1]:.3f}")
    output_lines.append(f"Nivel 2 - Promedio final: r={np.mean(r2[:,-1]):.3f}")
    output_lines.append(f"Nivel 3 - Promedio final: r={np.mean(r3[:,-1]):.3f}")

    # Crear una gráfica de ejemplo (la última)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    axes[0].plot(t, r1, 'b-', linewidth=2)
    axes[0].set_ylabel('Nivel 1')
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(t, r2.T, linewidth=1.5)
    axes[1].set_ylabel('Nivel 2')
    axes[1].set_ylim(0, 1.1)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(t, r3.T, linewidth=1, alpha=0.7)
    axes[2].set_xlabel('Tiempo (s)')
    axes[2].set_ylabel('Nivel 3')
    axes[2].set_ylim(0, 1.1)
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('Sincronización Jerárquica - Comparación de casos')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close('all')
    
    return "\n".join(output_lines), img_base64

# ============================================
# FUNCIÓN PARA KURAMOTO3 (Hito 3)
# ============================================
def ejecutar_kuramoto3():
    """Código de kuramoto3.py - Simulación de N niveles"""
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp

    def simular_niveles(niveles=6, metros_por_plataforma=3, K_intra=2.0, K_inter=1.5):
        plataformas_por_nivel = [3**i for i in range(niveles)]
        osciladores_por_nivel = [p * metros_por_plataforma for p in plataformas_por_nivel]
        total_osciladores = sum(osciladores_por_nivel)

        np.random.seed(42)
        frecuencias = []
        for nivel in range(niveles):
            std = 0.1 + 0.02 * nivel
            n_osc = osciladores_por_nivel[nivel]
            frecuencias.append(np.random.normal(1.0, std, n_osc))
        omega = np.concatenate(frecuencias)

        t_span = (0, 30)
        t_eval = np.linspace(0, 30, 300)

        indices = []
        start = 0
        for n in osciladores_por_nivel:
            indices.append(slice(start, start + n))
            start += n

        def dynamics(t, theta):
            dtheta = np.zeros_like(theta)
            
            # Nivel 0
            theta0 = theta[indices[0]]
            for i in range(len(theta0)):
                dtheta[indices[0]][i] = omega[indices[0]][i] + (K_intra/len(theta0)) * np.sum(np.sin(theta0 - theta0[i]))

            # Niveles superiores
            for nivel in range(1, niveles):
                n_plt = 3**nivel
                n_por_plt = metros_por_plataforma
                theta_n = theta[indices[nivel]].reshape(n_plt, n_por_plt)
                dtheta_n = np.zeros_like(theta_n)

                theta_prev = theta[indices[nivel-1]].reshape(3**(nivel-1), metros_por_plataforma)
                r_prev = np.mean(np.exp(1j * theta_prev), axis=1)
                phi_prev = np.angle(r_prev)

                for p in range(n_plt):
                    madre = p // 3
                    for i in range(n_por_plt):
                        intra = (K_intra/n_por_plt) * np.sum(np.sin(theta_n[p] - theta_n[p, i]))
                        inter = K_inter * np.sin(phi_prev[madre] - theta_n[p, i])
                        idx_global = indices[nivel].start + p*n_por_plt + i
                        dtheta_n[p, i] = omega[idx_global] + intra + inter

                dtheta[indices[nivel]] = dtheta_n.flatten()
            return dtheta

        theta0 = np.random.uniform(-np.pi, np.pi, total_osciladores)
        sol = solve_ivp(dynamics, t_span, theta0, t_eval=t_eval, method='RK45', rtol=1e-2)

        r_por_nivel = []
        for nivel in range(niveles):
            fases = sol.y[indices[nivel]]
            if len(fases.shape) == 1:
                r = np.abs(np.mean(np.exp(1j * fases), axis=0))
            else:
                r = np.abs(np.mean(np.exp(1j * fases), axis=0))
            r_por_nivel.append(r)

        return sol.t, r_por_nivel, plataformas_por_nivel

    # Ejecutar con 6 niveles
    output_lines = []
    output_lines.append(f"\n{'='*60}")
    output_lines.append(f"SIMULANDO 6 NIVELES JERÁRQUICOS")
    output_lines.append(f"Metrónomos por plataforma: 3")
    output_lines.append('='*60)

    t, r_niveles, plt_por_nivel = simular_niveles(niveles=6, metros_por_plataforma=3)

    # Gráfica
    fig, axes = plt.subplots(6, 1, figsize=(12, 16))
    for i in range(6):
        axes[i].plot(t, r_niveles[i], linewidth=2)
        axes[i].set_ylabel(f'N{i}')
        axes[i].set_ylim(0, 1.1)
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
    axes[-1].set_xlabel('Tiempo')
    plt.suptitle('Sincronización en 6 Niveles Jerárquicos')
    plt.tight_layout()

    # Capturar gráfica
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close('all')

    # Resultados
    r_final = [r[-1] for r in r_niveles]
    output_lines.append("\n=== RESULTADOS POR NIVEL ===")
    for i, r in enumerate(r_final):
        estado = "✅ Coherente" if r > 0.8 else "⚠️ Difuso" if r > 0.5 else "❌ Caótico"
        output_lines.append(f"Nivel {i}: r={r:.3f} {estado}")

    return "\n".join(output_lines), img_base64

# ============================================
# FUNCIÓN PARA KURAMOTO4 (Hito 4 - Modelo 46 capas)
# ============================================
def ejecutar_kuramoto4():
    """Código de kuramoto4.py - Modelo efectivo 46 capas"""
    import numpy as np
    import matplotlib.pyplot as plt

    n_capas = 46
    K_intra = 2.0
    K_inter_base = 1.5
    gamma_acumulado = np.ones(n_capas)

    r = np.zeros(n_capas)
    r[0] = 0.98
    tiempo_efectivo = np.zeros(n_capas)

    output_lines = []
    
    for capa in range(1, n_capas):
        K_inter = K_inter_base * np.exp(-0.1 * capa)
        r_capa_anterior = r[capa-1]
        r[capa] = r_capa_anterior * (1 - np.exp(-K_inter)) * np.tanh(K_intra)
        gamma_acumulado[capa] = gamma_acumulado[capa-1] * (1 + 0.05 * (1 - r[capa]))
        tiempo_efectivo[capa] = tiempo_efectivo[capa-1] + 1.0 / gamma_acumulado[capa]

    # Visualización
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    capas = np.arange(n_capas)

    axes[0].plot(capas, r, 'b-', linewidth=2)
    axes[0].set_xlabel('Nivel de capa')
    axes[0].set_ylabel('Sincronización (r)')
    axes[0].set_title('Degradación de la sincronización en 46 capas')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0.5, color='r', linestyle='--', label='Umbral de coherencia')
    axes[0].legend()

    axes[1].plot(capas, tiempo_efectivo, 'g-', linewidth=2)
    axes[1].set_xlabel('Nivel de capa')
    axes[1].set_ylabel('Tiempo efectivo acumulado')
    axes[1].set_title('Ralentización del tiempo por capa')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Capturar gráfica
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close('all')

    # Resultados
    capa_critica = np.where(r < 0.5)[0][0] if np.any(r < 0.5) else n_capas
    output_lines.append(f"Pérdida de coherencia (r<0.5) en capa: {capa_critica}")
    output_lines.append(f"Sincronización final capa 46: r={r[-1]:.4f}")

    return "\n".join(output_lines), img_base64

# ============================================
# FUNCIÓN PARA KURAMOTO5 (Hito 5)
# ============================================
def ejecutar_kuramoto5():
    """Código de kuramoto5.py - Experimentos de topología"""
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp

    def generar_scale_free(n_osc, m=2):
        if n_osc <= m:
            return np.ones((n_osc, n_osc)) - np.eye(n_osc)

        grados = np.zeros(n_osc)
        A = np.zeros((n_osc, n_osc))

        for i in range(m+1):
            for j in range(i+1, m+1):
                A[i, j] = 1
                A[j, i] = 1
                grados[i] += 1
                grados[j] += 1

        for nuevo in range(m+1, n_osc):
            prob = grados[:nuevo] / np.sum(grados[:nuevo])
            elegidos = np.random.choice(nuevo, size=m, replace=False, p=prob)
            for viejo in elegidos:
                A[nuevo, viejo] = 1
                A[viejo, nuevo] = 1
                grados[nuevo] += 1
                grados[viejo] += 1
        return A

    class KuramotoJerarquico:
        def __init__(self, niveles=5, metros_por_plataforma=3,
                     topologia_intra='global', topologia_inter='jerarquica',
                     K_intra=2.0, K_inter=1.5, K_lateral=0.5):
            self.niveles = niveles
            self.mpp = metros_por_plataforma
            self.topologia_intra = topologia_intra
            self.topologia_inter = topologia_inter
            self.K_intra = K_intra
            self.K_inter = K_inter
            self.K_lateral = K_lateral

            self.plataformas_por_nivel = [3**i for i in range(niveles)]
            self.osciladores_por_nivel = [p * metros_por_plataforma for p in self.plataformas_por_nivel]
            self.total = sum(self.osciladores_por_nivel)

            np.random.seed(42)
            self.omega = np.concatenate([
                np.random.normal(1.0, 0.1 + 0.02*i, n)
                for i, n in enumerate(self.osciladores_por_nivel)
            ])

            self.adjacencias = self._construir_adjacencias()
            self.indices = self._construir_indices()

        def _construir_indices(self):
            indices = []
            start = 0
            for n in self.osciladores_por_nivel:
                indices.append(slice(start, start + n))
                start += n
            return indices

        def _construir_adjacencias(self):
            adj = []
            for nivel in range(self.niveles):
                n_plt = self.plataformas_por_nivel[nivel]
                adj_nivel = []
                for p in range(n_plt):
                    n_osc = self.mpp
                    if self.topologia_intra == 'global':
                        A = np.ones((n_osc, n_osc)) - np.eye(n_osc)
                    elif self.topologia_intra == 'scale_free':
                        if n_osc > 2:
                            A = generar_scale_free(n_osc, m=2)
                        else:
                            A = np.ones((n_osc, n_osc)) - np.eye(n_osc)
                    else:
                        A = np.ones((n_osc, n_osc)) - np.eye(n_osc)
                    adj_nivel.append(A)
                adj.append(adj_nivel)
            return adj

        def dynamics(self, t, theta):
            dtheta = np.zeros_like(theta)

            theta0 = theta[self.indices[0]]
            n0 = len(theta0)
            A0 = self.adjacencias[0][0]
            for i in range(n0):
                suma = np.sum(A0[i] * np.sin(theta0 - theta0[i]))
                dtheta[self.indices[0]][i] = self.omega[self.indices[0]][i] + (self.K_intra/n0) * suma

            for nivel in range(1, self.niveles):
                n_plt = self.plataformas_por_nivel[nivel]
                n_por_plt = self.mpp
                theta_n = theta[self.indices[nivel]].reshape(n_plt, n_por_plt)
                dtheta_n = np.zeros_like(theta_n)

                theta_prev = theta[self.indices[nivel-1]].reshape(self.plataformas_por_nivel[nivel-1], self.mpp)
                r_prev = np.mean(np.exp(1j * theta_prev), axis=1)
                phi_prev = np.angle(r_prev)

                conexiones_laterales = []
                if self.topologia_inter == 'malla':
                    for p in range(n_plt):
                        if p > 0:
                            conexiones_laterales.append((p, p-1))
                        if p < n_plt - 1:
                            conexiones_laterales.append((p, p+1))

                norm_lateral = max(1, len(conexiones_laterales) * n_por_plt)

                for p in range(n_plt):
                    madre = p // 3
                    A_plt = self.adjacencias[nivel][p]
                    for i in range(n_por_plt):
                        intra = 0
                        for j in range(n_por_plt):
                            if A_plt[i, j] > 0:
                                intra += np.sin(theta_n[p, j] - theta_n[p, i])
                        intra = (self.K_intra / n_por_plt) * intra
                        inter = self.K_inter * np.sin(phi_prev[madre] - theta_n[p, i])
                        
                        lateral = 0
                        for (p1, p2) in conexiones_laterales:
                            if p == p1:
                                for j in range(n_por_plt):
                                    lateral += np.sin(theta_n[p2, j] - theta_n[p, i])
                            elif p == p2:
                                for j in range(n_por_plt):
                                    lateral += np.sin(theta_n[p1, j] - theta_n[p, i])
                        lateral = self.K_lateral * lateral / norm_lateral

                        idx = self.indices[nivel].start + p*n_por_plt + i
                        dtheta_n[p, i] = self.omega[idx] + intra + inter + lateral

                dtheta[self.indices[nivel]] = dtheta_n.flatten()
            return dtheta

        def simular(self, T=25, puntos=150):
            theta0 = np.random.uniform(-np.pi, np.pi, self.total)
            sol = solve_ivp(self.dynamics, (0, T), theta0,
                            t_eval=np.linspace(0, T, puntos),
                            method='RK45', rtol=1e-2)
            r_nivel = []
            for nivel in range(self.niveles):
                fases = sol.y[self.indices[nivel]]
                if len(fases.shape) == 1:
                    r = np.abs(np.mean(np.exp(1j * fases), axis=0))
                else:
                    r = np.abs(np.mean(np.exp(1j * fases), axis=0))
                r_nivel.append(r)
            return sol.t, r_nivel

    # Ejecutar opción 6 (Scale-free + Malla)
    output_lines = []
    output_lines.append("\n🔬 EXPERIMENTO: Scale-free + Malla (Opción 6)")
    
    sim = KuramotoJerarquico(niveles=5, metros_por_plataforma=3,
                             topologia_intra='scale_free',
                             topologia_inter='malla',
                             K_lateral=0.3)
    
    t, r_nivel = sim.simular(T=25, puntos=150)
    
    r_final = [r[-1] for r in r_nivel]
    output_lines.append("\n=== RESULTADOS ===")
   
