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
CORS(app)  # Permite que tu web en Neocities se conecte

# ============================================
# AQUÍ PEGAS TODO EL CÓDIGO DE TUS SCRIPTS
# ============================================
# 1. Código de kuramoto1.py COMPLETO
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation

# ============================================
# PARÁMETROS DEL SISTEMA
# ============================================

# Estructura de capas: [nivel1, nivel2, nivel3]
# nivel1: 1 plataforma con N1 osciladores
# nivel2: 3 plataformas con N2 osciladores cada una
# nivel3: 9 plataformas con N3 osciladores cada una

N1 = 5      # metrónomos en plataforma raíz
N2 = 4      # metrónomos en cada plataforma de nivel 2
N3 = 3      # metrónomos en cada plataforma de nivel 3

# Frecuencias naturales (distribución Lorentziana para variabilidad)
np.random.seed(42)  # reproducibilidad
omega1 = np.random.normal(1.0, 0.1, N1)           # nivel 1
omega2 = np.random.normal(1.0, 0.15, 3 * N2)      # nivel 2 (3 plataformas)
omega3 = np.random.normal(1.0, 0.2, 9 * N3)        # nivel 3 (9 plataformas)

# Fuerzas de acoplamiento
K_intra = 2.0      # acoplamiento dentro de cada plataforma
K_inter = 1.5      # fuerza con que la plataforma inferior influye en la superior

# Tiempo de simulación
t_span = (0, 50)
t_eval = np.linspace(0, 50, 1000)

# ============================================
# CONSTRUCCIÓN DEL VECTOR DE ESTADO
# ============================================
# El vector de estado tendrá la forma:
# [fases_nivel1 (N1), fases_nivel2 (3*N2), fases_nivel3 (9*N3)]

def indices():
    """Devuelve slices para acceder a cada nivel en el vector de estado"""
    idx1 = slice(0, N1)
    idx2 = slice(N1, N1 + 3*N2)
    idx3 = slice(N1 + 3*N2, N1 + 3*N2 + 9*N3)
    return idx1, idx2, idx3

idx1, idx2, idx3 = indices()

# ============================================
# ECUACIONES DIFERENCIALES (MODELO COMPLETO)
# ============================================

def kuramoto_hierarchical(t, theta):
    dtheta = np.zeros_like(theta)

    # ---- NIVEL 1: solo acoplamiento intra-plataforma ----
    theta1 = theta[idx1]
    for i in range(N1):
        # Kuramoto estándar dentro de la plataforma 1
        dtheta[i] = omega1[i] + (K_intra/N1) * np.sum(np.sin(theta1 - theta1[i]))

    # ---- NIVEL 2: acoplamiento intra + influencia del nivel 1 ----
    theta2 = theta[idx2].reshape(3, N2)
    dtheta2 = np.zeros_like(theta2)

    # Fase colectiva del nivel 1 (influye en TODAS las plataformas del nivel 2)
    r1 = np.mean(np.exp(1j * theta1))
    phi1 = np.angle(r1)  # fase promedio del nivel 1

    for p in range(3):  # para cada plataforma del nivel 2
        for i in range(N2):
            # Término intra-plataforma
            intra = (K_intra/N2) * np.sum(np.sin(theta2[p] - theta2[p, i]))
            # Término inter-capa (influencia de la base)
            inter = K_inter * np.sin(phi1 - theta2[p, i])
            dtheta2[p, i] = omega2[p*N2 + i] + intra + inter

    dtheta[idx2] = dtheta2.flatten()

    # ---- NIVEL 3: acoplamiento intra + influencia de su plataforma madre en nivel 2 ----
    theta3 = theta[idx3].reshape(9, N3)
    dtheta3 = np.zeros_like(theta3)

    # Fases colectivas de CADA plataforma del nivel 2 (cada una influye en sus 3 hijas)
    r2 = np.mean(np.exp(1j * theta2), axis=1)  # 3 fases colectivas
    phi2 = np.angle(r2)

    for p in range(9):  # 9 plataformas en nivel 3
        madre = p // 3  # qué plataforma del nivel 2 es su madre (0,1,2)
        for i in range(N3):
            intra = (K_intra/N3) * np.sum(np.sin(theta3[p] - theta3[p, i]))
            inter = K_inter * np.sin(phi2[madre] - theta3[p, i])
            dtheta3[p, i] = omega3[p*N3 + i] + intra + inter

    dtheta[idx3] = dtheta3.flatten()

    return dtheta

# ============================================
# SIMULACIÓN
# ============================================

# Condición inicial: fases aleatorias entre -pi y pi
theta0 = np.concatenate([
    np.random.uniform(-np.pi, np.pi, N1),
    np.random.uniform(-np.pi, np.pi, 3*N2),
    np.random.uniform(-np.pi, np.pi, 9*N3)
])

# Resolver EDOs
print("Simulando sistema jerárquico de 44 osciladores...")
sol = solve_ivp(kuramoto_hierarchical, t_span, theta0, t_eval=t_eval, method='RK45')

# ============================================
# CÁLCULO DE PARÁMETROS DE ORDEN POR NIVEL
# ============================================

# Parámetro de orden para el nivel 1
r1_t = np.abs(np.mean(np.exp(1j * sol.y[idx1]), axis=0))

# Parámetros de orden para cada plataforma del nivel 2
theta2_t = sol.y[idx2].reshape(3, N2, -1)
r2_plataformas = np.abs(np.mean(np.exp(1j * theta2_t), axis=1))  # (3, time)

# Parámetros de orden para cada plataforma del nivel 3
theta3_t = sol.y[idx3].reshape(9, N3, -1)
r3_plataformas = np.abs(np.mean(np.exp(1j * theta3_t), axis=1))  # (9, time)

# ============================================
# VISUALIZACIÓN
# ============================================

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Nivel 1
axes[0].plot(sol.t, r1_t, 'b-', linewidth=2)
axes[0].set_ylabel('r1 (Nivel 1)')
axes[0].set_ylim(0, 1.1)
axes[0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
axes[0].grid(True, alpha=0.3)
axes[0].set_title('Sincronización en Plataforma Raíz (Nivel 1)')

# Nivel 2
for p in range(3):
    axes[1].plot(sol.t, r2_plataformas[p], label=f'Plataforma {p+1}', linewidth=1.5)
axes[1].set_ylabel('r2 (Nivel 2)')
axes[1].set_ylim(0, 1.1)
axes[1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
axes[1].grid(True, alpha=0.3)
axes[1].legend(loc='upper right')
axes[1].set_title('Sincronización en 3 Plataformas del Nivel 2')

# Nivel 3
for p in range(9):
    axes[2].plot(sol.t, r3_plataformas[p], label=f'P{p+1}', linewidth=1, alpha=0.7)
axes[2].set_xlabel('Tiempo')
axes[2].set_ylabel('r3 (Nivel 3)')
axes[2].set_ylim(0, 1.1)
axes[2].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
axes[2].grid(True, alpha=0.3)
axes[2].set_title('Sincronización en 9 Plataformas del Nivel 3')

plt.tight_layout()
plt.show()

# ============================================
# ANÁLISIS: ¿CUÁNDO SE SINCRONIZA CADA NIVEL?
# ============================================

# Definir sincronización como r > 0.9
umbral = 0.9

t_sync1 = sol.t[np.where(r1_t > umbral)[0][0]] if np.any(r1_t > umbral) else np.inf

t_sync2 = []
for p in range(3):
    idx = np.where(r2_plataformas[p] > umbral)[0]
    t_sync2.append(sol.t[idx[0]] if len(idx) > 0 else np.inf)

t_sync3 = []
for p in range(9):
    idx = np.where(r3_plataformas[p] > umbral)[0]
    t_sync3.append(sol.t[idx[0]] if len(idx) > 0 else np.inf)

print("\n=== TIEMPOS DE SINCRONIZACIÓN (r > 0.9) ===")
print(f"Nivel 1: t = {t_sync1:.2f}")
print("\nNivel 2:")
for p, t in enumerate(t_sync2):
    print(f"  Plataforma {p+1}: t = {t:.2f}" if t != np.inf else f"  Plataforma {p+1}: NO sincroniza")
print("\nNivel 3:")
for p, t in enumerate(t_sync3):
    print(f"  Plataforma {p+1}: t = {t:.2f}" if t != np.inf else f"  Plataforma {p+1}: NO sincroniza")

# ============================================
# ANIMACIÓN DE FASES (OPCIONAL - REQUIERE FFMPEG)
# ============================================
"""
def update(frame):
    plt.clf()
    theta_frame = sol.y[:, frame]

    # Proyectar fases en círculo unitario
    x1 = np.cos(theta_frame[idx1])
    y1 = np.sin(theta_frame[idx1])

    x2 = np.cos(theta_frame[idx2])
    y2 = np.sin(theta_frame[idx2])

    x3 = np.cos(theta_frame[idx3])
    y3 = np.sin(theta_frame[idx3])

    plt.scatter(x1, y1, c='blue', s=50, label='Nivel 1')
    plt.scatter(x2, y2, c='red', s=30, label='Nivel 2', alpha=0.7)
    plt.scatter(x3, y3, c='green', s=20, label='Nivel 3', alpha=0.5)

    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.gca().set_aspect('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(f't = {sol.t[frame]:.2f}')

ani = animation.FuncAnimation(plt.figure(), update, frames=len(sol.t), interval=50)
ani.save('sincronizacion_fractal.mp4', writer='ffmpeg')
print("Animación guardada como 'sincronizacion_fractal.mp4'")
"""

# 2. Código de kuramoto2.py COMPLETO
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def simular_kuramoto(N1=5, N2=4, N3=3, K_intra=2.0, K_inter=1.5,
                     T=50, puntos=1000, semilla=42):
    """
    Simula el sistema jerárquico de Kuramoto con parámetros ajustables
    """
    print(f"\n{'='*50}")
    print(f"Simulando: {N1} + 3×{N2} + 9×{N3} = {N1 + 3*N2 + 9*N3} osciladores")
    print(f"K_intra={K_intra}, K_inter={K_inter}, T={T}s")
    print('='*50)

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

    # Mostrar resultados
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    axes[0].plot(sol.t, r1, 'b-', linewidth=2)
    axes[0].set_ylabel('Nivel 1 (Raíz)')
    axes[0].set_ylim(0, 1.1)
    axes[0].axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(sol.t, r2.T, linewidth=1.5)
    axes[1].set_ylabel('Nivel 2 (3 plataformas)')
    axes[1].set_ylim(0, 1.1)
    axes[1].axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(sol.t, r3.T, linewidth=1, alpha=0.7)
    axes[2].set_xlabel('Tiempo (s)')
    axes[2].set_ylabel('Nivel 3 (9 plataformas)')
    axes[2].set_ylim(0, 1.1)
    axes[2].axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f'Sincronización Jerárquica (K_intra={K_intra}, K_inter={K_inter})')
    plt.tight_layout()
    plt.show()

    # Análisis cuantitativo
    print("\n=== ANÁLISIS DE SINCRONIZACIÓN ===")
    print(f"Nivel 1 - Final: r={r1[-1]:.3f}")
    print(f"Nivel 2 - Promedio final: r={np.mean(r2[:,-1]):.3f}")
    print(f"Nivel 3 - Promedio final: r={np.mean(r3[:,-1]):.3f}")

    return sol, (r1, r2, r3)

# ============================================
# EXPLORACIÓN DE PARÁMETROS
# ============================================

if __name__ == "__main__":
    # Caso 1: Configuración original
    simular_kuramoto(N1=5, N2=4, N3=3, K_intra=2.0, K_inter=1.5, T=50)

    # Caso 2: Acoplamiento inter-capa débil
    simular_kuramoto(N1=5, N2=4, N3=3, K_intra=2.0, K_inter=0.5, T=50)

    # Caso 3: Acoplamiento inter-capa fuerte
    simular_kuramoto(N1=5, N2=4, N3=3, K_intra=2.0, K_inter=3.0, T=50)

    # Caso 4: Sistema más grande (si quieres)
    # simular_kuramoto(N1=8, N2=6, N3=4, K_intra=2.0, K_inter=1.5, T=50)

# 3. Código de kuramoto3.py COMPLETO
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def simular_niveles(niveles=6, metros_por_plataforma=3, K_intra=2.0, K_inter=1.5):
    """
    Simula sistema jerárquico hasta N niveles y muestra TODOS
    """
    print(f"\n{'='*60}")
    print(f"SIMULANDO {niveles} NIVELES JERÁRQUICOS")
    print(f"Metrónomos por plataforma: {metros_por_plataforma}")

    # Calcular estructura
    plataformas_por_nivel = [3**i for i in range(niveles)]
    osciladores_por_nivel = [p * metros_por_plataforma for p in plataformas_por_nivel]
    total_osciladores = sum(osciladores_por_nivel)

    print(f"Plataformas por nivel: {plataformas_por_nivel}")
    print(f"OSCILADORES TOTALES: {total_osciladores}")
    print(f"Memoria estimada: {total_osciladores * 1000 * 8 / 1e6:.1f} MB")
    print('='*60)

    # Generar frecuencias
    np.random.seed(42)
    frecuencias = []
    for nivel in range(niveles):
        std = 0.1 + 0.02 * nivel
        n_osc = osciladores_por_nivel[nivel]
        frecuencias.append(np.random.normal(1.0, std, n_osc))

    omega = np.concatenate(frecuencias)

    # Configuración temporal
    t_span = (0, 30)
    t_eval = np.linspace(0, 30, 300)

    # Construir índices
    indices = []
    start = 0
    for n in osciladores_por_nivel:
        indices.append(slice(start, start + n))
        start += n

    # Función de dinámica
    def dynamics(t, theta):
        dtheta = np.zeros_like(theta)

        # Nivel 0 (base)
        theta0 = theta[indices[0]]
        for i in range(len(theta0)):
            dtheta[indices[0]][i] = omega[indices[0]][i] + (K_intra/len(theta0)) * np.sum(np.sin(theta0 - theta0[i]))

        # Niveles superiores
        for nivel in range(1, niveles):
            n_plataformas = 3**nivel
            n_por_plataforma = metros_por_plataforma
            theta_n = theta[indices[nivel]].reshape(n_plataformas, n_por_plataforma)
            dtheta_n = np.zeros_like(theta_n)

            theta_prev = theta[indices[nivel-1]].reshape(3**(nivel-1), metros_por_plataforma)
            r_prev = np.mean(np.exp(1j * theta_prev), axis=1)
            phi_prev = np.angle(r_prev)

            for p in range(n_plataformas):
                madre = p // 3
                for i in range(n_por_plataforma):
                    intra = (K_intra/n_por_plataforma) * np.sum(np.sin(theta_n[p] - theta_n[p, i]))
                    inter = K_inter * np.sin(phi_prev[madre] - theta_n[p, i])
                    idx_global = indices[nivel].start + p*n_por_plataforma + i
                    dtheta_n[p, i] = omega[idx_global] + intra + inter

            dtheta[indices[nivel]] = dtheta_n.flatten()

        return dtheta

    # Condición inicial
    theta0 = np.random.uniform(-np.pi, np.pi, total_osciladores)

    # Simular
    print("Iniciando simulación...")
    sol = solve_ivp(dynamics, t_span, theta0, t_eval=t_eval, method='RK45', rtol=1e-2)

    # Calcular sincronización por nivel
    r_por_nivel = []
    for nivel in range(niveles):
        fases_nivel = sol.y[indices[nivel]]
        if len(fases_nivel.shape) == 1:
            r = np.abs(np.mean(np.exp(1j * fases_nivel), axis=0))
        else:
            r = np.abs(np.mean(np.exp(1j * fases_nivel), axis=0))
        r_por_nivel.append(r)

    # ===== VISUALIZACIÓN CORREGIDA - TODOS LOS NIVELES =====
    fig, axes = plt.subplots(niveles, 1, figsize=(12, 2*niveles + 4))

    # Asegurar que axes sea una lista aunque haya 1 nivel
    if niveles == 1:
        axes = [axes]

    for i in range(niveles):
        axes[i].plot(sol.t, r_por_nivel[i], linewidth=2, color=plt.cm.viridis(i/niveles))
        axes[i].set_ylabel(f'Nivel {i}\n({plataformas_por_nivel[i]} pltas)', fontsize=9)
        axes[i].set_ylim(0, 1.1)
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        axes[i].set_yticks([0, 0.5, 1.0])

        # Añadir etiqueta con el valor final
        axes[i].text(0.95, 0.1, f'r={r_por_nivel[i][-1]:.2f}',
                    transform=axes[i].transAxes, ha='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[-1].set_xlabel('Tiempo (s)')
    plt.suptitle(f'Sincronización en {niveles} Niveles Jerárquicos', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()

    # ===== GRÁFICO ADICIONAL: EVOLUCIÓN POR NIVEL =====
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    # Mostrar la sincronización final vs nivel
    niveles_x = np.arange(niveles)
    r_final = [r[-1] for r in r_por_nivel]

    ax2.plot(niveles_x, r_final, 'bo-', linewidth=2, markersize=8)
    ax2.set_xlabel('Nivel jerárquico')
    ax2.set_ylabel('Sincronización final (r)')
    ax2.set_title('Degradación de la coherencia con la profundidad')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(niveles_x)
    ax2.set_xticklabels([f'N{i}\n({plataformas_por_nivel[i]})' for i in niveles_x])
    ax2.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='Umbral coherencia')
    ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Umbral caos')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # Resumen
    print("\n=== RESULTADOS POR NIVEL ===")
    for i in range(niveles):
        estado = "✅ Coherente" if r_final[i] > 0.8 else "⚠️  Difuso" if r_final[i] > 0.5 else "❌ Caótico"
        print(f"Nivel {i:2d} ({plataformas_por_nivel[i]:3d} pltas): r={r_final[i]:.3f} {estado}")

    return sol, r_por_nivel

# ============================================
# EJECUTAR
# ============================================
if __name__ == "__main__":
    print("¿Cuántos niveles quieres simular?")
    print("• 5 niveles: 363 osciladores (30s)")
    print("• 6 niveles: 1,092 osciladores (5min)")
    print("• 7 niveles: 3,279 osciladores (30min)")

    try:
        n = int(input("Niveles (5-7): "))
        if 5 <= n <= 7:
            simular_niveles(n, 3, K_intra=2.0, K_inter=1.5)
        else:
            print("Por favor elige entre 5 y 7")
    except:
        print("Usando 6 niveles por defecto")
        simular_niveles(6, 3, K_intra=2.0, K_inter=1.5)

# 4. Código de kuramoto4.py COMPLETO
import numpy as np
import matplotlib.pyplot as plt

def modelo_efectivo_46_capas():
    """
    Modela 46 capas SIN simular cada oscilador individual
    Usa parámetros de orden promedio por capa
    """
    n_capas = 46
    K_intra = 2.0
    K_inter_base = 1.5
    gamma_acumulado = np.ones(n_capas)  # factor de Lorentz acumulado

    # Parámetro de orden por capa (r)
    r = np.zeros(n_capas)
    r[0] = 0.98  # capa base casi sincronizada

    # Tiempo efectivo por capa (se ralentiza)
    tiempo_efectivo = np.zeros(n_capas)

    for capa in range(1, n_capas):
        # El acoplamiento se degrada con la profundidad
        K_inter = K_inter_base * np.exp(-0.1 * capa)

        # La sincronización se propaga con pérdidas
        r_capa_anterior = r[capa-1]

        # Modelo efectivo: r nueva = f(r_anterior, K_inter)
        r[capa] = r_capa_anterior * (1 - np.exp(-K_inter)) * np.tanh(K_intra)

        # Tiempo se ralentiza (efecto de masa informacional)
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
    plt.show()

    # Punto crítico donde se pierde la coherencia
    capa_critica = np.where(r < 0.5)[0][0] if np.any(r < 0.5) else n_capas
    print(f"Pérdida de coherencia (r<0.5) en capa: {capa_critica}")
    print(f"Sincronización final capa 46: r={r[-1]:.4f}")

    return r, tiempo_efectivo

# Ejecutar
r, t_efectivo = modelo_efectivo_46_capas()

# 5. Código de kuramoto5.py COMPLETO
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def generar_scale_free(n_osc, m=2):
    """
    Genera matriz de adyacencia scale-free (Barabási-Albert)
    sin usar networkx
    """
    if n_osc <= m:
        # Si es muy pequeño, devolver completo
        A = np.ones((n_osc, n_osc)) - np.eye(n_osc)
        return A

    # Inicializar con un pequeño clique
    grados = np.zeros(n_osc)
    A = np.zeros((n_osc, n_osc))

    # Conectar los primeros m+1 nodos en un clique
    for i in range(m+1):
        for j in range(i+1, m+1):
            A[i, j] = 1
            A[j, i] = 1
            grados[i] += 1
            grados[j] += 1

    # Añadir nodos restantes con preferential attachment
    for nuevo in range(m+1, n_osc):
        # Calcular probabilidades proporcionales al grado
        prob = grados[:nuevo] / np.sum(grados[:nuevo])

        # Seleccionar m nodos existentes para conectar
        elegidos = np.random.choice(nuevo, size=m, replace=False, p=prob)

        for viejo in elegidos:
            A[nuevo, viejo] = 1
            A[viejo, nuevo] = 1
            grados[nuevo] += 1
            grados[viejo] += 1

    return A

class KuramotoJerarquico:
    def __init__(self, niveles=6, metros_por_plataforma=3,
                 topologia_intra='global',     # global, anillo, malla, scale_free, estrella
                 topologia_inter='jerarquica', # jerarquica, malla, estrella
                 K_intra=2.0, K_inter=1.5, K_lateral=0.5):

        self.niveles = niveles
        self.mpp = metros_por_plataforma
        self.topologia_intra = topologia_intra
        self.topologia_inter = topologia_inter
        self.K_intra = K_intra
        self.K_inter = K_inter
        self.K_lateral = K_lateral

        # Estructura de plataformas
        self.plataformas_por_nivel = [3**i for i in range(niveles)]
        self.osciladores_por_nivel = [p * metros_por_plataforma for p in self.plataformas_por_nivel]
        self.total = sum(self.osciladores_por_nivel)

        print(f"\n{'='*60}")
        print(f"📊 CONFIGURACIÓN:")
        print(f"  Niveles: {niveles}")
        print(f"  Total osciladores: {self.total}")
        print(f"  Topología intra-plataforma: {topologia_intra}")
        print(f"  Topología inter-plataforma: {topologia_inter}")
        print(f"{'='*60}")

        # Generar frecuencias
        np.random.seed(42)
        self.omega = np.concatenate([
            np.random.normal(1.0, 0.1 + 0.02*i, n)
            for i, n in enumerate(self.osciladores_por_nivel)
        ])

        # Construir matrices de adyacencia para cada plataforma
        self.adjacencias = self._construir_adjacencias()

        # Índices globales
        self.indices = self._construir_indices()

    def _construir_indices(self):
        indices = []
        start = 0
        for n in self.osciladores_por_nivel:
            indices.append(slice(start, start + n))
            start += n
        return indices

    def _construir_adjacencias(self):
        """Construye matriz de adyacencia para cada plataforma"""
        adj = []

        for nivel in range(self.niveles):
            n_plt = self.plataformas_por_nivel[nivel]
            adj_nivel = []

            for p in range(n_plt):
                n_osc = self.mpp

                if self.topologia_intra == 'global':
                    # Todos con todos (menos sí mismo)
                    A = np.ones((n_osc, n_osc)) - np.eye(n_osc)

                elif self.topologia_intra == 'anillo':
                    # Cada uno con sus dos vecinos
                    A = np.zeros((n_osc, n_osc))
                    for i in range(n_osc):
                        A[i, (i-1)%n_osc] = 1
                        A[i, (i+1)%n_osc] = 1

                elif self.topologia_intra == 'malla_1d':
                    # Línea (extremos no conectados)
                    A = np.zeros((n_osc, n_osc))
                    for i in range(n_osc-1):
                        A[i, i+1] = 1
                        A[i+1, i] = 1

                elif self.topologia_intra == 'estrella':
                    # Nodo central conectado a todos
                    A = np.zeros((n_osc, n_osc))
                    centro = 0  # primer nodo como centro
                    for i in range(1, n_osc):
                        A[centro, i] = 1
                        A[i, centro] = 1

                elif self.topologia_intra == 'scale_free':
                    # Red libre de escala (implementación manual)
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

        # Nivel 0 (base)
        theta0 = theta[self.indices[0]]
        n0 = len(theta0)
        A0 = self.adjacencias[0][0]  # única plataforma

        for i in range(n0):
            suma = np.sum(A0[i] * np.sin(theta0 - theta0[i]))
            dtheta[self.indices[0]][i] = self.omega[self.indices[0]][i] + (self.K_intra/n0) * suma

        # Niveles superiores
        for nivel in range(1, self.niveles):
            n_plt = self.plataformas_por_nivel[nivel]
            n_por_plt = self.mpp

            theta_n = theta[self.indices[nivel]].reshape(n_plt, n_por_plt)
            dtheta_n = np.zeros_like(theta_n)

            # Fases del nivel anterior (madres)
            theta_prev = theta[self.indices[nivel-1]].reshape(self.plataformas_por_nivel[nivel-1], self.mpp)
            r_prev = np.mean(np.exp(1j * theta_prev), axis=1)
            phi_prev = np.angle(r_prev)

            # Construir conexiones laterales según topología
            conexiones_laterales = []
            if self.topologia_inter == 'malla':
                # Conectar plataformas vecinas
                for p in range(n_plt):
                    if p > 0:
                        conexiones_laterales.append((p, p-1))
                    if p < n_plt - 1:
                        conexiones_laterales.append((p, p+1))
            elif self.topologia_inter == 'estrella' and n_plt > 1:
                # Una plataforma central conectada a todas
                centro = n_plt // 2
                for p in range(n_plt):
                    if p != centro:
                        conexiones_laterales.append((centro, p))
            elif self.topologia_inter == 'global':
                # Todas conectadas con todas
                for p1 in range(n_plt):
                    for p2 in range(p1+1, n_plt):
                        conexiones_laterales.append((p1, p2))

            # Normalización para laterales
            norm_lateral = max(1, len(conexiones_laterales) * n_por_plt)

            for p in range(n_plt):
                madre = p // 3
                A_plt = self.adjacencias[nivel][p]

                # Obtener r de esta plataforma (para feedback)
                r_local = np.abs(np.mean(np.exp(1j * theta_n[p])))

                # K_inter ajustado por feedback (si r_local bajo, más acoplamiento)
                K_inter_efectivo = self.K_inter * (1 + 0.5 * (1 - r_local))

                for i in range(n_por_plt):
                    # Término intra-plataforma (con topología)
                    intra = 0
                    for j in range(n_por_plt):
                        if A_plt[i, j] > 0:
                            intra += np.sin(theta_n[p, j] - theta_n[p, i])
                    intra = (self.K_intra / n_por_plt) * intra

                    # Término jerárquico (madre)
                    inter = K_inter_efectivo * np.sin(phi_prev[madre] - theta_n[p, i])

                    # Términos laterales (entre plataformas)
                    lateral = 0
                    for (p1, p2) in conexiones_laterales:
                        if p == p1:
                            # Conectar con la otra plataforma
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

    def simular(self, T=30, puntos=200):
        theta0 = np.random.uniform(-np.pi, np.pi, self.total)

        print("\n⏳ Simulando... (puede tomar varios minutos)")
        print(f"   T={T}s, puntos={puntos}")

        sol = solve_ivp(self.dynamics, (0, T), theta0,
                        t_eval=np.linspace(0, T, puntos),
                        method='RK45', rtol=1e-2)

        # Calcular sincronización por nivel
        r_nivel = []
        for nivel in range(self.niveles):
            fases = sol.y[self.indices[nivel]]
            if len(fases.shape) == 1:
                r = np.abs(np.mean(np.exp(1j * fases), axis=0))
            else:
                r = np.abs(np.mean(np.exp(1j * fases), axis=0))
            r_nivel.append(r)

        return sol.t, r_nivel

    def graficar(self, t, r_nivel):
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Evolución temporal
        colores = plt.cm.viridis(np.linspace(0, 1, len(r_nivel)))
        for i, r in enumerate(r_nivel):
            axes[0].plot(t, r, color=colores[i], label=f'N{i}', linewidth=1.5)
        axes[0].set_xlabel('Tiempo')
        axes[0].set_ylabel('Sincronización r')
        axes[0].set_ylim(0, 1.1)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='upper right', ncol=3)
        axes[0].set_title(f'Evolución temporal (intra={self.topologia_intra}, inter={self.topologia_inter})')

        # Valores finales
        r_final = [r[-1] for r in r_nivel]
        barras = axes[1].bar(range(self.niveles), r_final, color=colores)
        axes[1].set_xlabel('Nivel')
        axes[1].set_ylabel('r final')
        axes[1].set_ylim(0, 1.1)
        axes[1].axhline(y=0.8, color='g', linestyle='--', alpha=0.5, label='Coherente')
        axes[1].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Difuso')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        # Añadir valores sobre las barras
        for i, (bar, r) in enumerate(zip(barras, r_final)):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{r:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.show()

        # Mostrar resultados
        print("\n=== RESULTADOS ===")
        print("Nivel | Plataformas | r_final | Estado")
        print("-" * 45)
        for i, r in enumerate(r_final):
            estado = "✅ COHERENTE" if r > 0.8 else "⚠️ DIFUSO" if r > 0.5 else "❌ CAÓTICO"
            print(f"{i:3d}   | {self.plataformas_por_nivel[i]:6d}      | {r:.3f}   | {estado}")

# ============================================
# EXPERIMENTOS
# ============================================

if __name__ == "__main__":
    print("\n🔬 EXPERIMENTOS DE TOPOLOGÍA (sin networkx)")
    print("1. Control (global + jerárquica)")
    print("2. Anillo intra + jerárquica")
    print("3. Scale-free intra + jerárquica")
    print("4. Estrella intra + jerárquica")
    print("5. Global + malla inter (laterales)")
    print("6. Scale-free + malla inter")
    print("7. Scale-free + global inter (todas conectadas)")

    try:
        opcion = input("\nElige (1-7): ")

        if opcion == '1':
            sim = KuramotoJerarquico(niveles=5, metros_por_plataforma=3,
                                     topologia_intra='global',
                                     topologia_inter='jerarquica')
        elif opcion == '2':
            sim = KuramotoJerarquico(niveles=5, metros_por_plataforma=3,
                                     topologia_intra='anillo',
                                     topologia_inter='jerarquica')
        elif opcion == '3':
            sim = KuramotoJerarquico(niveles=5, metros_por_plataforma=3,
                                     topologia_intra='scale_free',
                                     topologia_inter='jerarquica')
        elif opcion == '4':
            sim = KuramotoJerarquico(niveles=5, metros_por_plataforma=3,
                                     topologia_intra='estrella',
                                     topologia_inter='jerarquica')
        elif opcion == '5':
            sim = KuramotoJerarquico(niveles=5, metros_por_plataforma=3,
                                     topologia_intra='global',
                                     topologia_inter='malla',
                                     K_lateral=0.3)
        elif opcion == '6':
            sim = KuramotoJerarquico(niveles=5, metros_por_plataforma=3,
                                     topologia_intra='scale_free',
                                     topologia_inter='malla',
                                     K_lateral=0.3)
        elif opcion == '7':
            sim = KuramotoJerarquico(niveles=5, metros_por_plataforma=3,
                                     topologia_intra='scale_free',
                                     topologia_inter='global',
                                     K_lateral=0.5)
        else:
            print("Opción no válida, usando control")
            sim = KuramotoJerarquico(niveles=5, metros_por_plataforma=3)

        t, r = sim.simular(T=25, puntos=150)
        sim.graficar(t, r)

    except KeyboardInterrupt:
        print("\n\n⚠️ Simulación interrumpida por usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")

# 6. Código de kuramoto6.py COMPLETO
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation

def generar_scale_free(n_osc, m=2):
    """Genera matriz de adyacencia scale-free"""
    if n_osc <= m:
        return np.ones((n_osc, n_osc)) - np.eye(n_osc)

    grados = np.zeros(n_osc)
    A = np.zeros((n_osc, n_osc))

    # Inicializar con clique
    for i in range(m+1):
        for j in range(i+1, m+1):
            A[i, j] = 1
            A[j, i] = 1
            grados[i] += 1
            grados[j] += 1

    # Preferential attachment
    for nuevo in range(m+1, n_osc):
        prob = grados[:nuevo] / np.sum(grados[:nuevo])
        elegidos = np.random.choice(nuevo, size=m, replace=False, p=prob)
        for viejo in elegidos:
            A[nuevo, viejo] = 1
            A[viejo, nuevo] = 1
            grados[nuevo] += 1
            grados[viejo] += 1

    return A

class KuramotoPODB:
    """
    Cada conexión entre osciladores tiene su propio estado P-O-D-B
    basado en su diferencia de fase:

    - Estado P (Partícula):   Δθ ≈ 0     → K máximo
    - Estado O (Onda):        Δθ ≈ π/2   → K medio
    - Estado D (Difuso):      Δθ aleatorio → K bajo
    - Estado B (Borrado):     Δθ ≈ π     → K = 0
    """

    # Definición de los estados
    ESTADOS = {
        'P': {'color': 'blue', 'rango': (0.8, 1.0), 'desc': 'Partícula'},
        'O': {'color': 'green', 'rango': (0.3, 0.8), 'desc': 'Onda'},
        'D': {'color': 'orange', 'rango': (0.1, 0.3), 'desc': 'Difuso'},
        'B': {'color': 'red', 'rango': (0.0, 0.1), 'desc': 'Borrado'}
    }

    def __init__(self, niveles=3, metros_por_plataforma=3,
                 topologia_intra='scale_free',
                 topologia_inter='malla',
                 K_base=2.0,
                 K_inter_base=1.5,
                 K_lateral_base=0.3):

        self.niveles = niveles
        self.mpp = metros_por_plataforma
        self.topologia_intra = topologia_intra
        self.topologia_inter = topologia_inter
        self.K_base = K_base
        self.K_inter_base = K_inter_base
        self.K_lateral_base = K_lateral_base

        # Estructura
        self.plataformas_por_nivel = [3**i for i in range(niveles)]
        self.osciladores_por_nivel = [p * metros_por_plataforma for p in self.plataformas_por_nivel]
        self.total = sum(self.osciladores_por_nivel)

        print(f"\n{'='*70}")
        print(f"🌀 KURAMOTO CON ESTADOS P-O-D-B POR ENLACE")
        print(f"  Niveles: {niveles}")
        print(f"  Total osciladores: {self.total}")
        print(f"  Topología intra: {topologia_intra}")
        print(f"  Topología inter: {topologia_inter}")
        print(f"{'='*70}\n")

        # Frecuencias naturales
        np.random.seed(42)
        self.omega = np.concatenate([
            np.random.normal(1.0, 0.1 + 0.02*i, n)
            for i, n in enumerate(self.osciladores_por_nivel)
        ])

        # Matrices de adyacencia fijas (quién se puede conectar)
        self.adjacencias = self._construir_adjacencias()

        # Matrices de acoplamiento (evolucionan con el estado)
        self.K_matrices = self._inicializar_K()

        # Para tracking histórico de estados
        self.historial_estados = []

        # Índices
        self.indices = self._construir_indices()

    def _construir_indices(self):
        indices = []
        start = 0
        for n in self.osciladores_por_nivel:
            indices.append(slice(start, start + n))
            start += n
        return indices

    def _construir_adjacencias(self):
        """Matrices binarias de conexiones posibles"""
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

    def _inicializar_K(self):
        """Inicializa todas las K = K_base (estado P inicial)"""
        K_mat = []
        for nivel in range(self.niveles):
            n_plt = self.plataformas_por_nivel[nivel]
            K_nivel = []
            for p in range(n_plt):
                n_osc = self.mpp
                A = self.adjacencias[nivel][p]
                K_plt = self.K_base * A  # Todas las conexiones comienzan en P
                K_nivel.append(K_plt)
            K_mat.append(K_nivel)
        return K_mat

    def _estado_a_partir_de_fase(self, delta_theta):
        """
        Determina el estado P-O-D-B a partir de la diferencia de fase
        """
        # Normalizar delta_theta a [0, π]
        delta = np.abs(delta_theta) % (2*np.pi)
        if delta > np.pi:
            delta = 2*np.pi - delta

        # El coseno determina el estado
        cos_delta = np.cos(delta)

        # Mapear coseno a [0, 1] donde 1 = P, 0 = B
        estado_valor = (cos_delta + 1) / 2

        return estado_valor

    def _estado_a_letra(self, valor):
        """Convierte valor numérico a letra del estado"""
        if valor > 0.8:
            return 'P'
        elif valor > 0.3:
            return 'O'
        elif valor > 0.1:
            return 'D'
        else:
            return 'B'

    def actualizar_K_por_estado(self, theta):
        """
        Actualiza cada K_ij basado en el estado de la conexión
        K_ij = K_base * (1 + cos(Δθ)) / 2
        """
        for nivel in range(self.niveles):
            if nivel == 0:
                theta_nivel = theta[self.indices[0]]
                n_plt = 1
                n_osc = len(theta_nivel)
                theta_reshaped = theta_nivel.reshape(1, n_osc)
            else:
                n_plt = self.plataformas_por_nivel[nivel]
                n_osc = self.mpp
                theta_reshaped = theta[self.indices[nivel]].reshape(n_plt, n_osc)

            for p in range(n_plt):
                A = self.adjacencias[nivel][p]
                K_actual = self.K_matrices[nivel][p]

                for i in range(n_osc):
                    for j in range(n_osc):
                        if A[i, j] > 0 and i != j:
                            # Diferencia de fase
                            delta = theta_reshaped[p, i] - theta_reshaped[p, j]

                            # Estado de la conexión (0 a 1)
                            estado_valor = self._estado_a_partir_de_fase(delta)

                            # K proporcional al estado
                            K_actual[i, j] = self.K_base * estado_valor

                self.K_matrices[nivel][p] = K_actual

    def dynamics(self, t, theta):
        # Actualizar todas las K basado en estados actuales
        self.actualizar_K_por_estado(theta)

        # Registrar para visualización (cada 10 pasos)
        if len(self.historial_estados) < 100 and np.random.random() < 0.1:
            self.historial_estados.append(theta.copy())

        dtheta = np.zeros_like(theta)

        # Nivel 0
        theta0 = theta[self.indices[0]]
        n0 = len(theta0)
        K0 = self.K_matrices[0][0]

        for i in range(n0):
            suma = 0
            for j in range(n0):
                if i != j:
                    suma += K0[i, j] * np.sin(theta0[j] - theta0[i])
            dtheta[self.indices[0]][i] = self.omega[self.indices[0]][i] + suma

        # Niveles superiores
        for nivel in range(1, self.niveles):
            n_plt = self.plataformas_por_nivel[nivel]
            n_osc = self.mpp

            theta_n = theta[self.indices[nivel]].reshape(n_plt, n_osc)
            dtheta_n = np.zeros_like(theta_n)

            # Madres (usamos K fijo para inter-capa por ahora)
            theta_prev = theta[self.indices[nivel-1]].reshape(self.plataformas_por_nivel[nivel-1], self.mpp)
            r_prev = np.mean(np.exp(1j * theta_prev), axis=1)
            phi_prev = np.angle(r_prev)

            # Conexiones laterales
            conexiones_laterales = []
            if self.topologia_inter == 'malla':
                for p in range(n_plt):
                    if p > 0:
                        conexiones_laterales.append((p, p-1))
                    if p < n_plt - 1:
                        conexiones_laterales.append((p, p+1))
            elif self.topologia_inter == 'global':
                for p1 in range(n_plt):
                    for p2 in range(p1+1, n_plt):
                        conexiones_laterales.append((p1, p2))

            for p in range(n_plt):
                madre = p // 3
                K_plt = self.K_matrices[nivel][p]

                for i in range(n_osc):
                    # Término intra (con K dinámico por estado)
                    intra = 0
                    for j in range(n_osc):
                        if i != j:
                            intra += K_plt[i, j] * np.sin(theta_n[p, j] - theta_n[p, i])

                    # Término jerárquico
                    inter = self.K_inter_base * np.sin(phi_prev[madre] - theta_n[p, i])

                    # Laterales
                    lateral = 0
                    norm = max(1, len(conexiones_laterales))
                    for (p1, p2) in conexiones_laterales:
                        if p == p1:
                            for j in range(n_osc):
                                delta = theta_n[p2, j] - theta_n[p, i]
                                lateral += np.sin(delta)
                        elif p == p2:
                            for j in range(n_osc):
                                delta = theta_n[p1, j] - theta_n[p, i]
                                lateral += np.sin(delta)
                    lateral = self.K_lateral_base * lateral / norm

                    idx = self.indices[nivel].start + p*n_osc + i
                    dtheta_n[p, i] = self.omega[idx] + intra + inter + lateral

            dtheta[self.indices[nivel]] = dtheta_n.flatten()

        return dtheta

    def simular(self, T=30, puntos=200):
        theta0 = np.random.uniform(-np.pi, np.pi, self.total)

        print("⏳ Simulando con estados P-O-D-B por enlace...")
        print(f"   T={T}s, puntos={puntos}")

        sol = solve_ivp(self.dynamics, (0, T), theta0,
                        t_eval=np.linspace(0, T, puntos),
                        method='RK45', rtol=1e-2)

        # Calcular sincronización por nivel
        r_nivel = []
        for nivel in range(self.niveles):
            fases = sol.y[self.indices[nivel]]
            if len(fases.shape) == 1:
                r = np.abs(np.mean(np.exp(1j * fases), axis=0))
            else:
                r = np.abs(np.mean(np.exp(1j * fases), axis=0))
            r_nivel.append(r)

        return sol.t, r_nivel, sol.y

    def visualizar_estados_conexiones(self, theta, nivel=1, plataforma=0):
        """
        Visualiza la matriz de estados de las conexiones de una plataforma
        """
        if nivel == 0:
            theta_plt = theta[self.indices[0]]
            n_osc = len(theta_plt)
            K_plt = self.K_matrices[0][0]
            titulo = "Nivel 0 (base)"
        else:
            idx = self.indices[nivel].start + plataforma * self.mpp
            theta_plt = theta[idx:idx+self.mpp]
            n_osc = self.mpp
            K_plt = self.K_matrices[nivel][plataforma]
            titulo = f"Nivel {nivel}, Plataforma {plataforma}"

        # Calcular matriz de estados
        matriz_estados = np.zeros((n_osc, n_osc))
        matriz_letras = [['' for _ in range(n_osc)] for _ in range(n_osc)]

        for i in range(n_osc):
            for j in range(n_osc):
                if i != j:
                    delta = theta_plt[i] - theta_plt[j]
                    valor = self._estado_a_partir_de_fase(delta)
                    matriz_estados[i, j] = valor
                    matriz_letras[i][j] = self._estado_a_letra(valor)

        # Visualizar
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Mapa de calor de estados
        im = axes[0].imshow(matriz_estados, cmap='RdYlGn', vmin=0, vmax=1)
        axes[0].set_title(f'Estados de conexiones - {titulo}')
        axes[0].set_xlabel('Oscilador j')
        axes[0].set_ylabel('Oscilador i')
        plt.colorbar(im, ax=axes[0], label='Estado (1=P, 0=B)')

        # Añadir letras en cada celda
        for i in range(n_osc):
            for j in range(n_osc):
                if i != j:
                    axes[0].text(j, i, matriz_letras[i][j],
                                ha='center', va='center',
                                color='black', fontweight='bold')

        # Gráfico de barras de distribución de estados
        valores = matriz_estados[matriz_estados > 0].flatten()
        axes[1].hist(valores, bins=20, color='skyblue', edgecolor='black')
        axes[1].axvline(x=0.8, color='blue', linestyle='--', label='P')
        axes[1].axvline(x=0.3, color='green', linestyle='--', label='O')
        axes[1].axvline(x=0.1, color='orange', linestyle='--', label='D')
        axes[1].set_xlabel('Valor de estado')
        axes[1].set_ylabel('Frecuencia')
        axes[1].set_title('Distribución de estados')
        axes[1].legend()

        plt.tight_layout()
        plt.show()

        # Estadísticas
        print(f"\n📊 Estadísticas de estados - {titulo}")
        print(f"  Estado P (≥0.8): {np.sum(valores >= 0.8)} conexiones")
        print(f"  Estado O (0.3-0.8): {np.sum((valores >= 0.3) & (valores < 0.8))} conexiones")
        print(f"  Estado D (0.1-0.3): {np.sum((valores >= 0.1) & (valores < 0.3))} conexiones")
        print(f"  Estado B (<0.1): {np.sum(valores < 0.1)} conexiones")

# ============================================
# EJECUTAR
# ============================================
if __name__ == "__main__":
    print("\n🌀 KURAMOTO CON ESTADOS P-O-D-B POR CONEXIÓN")
    print("Cada enlace individual tiene su propio estado")
    print("basado en la diferencia de fase local.\n")

    # Crear simulador con 3 niveles para poder visualizar
    sim = KuramotoPODB(
        niveles=3,
        metros_por_plataforma=4,
        topologia_intra='scale_free',
        topologia_inter='malla',
        K_base=2.0,
        K_inter_base=1.5,
        K_lateral_base=0.3
    )

    # Simular
    t, r_nivel, theta_hist = sim.simular(T=20, puntos=150)

    # Mostrar resultados globales
    print("\n=== SINCRONIZACIÓN POR NIVEL ===")
    for i, r in enumerate([r[-1] for r in r_nivel]):
        estado = "✅" if r > 0.8 else "⚠️" if r > 0.5 else "❌"
        print(f"Nivel {i}: r={r:.3f} {estado}")

    # Visualizar estados de conexiones en el último instante
    theta_final = theta_hist[:, -1]

    print("\n" + "="*60)
    print("VISUALIZANDO ESTADOS DE CONEXIONES (tiempo final)")
    print("="*60)

    # Nivel 0
    sim.visualizar_estados_conexiones(theta_final, nivel=0)

    # Nivel 1, primera plataforma
    sim.visualizar_estados_conexiones(theta_final, nivel=1, plataforma=0)

    # Nivel 2, primera plataforma
    if sim.niveles > 2:
        sim.visualizar_estados_conexiones(theta_final, nivel=2, plataforma=0)

    # Gráfica de sincronización
    plt.figure(figsize=(10, 5))
    for i, r in enumerate(r_nivel):
        plt.plot(t, r, label=f'Nivel {i}', linewidth=2)
    plt.xlabel('Tiempo')
    plt.ylabel('Sincronización r')
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Evolución de la sincronización por nivel')
    plt.show()


# 7. Código de kuramoto7.py COMPLETO
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import networkx as nx
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================
# FUNCIONES AUXILIARES DE TOPOLOGÍA
# ============================================

def generar_scale_free(n_osc, m=2):
    """Genera matriz de adyacencia scale-free usando networkx"""
    if n_osc <= 2:
        return np.ones((n_osc, n_osc)) - np.eye(n_osc)
    try:
        G = nx.barabasi_albert_graph(n_osc, min(m, n_osc-1))
        return nx.to_numpy_array(G)
    except:
        # Fallback a implementación manual
        return generar_scale_free_manual(n_osc, m)

def generar_scale_free_manual(n_osc, m=2):
    """Versión manual sin networkx (por si acaso)"""
    if n_osc <= m:
        return np.ones((n_osc, n_osc)) - np.eye(n_osc)

    grados = np.zeros(n_osc)
    A = np.zeros((n_osc, n_osc))

    # Inicializar con clique
    for i in range(m+1):
        for j in range(i+1, m+1):
            A[i, j] = 1
            A[j, i] = 1
            grados[i] += 1
            grados[j] += 1

    # Preferential attachment
    for nuevo in range(m+1, n_osc):
        prob = grados[:nuevo] / np.sum(grados[:nuevo])
        elegidos = np.random.choice(nuevo, size=m, replace=False, p=prob)
        for viejo in elegidos:
            A[nuevo, viejo] = 1
            A[viejo, nuevo] = 1
            grados[nuevo] += 1
            grados[viejo] += 1

    return A

def generar_small_world(n_osc, k=2, p=0.1):
    """Genera matriz de adyacencia small-world (Watts-Strogatz)"""
    if n_osc <= 2:
        return np.ones((n_osc, n_osc)) - np.eye(n_osc)
    try:
        G = nx.watts_strogatz_graph(n_osc, min(k*2, n_osc-1), p)
        return nx.to_numpy_array(G)
    except:
        # Fallback: anillo simple
        A = np.zeros((n_osc, n_osc))
        for i in range(n_osc):
            A[i, (i+1)%n_osc] = 1
            A[(i+1)%n_osc, i] = 1
        return A

# ============================================
# CLASE PRINCIPAL UNIFICADA
# ============================================

class KuramotoUnificado:
    """
    Macroclase que integra TODAS las opciones:
    - 7 topologías diferentes
    - Modo clásico (K fijo) o modo PODB (estados por conexión)
    - Tracking temporal detallado
    """

    # Definición de estados PODB
    ESTADOS = {
        'P': {'color': 'blue', 'rango': (0.8, 1.0), 'desc': 'Partícula', 'simbolo': '🔵'},
        'O': {'color': 'green', 'rango': (0.3, 0.8), 'desc': 'Onda', 'simbolo': '🟢'},
        'D': {'color': 'orange', 'rango': (0.1, 0.3), 'desc': 'Difuso', 'simbolo': '🟠'},
        'B': {'color': 'red', 'rango': (0.0, 0.1), 'desc': 'Borrado', 'simbolo': '🔴'}
    }

    # Diccionario de opciones disponibles
    OPCIONES = {
        1: {'nombre': 'Control', 'intra': 'global', 'inter': 'jerarquica', 'desc': 'Global + Jerárquica (original)'},
        2: {'nombre': 'Anillo', 'intra': 'anillo', 'inter': 'jerarquica', 'desc': 'Anillo intra + Jerárquica'},
        3: {'nombre': 'Scale-free', 'intra': 'scale_free', 'inter': 'jerarquica', 'desc': 'Scale-free intra + Jerárquica'},
        4: {'nombre': 'Estrella', 'intra': 'estrella', 'inter': 'jerarquica', 'desc': 'Estrella intra + Jerárquica'},
        5: {'nombre': 'Global+Malla', 'intra': 'global', 'inter': 'malla', 'desc': 'Global + Malla inter (laterales)'},
        6: {'nombre': 'Scale-free+Malla', 'intra': 'scale_free', 'inter': 'malla', 'desc': 'Scale-free + Malla inter (LA MEJOR)'},
        7: {'nombre': 'Scale-free+Global', 'intra': 'scale_free', 'inter': 'global', 'desc': 'Scale-free + Global inter (todas conectadas)'},
        8: {'nombre': 'Small-World', 'intra': 'small_world', 'inter': 'malla', 'desc': 'Small-world + Malla'},
        9: {'nombre': 'Personalizada', 'intra': 'custom', 'inter': 'custom', 'desc': 'Configuración manual'}
    }

    def __init__(self,
                 opcion=6,                    # Opción del 1-9
                 niveles=6,                    # Número de niveles
                 metros_por_plataforma=3,      # Metrónomos por plataforma
                 modo='podb',                   # 'clasico' o 'podb'
                 K_base=2.0,                    # Acoplamiento base
                 K_inter_base=1.5,               # Acoplamiento inter-capa
                 K_lateral_base=0.3,             # Acoplamiento lateral
                 tracking_detallado=True):       # Guardar historial completo

        self.opcion = opcion
        self.niveles = niveles
        self.mpp = metros_por_plataforma
        self.modo = modo
        self.K_base = K_base
        self.K_inter_base = K_inter_base
        self.K_lateral_base = K_lateral_base
        self.tracking_detallado = tracking_detallado

        # Cargar configuración de la opción
        config = self.OPCIONES[opcion]
        self.topologia_intra = config['intra']
        self.topologia_inter = config['inter']
        self.nombre_config = config['nombre']
        self.desc_config = config['desc']

        # Estructura fractal
        self.plataformas_por_nivel = [3**i for i in range(niveles)]
        self.osciladores_por_nivel = [p * metros_por_plataforma for p in self.plataformas_por_nivel]
        self.total = sum(self.osciladores_por_nivel)

        self._mostrar_configuracion()

        # Frecuencias naturales (con más variabilidad en niveles altos)
        np.random.seed(42)  # Reproducibilidad
        self.omega = np.concatenate([
            np.random.normal(1.0, 0.1 + 0.02*i, n)
            for i, n in enumerate(self.osciladores_por_nivel)
        ])

        # Construir matrices de adyacencia
        self.adjacencias = self._construir_adjacencias()

        # Inicializar matrices de acoplamiento
        self.K_matrices = self._inicializar_K()

        # Índices globales
        self.indices = self._construir_indices()

        # Tracking histórico
        self.historial_tiempos = []
        self.historial_r_por_nivel = defaultdict(list)
        self.historial_estados_por_nivel = defaultdict(list)
        self.paso_actual = 0

    def _mostrar_configuracion(self):
        """Muestra la configuración de la simulación"""
        print(f"\n{'='*80}")
        print(f"🌀 KURAMOTO UNIFICADO - Opción {self.opcion}: {self.nombre_config}")
        print(f"{'='*80}")
        print(f"  Descripción: {self.desc_config}")
        print(f"  Modo: {self.modo.upper()} ({'K variable por estado' if self.modo=='podb' else 'K fijo'})")
        print(f"  Niveles: {self.niveles}")
        print(f"  Metrónomos por plataforma: {self.mpp}")
        print(f"  Plataformas por nivel: {self.plataformas_por_nivel}")
        print(f"  OSCILADORES TOTALES: {self.total}")
        print(f"  Topología intra-plataforma: {self.topologia_intra}")
        print(f"  Topología inter-plataforma: {self.topologia_inter}")
        print(f"{'='*80}\n")

    def _construir_indices(self):
        """Construye slices para acceder a cada nivel"""
        indices = []
        start = 0
        for n in self.osciladores_por_nivel:
            indices.append(slice(start, start + n))
            start += n
        return indices

    def _construir_adjacencias_intra(self, n_osc, topologia):
        """Construye matriz de adyacencia intra-plataforma"""
        if n_osc <= 1:
            return np.zeros((n_osc, n_osc))

        if topologia == 'global':
            return np.ones((n_osc, n_osc)) - np.eye(n_osc)

        elif topologia == 'anillo':
            A = np.zeros((n_osc, n_osc))
            for i in range(n_osc):
                A[i, (i-1)%n_osc] = 1
                A[i, (i+1)%n_osc] = 1
            return A

        elif topologia == 'malla_1d':
            A = np.zeros((n_osc, n_osc))
            for i in range(n_osc-1):
                A[i, i+1] = 1
                A[i+1, i] = 1
            return A

        elif topologia == 'estrella':
            A = np.zeros((n_osc, n_osc))
            centro = 0
            for i in range(1, n_osc):
                A[centro, i] = 1
                A[i, centro] = 1
            return A

        elif topologia == 'scale_free':
            return generar_scale_free(n_osc, m=2)

        elif topologia == 'small_world':
            return generar_small_world(n_osc, k=2, p=0.2)

        else:
            # Por defecto: global
            return np.ones((n_osc, n_osc)) - np.eye(n_osc)

    def _construir_adjacencias(self):
        """Construye todas las matrices de adyacencia"""
        adj = []

        for nivel in range(self.niveles):
            n_plt = self.plataformas_por_nivel[nivel]
            adj_nivel = []

            for p in range(n_plt):
                n_osc = self.mpp
                A = self._construir_adjacencias_intra(n_osc, self.topologia_intra)
                adj_nivel.append(A)

            adj.append(adj_nivel)

        return adj

    def _inicializar_K(self):
        """Inicializa matrices de acoplamiento"""
        K_mat = []

        for nivel in range(self.niveles):
            n_plt = self.plataformas_por_nivel[nivel]
            K_nivel = []

            for p in range(n_plt):
                A = self.adjacencias[nivel][p]
                if self.modo == 'podb':
                    # En PODB, K comienza en K_base para todas las conexiones
                    K_plt = self.K_base * A
                else:
                    # Modo clásico: K fijo
                    K_plt = self.K_base * A
                K_nivel.append(K_plt)

            K_mat.append(K_nivel)

        return K_mat

    def _estado_a_partir_de_fase(self, delta_theta):
        """Convierte diferencia de fase a valor de estado (0-1)"""
        delta = np.abs(delta_theta) % (2*np.pi)
        if delta > np.pi:
            delta = 2*np.pi - delta
        cos_delta = np.cos(delta)
        return (cos_delta + 1) / 2

    def _estado_a_letra(self, valor):
        """Convierte valor numérico a letra del estado"""
        if valor > 0.8:
            return 'P'
        elif valor > 0.3:
            return 'O'
        elif valor > 0.1:
            return 'D'
        else:
            return 'B'

    def _estado_a_simbolo(self, valor):
        """Convierte valor a símbolo para visualización"""
        if valor > 0.8:
            return '🔵'
        elif valor > 0.3:
            return '🟢'
        elif valor > 0.1:
            return '🟠'
        else:
            return '🔴'

    def actualizar_K_por_estado(self, theta):
        """Actualiza K basado en estados (solo modo PODB)"""
        if self.modo != 'podb':
            return

        for nivel in range(self.niveles):
            if nivel == 0:
                theta_nivel = theta[self.indices[0]]
                n_plt = 1
                n_osc = len(theta_nivel)
                theta_reshaped = theta_nivel.reshape(1, n_osc)
            else:
                n_plt = self.plataformas_por_nivel[nivel]
                n_osc = self.mpp
                theta_reshaped = theta[self.indices[nivel]].reshape(n_plt, n_osc)

            for p in range(n_plt):
                A = self.adjacencias[nivel][p]
                K_actual = self.K_matrices[nivel][p]

                for i in range(n_osc):
                    for j in range(n_osc):
                        if A[i, j] > 0 and i != j:
                            delta = theta_reshaped[p, i] - theta_reshaped[p, j]
                            estado_valor = self._estado_a_partir_de_fase(delta)
                            K_actual[i, j] = self.K_base * estado_valor

                self.K_matrices[nivel][p] = K_actual

    def _construir_conexiones_laterales(self, n_plt):
        """Construye lista de conexiones laterales según topología inter"""
        conexiones = []

        if self.topologia_inter == 'malla':
            for p in range(n_plt):
                if p > 0:
                    conexiones.append((p, p-1))
                if p < n_plt - 1:
                    conexiones.append((p, p+1))

        elif self.topologia_inter == 'global':
            for p1 in range(n_plt):
                for p2 in range(p1+1, n_plt):
                    conexiones.append((p1, p2))

        elif self.topologia_inter == 'estrella' and n_plt > 1:
            centro = n_plt // 2
            for p in range(n_plt):
                if p != centro:
                    conexiones.append((centro, p))

        # 'jerarquica' no tiene conexiones laterales

        return conexiones

    def registrar_estados(self, theta, t):
        """Registra el estado actual para tracking"""
        if not self.tracking_detallado:
            return

        self.historial_tiempos.append(t)

        for nivel in range(self.niveles):
            if nivel == 0:
                theta_nivel = theta[self.indices[0]]
                n_plt = 1
                n_osc = len(theta_nivel)
                theta_reshaped = theta_nivel.reshape(1, n_osc)
            else:
                n_plt = self.plataformas_por_nivel[nivel]
                n_osc = self.mpp
                theta_reshaped = theta[self.indices[nivel]].reshape(n_plt, n_osc)

            # Calcular r de este nivel
            fases_nivel = theta[self.indices[nivel]]
            if len(fases_nivel.shape) == 1:
                r_nivel = np.abs(np.mean(np.exp(1j * fases_nivel)))
            else:
                r_nivel = np.abs(np.mean(np.exp(1j * fases_nivel), axis=0))
                r_nivel = np.mean(r_nivel) if hasattr(r_nivel, '__len__') else r_nivel

            self.historial_r_por_nivel[nivel].append(r_nivel)

            if self.modo == 'podb':
                # Contar estados
                conteo = {'P': 0, 'O': 0, 'D': 0, 'B': 0}
                total = 0

                for p in range(n_plt):
                    A = self.adjacencias[nivel][p]
                    for i in range(n_osc):
                        for j in range(n_osc):
                            if A[i, j] > 0 and i != j:
                                delta = theta_reshaped[p, i] - theta_reshaped[p, j]
                                valor = self._estado_a_partir_de_fase(delta)
                                letra = self._estado_a_letra(valor)
                                conteo[letra] += 1
                                total += 1

                if total > 0:
                    self.historial_estados_por_nivel[nivel].append({
                        't': t,
                        'P': conteo['P'] / total,
                        'O': conteo['O'] / total,
                        'D': conteo['D'] / total,
                        'B': conteo['B'] / total,
                        'total': total
                    })

    def dynamics(self, t, theta):
        """Ecuaciones de movimiento"""
        # Actualizar K en modo PODB
        self.actualizar_K_por_estado(theta)

        # Registrar cada 0.5s
        self.paso_actual += 1
        if abs(t - round(t*2)/2) < 0.01:
            self.registrar_estados(theta, t)

        dtheta = np.zeros_like(theta)

        # NIVEL 0 (base)
        theta0 = theta[self.indices[0]]
        n0 = len(theta0)
        K0 = self.K_matrices[0][0]

        for i in range(n0):
            suma = 0
            for j in range(n0):
                if i != j:
                    suma += K0[i, j] * np.sin(theta0[j] - theta0[i])
            dtheta[self.indices[0]][i] = self.omega[self.indices[0]][i] + suma

        # NIVELES SUPERIORES
        for nivel in range(1, self.niveles):
            n_plt = self.plataformas_por_nivel[nivel]
            n_osc = self.mpp

            theta_n = theta[self.indices[nivel]].reshape(n_plt, n_osc)
            dtheta_n = np.zeros_like(theta_n)

            # Fases del nivel anterior (madres)
            theta_prev = theta[self.indices[nivel-1]].reshape(self.plataformas_por_nivel[nivel-1], self.mpp)
            r_prev = np.mean(np.exp(1j * theta_prev), axis=1)
            phi_prev = np.angle(r_prev)

            # Conexiones laterales
            conexiones = self._construir_conexiones_laterales(n_plt)

            for p in range(n_plt):
                madre = p // 3
                K_plt = self.K_matrices[nivel][p]

                for i in range(n_osc):
                    # Término intra-plataforma
                    intra = 0
                    for j in range(n_osc):
                        if i != j:
                            intra += K_plt[i, j] * np.sin(theta_n[p, j] - theta_n[p, i])

                    # Término jerárquico (madre)
                    inter = self.K_inter_base * np.sin(phi_prev[madre] - theta_n[p, i])

                    # Términos laterales
                    lateral = 0
                    if conexiones:
                        for (p1, p2) in conexiones:
                            if p == p1:
                                for j in range(n_osc):
                                    lateral += np.sin(theta_n[p2, j] - theta_n[p, i])
                            elif p == p2:
                                for j in range(n_osc):
                                    lateral += np.sin(theta_n[p1, j] - theta_n[p, i])
                        lateral = self.K_lateral_base * lateral / (len(conexiones) * n_osc)

                    idx = self.indices[nivel].start + p*n_osc + i
                    dtheta_n[p, i] = self.omega[idx] + intra + inter + lateral

            dtheta[self.indices[nivel]] = dtheta_n.flatten()

        return dtheta

    def simular(self, T=30, puntos=200):
        """Ejecuta la simulación"""
        theta0 = np.random.uniform(-np.pi, np.pi, self.total)

        print(f"⏳ Simulando Opción {self.opcion} - {self.nombre_config}...")
        print(f"   Modo: {self.modo.upper()}, T={T}s, puntos={puntos}")
        print(f"   Total osciladores: {self.total}")
        print("   (Esto puede tomar varios minutos)\n")

        sol = solve_ivp(self.dynamics, (0, T), theta0,
                        t_eval=np.linspace(0, T, puntos),
                        method='RK45', rtol=1e-2)

        # Registrar estado final
        self.registrar_estados(sol.y[:, -1], T)

        return sol.t, sol.y

    def imprimir_resultados(self):
        """Imprime resultados detallados por consola"""

        print(f"\n{'='*80}")
        print(f"📊 RESULTADOS - Opción {self.opcion}: {self.nombre_config}")
        print(f"{'='*80}")

        # Tabla de sincronización
        print(f"\n📈 SINCRONIZACIÓN POR NIVEL (r):")
        print(f"{'-'*60}")
        print(f"Nivel | Plataformas | r_inicial | r_final | Estado")
        print(f"{'-'*60}")

        for nivel in range(self.niveles):
            if nivel in self.historial_r_por_nivel and self.historial_r_por_nivel[nivel]:
                r_inicial = self.historial_r_por_nivel[nivel][0]
                r_final = self.historial_r_por_nivel[nivel][-1]
            else:
                r_inicial = r_final = 0

            if r_final > 0.8:
                estado = "✅ COHERENTE"
            elif r_final > 0.5:
                estado = "⚠️ DIFUSO"
            else:
                estado = "❌ CAÓTICO"

            print(f"{nivel:3d}   | {self.plataformas_por_nivel[nivel]:6d}      | {r_inicial:.3f}     | {r_final:.3f}   | {estado}")

        # Si es modo PODB, mostrar distribución de estados
        if self.modo == 'podb' and self.historial_estados_por_nivel:
            print(f"\n{'='*80}")
            print(f"🌀 DISTRIBUCIÓN DE ESTADOS P-O-D-B (tiempo final)")
            print(f"{'='*80}")

            for nivel in range(self.niveles):
                if nivel not in self.historial_estados_por_nivel or not self.historial_estados_por_nivel[nivel]:
                    continue

                final = self.historial_estados_por_nivel[nivel][-1]

                print(f"\nNIVEL {nivel} ({self.plataformas_por_nivel[nivel]} plataformas):")
                print(f"  🔵 Partícula (P): {final['P']*100:5.1f}%")
                print(f"  🟢 Onda (O):      {final['O']*100:5.1f}%")
                print(f"  🟠 Difuso (D):    {final['D']*100:5.1f}%")
                print(f"  🔴 Borrado (B):   {final['B']*100:5.1f}%")
                print(f"  Total conexiones: {final['total']}")

            # Identificar nivel SOC (máxima mezcla)
            max_mezcla = 0
            nivel_soc = -1
            for nivel in range(self.niveles):
                if nivel not in self.historial_estados_por_nivel:
                    continue
                final = self.historial_estados_por_nivel[nivel][-1]
                # Producto de proporciones (máximo cuando todas son ~0.25)
                mezcla = final['P'] * final['O'] * final['D'] * final['B'] * 10000
                if mezcla > max_mezcla:
                    max_mezcla = mezcla
                    nivel_soc = nivel

            if nivel_soc >= 0:
                final_soc = self.historial_estados_por_nivel[nivel_soc][-1]
                print(f"\n🔬 PUNTO CRÍTICO (SOC) IDENTIFICADO:")
                print(f"   Nivel {nivel_soc} presenta la máxima mezcla de estados:")
                print(f"   P={final_soc['P']*100:.1f}%, O={final_soc['O']*100:.1f}%, "
                      f"D={final_soc['D']*100:.1f}%, B={final_soc['B']*100:.1f}%")

        print(f"\n{'='*80}")

    def graficar(self, guardar=False):
        """Genera gráficas de resultados"""

        # Gráfica de sincronización
        plt.figure(figsize=(12, 5))
        for nivel in range(self.niveles):
            if nivel in self.historial_r_por_nivel and self.historial_r_por_nivel[nivel]:
                plt.plot(self.historial_tiempos[:len(self.historial_r_por_nivel[nivel])],
                        self.historial_r_por_nivel[nivel],
                        label=f'N{nivel}', linewidth=2)

        plt.xlabel('Tiempo')
        plt.ylabel('Sincronización r')
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.title(f'Opción {self.opcion}: {self.nombre_config} - Evolución de r')
        if guardar:
            plt.savefig(f'sincronizacion_op{self.opcion}.png')
        plt.show()

        # Si es modo PODB, gráfica de evolución de estados para nivel SOC
        if self.modo == 'podb' and self.historial_estados_por_nivel:
            # Encontrar nivel con más datos
            niveles_con_datos = [n for n in range(self.niveles)
                               if n in self.historial_estados_por_nivel and self.historial_estados_por_nivel[n]]

            #for nivel in niveles_con_datos[:3]:  # Mostrar primeros 3 niveles
            for nivel in niveles_con_datos:  # TODOS los niveles con datos
                datos = self.historial_estados_por_nivel[nivel]
                tiempos = [d['t'] for d in datos]

                plt.figure(figsize=(12, 4))
                plt.fill_between(tiempos, 0, [d['P'] for d in datos],
                                color='blue', alpha=0.7, label='P')

                bottom_P = [d['P'] for d in datos]
                plt.fill_between(tiempos, bottom_P,
                                [bottom_P[i] + d['O'] for i, d in enumerate(datos)],
                                color='green', alpha=0.7, label='O')

                bottom_O = [bottom_P[i] + d['O'] for i, d in enumerate(datos)]
                plt.fill_between(tiempos, bottom_O,
                                [bottom_O[i] + d['D'] for i, d in enumerate(datos)],
                                color='orange', alpha=0.7, label='D')

                bottom_D = [bottom_O[i] + d['D'] for i, d in enumerate(datos)]
                plt.fill_between(tiempos, bottom_D,
                                [1.0 for _ in datos],
                                color='red', alpha=0.7, label='B')

                plt.xlabel('Tiempo')
                plt.ylabel('Proporción')
                plt.ylim(0, 1)
                plt.title(f'Nivel {nivel} - Evolución Estados P-O-D-B')
                plt.grid(True, alpha=0.3)
                plt.legend()
                if guardar:
                    plt.savefig(f'estados_n{nivel}_op{self.opcion}.png')
                plt.show()

# ============================================
# MENÚ INTERACTIVO
# ============================================

def mostrar_menu():
    """Muestra el menú de opciones"""
    print("\n" + "🎛️  MENÚ PRINCIPAL - KURAMOTO UNIFICADO")
    print("="*60)
    print("OPCIONES DE TOPOLOGÍA:")
    print("-"*60)
    for num, config in KuramotoUnificado.OPCIONES.items():
        print(f"  {num}. {config['nombre']:20s} - {config['desc']}")

    print("\n" + "⚙️  CONFIGURACIÓN ADICIONAL:")
    print("-"*60)
    print("  • Modo clásico: K fijo (como experimentos originales)")
    print("  • Modo PODB:    K variable por estado P-O-D-B")
    print("  • Niveles: 3-6 (más niveles = más lento)")
    print("  • Metrónomos/plataforma: 2-4")
    print("="*60)

def ejecutar():
    """Función principal"""
    mostrar_menu()

    try:
        # Seleccionar opción
        opcion = int(input("\n🔹 Elige opción (1-9): "))
        if opcion not in KuramotoUnificado.OPCIONES:
            print("❌ Opción no válida. Usando opción 6.")
            opcion = 6

        # Seleccionar modo
        modo = input("🔹 Modo (c=clásico, p=podb) [p]: ").lower()
        modo = 'podb' if modo in ['p', 'podb', ''] else 'clasico'

        # Niveles
        niveles = int(input("🔹 Niveles (3-6) [5]: ") or "5")
        niveles = max(3, min(6, niveles))

        # Metrónomos por plataforma
        mpp = int(input("🔹 Metrónomos/plataforma (2-4) [3]: ") or "3")
        mpp = max(2, min(4, mpp))

        # Crear y ejecutar simulación
        sim = KuramotoUnificado(
            opcion=opcion,
            niveles=niveles,
            metros_por_plataforma=mpp,
            modo=modo,
            K_base=2.0,
            K_inter_base=1.5,
            K_lateral_base=0.3,
            tracking_detallado=True
        )

        t, theta = sim.simular(T=25, puntos=150)
        sim.imprimir_resultados()

        # Preguntar si graficar
        graficar = input("\n🔹 ¿Generar gráficas? (s/n) [s]: ").lower()
        if graficar in ['s', 'si', 'sí', '']:
            sim.graficar(guardar=False)

        # Preguntar si guardar resultados
        guardar = input("\n🔹 ¿Guardar resultados en archivo? (s/n) [n]: ").lower()
        if guardar in ['s', 'si', 'sí']:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"resultados_op{opcion}_{timestamp}.txt"

            with open(filename, 'w') as f:
                import sys
                from io import StringIO

                old_stdout = sys.stdout
                sys.stdout = StringIO()

                sim.imprimir_resultados()

                output = sys.stdout.getvalue()
                sys.stdout = old_stdout

                f.write(output)

            print(f"✅ Resultados guardados en {filename}")

        return sim

    except KeyboardInterrupt:
        print("\n\n⚠️ Simulación interrumpida por usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

# ============================================
# EJECUCIÓN
# ============================================

if __name__ == "__main__":
    print("\n" + "🌟"*40)
    print("🌟  KURAMOTO UNIFICADO - TODAS LAS OPCIONES EN UNA  🌟")
    print("🌟"*40)

    sim = ejecutar()

    # Si quieres ejecutar múltiples opciones automáticamente:
    # for op in [1, 6, 7]:
    #     sim = KuramotoUnificado(opcion=op, niveles=5, modo='podb')
    #     t, theta = sim.simular(T=20, puntos=100)
    #     sim.imprimir_resultados()


#
#
#
# ============================================
# FUNCIONES PARA CADA SCRIPT
# ============================================
def ejecutar_kuramoto_explorer():
    """Ejecuta kuramoto_explorer.py"""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    # Llama a la función principal de tu script
    simular_kuramoto(N1=5, N2=4, N3=3, K_intra=2.0, K_inter=1.5, T=25, puntos=150)

    output = sys.stdout.getvalue()
    sys.stdout = old_stdout

    # Capturar gráfica
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close('all')

    return output, img_base64

def ejecutar_modelo_46():
    """Ejecuta modelo_efectivo_46_capas.py"""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    r, t_efectivo = modelo_efectivo_46_capas()

    output = sys.stdout.getvalue()
    sys.stdout = old_stdout

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close('all')

    return output, img_base64

def ejecutar_macro_podb():
    """Ejecuta MacroPODB.py con opción 6 (la mejor)"""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    sim = KuramotoUnificado(
        opcion=6,
        niveles=5,
        metros_por_plataforma=3,
        modo='podb',
        K_base=2.0,
        K_inter_base=1.5,
        K_lateral_base=0.3
    )
    t, theta = sim.simular(T=25, puntos=150)
    sim.imprimir_resultados()

    output = sys.stdout.getvalue()
    sys.stdout = old_stdout

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close('all')

    return output, img_base64

# ============================================
# API ENDPOINTS
# ============================================
@app.route('/ejecutar/<script>', methods=['GET'])
def ejecutar_script(script):
    try:
        if script == 'kuramoto_explorer':
            output, img = ejecutar_kuramoto_explorer()
        elif script == 'modelo_46':
            output, img = ejecutar_modelo_46()
        elif script == 'macro_podb':
            output, img = ejecutar_macro_podb()
        else:
            return jsonify({'success': False, 'error': 'Script no encontrado'})

        # Preparar respuesta
        respuesta = {
            'success': True,
            'output': output,
            'imagen': img
        }
        
        # Si hay callback (JSONP), devolver con ese nombre
        callback = request.args.get('callback')
        if callback:
            from flask import json
            return f"{callback}({json.dumps(respuesta)})"
        else:
            return jsonify(respuesta)

    except Exception as e:
        respuesta = {'success': False, 'error': str(e)}
        callback = request.args.get('callback')
        if callback:
            from flask import json
            return f"{callback}({json.dumps(respuesta)})"
        else:
            return jsonify(respuesta)
