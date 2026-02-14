from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import defaultdict
import sys
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# ============================================
# FUNCIONES AUXILIARES COMUNES
# ============================================

def figura_a_base64(fig):
    """Convierte una figura de matplotlib a base64"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

def generar_scale_free(n_osc, m=2):
    """Genera matriz de adyacencia scale-free"""
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

# ============================================
# HITO 1: KuramotoPRO (independiente)
# ============================================
def ejecutar_hito1(params=None):
    """Hito 1: KuramotoPRO - Modelo básico de 3 niveles"""
    output_lines = []
    imagenes = []
    
    # Parámetros fijos de este hito
    N1, N2, N3 = 5, 4, 3
    np.random.seed(42)
    omega1 = np.random.normal(1.0, 0.1, N1)
    omega2 = np.random.normal(1.0, 0.15, 3 * N2)
    omega3 = np.random.normal(1.0, 0.2, 9 * N3)
    
    K_intra, K_inter = 2.0, 1.5
    t_span = (0, 50)
    t_eval = np.linspace(0, 50, 1000)
    
    idx1 = slice(0, N1)
    idx2 = slice(N1, N1 + 3*N2)
    idx3 = slice(N1 + 3*N2, N1 + 3*N2 + 9*N3)
    
    def dynamics(t, theta):
        dtheta = np.zeros_like(theta)
        theta1 = theta[idx1]
        for i in range(N1):
            dtheta[i] = omega1[i] + (K_intra/N1) * np.sum(np.sin(theta1 - theta1[i]))
        
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
    
    theta0 = np.concatenate([
        np.random.uniform(-np.pi, np.pi, N1),
        np.random.uniform(-np.pi, np.pi, 3*N2),
        np.random.uniform(-np.pi, np.pi, 9*N3)
    ])
    
    sol = solve_ivp(dynamics, t_span, theta0, t_eval=t_eval, method='RK45')
    
    r1 = np.abs(np.mean(np.exp(1j * sol.y[idx1]), axis=0))
    theta2_hist = sol.y[idx2].reshape(3, N2, -1)
    r2 = np.abs(np.mean(np.exp(1j * theta2_hist), axis=1))
    theta3_hist = sol.y[idx3].reshape(9, N3, -1)
    r3 = np.abs(np.mean(np.exp(1j * theta3_hist), axis=1))
    
    # Gráfica
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(sol.t, r1, 'b-', linewidth=2)
    axes[0].set_ylabel('r1 (Nivel 1)')
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Sincronización en Plataforma Raíz')
    
    for p in range(3):
        axes[1].plot(sol.t, r2[p], label=f'Plataforma {p+1}', linewidth=1.5)
    axes[1].set_ylabel('r2 (Nivel 2)')
    axes[1].set_ylim(0, 1.1)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_title('Sincronización en 3 Plataformas')
    
    for p in range(9):
        axes[2].plot(sol.t, r3[p], alpha=0.7, linewidth=1)
    axes[2].set_xlabel('Tiempo')
    axes[2].set_ylabel('r3 (Nivel 3)')
    axes[2].set_ylim(0, 1.1)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Sincronización en 9 Plataformas')
    
    plt.tight_layout()
    imagenes.append(figura_a_base64(fig))
    
    # Output
    output_lines.append(f"\n{'='*50}")
    output_lines.append(f"HITO 1: KURAMOTO PRO")
    output_lines.append(f"{'='*50}")
    output_lines.append(f"Nivel 1 - r_final: {r1[-1]:.3f}")
    output_lines.append(f"Nivel 2 - r_final promedio: {np.mean(r2[:,-1]):.3f}")
    output_lines.append(f"Nivel 3 - r_final promedio: {np.mean(r3[:,-1]):.3f}")
    
    return "\n".join(output_lines), imagenes


# ============================================
# HITO 2: Kuramoto Explorer (independiente)
# ============================================
def ejecutar_hito2(params=None):
    """Hito 2: Kuramoto Explorer - Exploración de parámetros"""
    output_lines = []
    imagenes = []
    
    def simular_caso(N1, N2, N3, K_intra, K_inter, T, titulo):
        np.random.seed(42)
        omega1 = np.random.normal(1.0, 0.1, N1)
        omega2 = np.random.normal(1.0, 0.15, 3 * N2)
        omega3 = np.random.normal(1.0, 0.2, 9 * N3)
        
        t_span = (0, T)
        t_eval = np.linspace(0, T, 300)
        
        idx1 = slice(0, N1)
        idx2 = slice(N1, N1 + 3*N2)
        idx3 = slice(N1 + 3*N2, N1 + 3*N2 + 9*N3)
        
        def dynamics(t, theta):
            dtheta = np.zeros_like(theta)
            theta1 = theta[idx1]
            for i in range(N1):
                dtheta[i] = omega1[i] + (K_intra/N1) * np.sum(np.sin(theta1 - theta1[i]))
            
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
        
        theta0 = np.random.uniform(-np.pi, np.pi, N1 + 3*N2 + 9*N3)
        sol = solve_ivp(dynamics, t_span, theta0, t_eval=t_eval, method='RK45')
        
        r1 = np.abs(np.mean(np.exp(1j * sol.y[idx1]), axis=0))
        theta2_hist = sol.y[idx2].reshape(3, N2, -1)
        r2 = np.abs(np.mean(np.exp(1j * theta2_hist), axis=1))
        theta3_hist = sol.y[idx3].reshape(9, N3, -1)
        r3 = np.abs(np.mean(np.exp(1j * theta3_hist), axis=1))
        
        return sol.t, r1, r2, r3
    
    # Caso 1: Original
    output_lines.append(f"\n{'='*50}")
    output_lines.append(f"CASO 1: K_intra=2.0, K_inter=1.5")
    output_lines.append('='*50)
    
    t, r1, r2, r3 = simular_caso(5, 4, 3, 2.0, 1.5, 25, "Original")
    
    output_lines.append(f"Nivel 1 - Final: r={r1[-1]:.3f}")
    output_lines.append(f"Nivel 2 - Promedio: r={np.mean(r2[:,-1]):.3f}")
    output_lines.append(f"Nivel 3 - Promedio: r={np.mean(r3[:,-1]):.3f}")
    
    # Gráfica caso 1
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    axes[0].plot(t, r1, 'b-', linewidth=2)
    axes[0].set_ylabel('Nivel 1')
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Caso 1: K_intra=2.0, K_inter=1.5')
    
    axes[1].plot(t, r2.T, linewidth=1.5)
    axes[1].set_ylabel('Nivel 2')
    axes[1].set_ylim(0, 1.1)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(t, r3.T, linewidth=1, alpha=0.7)
    axes[2].set_xlabel('Tiempo (s)')
    axes[2].set_ylabel('Nivel 3')
    axes[2].set_ylim(0, 1.1)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    imagenes.append(figura_a_base64(fig))
    
    # Caso 2: Débil
    output_lines.append(f"\n{'='*50}")
    output_lines.append(f"CASO 2: K_intra=2.0, K_inter=0.5")
    output_lines.append('='*50)
    
    t, r1, r2, r3 = simular_caso(5, 4, 3, 2.0, 0.5, 25, "Débil")
    
    output_lines.append(f"Nivel 1 - Final: r={r1[-1]:.3f}")
    output_lines.append(f"Nivel 2 - Promedio: r={np.mean(r2[:,-1]):.3f}")
    output_lines.append(f"Nivel 3 - Promedio: r={np.mean(r3[:,-1]):.3f}")
    
    # Gráfica caso 2
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    axes[0].plot(t, r1, 'b-', linewidth=2)
    axes[0].set_ylabel('Nivel 1')
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Caso 2: K_intra=2.0, K_inter=0.5')
    
    axes[1].plot(t, r2.T, linewidth=1.5)
    axes[1].set_ylabel('Nivel 2')
    axes[1].set_ylim(0, 1.1)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(t, r3.T, linewidth=1, alpha=0.7)
    axes[2].set_xlabel('Tiempo (s)')
    axes[2].set_ylabel('Nivel 3')
    axes[2].set_ylim(0, 1.1)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    imagenes.append(figura_a_base64(fig))
    
    # Caso 3: Fuerte
    output_lines.append(f"\n{'='*50}")
    output_lines.append(f"CASO 3: K_intra=2.0, K_inter=3.0")
    output_lines.append('='*50)
    
    t, r1, r2, r3 = simular_caso(5, 4, 3, 2.0, 3.0, 25, "Fuerte")
    
    output_lines.append(f"Nivel 1 - Final: r={r1[-1]:.3f}")
    output_lines.append(f"Nivel 2 - Promedio: r={np.mean(r2[:,-1]):.3f}")
    output_lines.append(f"Nivel 3 - Promedio: r={np.mean(r3[:,-1]):.3f}")
    
    # Gráfica caso 3
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    axes[0].plot(t, r1, 'b-', linewidth=2)
    axes[0].set_ylabel('Nivel 1')
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Caso 3: K_intra=2.0, K_inter=3.0')
    
    axes[1].plot(t, r2.T, linewidth=1.5)
    axes[1].set_ylabel('Nivel 2')
    axes[1].set_ylim(0, 1.1)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(t, r3.T, linewidth=1, alpha=0.7)
    axes[2].set_xlabel('Tiempo (s)')
    axes[2].set_ylabel('Nivel 3')
    axes[2].set_ylim(0, 1.1)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    imagenes.append(figura_a_base64(fig))
    
    return "\n".join(output_lines), imagenes


# ============================================
# HITO 3: Kuramoto7_1 (con parámetros)
# ============================================
def ejecutar_hito3(params):
    """Hito 3: Kuramoto7_1 - Simulación de N niveles"""
    output_lines = []
    imagenes = []
    
    # Recoger parámetros
    niveles = int(params.get('niveles', 6))
    
    def simular_niveles(niv, metros_por_plataforma=3, K_intra=2.0, K_inter=1.5):
        plataformas_por_nivel = [3**i for i in range(niv)]
        osciladores_por_nivel = [p * metros_por_plataforma for p in plataformas_por_nivel]
        total = sum(osciladores_por_nivel)
        
        np.random.seed(42)
        frecuencias = []
        for nivel in range(niv):
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
            
            theta0 = theta[indices[0]]
            for i in range(len(theta0)):
                dtheta[indices[0]][i] = omega[indices[0]][i] + (K_intra/len(theta0)) * np.sum(np.sin(theta0 - theta0[i]))
            
            for nivel in range(1, niv):
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
        
        theta0 = np.random.uniform(-np.pi, np.pi, total)
        sol = solve_ivp(dynamics, t_span, theta0, t_eval=t_eval, method='RK45', rtol=1e-2)
        
        r_por_nivel = []
        for nivel in range(niv):
            fases = sol.y[indices[nivel]]
            if len(fases.shape) == 1:
                r = np.abs(np.mean(np.exp(1j * fases), axis=0))
            else:
                r = np.abs(np.mean(np.exp(1j * fases), axis=0))
            r_por_nivel.append(r)
        
        return sol.t, r_por_nivel, plataformas_por_nivel
    
    output_lines.append(f"\n{'='*60}")
    output_lines.append(f"HITO 3: SIMULANDO {niveles} NIVELES")
    output_lines.append(f"{'='*60}")
    
    t, r_niveles, plt_por_nivel = simular_niveles(niveles)
    
    # Gráfica 1: Evolución temporal
    fig, axes = plt.subplots(niveles, 1, figsize=(12, 2*niveles+4))
    if niveles == 1:
        axes = [axes]
    
    for i in range(niveles):
        axes[i].plot(t, r_niveles[i], linewidth=2)
        axes[i].set_ylabel(f'N{i}')
        axes[i].set_ylim(0, 1.1)
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
    axes[-1].set_xlabel('Tiempo')
    plt.suptitle(f'Sincronización en {niveles} Niveles')
    plt.tight_layout()
    imagenes.append(figura_a_base64(fig))
    
    # Gráfica 2: Degradación
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    r_final = [r[-1] for r in r_niveles]
    niveles_x = np.arange(niveles)
    
    ax2.plot(niveles_x, r_final, 'bo-', linewidth=2, markersize=8)
    ax2.set_xlabel('Nivel')
    ax2.set_ylabel('r final')
    ax2.set_title('Degradación de coherencia')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='Coherente')
    ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Difuso')
    ax2.legend()
    plt.tight_layout()
    imagenes.append(figura_a_base64(fig2))
    
    # Output
    output_lines.append("\n=== RESULTADOS ===")
    for i, r in enumerate(r_final):
        estado = "✅ Coherente" if r > 0.8 else "⚠️ Difuso" if r > 0.5 else "❌ Caótico"
        output_lines.append(f"Nivel {i}: r={r:.3f} {estado}")
    
    return "\n".join(output_lines), imagenes


# ============================================
# HITO 4: Modelo 46 Capas (independiente)
# ============================================
def ejecutar_hito4(params=None):
    """Hito 4: Modelo efectivo 46 capas"""
    output_lines = []
    imagenes = []
    
    n_capas = 46
    K_intra = 2.0
    K_inter_base = 1.5
    gamma_acumulado = np.ones(n_capas)
    
    r = np.zeros(n_capas)
    r[0] = 0.98
    tiempo_efectivo = np.zeros(n_capas)
    
    for capa in range(1, n_capas):
        K_inter = K_inter_base * np.exp(-0.1 * capa)
        r_capa_anterior = r[capa-1]
        r[capa] = r_capa_anterior * (1 - np.exp(-K_inter)) * np.tanh(K_intra)
        gamma_acumulado[capa] = gamma_acumulado[capa-1] * (1 + 0.05 * (1 - r[capa]))
        tiempo_efectivo[capa] = tiempo_efectivo[capa-1] + 1.0 / gamma_acumulado[capa]
    
    # Gráfica
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    capas = np.arange(n_capas)
    
    axes[0].plot(capas, r, 'b-', linewidth=2)
    axes[0].set_xlabel('Nivel')
    axes[0].set_ylabel('Sincronización r')
    axes[0].set_title('Degradación en 46 capas')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0.5, color='r', linestyle='--', label='Umbral')
    axes[0].legend()
    
    axes[1].plot(capas, tiempo_efectivo, 'g-', linewidth=2)
    axes[1].set_xlabel('Nivel')
    axes[1].set_ylabel('Tiempo efectivo')
    axes[1].set_title('Ralentización del tiempo')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    imagenes.append(figura_a_base64(fig))
    
    # Output
    capa_critica = np.where(r < 0.5)[0][0] if np.any(r < 0.5) else n_capas
    output_lines.append(f"\n{'='*60}")
    output_lines.append(f"HITO 4: MODELO 46 CAPAS")
    output_lines.append(f"{'='*60}")
    output_lines.append(f"Pérdida coherencia (r<0.5) en capa: {capa_critica}")
    output_lines.append(f"r final capa 46: {r[-1]:.4f}")
    output_lines.append(f"Tiempo efectivo acumulado: {tiempo_efectivo[-1]:.2f}")
    
    return "\n".join(output_lines), imagenes


# ============================================
# HITO 5: Kuramoto7_4 (con parámetros)
# ============================================
def ejecutar_hito5(params):
    """Hito 5: Kuramoto7_4 - Exploración de topologías"""
    output_lines = []
    imagenes = []
    
    # Recoger parámetros
    opcion = params.get('opcion', '6')
    
    # Clase específica para este hito (no compartida)
    class Hito5_KuramotoJerarquico:
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
    
    output_lines.append(f"\n{'='*60}")
    output_lines.append(f"HITO 5: OPCIÓN {opcion}")
    output_lines.append(f"{'='*60}")
    
    # Mapeo de opciones
    configs = {
        '1': ('global', 'jerarquica'),
        '2': ('anillo', 'jerarquica'),
        '3': ('scale_free', 'jerarquica'),
        '4': ('estrella', 'jerarquica'),
        '5': ('global', 'malla'),
        '6': ('scale_free', 'malla'),
        '7': ('scale_free', 'global')
    }
    
    intra, inter = configs.get(opcion, ('scale_free', 'malla'))
    K_lat = 0.3 if opcion in ['5', '6', '7'] else 0.5
    
    sim = Hito5_KuramotoJerarquico(niveles=5, metros_por_plataforma=3,
                                    topologia_intra=intra,
                                    topologia_inter=inter,
                                    K_lateral=K_lat)
    
    t, r_nivel = sim.simular(T=25, puntos=150)
    
    r_final = [r[-1] for r in r_nivel]
    for i, r in enumerate(r_final):
        estado = "✅ COHERENTE" if r > 0.8 else "⚠️ DIFUSO" if r > 0.5 else "❌ CAÓTICO"
        output_lines.append(f"Nivel {i}: r={r:.3f} {estado}")
    
    # Gráfica
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    colores = plt.cm.viridis(np.linspace(0, 1, len(r_nivel)))
    for i, r in enumerate(r_nivel):
        axes[0].plot(t, r, color=colores[i], label=f'N{i}', linewidth=1.5)
    axes[0].set_xlabel('Tiempo')
    axes[0].set_ylabel('r')
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_title(f'Opción {opcion}: {intra} + {inter}')
    
    axes[1].bar(range(len(r_final)), r_final, color=colores)
    axes[1].set_xlabel('Nivel')
    axes[1].set_ylabel('r final')
    axes[1].set_ylim(0, 1.1)
    axes[1].axhline(y=0.8, color='g', linestyle='--', alpha=0.5)
    axes[1].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    imagenes.append(figura_a_base64(fig))
    
    return "\n".join(output_lines), imagenes


# ============================================
# HITO 6: KuramotoPODB (con parámetros)
# ============================================
def ejecutar_hito6(params):
    """Hito 6: KuramotoPODB - Estados por conexión"""
    output_lines = []
    imagenes = []
    
    # Recoger parámetros
    niveles = int(params.get('niveles', 4))
    metros = int(params.get('metros', 4))
    
    # Clase específica para este hito (no compartida)
    class Hito6_KuramotoPODB:
        def __init__(self, niveles=3, metros_por_plataforma=4,
                     topologia_intra='scale_free', topologia_inter='malla',
                     K_base=2.0, K_inter_base=1.5, K_lateral_base=0.3):
            self.niveles = niveles
            self.mpp = metros_por_plataforma
            self.topologia_intra = topologia_intra
            self.topologia_inter = topologia_inter
            self.K_base = K_base
            self.K_inter_base = K_inter_base
            self.K_lateral_base = K_lateral_base
            
            self.plataformas_por_nivel = [3**i for i in range(niveles)]
            self.osciladores_por_nivel = [p * metros_por_plataforma for p in self.plataformas_por_nivel]
            self.total = sum(self.osciladores_por_nivel)
            
            np.random.seed(42)
            self.omega = np.concatenate([
                np.random.normal(1.0, 0.1 + 0.02*i, n)
                for i, n in enumerate(self.osciladores_por_nivel)
            ])
            
            self.adjacencias = self._construir_adjacencias()
            self.K_matrices = self._inicializar_K()
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
                    if self.topologia_intra == 'scale_free':
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
            K_mat = []
            for nivel in range(self.niveles):
                n_plt = self.plataformas_por_nivel[nivel]
                K_nivel = []
                for p in range(n_plt):
                    n_osc = self.mpp
                    A = self.adjacencias[nivel][p]
                    K_plt = self.K_base * A
                    K_nivel.append(K_plt)
                K_mat.append(K_nivel)
            return K_mat
        
        def _estado_a_partir_de_fase(self, delta_theta):
            delta = np.abs(delta_theta) % (2*np.pi)
            if delta > np.pi:
                delta = 2*np.pi - delta
            return (np.cos(delta) + 1) / 2
        
        def actualizar_K_por_estado(self, theta):
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
        
        def dynamics(self, t, theta):
            self.actualizar_K_por_estado(theta)
            dtheta = np.zeros_like(theta)
            
            theta0 = theta[self.indices[0]]
            n0 = len(theta0)
            K0 = self.K_matrices[0][0]
            for i in range(n0):
                suma = 0
                for j in range(n0):
                    if i != j:
                        suma += K0[i, j] * np.sin(theta0[j] - theta0[i])
                dtheta[self.indices[0]][i] = self.omega[self.indices[0]][i] + suma
            
            for nivel in range(1, self.niveles):
                n_plt = self.plataformas_por_nivel[nivel]
                n_osc = self.mpp
                theta_n = theta[self.indices[nivel]].reshape(n_plt, n_osc)
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
                
                for p in range(n_plt):
                    madre = p // 3
                    K_plt = self.K_matrices[nivel][p]
                    for i in range(n_osc):
                        intra = 0
                        for j in range(n_osc):
                            if i != j:
                                intra += K_plt[i, j] * np.sin(theta_n[p, j] - theta_n[p, i])
                        inter = self.K_inter_base * np.sin(phi_prev[madre] - theta_n[p, i])
                        
                        lateral = 0
                        norm = max(1, len(conexiones_laterales))
                        for (p1, p2) in conexiones_laterales:
                            if p == p1:
                                for j in range(n_osc):
                                    lateral += np.sin(theta_n[p2, j] - theta_n[p, i])
                            elif p == p2:
                                for j in range(n_osc):
                                    lateral += np.sin(theta_n[p1, j] - theta_n[p, i])
                        lateral = self.K_lateral_base * lateral / norm
                        
                        idx = self.indices[nivel].start + p*n_osc + i
                        dtheta_n[p, i] = self.omega[idx] + intra + inter + lateral
                
                dtheta[self.indices[nivel]] = dtheta_n.flatten()
            return dtheta
        
        def simular(self, T=20, puntos=150):
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
            return sol.t, r_nivel, sol.y
    
    output_lines.append(f"\n{'='*70}")
    output_lines.append(f"HITO 6: PODB (niveles={niveles}, metros={metros})")
    output_lines.append(f"{'='*70}")
    
    sim = Hito6_KuramotoPODB(niveles=niveles, metros_por_plataforma=metros,
                              topologia_intra='scale_free',
                              topologia_inter='malla')
    
    t, r_nivel, theta_hist = sim.simular(T=20, puntos=150)
    
    output_lines.append("\n=== SINCRONIZACIÓN POR NIVEL ===")
    for i, r in enumerate([r[-1] for r in r_nivel]):
        estado = "✅" if r > 0.8 else "⚠️" if r > 0.5 else "❌"
        output_lines.append(f"Nivel {i}: r={r:.3f} {estado}")
    
    # ============================================
    # GRÁFICA 1: Evolución de sincronización
    # ============================================
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    for i, r in enumerate(r_nivel):
        ax1.plot(t, r, label=f'Nivel {i}', linewidth=2)
    ax1.set_xlabel('Tiempo')
    ax1.set_ylabel('Sincronización r')
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title(f'PODB - {niveles} niveles, {metros} metrónomos')
    imagenes.append(figura_a_base64(fig1))
    
    # ============================================
    # GRÁFICA 2: Visualización de estados Nivel 0
    # ============================================
    theta_final = theta_hist[:, -1]
    
    # Nivel 0
    idx0 = sim.indices[0]
    theta_nivel0 = theta_final[idx0]
    n_osc0 = len(theta_nivel0)
    
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    
    # Matriz de estados
    matriz_estados0 = np.zeros((n_osc0, n_osc0))
    for i in range(n_osc0):
        for j in range(n_osc0):
            if i != j:
                delta = theta_nivel0[i] - theta_nivel0[j]
                matriz_estados0[i, j] = sim._estado_a_partir_de_fase(delta)
    
    im0 = axes2[0].imshow(matriz_estados0, cmap='RdYlGn', vmin=0, vmax=1)
    axes2[0].set_title('Estados - Nivel 0 (Base)')
    axes2[0].set_xlabel('Oscilador j')
    axes2[0].set_ylabel('Oscilador i')
    plt.colorbar(im0, ax=axes2[0])
    
    # Histograma
    valores0 = matriz_estados0[matriz_estados0 > 0].flatten()
    axes2[1].hist(valores0, bins=20, color='skyblue', edgecolor='black')
    axes2[1].axvline(x=0.8, color='blue', linestyle='--', label='P')
    axes2[1].axvline(x=0.3, color='green', linestyle='--', label='O')
    axes2[1].axvline(x=0.1, color='orange', linestyle='--', label='D')
    axes2[1].set_xlabel('Valor de estado')
    axes2[1].set_ylabel('Frecuencia')
    axes2[1].set_title('Distribución de estados - Nivel 0')
    axes2[1].legend()
    
    plt.tight_layout()
    imagenes.append(figura_a_base64(fig2))

    # 📝 ESTADÍSTICAS NIVEL 0
    valores_array0 = np.array(valores0)
    output_lines.append(f"\n📊 Estadísticas de estados - Nivel 0 (base)")
    output_lines.append(f"  Estado P (≥0.8): {np.sum(valores_array0 >= 0.8)} ({np.sum(valores_array0 >= 0.8)/len(valores_array0)*100:.1f}%)")
    output_lines.append(f"  Estado O (0.3-0.8): {np.sum((valores_array0 >= 0.3) & (valores_array0 < 0.8))} ({np.sum((valores_array0 >= 0.3) & (valores_array0 < 0.8))/len(valores_array0)*100:.1f}%)")
    output_lines.append(f"  Estado D (0.1-0.3): {np.sum((valores_array0 >= 0.1) & (valores_array0 < 0.3))} ({np.sum((valores_array0 >= 0.1) & (valores_array0 < 0.3))/len(valores_array0)*100:.1f}%)")
    output_lines.append(f"  Estado B (<0.1): {np.sum(valores_array0 < 0.1)} ({np.sum(valores_array0 < 0.1)/len(valores_array0)*100:.1f}%)")
    
    # ============================================
    # GRÁFICA 3: Visualización de estados Nivel 1
    # ============================================
    if niveles > 1:
        idx1_start = sim.indices[1].start
        theta_nivel1 = theta_final[idx1_start:idx1_start + metros]
        n_osc1 = len(theta_nivel1)
        
        fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
        
        matriz_estados1 = np.zeros((n_osc1, n_osc1))
        for i in range(n_osc1):
            for j in range(n_osc1):
                if i != j:
                    delta = theta_nivel1[i] - theta_nivel1[j]
                    matriz_estados1[i, j] = sim._estado_a_partir_de_fase(delta)
        
        im1 = axes3[0].imshow(matriz_estados1, cmap='RdYlGn', vmin=0, vmax=1)
        axes3[0].set_title('Estados - Nivel 1, Plataforma 0')
        axes3[0].set_xlabel('Oscilador j')
        axes3[0].set_ylabel('Oscilador i')
        plt.colorbar(im1, ax=axes3[0])
        
        valores1 = matriz_estados1[matriz_estados1 > 0].flatten()
        axes3[1].hist(valores1, bins=20, color='skyblue', edgecolor='black')
        axes3[1].axvline(x=0.8, color='blue', linestyle='--', label='P')
        axes3[1].axvline(x=0.3, color='green', linestyle='--', label='O')
        axes3[1].axvline(x=0.1, color='orange', linestyle='--', label='D')
        axes3[1].set_xlabel('Valor de estado')
        axes3[1].set_ylabel('Frecuencia')
        axes3[1].set_title('Distribución de estados - Nivel 1')
        axes3[1].legend()
        
        plt.tight_layout()
        imagenes.append(figura_a_base64(fig3))

        if niveles > 1:
        # ... (tu código de gráfica) ...
        # 📝 ESTADÍSTICAS NIVEL 1
            valores_array1 = np.array(valores1)
            output_lines.append(f"\n📊 Estadísticas de estados - Nivel 1, Plataforma 0")
            output_lines.append(f"  Estado P (≥0.8): {np.sum(valores_array1 >= 0.8)} ({np.sum(valores_array1 >= 0.8)/len(valores_array1)*100:.1f}%)")
            output_lines.append(f"  Estado O (0.3-0.8): {np.sum((valores_array1 >= 0.3) & (valores_array1 < 0.8))} ({np.sum((valores_array1 >= 0.3) & (valores_array1 < 0.8))/len(valores_array1)*100:.1f}%)")
            output_lines.append(f"  Estado D (0.1-0.3): {np.sum((valores_array1 >= 0.1) & (valores_array1 < 0.3))} ({np.sum((valores_array1 >= 0.1) & (valores_array1 < 0.3))/len(valores_array1)*100:.1f}%)")
            output_lines.append(f"  Estado B (<0.1): {np.sum(valores_array1 < 0.1)} ({np.sum(valores_array1 < 0.1)/len(valores_array1)*100:.1f}%)")
    
    # ============================================
    # GRÁFICA 4: Visualización de estados Nivel 2 (si existe)
    # ============================================
    if niveles > 2:
        idx2_start = sim.indices[2].start
        theta_nivel2 = theta_final[idx2_start:idx2_start + metros]
        n_osc2 = len(theta_nivel2)
        
        fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
        
        matriz_estados2 = np.zeros((n_osc2, n_osc2))
        for i in range(n_osc2):
            for j in range(n_osc2):
                if i != j:
                    delta = theta_nivel2[i] - theta_nivel2[j]
                    matriz_estados2[i, j] = sim._estado_a_partir_de_fase(delta)
        
        im2 = axes4[0].imshow(matriz_estados2, cmap='RdYlGn', vmin=0, vmax=1)
        axes4[0].set_title('Estados - Nivel 2, Plataforma 0')
        axes4[0].set_xlabel('Oscilador j')
        axes4[0].set_ylabel('Oscilador i')
        plt.colorbar(im2, ax=axes4[0])
        
        valores2 = matriz_estados2[matriz_estados2 > 0].flatten()
        axes4[1].hist(valores2, bins=20, color='skyblue', edgecolor='black')
        axes4[1].axvline(x=0.8, color='blue', linestyle='--', label='P')
        axes4[1].axvline(x=0.3, color='green', linestyle='--', label='O')
        axes4[1].axvline(x=0.1, color='orange', linestyle='--', label='D')
        axes4[1].set_xlabel('Valor de estado')
        axes4[1].set_ylabel('Frecuencia')
        axes4[1].set_title('Distribución de estados - Nivel 2')
        axes4[1].legend()
        
        plt.tight_layout()
        imagenes.append(figura_a_base64(fig4))

        if niveles > 2:
        # ... (tu código de gráfica) ...
        # 📝 ESTADÍSTICAS NIVEL 2
            valores_array2 = np.array(valores2)
            output_lines.append(f"\n📊 Estadísticas de estados - Nivel 2, Plataforma 0")
            output_lines.append(f"  Estado P (≥0.8): {np.sum(valores_array2 >= 0.8)} ({np.sum(valores_array2 >= 0.8)/len(valores_array2)*100:.1f}%)")
            output_lines.append(f"  Estado O (0.3-0.8): {np.sum((valores_array2 >= 0.3) & (valores_array2 < 0.8))} ({np.sum((valores_array2 >= 0.3) & (valores_array2 < 0.8))/len(valores_array2)*100:.1f}%)")
            output_lines.append(f"  Estado D (0.1-0.3): {np.sum((valores_array2 >= 0.1) & (valores_array2 < 0.3))} ({np.sum((valores_array2 >= 0.1) & (valores_array2 < 0.3))/len(valores_array2)*100:.1f}%)")
            output_lines.append(f"  Estado B (<0.1): {np.sum(valores_array2 < 0.1)} ({np.sum(valores_array2 < 0.1)/len(valores_array2)*100:.1f}%)")
        
    # ============================================#
    # GRÁFICA 5: Visualización de estados Nivel 3 (si existe)
    # ============================================#    
    if niveles > 3:
        idx3_start = sim.indices[3].start
        theta_nivel3 = theta_final[idx3_start:idx3_start + metros]
        n_osc3 = len(theta_nivel3)
    
        # Calcular matriz y valores
        matriz_estados3 = np.zeros((n_osc3, n_osc3))
        valores3 = []
        for i in range(n_osc3):
            for j in range(n_osc3):
                if i != j:
                    delta = theta_nivel3[i] - theta_nivel3[j]
                    valor = sim._estado_a_partir_de_fase(delta)
                    matriz_estados3[i, j] = valor
                    valores3.append(valor)
    
        # GRÁFICA NIVEL 3
        fig5, axes5 = plt.subplots(1, 2, figsize=(14, 5))
    
        # Matriz
        im3 = axes5[0].imshow(matriz_estados3, cmap='RdYlGn', vmin=0, vmax=1)
        axes5[0].set_title('Estados - Nivel 3, Plataforma 0')
        axes5[0].set_xlabel('Oscilador j')
        axes5[0].set_ylabel('Oscilador i')
        plt.colorbar(im3, ax=axes5[0])
    
        # Histograma
        axes5[1].hist(valores3, bins=20, color='skyblue', edgecolor='black')
        axes5[1].axvline(x=0.8, color='blue', linestyle='--', label='P')
        axes5[1].axvline(x=0.3, color='green', linestyle='--', label='O')
        axes5[1].axvline(x=0.1, color='orange', linestyle='--', label='D')
        axes5[1].set_xlabel('Valor de estado')
        axes5[1].set_ylabel('Frecuencia')
        axes5[1].set_title('Distribución - Nivel 3')
        axes5[1].legend()
    
        plt.tight_layout()
        imagenes.append(figura_a_base64(fig5))

        if niveles > 3:
        # 📝 ESTADÍSTICAS NIVEL 3
            valores_array3 = np.array(valores3)
            output_lines.append(f"\n📊 Estadísticas de estados - Nivel 3, Plataforma 0")
            output_lines.append(f"  Estado P (≥0.8): {np.sum(valores_array3 >= 0.8)} ({np.sum(valores_array3 >= 0.8)/len(valores_array3)*100:.1f}%)")
            output_lines.append(f"  Estado O (0.3-0.8): {np.sum((valores_array3 >= 0.3) & (valores_array3 < 0.8))} ({np.sum((valores_array3 >= 0.3) & (valores_array3 < 0.8))/len(valores_array3)*100:.1f}%)")
            output_lines.append(f"  Estado D (0.1-0.3): {np.sum((valores_array3 >= 0.1) & (valores_array3 < 0.3))} ({np.sum((valores_array3 >= 0.1) & (valores_array3 < 0.3))/len(valores_array3)*100:.1f}%)")
            output_lines.append(f"  Estado B (<0.1): {np.sum(valores_array3 < 0.1)} ({np.sum(valores_array3 < 0.1)/len(valores_array3)*100:.1f}%)")

    # ============================================
    # ESTADÍSTICAS EN OUTPUT
    # ============================================
    output_lines.append("\n=== ESTADÍSTICAS DE ESTADOS (Nivel 0) ===")
    # Aquí puedes añadir las estadísticas que quieras
    
    return "\n".join(output_lines), imagenes

# ============================================
# HITO 7: MacroPODB - VERSIÓN FUNCIONAL
# ============================================
def ejecutar_hito7(params):
    """Hito 7: MacroPODB - Versión completa funcional"""
    output_lines = []
    imagenes = []
    
    # Recoger parámetros
    opcion = int(params.get('opcion', '6'))
    niveles = int(params.get('niveles', 5))
    modo = params.get('modo', 'podb')
    
    output_lines.append(f"\n{'='*80}")
    output_lines.append(f"HITO 7: MACRO PODB")
    output_lines.append(f"{'='*80}")
    output_lines.append(f"Parámetros: opción={opcion}, niveles={niveles}, modo={modo}")
    
    # ============================================
    # CLASE COMPLETA (copiada de MacroPODB.py)
    # ============================================
    class KuramotoUnificado:
        # Definición de estados PODB
        ESTADOS = {
            'P': {'color': 'blue', 'rango': (0.8, 1.0), 'desc': 'Partícula', 'simbolo': '🔵'},
            'O': {'color': 'green', 'rango': (0.3, 0.8), 'desc': 'Onda', 'simbolo': '🟢'},
            'D': {'color': 'orange', 'rango': (0.1, 0.3), 'desc': 'Difuso', 'simbolo': '🟠'},
            'B': {'color': 'red', 'rango': (0.0, 0.1), 'desc': 'Borrado', 'simbolo': '🔴'}
        }

        # Diccionario de opciones disponibles
        OPCIONES = {
            1: {'nombre': 'Control', 'intra': 'global', 'inter': 'jerarquica', 'desc': 'Global + Jerárquica'},
            2: {'nombre': 'Anillo', 'intra': 'anillo', 'inter': 'jerarquica', 'desc': 'Anillo intra + Jerárquica'},
            3: {'nombre': 'Scale-free', 'intra': 'scale_free', 'inter': 'jerarquica', 'desc': 'Scale-free intra + Jerárquica'},
            4: {'nombre': 'Estrella', 'intra': 'estrella', 'inter': 'jerarquica', 'desc': 'Estrella intra + Jerárquica'},
            5: {'nombre': 'Global+Malla', 'intra': 'global', 'inter': 'malla', 'desc': 'Global + Malla inter'},
            6: {'nombre': 'Scale-free+Malla', 'intra': 'scale_free', 'inter': 'malla', 'desc': 'Scale-free + Malla inter'},
            7: {'nombre': 'Scale-free+Global', 'intra': 'scale_free', 'inter': 'global', 'desc': 'Scale-free + Global inter'}
        }

        def __init__(self,
                     opcion=6,
                     niveles=6,
                     metros_por_plataforma=3,
                     modo='podb',
                     K_base=2.0,
                     K_inter_base=1.5,
                     K_lateral_base=0.3,
                     tracking_detallado=True):

            self.opcion = opcion
            self.niveles = niveles
            self.mpp = metros_por_plataforma
            self.modo = modo
            self.K_base = K_base
            self.K_inter_base = K_inter_base
            self.K_lateral_base = K_lateral_base
            self.tracking_detallado = tracking_detallado

            config = self.OPCIONES[opcion]
            self.topologia_intra = config['intra']
            self.topologia_inter = config['inter']
            self.nombre_config = config['nombre']
            self.desc_config = config['desc']

            self.plataformas_por_nivel = [3**i for i in range(niveles)]
            self.osciladores_por_nivel = [p * metros_por_plataforma for p in self.plataformas_por_nivel]
            self.total = sum(self.osciladores_por_nivel)

            np.random.seed(42)
            self.omega = np.concatenate([
                np.random.normal(1.0, 0.1 + 0.02*i, n)
                for i, n in enumerate(self.osciladores_por_nivel)
            ])

            self.adjacencias = self._construir_adjacencias()
            self.K_matrices = self._inicializar_K()
            self.indices = self._construir_indices()

            self.historial_tiempos = []
            self.historial_r_por_nivel = defaultdict(list)
            self.historial_estados_por_nivel = defaultdict(list)
            self.paso_actual = 0

        def _construir_indices(self):
            indices = []
            start = 0
            for n in self.osciladores_por_nivel:
                indices.append(slice(start, start + n))
                start += n
            return indices

        def _construir_adjacencias_intra(self, n_osc, topologia):
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
            elif topologia == 'estrella':
                A = np.zeros((n_osc, n_osc))
                centro = 0
                for i in range(1, n_osc):
                    A[centro, i] = 1
                    A[i, centro] = 1
                return A
            elif topologia == 'scale_free':
                if n_osc > 2:
                    return generar_scale_free(n_osc, m=2)
                else:
                    return np.ones((n_osc, n_osc)) - np.eye(n_osc)
            elif topologia == 'small_world':
                return generar_small_world(n_osc, k=2, p=0.2)
            else:
                return np.ones((n_osc, n_osc)) - np.eye(n_osc)

        def _construir_adjacencias(self):
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
            K_mat = []
            for nivel in range(self.niveles):
                n_plt = self.plataformas_por_nivel[nivel]
                K_nivel = []
                for p in range(n_plt):
                    A = self.adjacencias[nivel][p]
                    K_plt = self.K_base * A
                    K_nivel.append(K_plt)
                K_mat.append(K_nivel)
            return K_mat

        def _estado_a_partir_de_fase(self, delta_theta):
            delta = np.abs(delta_theta) % (2*np.pi)
            if delta > np.pi:
                delta = 2*np.pi - delta
            return (np.cos(delta) + 1) / 2

        def actualizar_K_por_estado(self, theta):
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
            return conexiones

        def dynamics(self, t, theta):
            self.actualizar_K_por_estado(theta)
            dtheta = np.zeros_like(theta)

            theta0 = theta[self.indices[0]]
            n0 = len(theta0)
            K0 = self.K_matrices[0][0]
            for i in range(n0):
                suma = 0
                for j in range(n0):
                    if i != j:
                        suma += K0[i, j] * np.sin(theta0[j] - theta0[i])
                dtheta[self.indices[0]][i] = self.omega[self.indices[0]][i] + suma

            for nivel in range(1, self.niveles):
                n_plt = self.plataformas_por_nivel[nivel]
                n_osc = self.mpp
                theta_n = theta[self.indices[nivel]].reshape(n_plt, n_osc)
                dtheta_n = np.zeros_like(theta_n)

                theta_prev = theta[self.indices[nivel-1]].reshape(self.plataformas_por_nivel[nivel-1], self.mpp)
                r_prev = np.mean(np.exp(1j * theta_prev), axis=1)
                phi_prev = np.angle(r_prev)

                conexiones = self._construir_conexiones_laterales(n_plt)

                for p in range(n_plt):
                    madre = p // 3
                    K_plt = self.K_matrices[nivel][p]
                    for i in range(n_osc):
                        intra = 0
                        for j in range(n_osc):
                            if i != j:
                                intra += K_plt[i, j] * np.sin(theta_n[p, j] - theta_n[p, i])
                        inter = self.K_inter_base * np.sin(phi_prev[madre] - theta_n[p, i])
                        
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

        def simular(self, T=15, puntos=100):
            theta0 = np.random.uniform(-np.pi, np.pi, self.total)
            t_eval = np.linspace(0, T, puntos)
            sol = solve_ivp(self.dynamics, (0, T), theta0,
                            t_eval=t_eval,
                            method='RK45', rtol=1e-2)

            # Registrar TODOS los puntos (con la indentación correcta)
            if self.tracking_detallado:
                for i, t_point in enumerate(t_eval):
                    self.registrar_estados(sol.y[:, i], t_point)

            return sol.t, sol.y

        def registrar_estados(self, theta, t):
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

                fases_nivel = theta[self.indices[nivel]]
                if len(fases_nivel.shape) == 1:
                    r_nivel = np.abs(np.mean(np.exp(1j * fases_nivel)))
                else:
                    r_nivel = np.abs(np.mean(np.exp(1j * fases_nivel), axis=0))
                    r_nivel = np.mean(r_nivel) if hasattr(r_nivel, '__len__') else r_nivel

                self.historial_r_por_nivel[nivel].append(r_nivel)

                if self.modo == 'podb':
                    conteo = {'P': 0, 'O': 0, 'D': 0, 'B': 0}
                    total = 0
                    for p in range(n_plt):
                        A = self.adjacencias[nivel][p]
                        for i in range(n_osc):
                            for j in range(n_osc):
                                if A[i, j] > 0 and i != j:
                                    delta = theta_reshaped[p, i] - theta_reshaped[p, j]
                                    valor = self._estado_a_partir_de_fase(delta)
                                    if valor > 0.8:
                                        conteo['P'] += 1
                                    elif valor > 0.3:
                                        conteo['O'] += 1
                                    elif valor > 0.1:
                                        conteo['D'] += 1
                                    else:
                                        conteo['B'] += 1
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

        def obtener_resultados_texto(self):
            lines = []
            lines.append(f"\n{'='*80}")
            lines.append(f"📊 RESULTADOS - Opción {self.opcion}: {self.nombre_config}")
            lines.append(f"{'='*80}")
            lines.append(f"\n📈 SINCRONIZACIÓN POR NIVEL (r):")
            lines.append(f"{'-'*60}")
            lines.append(f"Nivel | Plataformas | r_inicial | r_final | Estado")
            lines.append(f"{'-'*60}")

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

                lines.append(f"{nivel:3d}   | {self.plataformas_por_nivel[nivel]:6d}      | {r_inicial:.3f}     | {r_final:.3f}   | {estado}")

            if self.modo == 'podb' and self.historial_estados_por_nivel:
                lines.append(f"\n{'='*80}")
                lines.append(f"🌀 DISTRIBUCIÓN DE ESTADOS P-O-D-B (tiempo final)")
                lines.append(f"{'='*80}")

                for nivel in range(self.niveles):
                    if nivel not in self.historial_estados_por_nivel or not self.historial_estados_por_nivel[nivel]:
                        continue
                    final = self.historial_estados_por_nivel[nivel][-1]
                    lines.append(f"\nNIVEL {nivel} ({self.plataformas_por_nivel[nivel]} plataformas):")
                    lines.append(f"  🔵 Partícula (P): {final['P']*100:5.1f}%")
                    lines.append(f"  🟢 Onda (O):      {final['O']*100:5.1f}%")
                    lines.append(f"  🟠 Difuso (D):    {final['D']*100:5.1f}%")
                    lines.append(f"  🔴 Borrado (B):   {final['B']*100:5.1f}%")
                    lines.append(f"  Total conexiones: {final['total']}")

                max_mezcla = 0
                nivel_soc = -1
                for nivel in range(self.niveles):
                    if nivel not in self.historial_estados_por_nivel:
                        continue
                    final = self.historial_estados_por_nivel[nivel][-1]
                    mezcla = final['P'] * final['O'] * final['D'] * final['B'] * 10000
                    if mezcla > max_mezcla:
                        max_mezcla = mezcla
                        nivel_soc = nivel

                if nivel_soc >= 0:
                    final_soc = self.historial_estados_por_nivel[nivel_soc][-1]
                    lines.append(f"\n🔬 PUNTO CRÍTICO (SOC) IDENTIFICADO:")
                    lines.append(f"   Nivel {nivel_soc} presenta la máxima mezcla de estados:")
                    lines.append(f"   P={final_soc['P']*100:.1f}%, O={final_soc['O']*100:.1f}%, "
                                 f"D={final_soc['D']*100:.1f}%, B={final_soc['B']*100:.1f}%")

            lines.append(f"\n{'='*80}")
            return "\n".join(lines)

        def capturar_graficas(self):
            imagenes = []
            
            # Gráfica de sincronización
            fig1 = plt.figure(figsize=(12, 5))
            for nivel in range(self.niveles):
                if nivel in self.historial_r_por_nivel and self.historial_r_por_nivel[nivel]:
                    tiempos = self.historial_tiempos[:len(self.historial_r_por_nivel[nivel])]
                    plt.plot(tiempos, self.historial_r_por_nivel[nivel],
                            label=f'N{nivel}', linewidth=2)
            plt.xlabel('Tiempo')
            plt.ylabel('Sincronización r')
            plt.ylim(0, 1.1)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.title(f'Opción {self.opcion}: {self.nombre_config} - Evolución de r')
            plt.tight_layout()
            imagenes.append(figura_a_base64(fig1))

            # Gráficas de evolución de estados (modo PODB)
            if self.modo == 'podb' and self.historial_estados_por_nivel:
                niveles_con_datos = [n for n in range(self.niveles)
                                   if n in self.historial_estados_por_nivel and self.historial_estados_por_nivel[n]]

                for nivel in niveles_con_datos:
                    datos = self.historial_estados_por_nivel[nivel]
                    tiempos = [d['t'] for d in datos]

                    fig2 = plt.figure(figsize=(12, 4))
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
                    plt.tight_layout()
                    imagenes.append(figura_a_base64(fig2))

            return imagenes

    # ============================================
    # EJECUTAR SIMULACIÓN
    # ============================================
    sim = KuramotoUnificado(
        opcion=opcion,
        niveles=niveles,
        metros_por_plataforma=3,
        modo=modo,
        K_base=2.0,
        K_inter_base=1.5,
        K_lateral_base=0.3,
        tracking_detallado=True
    )

    # Simular
    t, theta = sim.simular(T=15, puntos=100)
    
    # Obtener resultados en texto
    output_lines.append(sim.obtener_resultados_texto())
    
    # Capturar gráficas
    imagenes.extend(sim.capturar_graficas())
    
    return "\n".join(output_lines), imagenes

# ============================================
# ENDPOINT PRINCIPAL
# ============================================

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'ok',
        'mensaje': 'Backend Kuramoto - Framework P-O-D-B',
        'scripts_disponibles': ['hito1', 'hito2', 'hito3', 'hito4', 'hito5', 'hito6', 'hito7']
    })

@app.route('/ejecutar/<script>', methods=['GET'])
def ejecutar_script(script):
    try:
        # Recoger parámetros de la URL
        params = {
            'niveles': request.args.get('niveles', '6'),
            'opcion': request.args.get('opcion', '6'),
            'metros': request.args.get('metros', '4'),
            'modo': request.args.get('modo', 'podb')
        }
        
        # Mapeo directo - cada hito es completamente independiente
        if script == 'hito1':
            output, imagenes = ejecutar_hito1(params)
        elif script == 'hito2':
            output, imagenes = ejecutar_hito2(params)
        elif script == 'hito3':
            output, imagenes = ejecutar_hito3(params)
        elif script == 'hito4':
            output, imagenes = ejecutar_hito4(params)
        elif script == 'hito5':
            output, imagenes = ejecutar_hito5(params)
        elif script == 'hito6':
            output, imagenes = ejecutar_hito6(params)
        elif script == 'hito7':
            output, imagenes = ejecutar_hito7(params)
        else:
            return jsonify({'success': False, 'error': f'Script "{script}" no encontrado'})

        respuesta = {
            'success': True, 
            'output': output, 
            'imagenes': imagenes
        }
        
        callback = request.args.get('callback')
        if callback:
            from flask import json
            return f"{callback}({json.dumps(respuesta)})"
        else:
            return jsonify(respuesta)

    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        respuesta = {'success': False, 'error': error_msg}
        callback = request.args.get('callback')
        if callback:
            from flask import json
            return f"{callback}({json.dumps(respuesta)})"
        else:
            return jsonify(respuesta)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
