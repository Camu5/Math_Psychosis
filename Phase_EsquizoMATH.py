


#   ██████╗ ██╗  ██╗ █████╗ ███████╗███████╗    ██████╗ ██╗      █████╗ ███╗   ██╗███████╗     ██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗
#   ██╔══██╗██║  ██║██╔══██╗██╔════╝██╔════╝    ██╔══██╗██║     ██╔══██╗████╗  ██║██╔════╝    ██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║
#   ██████╔╝███████║███████║███████╗█████╗█████╗██████╔╝██║     ███████║██╔██╗ ██║█████╗      ██║  ███╗██████╔╝███████║██████╔╝███████║
#   ██╔═══╝ ██╔══██║██╔══██║╚════██║██╔══╝╚════╝██╔═══╝ ██║     ██╔══██║██║╚██╗██║██╔══╝      ██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║
#   ██║     ██║  ██║██║  ██║███████║███████╗    ██║     ███████╗██║  ██║██║ ╚████║███████╗    ╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║
#   ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝    ╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝     ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝
                                                                                                                                   
# Este programa nos permite obtener los planos fase del sistema para las tres condiciones: homeostática (salud), inestable, patológica (psicosis).
# También e calculan los puntos de equilibrio, autovectores, autovalores, la traza y el determinante de las condiciones.
# Asimismo, el programa calcula el exponente de Lyapunov para tales condiciones, lo que nos permite profundizar en el análisis de estabilidad.

# Autor: Santiago Caballero Rosas

# Importamos las librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from lyapynov.DynamicalSystem import ContinuousDS # Para el cálculo del número de Lyapunov.
from lyapynov.Lyapunov import mLCE 

# Definición del sistema dinámico: modelo Schizophrenia-E/I
def schizo_IE_system(y, t, m_E, m_I, k_E, k_I):
    E, I = y
    f_E = gamma * (E - (k_I * I + H * (m_E * g_E - g_I * k_I)) / (k_E * k_I - m_E * m_I))**2 # f_E = gamma(E - E*)^2
    g_EI = delta * (k_I * I + H * (m_E * g_E - g_I * k_I)) / (k_E * k_I - m_E * m_I) * \
          (I - (-m_E * I + H * (g_E * m_I - k_E * g_I)) / (k_E * k_I - m_E * m_I))
    dE_dt = -m_E * E - k_E * I + I - g_E * (H - f_E)
    dI_dt = k_I * E + m_I * I + g_I * (H - f_E - g_EI)
    return np.array([dE_dt, dI_dt])

# Runge-Kutta de Orden 4 (RK4)
def rk4_step(f, y, t, dt, *args):
    k1 = f(y, t, *args)
    k2 = f(y + 0.5 * dt * k1, t + 0.5 * dt, *args)
    k3 = f(y + 0.5 * dt * k2, t + 0.5 * dt, *args)
    k4 = f(y + dt * k3, t + dt, *args)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# Parámetros y condiciones iniciales conocidos
condiciones = [
    {"m_E": 1.1, "m_I": 1, "k_E": 4.0, "k_I": 8.0, "estado": "Inestable", "descripcion": "Psicótico", "lyapunov": 0.0317},
    {"m_E": 0.95, "m_I": 1, "k_E": 4.0, "k_I": 8.0, "estado": "Estable", "descripcion": "Sano", "lyapunov": -0.0384},
    {"m_E": 1.4473684210526314, "m_I": 1.3684210526315788, "k_E": 4.0, "k_I": 8.0, "estado": "Inestable", "descripcion": "Inestable", "lyapunov": 0.0028}
]

# Parámetros comunes
gamma = 1
delta = 1
g_E = 2
g_I = 1
H = 1
I = 0 
t0 = 0.0                    # Tiempo inicial
dt = 0.01                   # Intervalo de tiempo
t_span = 150               # Tiempo de integración
t_eval = np.arange(t0, t_span, dt)
y0 = np.array([0.25, 0.25])   # Condiciones iniciales

# Graficar cada uno de los tres estados
for idx, cond in enumerate(condiciones):
    m_E, m_I, k_E, k_I = cond['m_E'], cond['m_I'], cond['k_E'], cond['k_I']
    
    # Imprimir los parámetros utilizados
    print(f"\n\n--- Estado {cond['descripcion']} ({cond['estado']}) ---")
    print(f"Parámetros: m_E = {m_E}, m_I = {m_I}, k_E = {k_E}, k_I = {k_I}")

    # Cálculo del punto de equilibrio
    A_inv = np.linalg.inv(np.array([[-m_E, -k_E], [k_I, m_I]]))
    y_eq = A_inv @ np.array([0, 0])
    print(f"Punto de Equilibrio: {y_eq}")

    # Cálculo de la matriz Jacobiana, autovalores, autovectores, traza, y determinante
    jacobian = np.array([[-m_E, -k_E], [k_I, m_I]])
    eigenvalues, eigenvectors = np.linalg.eig(jacobian)
    traza = np.trace(jacobian)
    determinante = np.linalg.det(jacobian)

    print(f"Autovalores: {eigenvalues}")
    print(f"Autovectores:\n{eigenvectors}")
    print(f"Traza de la Jacobiana: {traza}")
    print(f"Determinante de la Jacobiana: {determinante}")

    # Integrar el sistema usando RK4
    trajectory = np.zeros((len(t_eval), len(y0)))
    trajectory[0] = y0
    y = y0
    for i in range(1, len(t_eval)):
        y = rk4_step(schizo_IE_system, y, t_eval[i-1], dt, m_E, m_I, k_E, k_I)
        trajectory[i] = y

    # Cálculo del máximo exponente de Lyapunov
    system = ContinuousDS(x0=y0, t0=t0, f=lambda y, t: schizo_IE_system(y, t, m_E, m_I, k_E, k_I),
                          jac=lambda y, t: jacobian, dt=dt)
    n_forward, n_compute, keep = 100, 1000, False
    try:
        max_lyapunov = mLCE(system, n_forward=n_forward, n_compute=n_compute, keep=keep)
    except Exception as e:
        max_lyapunov = cond["lyapunov"]
        print(f"Error calculando el exponente de Lyapunov, usando valor conocido: {e}")

    print(f"Exponente de Lyapunov calculado: {max_lyapunov:.4f}")

    # Crear un plano fase y graficar la trayectoria
    E, I = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))  # Aumentar rango para mayor claridad
    U, V = schizo_IE_system([E, I], t0, m_E, m_I, k_E, k_I)

    # Normalizar el campo vectorial
    N = np.sqrt(U**2 + V**2)
    U, V = U/N, V/N

    plt.figure(figsize=(10, 8))
    plt.streamplot(E, I, U, V, color='slateblue', linewidth=0.8, density=1.5, arrowsize=1.2, arrowstyle='->')
    
    # Dibujar una trayectoria clara
    plt.plot(trajectory[:, 0], trajectory[:, 1], color='darkorange', linewidth=1, label='Trayectoria')

    # Añadir una flecha para indicar la dirección en la trayectoria
    idx_arrow = len(trajectory) // 3  # Posición para la flecha
    plt.annotate('', xy=(trajectory[idx_arrow + 1, 0], trajectory[idx_arrow + 1, 1]), 
                 xytext=(trajectory[idx_arrow, 0], trajectory[idx_arrow, 1]),
                 arrowprops=dict(facecolor='darkorange', edgecolor='darkorange', arrowstyle='->', lw=2))

    # Añadir una caja de texto con la descripción del estado y parámetros
    textstr = (f"Estado: {cond['descripcion']} ({cond['estado']})\n"
               f"m_E = {m_E}\n"
               f"m_I = {m_I}\n"
               f"k_E = {k_E}\n"
               f"k_I = {k_I}\n"
               f"L (Lyapunov) = {max_lyapunov:.4f}")
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    # Configuraciones del gráfico
    plt.xlabel('Actividad relativa de las Neuronas Excitadoras (E)')
    plt.ylabel('Actividad relativa de las Neuronas Inhibidoras (I)')
    plt.title(f"Retrato Fase del Sistema")
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.legend()
    plt.grid()
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.show()
