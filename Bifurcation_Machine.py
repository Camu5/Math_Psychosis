
#   ▄▄▄▄    ██▓  █████▒█    ██  ██▀███   ▄████▄   ▄▄▄     ▄▄▄█████▓ ██▓ ▒█████   ███▄    █    ███▄ ▄███▓ ▄▄▄       ▄████▄   ██░ ██  ██▓ ███▄    █ ▓█████ 
# ▓█████▄ ▓██▒▓██   ▒ ██  ▓██▒▓██ ▒ ██▒▒██▀ ▀█  ▒████▄   ▓  ██▒ ▓▒▓██▒▒██▒  ██▒ ██ ▀█   █   ▓██▒▀█▀ ██▒▒████▄    ▒██▀ ▀█  ▓██░ ██▒▓██▒ ██ ▀█   █ ▓█   ▀ 
# ▒██▒ ▄██▒██▒▒████ ░▓██  ▒██░▓██ ░▄█ ▒▒▓█    ▄ ▒██  ▀█▄ ▒ ▓██░ ▒░▒██▒▒██░  ██▒▓██  ▀█ ██  ▒▓██    ▓██░▒██  ▀█▄  ▒▓█    ▄ ▒██▀▀██░▒██▒▓██  ▀█ ██▒▒███   
# ▒██░█▀  ░██░░▓█▒  ░▓▓█  ░██░▒██▀▀█▄  ▒▓▓▄ ▄██▒░██▄▄▄▄██░ ▓██▓ ░ ░██░▒██   ██░▓██▒  ▐▌██  ▒▒██    ▒██ ░██▄▄▄▄██ ▒▓▓▄ ▄██▒░▓█ ░██ ░██░▓██▒  ▐▌██▒▒▓█  ▄ 
# ░▓█  ▀█▓░██░░▒█░   ▒▒█████▓ ░██▓ ▒██▒▒ ▓███▀ ░ ▓█   ▓██▒ ▒██▒ ░ ░██░░ ████▓▒░▒██░   ▓██  ░▒██▒   ░██▒ ▓█   ▓██▒▒ ▓███▀ ░░▓█▒░██▓░██░▒██░   ▓██░░▒████▒
# ░▒▓███▀▒░▓   ▒ ░   ░▒▓▒ ▒ ▒ ░ ▒▓ ░▒▓░░ ░▒ ▒  ░ ▒▒   ▓▒█░ ▒ ░░   ░▓  ░ ▒░▒░▒░ ░ ▒░   ▒ ▒   ░ ▒░   ░  ░ ▒▒   ▓▒█░░ ░▒ ▒  ░ ▒ ░░▒░▒░▓  ░ ▒░   ▒ ▒ ░░ ▒░ ░
# ▒░▒   ░  ▒ ░ ░     ░░▒░ ░ ░   ░▒ ░ ▒░  ░  ▒     ▒   ▒▒ ░   ░     ▒ ░  ░ ▒ ▒░ ░ ░░   ░ ▒  ░░  ░      ░  ▒   ▒▒ ░  ░  ▒    ▒ ░▒░ ░ ▒ ░░ ░░   ░ ▒░ ░ ░  ░
#  ░    ░  ▒ ░ ░ ░    ░░░ ░ ░   ░░   ░ ░          ░   ▒    ░       ▒ ░░ ░ ░ ▒     ░   ░ ░  ░      ░     ░   ▒   ░         ░  ░░ ░ ▒ ░   ░   ░ ░    ░   
#  ░       ░            ░        ░     ░ ░            ░  ░         ░      ░ ░           ░         ░         ░  ░░ ░       ░  ░  ░ ░           ░    ░  ░
#       ░                              ░                                                                       ░                                      

# Autor: Santiago Caballero Rosas
# Este código sirve como prueba empírica-numérica de que las bifurcaciones ocurren cuando m_E = m_I.
# Así mismo, sirve como prueba de que se trata de una bifurcación de Andronov-Hopf dadas las dinámicas antes durante y después de la bifurcación.

# Librerias necesarias
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Definimos el sistema dinámico (E/I)
def schizo_IE_system(t, y, m_E, m_I, k_E, k_I, g_E, g_I, S, H, gamma, delta):
    E, I = y
    f_E = gamma * (E - (k_I * I + H * (m_E * g_E - g_I * k_I)) / (k_E * k_I - m_E * m_I))**2  # f_E = gamma(E - E*)^2
    g_EI = delta * (k_I * I + H * (m_E * g_E - g_I * k_I)) / (k_E * k_I - m_E * m_I) * \
           (I - (-m_E * I + H * (g_E * m_I - k_E * g_I)) / (k_E * k_I - m_E * m_I))
    dE_dt = -m_E * E - k_E * I + S - g_E * (H - f_E)
    dI_dt = k_I * E + m_I * I + g_I * (H - f_E - g_EI)
    return [dE_dt, dI_dt]

# Parámetros
k_E = 4
k_I = 8
m_I = 0.5
g_E = 0.1
g_I = 0.1
S = 2
H = 1.5
gamma = 0.05
delta = 0.05

# Simulación propiamente dicha
t_span = (0, 100)

def plot_phase_plane(m_E):
    # Condiciones iniciales
    y0 = [0, 0]  # Comenzamos en el equilibrio

    # Resolvemos el sistema usando solve_ivp
    sol = solve_ivp(schizo_IE_system, t_span, y0, args=(m_E, m_I, k_E, k_I, g_E, g_I, S, H, gamma, delta),
                    dense_output=True, t_eval=np.linspace(t_span[0], t_span[1], 1000))

    # Creamos la cuadrícula y el espacio lineal para el plano fase
    E_vals = np.linspace(-1, 1, 20)
    I_vals = np.linspace(-1, 1, 20)
    E, I = np.meshgrid(E_vals, I_vals)
    U = np.zeros(E.shape)
    V = np.zeros(I.shape)

    # Calculamos los vectores del campo en cada punto de la cuadrícula
    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            dE_dt, dI_dt = schizo_IE_system(0, [E[i, j], I[i, j]], m_E, m_I, k_E, k_I, g_E, g_I, S, H, gamma, delta)
            U[i, j] = dE_dt
            V[i, j] = dI_dt

    # Graficamos el plano fase y la trayectoria del sistema
    plt.figure(figsize=(8, 6))
    plt.streamplot(E, I, U, V, color='r', linewidth=1, density=1.5)
    plt.plot(sol.y[0], sol.y[1], 'b-', label='Trayectoria')
    plt.xlabel('Actividad Excitatoria (E)')
    plt.ylabel('Actividad Inhibitoria (I)')
    plt.title(f'Plano fase para $m_E = {m_E}$')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.grid(True)
    plt.legend()
    plt.show()

    # Cálculo del punto de equilibrio
    A_inv = np.linalg.inv(np.array([[-m_E, -k_E], [k_I, m_I]]))
    y_eq = A_inv @ np.array([0, 0])
    print(f"Punto de Equilibrio: {y_eq}")

    # Cálculo de la matriz Jacobiana, autovalores, autovectores, traza, y determinante
    jacobian = np.array([[-m_E, -k_E], [k_I, m_I]])
    eigenvalues, eigenvectors = np.linalg.eig(jacobian)
    traza = np.trace(jacobian)
    determinante = np.linalg.det(jacobian)
    
    # Imprimir lo valores calculados.
    print(f"Autovalores: {eigenvalues}")
    print(f"Autovectores:\n{eigenvectors}")
    print(f"Traza de la Jacobiana: {traza}")
    print(f"Determinante de la Jacobiana: {determinante}")

# Distintos valores de m_E
plot_phase_plane(0.6)  # Antes de la bifurcación
plot_phase_plane(0.5)  # Bifurcación
plot_phase_plane(0.4)  # Después de la bifurcación
