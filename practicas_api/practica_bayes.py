import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # <-- Añadimos la librería Pandas
from scipy.stats import multivariate_normal

# --- CONFIGURACIÓN DE LAS MATRICES ---
mean = [0, 0] 

covs = [
    [[0.1, 0], [0, 1]],         # (a)
    [[0.2, 0.1], [0.1, 0.2]],   # (b)
    [[0.7, 0.5], [0.5, 0.7]],   # (c)
    [[3, 0], [0, 0.5]],         # (d)
    [[1.1, -0.9], [-0.9, 1]],   # (e)
    [[5, 0], [0, 5]]            # (f)
]

titulos = ['(a) s1=0.1, s2=1, cov=0', '(b) s1=0.2, s2=0.2, cov=0.1', 
           '(c) s1=0.7, s2=0.7, cov=0.5', '(d) s1=3, s2=0.5, cov=0', 
           '(e) s1=1.1, s2=1, cov=-0.9', '(f) s1=5, s2=5, cov=0']

print("Ejecutando Ejercicio 1: Generando nubes de puntos 2D...")

# =======================================================
# EJERCICIO 1: Gráficas 2D (1000 muestras)
# =======================================================
# Aplicamos tu sugerencia de 2 filas y 3 columnas
fig1, axes1 = plt.subplots(2, 3, figsize=(15, 8)) 
axes1 = axes1.flatten()

for i in range(6):
    # Numpy genera los datos
    x, y = np.random.multivariate_normal(mean, covs[i], 1000).T
    
    # PANDAS EN ACCIÓN: Guardamos los datos en un DataFrame
    df_muestras = pd.DataFrame({'Eje_X': x, 'Eje_Y': y})
    
    # Leemos desde Pandas para graficar
    axes1[i].plot(df_muestras['Eje_X'], df_muestras['Eje_Y'], '.', alpha=0.3)
    axes1[i].set_title(titulos[i])
    axes1[i].set_xlim([-5, 5])
    axes1[i].set_ylim([-5, 5])
    axes1[i].set_aspect('equal') 
    axes1[i].grid(True)

plt.tight_layout()
plt.show() 

print("Ejecutando Ejercicio 2: Generando superficies 3D (10,000 muestras)...")

# =======================================================
# EJERCICIO 2: Distribución de Probabilidad 3D (10,000 muestras)
# =======================================================
x_3d = np.linspace(-4, 4, 100)
y_3d = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x_3d, y_3d)
pos = np.dstack((X, Y))

fig2 = plt.figure(figsize=(18, 10))

for i in range(6):
    # Ajustamos también el 3D a 2 filas y 3 columnas
    ax = fig2.add_subplot(2, 3, i+1, projection='3d')
    
    rv = multivariate_normal(mean, covs[i])
    Z = rv.pdf(pos)
    
    superficie = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title(titulos[i])
    ax.set_xlabel('Eje X')
    ax.set_ylabel('Eje Y')
    ax.set_zlabel('Probabilidad')

plt.tight_layout()
plt.show()