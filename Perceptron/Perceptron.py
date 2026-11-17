import csv
import numpy as np
import matplotlib.pyplot as plt

archivo_path = 'SIRTUIN6.csv'

with open(archivo_path, 'r') as archivo:
    datos_crudos = list(csv.reader(archivo))[1:]
num_muestras = len(datos_crudos)

X = np.zeros((num_muestras, 3))
Y = np.zeros(num_muestras)

for i in range(num_muestras):
    fila = datos_crudos[i]
    X[i, 0] = float(fila[1])
    X[i, 1] = float(fila[5])
    X[i, 2] = 1.0  # El sesgo (bias) fijo de tu libreta
    
    etiqueta = int(fila[6])
    Y[i] = -1 if etiqueta == 0 else etiqueta

w = np.random.uniform(-1, 1, 3) # Inicio w RANDOM
print(f"w inicial RANDOM: {w}\n")

iteracion = 1

while True: # Iterar
    hubo_actualizacion = False
    
    for i in range(num_muestras):
        x = X[i]
        etiqueta = Y[i]
        
        # S.P = funcion activacion (w . x)
        producto_punto = np.dot(w, x)
        sp = 1 if producto_punto >= 0 else -1
            
        # comparar S.P vs Etiqueta
        if sp != etiqueta:
            print(f"Error Iteración {iteracion}, muestra m_{i+1} | SP ({sp}) != Etiqueta ({etiqueta})")
            
            # Actualizacion: w = w + etiqueta * x
            w = w + (etiqueta * x)
            
            hubo_actualizacion = True
            break # Romper ciclo para volver a empezar desde m1
            
    if not hubo_actualizacion:
        print(f"\n¡ALGORITMO FINALIZADO CON ÉXITO!")
        print(f"Convergencia en iteración: {iteracion}")
        print(f"Vector w final: {w}")
        break
        
    iteracion += 1

eje_x = X[:, 0]
eje_y = X[:, 1]

colores = np.where(Y == 1, '#fa5f49', '#84b6f4')
plt.figure(figsize=(8, 6))
plt.scatter(eje_x, eje_y, c=colores, edgecolors='black', zorder=2)

x_min, x_max = np.min(eje_x) - 0.2, np.max(eje_x) + 0.2
y_min = -(w[0] * x_min + w[2]) / w[1]
y_max = -(w[0] * x_max + w[2]) / w[1]

plt.plot([x_min, x_max], [y_min, y_max], color='black', linestyle='--', linewidth=2.5, zorder=1, label='Frontera Perceptrón')

plt.title(f'Frontera de Decisión - Perceptrón (Iteración {iteracion})')
plt.xlabel('Característica 1 (SP-6)')
plt.ylabel('Característica 2 (FMF)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.show()