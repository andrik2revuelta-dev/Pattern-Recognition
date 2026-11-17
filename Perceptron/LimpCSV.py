import csv
import numpy as np
import matplotlib.pyplot as plt

archivo_path = 'SIRTUIN6.csv'

with open(archivo_path, 'r', newline='') as archivo:
    lector = csv.reader(archivo, delimiter=',')
    encabezados = next(lector)
    datos = list(lector)

datos_np = np.array(datos)
columna1 = datos_np[:, 1].astype(float)
columna2 = datos_np[:, 5].astype(float)
etiquetas = datos_np[:, 6].astype(int)

colores = np.where(etiquetas == 1, '#fa5f49', '#84b6f4')

fig, ax = plt.subplots()
scatter = ax.scatter(columna1, columna2, c=colores, picker=True)
indices_eliminados = set()
datos_actuales = [encabezados] + datos

def on_pick(event):
    if event.artist != scatter: return
    ind = event.ind[0]
    if ind in indices_eliminados: return
    
    colores_actuales = scatter.get_facecolor()
    colores_actuales[ind] = [0.5, 0.5, 0.5, 1]
    scatter.set_facecolor(colores_actuales)
    indices_eliminados.add(ind)
    fig.canvas.draw()

def on_key(event):
    global datos_actuales, scatter
    if event.key == 'enter' and indices_eliminados:
        nuevos_datos = [fila for i, fila in enumerate(datos_actuales) if i == 0 or (i-1) not in indices_eliminados]
        
        with open(archivo_path, 'w', newline='') as archivo:
            csv.writer(archivo).writerows(nuevos_datos)

        datos_actuales = nuevos_datos
        datos_np_nuevos = np.array(nuevos_datos[1:])
        
        nueva_c1 = datos_np_nuevos[:, 1].astype(float)
        nueva_c2 = datos_np_nuevos[:, 5].astype(float)
        nuevas_etiquetas = datos_np_nuevos[:, 6].astype(int)
        nuevos_colores = np.where(nuevas_etiquetas == 1, '#fa5f49', '#84b6f4')

        scatter.set_offsets(np.column_stack((nueva_c1, nueva_c2)))
        scatter.set_facecolor(nuevos_colores)
        
        print(f"Se eliminaron {len(indices_eliminados)} filas. CSV limpio.")
        indices_eliminados.clear()
        fig.canvas.draw()

fig.canvas.mpl_connect('pick_event', on_pick)
fig.canvas.mpl_connect('key_press_event', on_key)
plt.title("LimpCSV: Clic para grises | Enter para limpiar CSV")
plt.show()