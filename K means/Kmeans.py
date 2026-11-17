import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

img = plt.imread('img_original/sagrada-familia.jpg') 
alto, ancho, canales = img.shape
pixeles = img.reshape(-1, 3)

modelo_kmeans = KMeans(n_clusters=30, init='random', n_init=10, random_state=42)

print("Calculando 30 centroides... esto puede tardar un poco por la cantidad de matemáticas.")

etiquetas = modelo_kmeans.fit_predict(pixeles)
centroides_finales = modelo_kmeans.cluster_centers_

img_recoloreada_plana = centroides_finales[etiquetas]
img_final = img_recoloreada_plana.reshape(alto, ancho, canales).astype(np.uint8)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(img)
ax1.set_title('Imagen Original (Millones de colores)')
ax1.axis('off')
ax2.imshow(img_final)
ax2.set_title('Imagen Recoloreada (Solo 30 Colores)')
ax2.axis('off')

plt.tight_layout()
plt.show()
print("¡Recoloreado finalizado con éxito!")

print("Generando gráfica 3D en el espacio RGB...")

fig_3d = plt.figure(figsize=(10, 8))
ax3d = fig_3d.add_subplot(111, projection='3d')

num_puntos_3d = 5000
indices_muestra = np.random.choice(pixeles.shape[0], num_puntos_3d, replace=False)

pixeles_muestra = pixeles[indices_muestra]
etiquetas_muestra = etiquetas[indices_muestra]

colores_muestra = centroides_finales[etiquetas_muestra] / 255.0

ax3d.scatter(pixeles_muestra[:, 0], pixeles_muestra[:, 1], pixeles_muestra[:, 2], 
             c=colores_muestra, s=15, alpha=0.5, edgecolors='none')
ax3d.scatter(centroides_finales[:, 0], centroides_finales[:, 1], centroides_finales[:, 2], 
             c='red', marker='X', s=200, edgecolors='black', label='Centroides (Los 30 Colores)')
ax3d.set_title('Espacio de Características 3D (Clustering K-Means)')
ax3d.set_xlabel('Canal Rojo (R)')
ax3d.set_ylabel('Canal Verde (G)')
ax3d.set_zlabel('Canal Azul (B)')
plt.legend()
plt.show()