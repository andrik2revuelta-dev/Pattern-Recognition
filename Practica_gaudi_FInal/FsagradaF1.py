import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

carpeta = 'muestras_futuras'
archivos = os.listdir(carpeta)
num_muestras = len(archivos)
medias = np.zeros((num_muestras, 3))
covarianzas = np.zeros((num_muestras, 3, 3))
for i, nombre in enumerate(archivos):
    pixeles = plt.imread(os.path.join(carpeta, nombre)).reshape(-1, 3)
    medias[i] = np.mean(pixeles, axis=0)
    covarianzas[i] = np.cov(pixeles, rowvar=False)

img_test = plt.imread('img_original/sagrada-familia.jpg')
alto, ancho, canales = img_test.shape
pixeles_test = img_test.reshape(-1, 3)
probabilidades = np.zeros((num_muestras, pixeles_test.shape[0]))

for i in range(num_muestras):
    probabilidades[i] = multivariate_normal.pdf(pixeles_test, mean=medias[i], cov=covarianzas[i], allow_singular=True)
clase_ganadora = np.argmax(probabilidades, axis=0)
img_final = medias[clase_ganadora].reshape(alto, ancho, canales).astype(np.uint8)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(img_test); ax1.axis('off'); ax2.imshow(img_final); ax2.axis('off')
plt.show()