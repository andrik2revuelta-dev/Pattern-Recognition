import pandas as pd
import requests
from io import StringIO

print("Iniciando Scraper... extrayendo datos de Wikipedia...")

url = "https://es.wikipedia.org/wiki/Anexo:Entidades_federativas_de_M%C3%A9xico_por_PIB"
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

# 1. Descargamos la página web como si fuéramos un navegador
respuesta = requests.get(url, headers=headers)

# 2. Convertimos el texto a un formato que Pandas pueda leer perfectamente (StringIO)
# y le pedimos que busque la tabla que contenga la palabra "Entidad"
tablas_html = pd.read_html(StringIO(respuesta.text), match="Entidad")

# 3. Tomamos la tabla limpia
df_pib = tablas_html[0]

# 4. Imprimimos SOLO la tabla bonita (sin basura HTML)
print("\n--- ¡TABLA EXTRAÍDA CON ÉXITO! ---")
print(df_pib.head())

# 5. Guardamos el archivo CSV para tu proyecto
df_pib.to_csv("pib_por_estado.csv", index=False)
print("\nEl archivo 'pib_por_estado.csv' ya está guardado y listo.")