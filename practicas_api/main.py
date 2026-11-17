import requests
import pandas as pd

# 1. URL SIN la palabra "oportuno" para traer todo el historial
url = "https://www.banxico.org.mx/SieAPIRest/service/v1/series/SP68257/datos"

# 2. Tu Token (Asegúrate de poner el tuyo)
headers = {"Bmx-Token": "50167a4ba666fc5a033a851da1c81bbed6a4afebd7056973ceac274549226cb7"}

# 3. Petición a la API
respuesta = requests.get(url, headers=headers)

if respuesta.status_code == 200:
    datos_json = respuesta.json()
    
    # 4. Extraemos la lista de datos
    lista_datos = datos_json['bmx']['series'][0]['datos']
    
    # 5. Convertimos a una "tablota" de Pandas
    df_banxico = pd.DataFrame(lista_datos)
    
    print(f"¡Éxito! Se descargó una tabla con {len(df_banxico)} filas desde la API.\n")
    
    print("--- PRIMERAS 5 FILAS ---")
    print(df_banxico.head())
    
    print("\n--- ÚLTIMAS 5 FILAS ---")
    print(df_banxico.tail())

    # NUEVA LÍNEA: Exportar la "tablota" a un archivo CSV real
    df_banxico.to_csv("historial_banxico.csv", index=False)
    print("\n¡Archivo 'historial_banxico.csv' guardado con éxito en tu carpeta!")
    
else:
    print(f"Error en la conexión. Código: {respuesta.status_code}")