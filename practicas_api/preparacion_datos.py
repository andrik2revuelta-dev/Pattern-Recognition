import pandas as pd
import numpy as np

print("Iniciando procesamiento de datos. Esto puede tomar unos segundos...")

# --------------------------------------------------------
# FASE 1: CARGA Y SELECCIÓN DE CARACTERÍSTICAS
# --------------------------------------------------------
# Cargamos el archivo gigante de la ENOE (Ajusta el nombre si es diferente)
# Usamos low_memory=False porque el archivo tiene muchísimas columnas
df_bruto = pd.read_csv("ENOE_SDEMT226.csv", low_memory=False, encoding='latin-1')

# Seleccionamos estrictamente las columnas que importan para el modelo
columnas_necesarias = ['eda', 'sex', 'anios_esc', 'hrsocup', 'ingocup', 'cvegeo']
df_limpio = df_bruto[columnas_necesarias].copy()

# --------------------------------------------------------
# FASE 2: FILTRADO DEMOGRÁFICO Y LIMPIEZA
# --------------------------------------------------------
# Forzamos que los datos sean numéricos (por si el CSV trae espacios o letras ocultas)
for col in df_limpio.columns:
    df_limpio[col] = pd.to_numeric(df_limpio[col], errors='coerce')

# Filtramos nuestra demografía objetivo: Personas de 18 a 65 años que SÍ tienen ingresos
df_limpio = df_limpio[(df_limpio['eda'] >= 18) & (df_limpio['eda'] <= 65)]
df_limpio = df_limpio[df_limpio['ingocup'] > 0]
df_limpio = df_limpio.dropna() # Eliminamos cualquier fila con datos vacíos

# --------------------------------------------------------
# FASE 3: FEATURE ENGINEERING (Lógica de Negocios)
# --------------------------------------------------------
# 1. Variable Objetivo (Target): 1 si gana más de 15k, 0 si no.
df_limpio['gana_mas_de_15k'] = np.where(df_limpio['ingocup'] > 15000, 1, 0)

# 2. Variable Sintética: Años de Experiencia (Ecuación de Mincer)
# Fórmula: Edad - Años de escolaridad - 6 (edad promedio en que se inicia la primaria)
df_limpio['anios_experiencia'] = df_limpio['eda'] - df_limpio['anios_esc'] - 6
# Evitamos experiencias negativas (gente que estudió mucho y aún no trabaja)
df_limpio['anios_experiencia'] = df_limpio['anios_experiencia'].clip(lower=0)

# --------------------------------------------------------
# FASE 4: FUSIÓN DE DATOS (JOIN CON EL SCRAPER)
# --------------------------------------------------------
# La ENOE usa números del 1 al 32 para los estados. 
# Creamos un diccionario rápido para que coincida con los nombres de tu Scraper
mapa_estados = {
    1: 'Aguascalientes', 2: 'Baja California', 3: 'Baja California Sur', 4: 'Campeche', 
    5: 'Coahuila', 6: 'Colima', 7: 'Chiapas', 8: 'Chihuahua', 9: 'Ciudad de México', 
    10: 'Durango', 11: 'Guanajuato', 12: 'Guerrero', 13: 'Hidalgo', 14: 'Jalisco', 
    15: 'México', 16: 'Michoacán', 17: 'Morelos', 18: 'Nayarit', 19: 'Nuevo León', 
    20: 'Oaxaca', 21: 'Puebla', 22: 'Querétaro', 23: 'Quintana Roo', 24: 'San Luis Potosí', 
    25: 'Sinaloa', 26: 'Sonora', 27: 'Tabasco', 28: 'Tamaulipas', 29: 'Tlaxcala', 
    30: 'Veracruz', 31: 'Yucatán', 32: 'Zacatecas'
}
# Dividimos entre 1000 para quitar el municipio y quedarnos solo con la clave del Estado
df_limpio['clave_estado_real'] = df_limpio['cvegeo'] // 1000
df_limpio['nombre_estado'] = df_limpio['clave_estado_real'].map(mapa_estados)

# Cargamos el CSV que generó tu Web Scraper
try:
    df_pib = pd.read_csv("pib_por_estado.csv")
    
    # Hacemos el cruce (MERGE) uniendo ambos datasets por el nombre del estado
    # Usamos la primera columna del PIB (que es el nombre) y nuestra nueva columna
    nombre_col_estado_pib = df_pib.columns[0]
    df_final = pd.merge(df_limpio, df_pib, left_on='nombre_estado', right_on=nombre_col_estado_pib, how='left')
    
    # Eliminamos columnas redundantes para dejar el dataset perfecto
    columnas_finales = ['eda', 'sex', 'anios_esc', 'hrsocup', 'anios_experiencia', 'nombre_estado', 'ingocup', 'gana_mas_de_15k']
    df_final = df_final[columnas_finales]
    
except FileNotFoundError:
    print("Advertencia: No se encontró 'pib_por_estado.csv'. Guardando dataset solo con datos de la ENOE.")
    df_final = df_limpio

# --------------------------------------------------------
# FASE 5: EXPORTACIÓN DEL DATASET MAESTRO
# --------------------------------------------------------
df_final.to_csv("Dataset_Maestro_Proyecto.csv", index=False)
print(f"\n¡Éxito! Dataset Maestro creado con {len(df_final)} registros.")
print("\n--- VISTAZO AL DATASET FINAL ---")
print(df_final.head())