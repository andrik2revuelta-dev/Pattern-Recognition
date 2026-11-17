import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

print("=========================================================")
print("🤖 INICIANDO PIPELINE DE MACHINE LEARNING END-TO-END 🤖")
print("=========================================================\n")

print("Paso 1: Extrayendo y limpiando microdatos de la ENOE...")
df_bruto = pd.read_csv("../enoe_2026_trim2_csv/ENOE_SDEMT226.csv", low_memory=False, encoding='latin-1')

# 8 Atributos seguros
cols = ['eda', 'sex', 'anios_esc', 'hrsocup', 'ingocup', 'cvegeo', 'pos_ocu', 'seg_soc']
df = df_bruto[cols].copy()

# Limpieza inicial
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Filtrado de PEA (18 a 65 años)
df = df[(df['eda'] >= 18) & (df['eda'] <= 65) & (df['ingocup'] > 0)]

print("Paso 2: Generando Etiqueta y Feature Engineering...")
df['gana_mas_de_15k'] = np.where(df['ingocup'] > 15000, 1, 0)
df['anios_experiencia'] = (df['eda'] - df['anios_esc'] - 6).clip(lower=0)

print("Paso 3: Cruzando datos macroeconómicos (Scraper PIB)...")
mapa_estados = {1: 'Aguascalientes', 2: 'Baja California', 3: 'Baja California Sur', 4: 'Campeche', 5: 'Coahuila', 6: 'Colima', 7: 'Chiapas', 8: 'Chihuahua', 9: 'Ciudad de México', 10: 'Durango', 11: 'Guanajuato', 12: 'Guerrero', 13: 'Hidalgo', 14: 'Jalisco', 15: 'México', 16: 'Michoacán', 17: 'Morelos', 18: 'Nayarit', 19: 'Nuevo León', 20: 'Oaxaca', 21: 'Puebla', 22: 'Querétaro', 23: 'Quintana Roo', 24: 'San Luis Potosí', 25: 'Sinaloa', 26: 'Sonora', 27: 'Tabasco', 28: 'Tamaulipas', 29: 'Tlaxcala', 30: 'Veracruz', 31: 'Yucatán', 32: 'Zacatecas'}

# Forzamos que sea número entero para un cruce perfecto
df['clave_estado'] = (df['cvegeo'] // 1000).astype(int)
df['nombre_estado'] = df['clave_estado'].map(mapa_estados)

# Búsqueda Inteligente en el CSV de Wikipedia
df_pib = pd.read_csv("pib_por_estado.csv")
col_entidad = [c for c in df_pib.columns if 'ntidad' in c][0] 
col_pib_raw = [c for c in df_pib.columns if 'PIB' in c][0]

# Cruce
df = pd.merge(df, df_pib, left_on='nombre_estado', right_on=col_entidad, how='inner')

# Extraemos solo los números del PIB ignorando comas o signos
df['pib_estatal'] = pd.to_numeric(df[col_pib_raw].astype(str).str.replace(r'\D', '', regex=True), errors='coerce')
df['pib_estatal'] = df['pib_estatal'].fillna(df['pib_estatal'].median())

# =========================================================
# DEFINICIÓN FINAL DE LOS 8 ATRIBUTOS
# =========================================================
atributos_x = ['eda', 'sex', 'anios_esc', 'hrsocup', 'anios_experiencia', 'pos_ocu', 'seg_soc', 'pib_estatal']
etiqueta_y = 'gana_mas_de_15k'

df_final = df[atributos_x + [etiqueta_y]].dropna()
print(f"-> ¡Dataset Maestro completado con {len(df_final)} muestras y {len(atributos_x)} atributos!\n")

# =========================================================
# GUARDADO FÍSICO DEL ARCHIVO (¡EL AJUSTE CLAVE!)
# =========================================================
print("Guardando el Dataset_Maestro_Proyecto.csv actualizado en tu disco duro...")
df_final.to_csv("Dataset_Maestro_Proyecto.csv", index=False)
print("¡Archivo CSV guardado con éxito! Ya puedes abrirlo para ver tus 8 columnas.\n")


#print("Paso 4: Entrenando el algoritmo para responder a la pregunta principal...")
#X = df_final[atributos_x]
#y = df_final[etiqueta_y]

#modelo = DecisionTreeClassifier(max_depth=5, random_state=42)
#modelo.fit(X, y)

#print("\n=========================================================")
#print("🎯 RESPUESTA DEL PROGRAMA A LA PREGUNTA DE INVESTIGACIÓN")
#print("=========================================================")
#print("El Algoritmo dictamina que estos atributos determinan si se superan los $15k:\n")

#importancias = pd.DataFrame({'Atributo': atributos_x, 'Impacto (%)': modelo.feature_importances_ * 100})
#print(importancias.sort_values(by='Impacto (%)', ascending=False).to_string(index=False))
#print("=========================================================\n")