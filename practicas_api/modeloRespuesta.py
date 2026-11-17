import pandas as pd
# Importamos las herramientas de Machine Learning (Asegúrate de instalar: pip install scikit-learn)
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore') # Ocultar advertencias para que se vea limpio en terminal

print("Cargando el Dataset Maestro y entrenando al algoritmo (Reconocimiento de Patrones)...")

# 1. Cargar el dataset que limpiamos previamente
df = pd.read_csv("Dataset_Maestro_Proyecto.csv")

# 2. Separar las características (X) y lo que queremos predecir (Y)
# OJO: Quitamos la columna 'ingocup' porque sería hacer trampa decirle el sueldo al algoritmo
X = df[['eda', 'sex', 'anios_esc', 'hrsocup', 'anios_experiencia']]
Y = df['gana_mas_de_15k']

# 3. Entrenar el Modelo (Aquí ocurre la magia de los patrones)
modelo = DecisionTreeClassifier(max_depth=5, random_state=42)
modelo.fit(X, Y)

print("\n¡Algoritmo Entrenado con Éxito usando 115,430 registros!\n")

# =======================================================
# 4. LA RESPUESTA DEL PROGRAMA (PRUEBA EN VIVO)
# =======================================================
print("--- PRUEBA DE PREDICCIÓN EN TIEMPO REAL ---")
print("Ingresa los datos del trabajador para descubrir su patrón salarial:")

# Pedimos los datos al usuario desde la terminal
edad_input = int(input("1. Edad (ej. 28): "))
sexo_input = int(input("2. Sexo (1=Hombre, 2=Mujer): "))
esc_input = int(input("3. Años de estudio (ej. 16 para Universidad): "))
hrs_input = int(input("4. Horas trabajadas a la semana (ej. 48): "))
exp_input = int(input("5. Años de experiencia (ej. 5): "))

# Armamos el perfil
perfil_nuevo = [[edad_input, sexo_input, esc_input, hrs_input, exp_input]]

# EL PROGRAMA RESPONDE
prediccion = modelo.predict(perfil_nuevo)

print("\n" + "="*50)
if prediccion[0] == 1:
    print("🤖 RESPUESTA DEL ALGORITMO: \n¡ESTE PERFIL SÍ TIENE EL PATRÓN PARA GANAR MÁS DE $15,000 MXN!")
else:
    print("🤖 RESPUESTA DEL ALGORITMO: \nLamentablemente, este perfil SE QUEDARÁ POR DEBAJO de los $15,000 MXN.")
print("="*50 + "\n")