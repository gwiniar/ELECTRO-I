from funciones_electro import importar_datos_smaart, comparar_db, graficar_patron_polar, comparar_patron_polar
import numpy as np

# 1. Importar los datos por tercios de octava para cada ángulo
frecs_0, spl_0 = importar_datos_smaart(r"Acústico\polar x tercios\0° tercios.txt")
frecs_15, spl_15 = importar_datos_smaart(r"Acústico\polar x tercios\15° tercios.txt")
frecs_30, spl_30 = importar_datos_smaart(r"Acústico\polar x tercios\30° tercios.txt")
frecs_45, spl_45 = importar_datos_smaart(r"Acústico\polar x tercios\45° tercios.txt")
frecs_60, spl_60 = importar_datos_smaart(r"Acústico\polar x tercios\60° tercios.txt")
frecs_75, spl_75 = importar_datos_smaart(r"Acústico\polar x tercios\75° tercios.txt")
frecs_90, spl_90 = importar_datos_smaart(r"Acústico\polar x tercios\90° tercios.txt")

# 2. Agrupar las mediciones de SPL en una lista ordenada
lista_curvas_db = [spl_0, spl_15, spl_30, spl_45, spl_60, spl_75, spl_90]

# 3. Definir los ángulos correspondientes a cada archivo de la lista
angulos_medidos = [0, 15, 30, 45, 60, 75, 90]
angulos_labels = ['0°', '15°', '30°', '45°', '60°', '75°', '90°']

# 4. Definir la frecuencia central del tercio de octava a analizar o la lista de frecuencias
frecuencia_objetivo = 1000
frecuencias_a_comparar = [125, 250, 500, 1000, 2000, 4000, 8000, 16000]

# 5. Ejecutar el graficador polar por índice más cercano
graficar_patron_polar(
    frecuencias=frecs_0,          # Usamos el vector de frecs de 0° como base compartida
    lista_db=lista_curvas_db, 
    angulos=angulos_medidos, 
    f_centro=frecuencia_objetivo,
    title="Directividad para el tercio de octava con centro 1000 Hz")

comparar_patron_polar(
    frecuencias=frecs_0, 
    lista_db=lista_curvas_db, 
    angulos=angulos_medidos, 
    lista_f_centro=frecuencias_a_comparar,
    title="Directividad para distintas frecuencias")

comparar_db(frecs_0, lista_curvas_db, labels=angulos_labels,
            title="Valores de presion sonora para distintas angulaturas", ylabel='dB SPL')
