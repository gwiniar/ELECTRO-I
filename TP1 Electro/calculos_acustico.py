from funciones_electro import importar_datos_smaart, graficar_db, comparar_db
from funciones_electro import graficar_patron_polar, comparar_patron_polar

# 1. Importás los datos (asumiendo que todos comparten el mismo vector de frecuencias)
frecs_0, spl_0 = importar_datos_smaart(r"Acústico\0° s filtrar.txt")
frecs_15, spl_15 = importar_datos_smaart(r"Acústico\15° s filtrar.txt")
frecs_30, spl_30 = importar_datos_smaart(r"Acústico\30° s filtrar.txt")
frecs_45, spl_45 = importar_datos_smaart(r"Acústico\45° s filtrar.txt")
frecs_60, spl_60 = importar_datos_smaart(r"Acústico\60° (sin filtrar).txt")
frecs_75, spl_75 = importar_datos_smaart(r"Acústico\75° (sin filtrar).txt")
frecs_90, spl_90 = importar_datos_smaart(r"Acústico\90° (sin filtrar).txt")

# 2. Armás tus listas personalizadas de datos y etiquetas
curvas_pp = [spl_0, spl_15, spl_30, spl_45, spl_60, spl_75, spl_90]
etiquetas_pp = ["0°", "15°", "30°", "45°", "60°", "75°", "90°"]
angulos_pp = [0, 15, 30, 45, 60, 75, 90]
bandas_octava = [500, 1000, 2000, 4000, 8000]

# 3. Llamás a la función de comparación

comparar_db(frecs_0, curvas_pp, labels=etiquetas_pp,
            ylabel='dB SPL')
'''
graficar_patron_polar(
    frecuencias=frecs_0, lista_db=curvas_pp, angulos=angulos_pp, f_centro=1000,
    title="Patrón Polar - Octava de 1kHz")

graficar_patron_polar(
    frecuencias=frecs_0, lista_db=curvas_pp, angulos=angulos_pp, f_centro=200,
    title="Patrón Polar - Octava de 200 Hz")

graficar_patron_polar(
    frecuencias=frecs_0, lista_db=curvas_pp, angulos=angulos_pp, f_centro=5000,
    title="Patrón Polar - Octava de 5kHz")

graficar_patron_polar(
    frecuencias=frecs_0, lista_db=curvas_pp, angulos=angulos_pp, f_centro=8000,
    title="Patrón Polar - Octava de 8kHz")

frecs_sens, db_sens = importar_datos_smaart(r"Acústico\fs a 10fs.txt")
graficar_db(frecs_sens, db_sens)
'''

comparar_patron_polar(frecuencias=frecs_0, lista_db=curvas_pp, angulos=angulos_pp, lista_f_centro=bandas_octava, 
                      title="Evolución de Directividad por Octavas")