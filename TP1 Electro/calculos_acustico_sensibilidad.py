from funciones_electro import importar_datos_smaart, graficar_db, comparar_db, promedio_logaritmico_butterworth, escalar_curva, encontrar_desviacion
from funciones_electro import detectar_resonancia, importar_datos_limp
import numpy as np


# 1. Definimos la frecuencia de resonancia como variable base
frecuencias_sinmasa, magnitud_sinmasa, fase_sinmasa = importar_datos_limp(r"Electrico\SIN MASA.csv")
fs, mag_fs = detectar_resonancia(frecuencias_sinmasa, magnitud_sinmasa)

# 2. Importar y corregir la medición física de la banda, obtiene su promedio en la zona de interes.
frecs_filtrado, spl_filtrado = importar_datos_smaart(r"Acústico\sensibilidad y rta en fcia tercios\fs a 10fs tercios.txt")
spl_filtrado_corregido = np.array(spl_filtrado) - 6
valor_promedio = promedio_logaritmico_butterworth(frecs_filtrado, spl_filtrado_corregido, fs, 10.0 * fs)
# ----------------------------------

# 3. Importar la curva completa (relativa), obtiene su promedio en la zona de interes.
frecs_completo, spl_completo = importar_datos_smaart(r"Acústico\sensibilidad y rta en fcia tercios\sin filtrar tercios.txt")
valor_promedio_cc = promedio_logaritmico_butterworth(frecs_completo, spl_completo, fs, 10.0 * fs)

# 4. Escalar la curva completa.
spl_escalado, offset = escalar_curva(
    frecuencias=frecs_completo, 
    spl_curva_completa=spl_completo, 
    f_s=fs, 
    spl_curva_referencia=spl_filtrado_corregido
)

fmax_desvio, splmax_desvio, fmin_desvio, splmin_desvio, span_desvio = encontrar_desviacion(frecs_completo, spl_escalado, fs, 10*fs)

# --- REPORTES Y PRINTS ---
print(f"Frecuencia de Resonancia detectada (fs): {fs:.2f} Hz")
print(f"El valor de sensibilidad es (el valor promedio de la curva fs-10fs): {valor_promedio:.2f} dB")
print(f"El valor promedio de la curva completa (pero filtrada fs-10fs) era: {valor_promedio_cc:.2f} dB")
print(f"Offset calculado para el escalado (se le resta punto a punto a la curva completa): {offset:.2f} dB")
print(f"El valor maximo en el rango cae en: ({fmax_desvio:.2f} ; {splmax_desvio:.2f})")
print(f"El valor minimo en el rango cae en: ({fmin_desvio:.2f} ; {splmin_desvio:.2f})")

# 5. Graficar la comparación final
comparar_db(
    frecs_completo, 
    [spl_filtrado_corregido, spl_completo], 
    labels=['Señal filtrada de fs a 10*fs (con corrección -6 dB)', 'Señal sin filtrar'],
    title='Comparación de respuestas en frecuencia para el cálculo de offset',
    ylabel='dB SPL'
)

graficar_db(
    frecs_completo, 
    spl_escalado, 
    title='Sensibilidad (1 W - 1 m) del altoparlante', 
    ylabel='dB SPL'
)