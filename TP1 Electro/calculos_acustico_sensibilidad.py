from funciones_electro import importar_datos_smaart, graficar_db, comparar_db, promedio_logaritmico_butterworth, escalar_curva
from funciones_electro import detectar_resonancia, importar_datos_limp
import numpy as np


# 1. Definimos la frecuencia de resonancia como variable base
frecuencias_sinmasa, magnitud_sinmasa, fase_sinmasa = importar_datos_limp(r"Electrico\SIN MASA.csv")
fs, mag_fs = detectar_resonancia(frecuencias_sinmasa, magnitud_sinmasa)

# 2. Importar y corregir la medición física de la banda
frecs_filtrado, spl_filtrado = importar_datos_smaart(r"Acústico\fs a 10fs.txt")
spl_filtrado_corregido = np.array(spl_filtrado) - 6

# --- CÁLCULO SEGURO Y COMPACTO ---
# Mandamos toda la curva y las frecuencias de corte, la función se encarga del resto
valor_promedio = promedio_logaritmico_butterworth(frecs_filtrado, spl_filtrado_corregido, fs, 10.0 * fs)
# ----------------------------------

# 3. Importar la curva completa (relativa)
frecs_completo, spl_completo = importar_datos_smaart(r'Acústico\sin filtrar.txt')

# 4. Escalar la curva completa
spl_escalado, offset = escalar_curva(frecs_completo, spl_completo, fs, valor_promedio)

print(f"Frecuencia de Resonancia detectada (fs): {fs:.2f} Hz")
print(f"Offset calculado para el escalado: {offset:+.2f} dB")

# 5. Graficar la comparación final
comparar_db(
    frecs_completo, 
    [spl_completo, spl_filtrado_corregido, spl_escalado], 
    labels=['sin filtrar original', '10 a 10fs -6dB', 'sin filtrar escalado'],
    ylabel='dB SPL'
)

graficar_db(frecs_completo, spl_escalado, title='sensibilidad')