from funciones_electro import importar_datos_smaart, graficar_db, comparar_db, promedio_logaritmico_db, escalar_curva
from funciones_electro import detectar_resonancia
import numpy as np

# 1. Definimos la frecuencia de resonancia como variable base
f_s = 87.1 

# 2. Importar y corregir la medición física de la banda
frecs_filtrado, spl_filtrado = importar_datos_smaart(r"Acústico\fs a 10fs.txt")
spl_filtrado_corregido = np.array(spl_filtrado) - 6

# --- CORRECCIÓN CRUCIAL ---
# Creamos una máscara para que el promedio se calcule SÓLO en la banda útil (87.1 Hz a 871 Hz)
mascara_banda = (frecs_filtrado >= f_s) & (frecs_filtrado <= 10.0 * f_s)
valor_promedio = promedio_logaritmico_db(spl_filtrado_corregido[mascara_banda])
# ---------------------------

# 3. Importar la curva completa (relativa)
frecs_completo, spl_completo = importar_datos_smaart(r'Acústico\sin filtrar.txt')

# 4. Escalar la curva completa usando el valor promedio real (bien calculado)
spl_escalado, offset = escalar_curva(frecs_completo, spl_completo, f_s, valor_promedio)

print(f"Offset calculado para el escalado: {offset:+.2f} dB")

# 5. Graficar la comparación final
comparar_db(
    frecs_completo, 
    [spl_completo, spl_filtrado_corregido, spl_escalado], 
    labels=['sin filtrar original', '10 a 10fs -6dB', 'sin filtrar escalado'],
    ylabel='dB SPL'
)