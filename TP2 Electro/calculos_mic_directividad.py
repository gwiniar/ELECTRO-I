from funciones_electro_tp2 import importar_datos_smaart, suavizado, graficar_db, comparar_db, comparar_patron_polar_2, encontrar_desviacion
import numpy as np

# =============================================================================
# --- SHURE SM58 ---
# =============================================================================
# 1. Importa los datos completos para cada ángulo
frecs_0_sm58, spl_0_sm58 = importar_datos_smaart(r'SHURE\SM58 0g.txt')
frecs_15_sm58, spl_15_sm58 = importar_datos_smaart(r'SHURE\SM58 15g.txt')
frecs_30_sm58, spl_30_sm58 = importar_datos_smaart(r'SHURE\SM58 30g.txt')
frecs_45_sm58, spl_45_sm58 = importar_datos_smaart(r'SHURE\SM58 45g.txt')
frecs_60_sm58, spl_60_sm58 = importar_datos_smaart(r'SHURE\SM58 60g.txt')
frecs_75_sm58, spl_75_sm58 = importar_datos_smaart(r'SHURE\SM58 75g.txt')
frecs_90_sm58, spl_90_sm58 = importar_datos_smaart(r'SHURE\SM58 90g.txt')
frecs_105_sm58, spl_105_sm58 = importar_datos_smaart(r'SHURE\SM58 105g.txt')
frecs_120_sm58, spl_120_sm58 = importar_datos_smaart(r'SHURE\SM58 120g.txt')
frecs_135_sm58, spl_135_sm58 = importar_datos_smaart(r'SHURE\SM58 135g.txt')
frecs_150_sm58, spl_150_sm58 = importar_datos_smaart(r'SHURE\SM58 150g.txt')
frecs_165_sm58, spl_165_sm58 = importar_datos_smaart(r'SHURE\SM58 165g.txt')
frecs_180_sm58, spl_180_sm58 = importar_datos_smaart(r'SHURE\SM58 180g.txt')

# 2. Agrupar las mediciones originales en listas
lista_curvas_db_orig_sm58 = [spl_0_sm58, spl_15_sm58, spl_30_sm58, spl_45_sm58, spl_60_sm58, spl_75_sm58, spl_90_sm58, spl_105_sm58, spl_120_sm58, spl_135_sm58, spl_150_sm58, spl_165_sm58, spl_180_sm58]
lista_frecs_orig_sm58 = [frecs_0_sm58, frecs_15_sm58, frecs_30_sm58, frecs_45_sm58, frecs_60_sm58, frecs_75_sm58, frecs_90_sm58, frecs_105_sm58, frecs_120_sm58, frecs_135_sm58, frecs_150_sm58, frecs_165_sm58, frecs_180_sm58]

# --- RECORTADO AUTOMÁTICO ---
largo_minimo_sm58 = min(len(f) for f in lista_frecs_orig_sm58)
lista_frecs_sm58 = [f[:largo_minimo_sm58] for f in lista_frecs_orig_sm58]
lista_curvas_db_sm58 = [db[:largo_minimo_sm58] for db in lista_curvas_db_orig_sm58]

# 3. Se suavizan por 1/3 de octava las curvas ya recortadas
lista_curvas_suavizadas_sm58 = []
for frec_curva, spl_curva in zip(lista_frecs_sm58, lista_curvas_db_sm58):
    spl_suavizado = suavizado(frec_curva, spl_curva, 3)
    lista_curvas_suavizadas_sm58.append(spl_suavizado)
    
# 4. Definir los parámetros globales compartidos para ambos micrófonos
angulos_medidos = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
angulos_labels = ['0°', '15°', '30°', '45°', '60°', '75°', '90°', '105°', '120°', '135°', '150°', '165°', '180°']
frecuencias_a_comparar = [125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    
# 5. Graficar respuesta en frecuencia a los 0g.
graficar_db(lista_frecs_sm58[0], lista_curvas_suavizadas_sm58[0], title='Respuesta en Frecuencia SHURE SM58')
f_max_sm58, v_max_sm58, f_min_sm58, v_min_sm58, span_sm58 = encontrar_desviacion(lista_frecs_sm58[0], lista_curvas_suavizadas_sm58[0], 100, 1500)
print('--- SHURE SM58 ---')
print(f'Minimo: ({f_min_sm58} Hz , {v_min_sm58} dB)')
print(f'Maximo: ({f_max_sm58} Hz , {v_max_sm58} dB)')
print(f'Span: {span_sm58}')


# 6. Graficar comparativa de respuestas en frecuencia
comparar_db(lista_frecs_sm58[0], lista_curvas_suavizadas_sm58, labels=angulos_labels,
            title="SHURE SM58 \n Valores de presion sonora para distintas angulaturas (Suavizado 1/3 Oct)", ylabel='dB SPL')

# 7. Graficar patrones polares
comparar_patron_polar_2(lista_frecs_sm58[0], lista_curvas_suavizadas_sm58, angulos_medidos, frecuencias_a_comparar,
                        title=' SHURE SM58 | Patrón Polar Comparativo \n(Bajas a la Derecha - Altas a la Izquierda)')


# =============================================================================
# --- RODE NT2000 ---
# =============================================================================
# 1. Importa los datos completos para cada ángulo
frecs_0_nt2000, spl_0_nt2000 = importar_datos_smaart(r'RODE\NT2000 0g.txt')
frecs_15_nt2000, spl_15_nt2000 = importar_datos_smaart(r'RODE\NT2000 15g.txt')
frecs_30_nt2000, spl_30_nt2000 = importar_datos_smaart(r'RODE\NT2000 30g.txt')
frecs_45_nt2000, spl_45_nt2000 = importar_datos_smaart(r'RODE\NT2000 45g.txt')
frecs_60_nt2000, spl_60_nt2000 = importar_datos_smaart(r'RODE\NT2000 60g.txt')
frecs_75_nt2000, spl_75_nt2000 = importar_datos_smaart(r'RODE\NT2000 75g.txt')
frecs_90_nt2000, spl_90_nt2000 = importar_datos_smaart(r'RODE\NT2000 90g.txt')
frecs_105_nt2000, spl_105_nt2000 = importar_datos_smaart(r'RODE\NT2000 105g.txt')
frecs_120_nt2000, spl_120_nt2000 = importar_datos_smaart(r'RODE\NT2000 120g.txt')
frecs_135_nt2000, spl_135_nt2000 = importar_datos_smaart(r'RODE\NT2000 135g.txt')
frecs_150_nt2000, spl_150_nt2000 = importar_datos_smaart(r'RODE\NT2000 150g.txt')
frecs_165_nt2000, spl_165_nt2000 = importar_datos_smaart(r'RODE\NT2000 165g.txt')
frecs_180_nt2000, spl_180_nt2000 = importar_datos_smaart(r'RODE\NT2000 180g.txt')

# 2. Agrupar las mediciones originales en listas
lista_curvas_db_orig_nt2000 = [spl_0_nt2000, spl_15_nt2000, spl_30_nt2000, spl_45_nt2000, spl_60_nt2000, spl_75_nt2000, spl_90_nt2000, spl_105_nt2000, spl_120_nt2000, spl_135_nt2000, spl_150_nt2000, spl_165_nt2000, spl_180_nt2000]
lista_frecs_orig_nt2000 = [frecs_0_nt2000, frecs_15_nt2000, frecs_30_nt2000, frecs_45_nt2000, frecs_60_nt2000, frecs_75_nt2000, frecs_90_nt2000, frecs_105_nt2000, frecs_120_nt2000, frecs_135_nt2000, frecs_150_nt2000, frecs_165_nt2000, frecs_180_nt2000]

# --- RECORTADO AUTOMÁTICO ---
largo_minimo_nt2000 = min(len(f) for f in lista_frecs_orig_nt2000)
lista_frecs_nt2000 = [f[:largo_minimo_nt2000] for f in lista_frecs_orig_nt2000]
lista_curvas_db_nt2000 = [db[:largo_minimo_nt2000] for db in lista_curvas_db_orig_nt2000]

# 3. Se suavizan por 1/3 de octava las curvas ya recortadas
lista_curvas_suavizadas_nt2000 = []
for frec_curva, spl_curva in zip(lista_frecs_nt2000, lista_curvas_db_nt2000):
    spl_suavizado = suavizado(frec_curva, spl_curva, 3)
    lista_curvas_suavizadas_nt2000.append(spl_suavizado)
    
# 5. Graficar respuesta en frecuencia a los 0g.
graficar_db(lista_frecs_nt2000[0], lista_curvas_suavizadas_nt2000[0], title='Respuesta en Frecuencia RODE NT2000')
f_max_nt2000, v_max_nt2000, f_min_nt2000, v_min_nt2000, span_nt2000 = encontrar_desviacion(lista_frecs_nt2000[0], lista_curvas_suavizadas_nt2000[0], 100, 1500)
print('--- RODE NT2000 ---')
print(f'Minimo: ({f_min_nt2000} Hz , {v_min_nt2000} dB)')
print(f'Maximo: ({f_max_nt2000} Hz , {v_max_nt2000} dB)')
print(f'Span: {span_nt2000} dB')


# 6. Graficar comparativa de respuestas en frecuencia
comparar_db(lista_frecs_nt2000[0], lista_curvas_suavizadas_nt2000, labels=angulos_labels,
            title="RODE NT2000 \n Valores de presion sonora para distintas angulaturas (Suavizado 1/3 Oct)", ylabel='dB SPL')

# 7. Graficar patrones polares
comparar_patron_polar_2(lista_frecs_nt2000[0], lista_curvas_suavizadas_nt2000, angulos_medidos, frecuencias_a_comparar,
                        title=' RODE NT2000 | Patrón Polar Comparativo \n(Bajas a la Derecha - Altas a la Izquierda)')
    