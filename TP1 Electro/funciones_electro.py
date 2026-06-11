import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- FUNCIONES DE IMPORTACION DE DATOS ---
def importar_datos_limp(archivo_path):
    """
    Lee el archivo CSV de LIMP, limpia el formato regional de decimales,
    ignora metadatos y devuelve 3 vectores numéricos limpios.
    """
    # Usamos decimal=',' por si LIMP exportó con comas europeas/latinas.
    df = pd.read_csv(archivo_path, comment='*', header=None, names=['freq', 'mag', 'phase'], decimal=',')
    
    # Forzamos la conversión a numérico por si quedó algún texto huérfano (los errores se vuelven NaN y se limpian)
    df['freq'] = pd.to_numeric(df['freq'], errors='coerce')
    df['mag'] = pd.to_numeric(df['mag'], errors='coerce')
    df['phase'] = pd.to_numeric(df['phase'], errors='coerce')
    
    # Eliminamos filas vacías o con errores si las hubiera
    df = df.dropna()
    
    # Convertimos a vectores numéricos puros
    frecuencias = df['freq'].to_numpy()
    magnitud_imp = df['mag'].to_numpy()
    fase_imp = df['phase'].to_numpy()
    
    return frecuencias, magnitud_imp, fase_imp


def importar_datos_smaart(archivo_path):
    """
    Lee un archivo TXT exportado de Smaart y extrae la información 
    en 2 vectores: frecuencias y dbspl.
    """
    # Le asignamos nombres temporales del 0 al 5 (6 columnas potenciales) 
    # para que no se trabe si la línea de títulos tiene menos elementos que los datos.
    df = pd.read_csv(archivo_path, comment=';', header=None, sep=r'\s+', decimal=',', names=range(6))
    
    # Smaart siempre exporta la Frecuencia en la columna 0 y los dB en la 1
    df[0] = pd.to_numeric(df[0], errors='coerce')
    df[1] = pd.to_numeric(df[1], errors='coerce')
    
    # IMPORTANTE: Limpiamos filas vacías fijándonos SOLO en las columnas 0 y 1.
    # Así, si el archivo tiene 2 o 4 columnas en total, no altera el filtrado.
    df = df.dropna(subset=[0, 1])
    
    # Extraemos los dos vectores puros
    frecuencias = df[0].to_numpy()
    db = df[1].to_numpy()
    
    return frecuencias, db

# --- FUNCIONES DE GRAFICACION ELECTRICA ---
def detectar_resonancia(frecuencias, magnitudes):
    mascara_graves = frecuencias < 300
    f_res = None
    if any(mascara_graves):
        idx_max = magnitudes[mascara_graves].argmax()
        f_res = frecuencias[mascara_graves][idx_max]
        mag_res = magnitudes[mascara_graves][idx_max]
        
    return f_res, mag_res
        
def graficar_ohm(frecuencias, magnitudes, title="Respuesta de Magnitud (Impedancia)"):
    """
    Grafica la frecuencia vs Impedancia en escala logarítmica.
    - Muestra los ticks típicos de audio de 20 a 20kHz.
    - Eje Y limitado a un máximo de 40 Ohms con saltos de a 4.
    - Marca en negro y sin puntos (s=0): Inicio, Resonancia y Zmin.
    """
    plt.figure(figsize=(11, 6))
    
    # Graficado de la curva de impedancia
    plt.semilogx(frecuencias, magnitudes, linewidth=2.5, color='darkblue', zorder=3)
    
    # --- CONFIGURACIÓN DEL EJE X (Frecuencias de Audio) ---
    ticks_audio = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    labels_audio = ['20', '50', '100', '200', '500', '1k', '2k', '5k', '10k', '20k']
    plt.xlim(20, 20000)
    plt.xticks(ticks_audio, labels_audio)
    plt.xlabel('Frecuencia (Hz)', fontsize=11)
    
    # --- CONFIGURACIÓN DEL EJE Y (Máximo 40 Ohms, saltos de a 4) ---
    limite_y = 40
    ticks_y = list(range(0, limite_y + 4, 4))
    labels_y = [f'{val} $\\Omega$' for val in ticks_y]
    
    plt.ylim(0, limite_y) # Forzamos el corte estricto en 40 Ohms
    plt.yticks(ticks_y, labels_y)
    plt.ylabel('Impedancia (|Z|)', fontsize=11)
    
    # --- Re (INICIO DE CURVA) ---
    f_start = frecuencias[0]
    mag_start = magnitudes[0]
    if mag_start <= limite_y:
        plt.scatter(f_start, mag_start, color='k', s=0, zorder=4)
        plt.annotate(f'Re = {mag_start:.1f} $\\Omega$', 
                     xy=(f_start, mag_start), 
                     xytext=(20, -20), textcoords='offset points',
                     fontsize=9, fontweight='bold', color='k',
                     ha='left', va='bottom',
                     arrowprops=dict(arrowstyle="->", color='k', lw=1.2))
        
    # --- DETECTAR Y MARCAR LA RESONANCIA ---
    mascara_graves = frecuencias < 300
    f_res = None
    if any(mascara_graves):
        idx_max = magnitudes[mascara_graves].argmax()
        f_res = frecuencias[mascara_graves][idx_max]
        mag_res = magnitudes[mascara_graves][idx_max]
        
        if mag_res <= limite_y:
            plt.scatter(f_res, mag_res, color='k', s=0, zorder=4)
            # xytext=(0, 40) tira la flecha recto desde arriba para no pisar las laderas
            plt.annotate(f'Resonancia\n({f_res:.1f} Hz, {mag_res:.1f} $\\Omega$)', 
                         xy=(f_res, mag_res), 
                         xytext=(0, 40), textcoords='offset points',
                         fontsize=10, fontweight='bold', color='k',
                         ha='center', va='bottom',
                         arrowprops=dict(arrowstyle="->", color='k', lw=1.2))
            
    # --- REQUISITO B: VALOR DE MÍNIMA IMPEDANCIA (Zmin) DESPUÉS DE FS ---
    if f_res is not None:
        mascara_post_fs = frecuencias > f_res
        if any(mascara_post_fs):
            idx_min = magnitudes[mascara_post_fs].argmin()
            f_zmin = frecuencias[mascara_post_fs][idx_min]
            mag_zmin = magnitudes[mascara_post_fs][idx_min]
            
            if mag_zmin <= limite_y:
                plt.scatter(f_zmin, mag_zmin, color='k', s=0, zorder=4)
                # Se desplaza arriba y a la derecha para librar la zona del valle
                plt.annotate(f'$Z_{{min}}$\n({f_zmin:.1f} Hz, {mag_zmin:.1f} $\\Omega$)', 
                             xy=(f_zmin, mag_zmin), 
                             xytext=(30, -35), textcoords='offset points',
                             fontsize=10, fontweight='bold', color='k',
                             ha='left', va='bottom',
                             arrowprops=dict(arrowstyle="->", color='k', lw=1.2))
        
    plt.title(title, fontsize=13, pad=15)
    
    # Grilla limpia anti-aliasing (línea continua, transparente y al fondo absoluto)
    plt.grid(True, which="both", ls="-", alpha=0.15, color='gray', zorder=0)
    
    plt.tight_layout()
    plt.show()


def graficar_fase(frecuencias, fases, magnitudes, title="Respuesta de Fase"):
    """
    Grafica la frecuencia vs Fase en escala logarítmica.
    - Muestra los ticks típicos de audio de 20 a 20kHz.
    - Eje Y fijo entre -180° y 180° con saltos cada 45°.
    - Detecta la resonancia mediante el vector de magnitudes y marca fs con una línea vertical.
    - Permite personalizar el título del gráfico.
    """
    plt.figure(figsize=(11, 6))
    
    # Graficado de la curva de fase en color naranja
    plt.semilogx(frecuencias, fases, linewidth=2.5, color='darkorange', zorder=3)
    
    # --- CONFIGURACIÓN DEL EJE X (Frecuencias de Audio) ---
    ticks_audio = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    labels_audio = ['20', '50', '100', '200', '500', '1k', '2k', '5k', '10k', '20k']
    plt.xlim(20, 20000)
    plt.xticks(ticks_audio, labels_audio)
    plt.xlabel('Frecuencia (Hz)', fontsize=11)
    
    # --- CONFIGURACIÓN DEL EJE Y (Ángulos de fase estándar) ---
    ticks_y = [-180, -135, -90, -45, 0, 45, 90, 135, 180]
    labels_y = [f'{val}°' for val in ticks_y]
    plt.ylim(-180, 180)
    plt.yticks(ticks_y, labels_y)
    plt.ylabel('Fase (Grados)', fontsize=11)
    
    # Línea de referencia central en 0°
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1.2, zorder=1)
    
    # --- MARCAR LA FRECUENCIA DE RESONANCIA (fs) ---
    # Buscamos el pico de impedancia en graves (f < 300 Hz) para localizar fs
    mascara_graves = frecuencias < 300
    if any(mascara_graves):
        idx_max = magnitudes[mascara_graves].argmax()
        f_res = frecuencias[mascara_graves][idx_max]
        
        # Dibujamos una línea vertical punteada en la frecuencia de resonancia
        plt.axvline(x=f_res, color='k', linestyle='--', alpha=0.7, linewidth=1.5, zorder=2)
        plt.text(f_res * 1.1, 150, f'$f_s$: {f_res:.1f} Hz', 
                 fontsize=10, fontweight='bold', color='k', 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.title(title, fontsize=13, pad=15)
    plt.grid(True, which="both", ls=":", alpha=0.4, color='gray', zorder=2)
    
    plt.tight_layout()
    plt.show()
    
    
def comparar_impedancias(frecuencias, mags_1, mags_2, title="Comparación de Impedancias", label_1="Medición 1", label_2="Medición 2"):
    """
    Grafica DOS curvas de impedancia en el mismo gráfico para compararlas.
    - Curva 1: Azul. Curva 2: Rojo.
    - Eje Y limitado a 40 Ohms con saltos de a 4.
    - Las etiquetas de resonancia se abren simétricamente hacia arriba/lados 
      para evitar colisiones entre sí y con las curvas.
    """
    plt.figure(figsize=(12, 6.5))
    
    # --- GRAFICADO DE AMBAS CURVAS ---
    plt.semilogx(frecuencias, mags_1, linewidth=2.5, color='darkblue', label=label_1, zorder=3)
    plt.semilogx(frecuencias, mags_2, linewidth=2.5, color='red', label=label_2, zorder=3)
    
    # --- CONFIGURACIÓN DEL EJE X (Frecuencias de Audio) ---
    ticks_audio = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    labels_audio = ['20', '50', '100', '200', '500', '1k', '2k', '5k', '10k', '20k']
    plt.xlim(20, 20000)
    plt.xticks(ticks_audio, labels_audio)
    plt.xlabel('Frecuencia (Hz)', fontsize=11)
    
    # --- CONFIGURACIÓN DEL EJE Y (Máximo 40 Ohms, saltos de a 4) ---
    limite_y = 40
    ticks_y = list(range(0, limite_y + 4, 4))
    labels_y = [f'{val} $\\Omega$' for val in ticks_y]
    
    plt.ylim(0, limite_y)
    plt.yticks(ticks_y, labels_y)
    plt.ylabel('Impedancia Magnitud (|Z|)', fontsize=11)
    
    # --- DETECCION Y MARCADO DE RESONANCIAS ---
    mascara_graves = frecuencias < 300
    
    if any(mascara_graves):
        # --- Resonancia Curva 1 (Azul) -> Se anota hacia la DERECHA y ARRIBA ---
        idx_1 = mags_1[mascara_graves].argmax()
        f_res1 = frecuencias[mascara_graves][idx_1]
        mag_res1 = mags_1[mascara_graves][idx_1]
        
        if mag_res1 <= limite_y:
            plt.scatter(f_res1, mag_res1, color='k', s=0, zorder=4)
            
            # xytext=(100, 35) significa: mover 100 píxeles a la izquierda y 35 hacia arriba desde el pico
            plt.annotate(f'({f_res1:.1f} Hz, {mag_res1:.1f} $\\Omega$)', 
                         xy=(f_res1, mag_res1), 
                         xytext=(100, 35), textcoords='offset points',
                         fontsize=9, fontweight='bold', color='k',
                         ha='right', va='bottom', # Alinea el texto a la izquierda de la flecha
                         arrowprops=dict(arrowstyle="->", color='k', lw=1.2))
            
        # --- Resonancia Curva 2 (Rojo) -> Se anota hacia la IZQUIERDA y ARRIBA ---
        idx_2 = mags_2[mascara_graves].argmax()
        f_res2 = frecuencias[mascara_graves][idx_2]
        mag_res2 = mags_2[mascara_graves][idx_2]
        
        if mag_res2 <= limite_y:
            plt.scatter(f_res2, mag_res2, color='k', s=0, zorder=4)
            
            plt.annotate(f'({f_res2:.1f} Hz, {mag_res2:.1f} $\\Omega$)', 
                         xy=(f_res2, mag_res2), 
                         xytext=(-100, 35), textcoords='offset points',
                         fontsize=9, fontweight='bold', color='k',
                         ha='left', va='bottom', # Alinea el texto a la derecha de la flecha
                         arrowprops=dict(arrowstyle="->", color='k', lw=1.2))

    plt.title(title, fontsize=13, pad=15)
    plt.legend(loc="upper right", fontsize=10, framealpha=0.9)
    plt.grid(True, which="both", ls="-", alpha=0.15, color='gray', zorder=0)
    
    plt.tight_layout()
    plt.show()
    
    
# --- FUNCIONES DE GRAFICACION ACUSTICA ---
def promedio_logaritmico_db(valores_db):
    """
    Calcula el promedio energético (logarítmico) de un set de valores en dB.
    Equivale matemáticamente a la raíz cuadrada del promedio aritmético 
    de los cuadrados de las presiones.
    """
    # Pasamos de dB a escala lineal de energía (presión al cuadrado)
    energía = 10 ** (valores_db / 10.0)
    # Sacamos el promedio aritmético de la energía
    promedio_energia = np.mean(energía)
    # Volvemos a la escala logarítmica de dB
    return 10 * np.log10(promedio_energia)


def escalar_curva(frecuencias, spl_curva, f_s, promedio_filtrado_1m):
    """
    Ancla y escala una curva completa de respuesta en frecuencia basándose en
    una medición física real integrada en la banda de f_s a 10x f_s.
    
    - frecuencias: Array de NumPy con el espectro completo de frecuencias.
    - spl_curva: Array de NumPy con la curva relativa (ej: la que da 10 dB).
    - f_r: Frecuencia de resonancia o límite inferior del filtro.
    - promedio_filrado_1m: Nivel SPL (dB) medido físicamente con el ruido rosa filtrado (corregido a 1m).
    
    Returns:
    - spl_escalado: Array de NumPy con la curva final calibrada en SPL real.
    - offset: El offset calculado en dB (para control o logs).
    """
    # 1. Definir los límites de la banda según el procedimiento (f_s a 10*f_s)
    f_inf = f_s
    f_sup = 10.0 * f_s
    
    # 2. Crear la máscara para aislar los datos dentro de esa banda
    mascara = (frecuencias >= f_inf) & (frecuencias <= f_sup)
    
    if not any(mascara):
        raise ValueError(f"Error: No se encontraron frecuencias en el rango [{f_inf:.1f} Hz - {f_sup:.1f} Hz].")
    
    # 3. Aislar los dB de la curva en esa banda y usa promedio_logaritmico_db para calcular su promedio acústico real
    db_en_banda = spl_curva[mascara]
    promedio_curva_banda = promedio_logaritmico_db(db_en_banda)
    
    # 4. Calcular el Offset (Diferencia entre lo real físico y lo relativo de la curva)
    offset = promedio_filtrado_1m - promedio_curva_banda
    
    # 5. Escalado: Sumamos el offset a TODA la curva en un solo bloque (gracias a NumPy)
    spl_escalado = spl_curva + offset
    
    return spl_escalado, offset


def graficar_db(frecuencias, db, title="Respuesta en Frecuencia", ylabel="Nivel (dB)"):
    """
    Grafica Frecuencia vs dB para sistemas de audio.
    - Eje X: Logarítmico con ticks estándar de audio (20Hz a 20kHz).
    - Eje Y: Lineal y con etiqueta personalizable.
    """
    plt.figure(figsize=(11, 6))
    
    # Graficado de la curva de nivel
    plt.semilogx(frecuencias, db, linewidth=2, color='crimson', zorder=3)
    
    # --- CONFIGURACIÓN DEL EJE X ---
    ticks_audio = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    labels_audio = ['20', '50', '100', '200', '500', '1k', '2k', '5k', '10k', '20k']
    plt.xlim(20, 20000)
    plt.xticks(ticks_audio, labels_audio)
    plt.xlabel('Frecuencia (Hz)', fontsize=11)
    
    # --- CONFIGURACIÓN DEL EJE Y ---
    plt.ylabel(ylabel, fontsize=11) # <--- Ahora es una variable editable
    
    plt.title(title, fontsize=13, pad=15)
    plt.grid(True, which="both", ls="-", alpha=0.15, color='gray', zorder=0)
    
    plt.tight_layout()
    plt.show()


def comparar_db(frecuencias, lista_db, labels=None, title="Comparación de Respuestas en Frecuencia", ylabel="Nivel (dB)"):
    """
    Grafica MÚLTIPLES curvas de dB en el mismo gráfico para compararlas.
    - Eje X: Logarítmico con ticks estándar de audio (20Hz a 20kHz).
    - Eje Y: Lineal y con etiqueta personalizable.
    """
    plt.figure(figsize=(12, 6.5))
    
    if labels is None:
        labels = [f"Medición {i+1}" for i in range(len(lista_db))]
    elif len(labels) != len(lista_db):
        print("Advertencia: La cantidad de labels no coincide con la cantidad de curvas. Usando nombres automáticos.")
        labels = [f"Medición {i+1}" for i in range(len(lista_db))]

    # Graficado de todas las curvas
    for db_curva, label_curva in zip(lista_db, labels):
        plt.semilogx(frecuencias, db_curva, linewidth=2, label=label_curva, zorder=3)
            
    # --- CONFIGURACIÓN DEL EJE X ---
    ticks_audio = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    labels_audio = ['20', '50', '100', '200', '500', '1k', '2k', '5k', '10k', '20k']
    plt.xlim(20, 20000)
    plt.xticks(ticks_audio, labels_audio)
    plt.xlabel('Frecuencia (Hz)', fontsize=11)
    
    # --- CONFIGURACIÓN DEL EJE Y ---
    plt.ylabel(ylabel, fontsize=11) # <--- Ahora es una variable editable
    
    plt.title(title, fontsize=13, pad=15)
    plt.legend(loc="upper right", fontsize=10, framealpha=0.9)
    plt.grid(True, which="both", ls="-", alpha=0.15, color='gray', zorder=0)
    
    plt.tight_layout()
    plt.show()
    

def graficar_patron_polar(frecuencias, lista_db, angulos, f_centro, title=None):
    """
    Grafica el patrón polar normalizado a 0 dB en el eje (0°).
    - Espeja de 0-90° hacia 270-360° y deja nulo el sector trasero.
    - Grilla radial fija estrictamente con pasos de -10 dB.
    """
    # 1. Calcular límites matemáticos de la banda de octava
    f_inf = f_centro / np.sqrt(2)
    f_sup = f_centro * np.sqrt(2)
    mascara = (frecuencias >= f_inf) & (frecuencias <= f_sup)
    
    if not any(mascara):
        print(f"Error: No se encontraron frecuencias para la banda de {f_centro} Hz.")
        return
        
    # Promediar el nivel absoluto en la banda para cada ángulo medido
    valores_absolutos = []
    for db_curva in lista_db:
        valores_absolutos.append(np.mean(db_curva[mascara]))
        
    angulos = np.array(angulos)
    valores_absolutos = np.array(valores_absolutos)
    
    # NORMALIZACIÓN A 0 dB
    idx_cero = np.where(angulos == 0)[0]
    if len(idx_cero) == 0:
        print("Error: No se encontró la medición a 0 grados para normalizar.")
        return
    
    valor_referencia_0deg = valores_absolutos[idx_cero[0]]
    valores_medidos = valores_absolutos - valor_referencia_0deg
    
    # 2. LÓGICA DE ESPEJADO Y ZONA NULA (90° a 270°)
    angulos_completos = []
    valores_completos = []
    
    for ang, val in zip(angulos, valores_medidos):
        if 0 <= ang <= 90:
            angulos_completos.append(ang)
            valores_completos.append(val)
            
    angulos_completos.append(180)
    valores_completos.append(np.nan) # Quiebre de línea para zona nula
    
    for ang, val in zip(angulos, valores_medidos):
        if 0 < ang <= 90:
            angulos_completos.append(360 - ang)
            valores_completos.append(val)
            
    angulos_completos.append(360)
    valores_completos.append(0.0)
    
    # 3. ORDENAR LOS VECTORES
    angulos_completos = np.array(angulos_completos)
    valores_completos = np.array(valores_completos)
    indices_ordenados = np.argsort(angulos_completos)
    
    angulos_ordenados = angulos_completos[indices_ordenados]
    valores_ordenados = valores_completos[indices_ordenados]
    angulos_rad = np.radians(angulos_ordenados)
    
    # 4. CONFIGURACIÓN DEL GRÁFICO POLAR
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(7.5, 7.5))
    
    # Convención acústica estándar (0° arriba, giro horario)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    # Graficado de la curva
    ax.plot(angulos_rad, valores_ordenados, linewidth=2.5, color='crimson', zorder=3)
    
    # --- CONFIGURACIÓN DE LA ESCALA FIJA CADA -10 dB ---
    # Buscamos la máxima atenuación real (ignorando los NaN de la zona trasera)
    min_val = np.nanmin(valores_ordenados)
    
    # Redondeamos al siguiente múltiplo de -10 (ej: -23dB se va a -30dB)
    # Dejamos un piso estético mínimo de -30dB para que no quede un gráfico diminuto
    bottom_limit = min(-30, int(np.floor(min_val / 10.0) * 10))
    
    # Generamos la lista de pasos exactos desde el centro hacia el borde (0 dB)
    ticks_radial = list(range(bottom_limit, 10, 10))
    
    # Aplicamos los límites y los ticks al eje radial
    ax.set_rlim(bottom=bottom_limit, top=0)
    ax.set_rticks(ticks_radial)
    
    # Formateamos los carteles para que digan "dB" de forma elegante
    labels_radial = [f"{t} dB" if t != 0 else "0 dB" for t in ticks_radial]
    ax.set_yticklabels(labels_radial, fontsize=9, fontweight='bold', color='dimgray')
    
    # Configurar las marcas angulares exteriores (cada 30°)
    ax.set_xticks(np.radians([0, 15, 30, 45, 60, 75, 90, 180, 
                              270, 285, 300, 315, 330]))
    
    # Grilla tenue de fondo
    ax.grid(True, ls='-', alpha=0.15, color='gray', zorder=0)
    
    if title is None:
        title = f"Directividad Relativa (0 dB en eje) - Banda: {f_centro} Hz\n(Zona Trasera Sin Mediciones)"
    ax.set_title(title, fontsize=12, pad=25, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def comparar_patron_polar(frecuencias, lista_db, angulos, lista_f_centro, title=None):
    """
    Grafica MÚLTIPLES patrones polares normalizados a 0 dB en el mismo gráfico.
    
    - frecuencias: Vector común de frecuencias.
    - lista_db: Lista con los arrays de dB [db_0, db_30, db_60, db_90].
    - angulos: Lista o array con los ángulos medidos [0, 30, 60, 90].
    - lista_f_centro: Lista de frecuencias a comparar, ej: [250, 500, 1000, 4000].
    - title: Título personalizado.
    """
    # Inicializamos el gráfico polar común
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    
    # Convención acústica estándar (0° arriba, giro horario)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    # Variable para rastrear la máxima atenuación global y ajustar la escala al final
    max_atenuacion_global = -30
    
    # --- BUCLE PRINCIPAL: PROCESAR CADA FRECUENCIA SOLICITADA ---
    for f_centro in lista_f_centro:
        
        # 1. Calcular límites de la banda de octava
        f_inf = f_centro / np.sqrt(2)
        f_sup = f_centro * np.sqrt(2)
        mascara = (frecuencias >= f_inf) & (frecuencias <= f_sup)
        
        if not any(mascara):
            print(f"Advertencia: No se encontraron frecuencias para la banda de {f_centro} Hz. Saltando.")
            continue
            
        # Promediar nivel absoluto para esta banda específica
        valores_absolutos = []
        for db_curva in lista_db:
            valores_absolutos.append(np.mean(db_curva[mascara]))
            
        angulos_arr = np.array(angulos)
        valores_absolutos = np.array(valores_absolutos)
        
        # Normalización a 0 dB (Cada banda respecto a su propio eje)
        idx_cero = np.where(angulos_arr == 0)[0]
        if len(idx_cero) == 0:
            print(f"Error: No se encontró la medición a 0 grados para la banda de {f_centro} Hz.")
            return
        
        valor_referencia_0deg = valores_absolutos[idx_cero[0]]
        valores_medidos = valores_absolutos - valor_referencia_0deg
        
        # 2. Lógica de espejado y zona nula (90° a 270°)
        angulos_completos = []
        valores_completos = []
        
        for ang, val in zip(angulos_arr, valores_medidos):
            if 0 <= ang <= 90:
                angulos_completos.append(ang)
                valores_completos.append(val)
                
        angulos_completos.append(180)
        valores_completos.append(np.nan) # Quiebre de línea para zona nula
        
        for ang, val in zip(angulos_arr, valores_medidos):
            if 0 < ang <= 90:
                angulos_completos.append(360 - ang)
                valores_completos.append(val)
                
        angulos_completos.append(360)
        valores_completos.append(0.0)
        
        # 3. Ordenar los vectores para el graficado continuo
        angulos_completos = np.array(angulos_completos)
        valores_completos = np.array(valores_completos)
        indices_ordenados = np.argsort(angulos_completos)
        
        angulos_ordenados = angulos_completos[indices_ordenados]
        valores_ordenados = valores_completos[indices_ordenados]
        angulos_rad = np.radians(angulos_ordenados)
        
        # Actualizar el mínimo global si esta curva atenúa más que las anteriores
        min_val = np.nanmin(valores_ordenados)
        if min_val < max_atenuacion_global:
            max_atenuacion_global = min_val
            
        # Formatear el nombre de la curva para la leyenda (ej: 1000 -> 1 kHz)
        label_f = f"{f_centro} Hz" if f_centro < 1000 else f"{f_centro/1000:.1f} kHz".replace(".0", "")
        
        # Graficar esta línea en el plano común (Matplotlib cambia el color solo)
        ax.plot(angulos_rad, valores_ordenados, linewidth=2, label=label_f, zorder=3)

    # --- 4. CONFIGURACIÓN FINAL DE LA ESCALA Y LA GRILLA (POST-BUCLE) ---
    # Redondeamos la peor atenuación al siguiente múltiplo de -10 dB
    bottom_limit = min(-30, int(np.floor(max_atenuacion_global / 10.0) * 10))
    ticks_radial = list(range(bottom_limit, 10, 10))
    
    # Aplicar límites fijos y pasos cada -10 dB
    ax.set_rlim(bottom=bottom_limit, top=0)
    ax.set_rticks(ticks_radial)
    
    labels_radial = [f"{t} dB" if t != 0 else "0 dB" for t in ticks_radial]
    ax.set_yticklabels(labels_radial, fontsize=9, fontweight='bold', color='dimgray')
    
    # Configurar marcas angulares (cada 30°)
    ax.set_xticks(np.radians([0, 15, 30, 45, 60, 75, 90, 180, 
                              270, 285, 300, 315, 330]))
    ax.grid(True, ls='-', alpha=0.15, color='gray', zorder=0)
    
    # Agregar la leyenda de colores (ubicada estratégicamente fuera del círculo para no tapar)
    ax.legend(loc="upper left", bbox_to_anchor=(1.08, 1.0), fontsize=10, framealpha=0.9)
    
    if title is None:
        title = "Comparación de Patrones Polares (Directividad)"
    ax.set_title(title, fontsize=12, pad=25, fontweight='bold')
    
    plt.tight_layout()
    plt.show()