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

def detectar_zmin(frecuencias, magnitudes, f_res):
    """
    Detecta la impedancia mínima (Zmin) y su frecuencia correspondiente,
    buscando únicamente a la derecha de la frecuencia de resonancia (f_res).
    
    - frecuencias: Array de NumPy con el espectro de frecuencias.
    - magnitudes: Array de NumPy con la magnitud de la impedancia en Ohms.
    - f_res: Frecuencia de resonancia (fs) obtenida previamente.
    
    Returns:
    - f_zmin: Frecuencia del mínimo de impedancia (Hz).
    - mag_zmin: Valor de la impedancia mínima (Ohms).
      (Devuelve None, None si no se encuentra o f_res es inválido).
    """
    if f_res is not None:
        mascara_post_fs = frecuencias > f_res
        if any(mascara_post_fs):
            idx_min = magnitudes[mascara_post_fs].argmin()
            f_zmin = frecuencias[mascara_post_fs][idx_min]
            mag_zmin = magnitudes[mascara_post_fs][idx_min]
            return f_zmin, mag_zmin
            
    return None, None
        
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
        plt.annotate(f'Re = {mag_start:.2f} $\\Omega$', 
                     xy=(f_start, mag_start), 
                     xytext=(20, -20), textcoords='offset points',
                     fontsize=9, fontweight='bold', color='k',
                     ha='left', va='bottom',
                     arrowprops=dict(arrowstyle="->", color='k', lw=1.2))
        
    # --- DETECTAR Y MARCAR LA RESONANCIA ---
    f_res, mag_res = detectar_resonancia(frecuencias, magnitudes)
        
    if mag_res <= limite_y:
        plt.scatter(f_res, mag_res, color='k', s=0, zorder=4)
        # xytext=(0, 40) tira la flecha recto desde arriba para no pisar las laderas
        plt.annotate(f'Resonancia\n({f_res:.2f} Hz, {mag_res:.2f} $\\Omega$)', 
                     xy=(f_res, mag_res), 
                     xytext=(0, 40), textcoords='offset points',
                     fontsize=10, fontweight='bold', color='k',
                     ha='center', va='bottom',
                     arrowprops=dict(arrowstyle="->", color='k', lw=1.2))
            
    # --- VALOR DE MÍNIMA IMPEDANCIA (Zmin) DESPUÉS DE FS ---
    f_zmin, mag_zmin = detectar_zmin(frecuencias, magnitudes, f_res)
            
    if mag_zmin <= limite_y:
        plt.scatter(f_zmin, mag_zmin, color='k', s=0, zorder=4)
        # Se desplaza arriba y a la derecha para librar la zona del valle
        plt.annotate(f'$Z_{{min}}$\n({f_zmin:.2f} Hz, {mag_zmin:.2f} $\\Omega$)', 
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
    f_res, mag_res = detectar_resonancia(frecuencias, magnitudes)
        
    # Dibujamos una línea vertical punteada en la frecuencia de resonancia
    plt.axvline(x=f_res, color='k', linestyle='--', alpha=0.7, linewidth=1.5, zorder=2)
    plt.text(f_res * 1.1, 150, f'$f_s$: {f_res:.2f} Hz', 
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
    f_res1, mag_res1 = detectar_resonancia(frecuencias, mags_1)
        
    if mag_res1 <= limite_y:
        plt.scatter(f_res1, mag_res1, color='k', s=0, zorder=4)
            
        # xytext=(100, 35) significa: mover 100 píxeles a la izquierda y 35 hacia arriba desde el pico
        plt.annotate(f'({f_res1:.2f} Hz, {mag_res1:.2f} $\\Omega$)', 
                     xy=(f_res1, mag_res1), 
                     xytext=(100, 35), textcoords='offset points',
                     fontsize=9, fontweight='bold', color='k',
                     ha='right', va='bottom', # Alinea el texto a la izquierda de la flecha
                     arrowprops=dict(arrowstyle="->", color='k', lw=1.2))
            
        # --- Resonancia Curva 2 (Rojo) -> Se anota hacia la IZQUIERDA y ARRIBA ---
        f_res2, mag_res2 = detectar_resonancia(frecuencias, mags_2)
        
        if mag_res2 <= limite_y:
            plt.scatter(f_res2, mag_res2, color='k', s=0, zorder=4)
            
            plt.annotate(f'({f_res2:.2f} Hz, {mag_res2:.2f} $\\Omega$)', 
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
def promedio_logaritmico_butterworth(frecuencias, valores_db, f_inf, f_sup):
    """
    Calculates the logarithmic energy average weighted by a 4th-order Butterworth filter.
    """
    # 1. Generar el filtro interno
    h_hp2 = 1.0 / (1.0 + (f_inf / frecuencias) ** 8)
    h_lp2 = 1.0 / (1.0 + (frecuencias / f_sup) ** 8)
    filtro_energia = h_hp2 * h_lp2
    
    # 2. Pasar a lineal, ponderar, promediar y volver a dB
    energia = 10 ** (np.array(valores_db) / 10.0)
    promedio_energia = np.sum(energia * filtro_energia) / np.sum(filtro_energia)
    
    return 10 * np.log10(promedio_energia)


def encontrar_desviacion(frecuencias, valores_db, f_inf, f_sup):
    """
    Encuentra el valor máximo, mínimo y sus respectivas frecuencias
    dentro de una banda de interés [f_inf, f_sup].
    
    Returns:
    - f_max: Frecuencia del pico máximo (Hz).
    - v_max: Valor máximo en dB SPL.
    - f_min: Frecuencia del valle mínimo (Hz).
    - v_min: Valor mínimo en dB SPL.
    - span: Desviación total pico a pico (dB).
    """
    # 1. Asegurar que sean arrays de NumPy
    frecuencias = np.array(frecuencias)
    valores_db = np.array(valores_db)
    
    # 2. Crear la máscara para aislar la banda
    mascara = (frecuencias >= f_inf) & (frecuencias <= f_sup)
    
    # 3. Filtrar los vectores aplicando la máscara
    frecs_banda = frecuencias[mascara]
    db_banda = valores_db[mascara]
    
    if len(db_banda) == 0:
        raise ValueError("Error: No hay datos en el rango de frecuencias especificado.")
    
    # 4. Encontrar los índices del máximo y mínimo dentro de la banda
    idx_max = np.argmax(db_banda)
    idx_min = np.argmin(db_banda)
    
    # 5. Extraer los valores correspondientes
    f_max = frecs_banda[idx_max]
    v_max = db_banda[idx_max]
    
    f_min = frecs_banda[idx_min]
    v_min = db_banda[idx_min]
    
    # 6. Calcular la variación total pico a pico
    span = v_max - v_min
    
    return f_max, v_max, f_min, v_min, span


def escalar_curva(frecuencias, spl_curva_completa, f_s, spl_curva_referencia):
    """
    Ancla y escala una curva teórica basándose en el espectro de una medición física real,
    procesando AMBOS arrays con el mismo filtro Butterworth de 24 dB/oct.
    
    - frecuencias: Array de NumPy con el espectro común de frecuencias.
    - spl_curva_completa: Array con la curva relativa a escalar (ej. datos de simulación).
    - f_s: Frecuencia de resonancia o límite inferior del filtro.
    - spl_curva_referencia: Array con el espectro de la medición real de la curva de referencia (ya corregida a 1m!).
    
    Returns:
    - spl_escalado: Curva completa calibrada en SPL real.
    - offset: El offset calculado en dB.
    """
    # 1. Definir los límites de la banda de paso (f_s a 10x f_s)
    f_inf = f_s
    f_sup = 10.0 * f_s
    
    # 2. Calcular el promedio energético de la curva TEÓRICA
    promedio_curva_completa = promedio_logaritmico_butterworth(frecuencias, spl_curva_completa, f_inf, f_sup)
    
    # 3. Calcular el promedio energético de la curva MEDIDA (¡Tu mejora!)
    promedio_curva_referencia = promedio_logaritmico_butterworth(frecuencias, spl_curva_referencia, f_inf, f_sup)
    
    # 4. El offset es la diferencia pura entre ambos promedios procesados bajo la misma ventana
    offset = promedio_curva_completa - promedio_curva_referencia
    
    # 5. Escalado de la curva original
    spl_escalado = spl_curva_completa - offset
    
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
    Busca el centro de banda de tercio de octava más cercano en los datos ya promediados.
    
    - Espeja de 0-90° hacia 270-360° y deja nulo el sector trasero.
    - Grilla radial fija estrictamente con pasos de -10 dB.
    """
    # Aseguramos que frecuencias sea un array de NumPy
    frecuencias = np.asarray(frecuencias)
    
    # 1. ENCONTRAR EL ÍNDICE DE LA FRECUENCIA MÁS CERCANA
    # (Mapea tu f_centro nominal con el centro real que guardó Smaart)
    idx_f = np.argmin(np.abs(frecuencias - f_centro))
    f_real = frecuencias[idx_f]
    
    print(f"-> Frecuencia solicitada: {f_centro} Hz | Centro de tercio encontrado: {f_real:.2f} Hz")
        
    # Extraer el nivel absoluto en ese índice específico para cada curva/ángulo
    valores_absolutos = []
    for db_curva in lista_db:
        valores_absolutos.append(db_curva[idx_f])
        
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
    valores_completos.append(np.nan) # Quiebre de línea para la zona trasera vacía
    
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
    min_val = np.nanmin(valores_ordenados)
    bottom_limit = min(-30, int(np.floor(min_val / 10.0) * 10))
    ticks_radial = list(range(bottom_limit, 10, 10))
    
    ax.set_rlim(bottom=bottom_limit, top=0)
    ax.set_rticks(ticks_radial)
    
    labels_radial = [f"{t} dB" if t != 0 else "0 dB" for t in ticks_radial]
    ax.set_yticklabels(labels_radial, fontsize=9, fontweight='bold', color='dimgray')
    
    ax.set_xticks(np.radians([0, 15, 30, 45, 60, 75, 90, 180, 
                              270, 285, 300, 315, 330]))
    
    ax.grid(True, ls='-', alpha=0.15, color='gray', zorder=0)
    
    if title is None:
        title = f"Directividad Relativa (0 dB en eje) - Tercio de Octava: {f_centro} Hz"
    ax.set_title(title, fontsize=12, pad=25, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def comparar_patron_polar(frecuencias, lista_db, angulos, lista_f_centro, title=None):
    """
    Grafica MÚLTIPLES patrones polares normalizados a 0 dB en el mismo gráfico.
    Busca los centros de banda de tercio de octava más cercanos en los datos ya promediados.
    
    - frecuencias: Vector común de frecuencias de Smaart.
    - lista_db: Lista con los arrays de dB [db_0, db_15, db_30, ...].
    - angulos: Lista o array con los ángulos medidos [0, 15, 30, ...].
    - lista_f_centro: Lista de frecuencias nominales a comparar, ej: [1000, 2000, 4000].
    - title: Título personalizado.
    """
    frecuencias = np.asarray(frecuencias)
    
    # Inicializamos el gráfico polar común
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    
    # Convención acústica estándar (0° arriba, giro horario)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    # Variable para rastrear la máxima atenuación global y ajustar la escala al final
    max_atenuacion_global = -30
    
    # --- BUCLE PRINCIPAL: PROCESAR CADA FRECUENCIA SOLICITADA ---
    for f_centro in lista_f_centro:
        
        # 1. ENCONTRAR EL ÍNDICE DE LA FRECUENCIA MÁS CERCANA
        idx_f = np.argmin(np.abs(frecuencias - f_centro))
        f_real = frecuencias[idx_f]
        
        print(f"-> Procesando: {f_centro} Hz | Centro de tercio real: {f_real:.2f} Hz")
            
        # Extraer el nivel absoluto en ese índice específico para cada ángulo
        valores_absolutos = []
        for db_curva in lista_db:
            valores_absolutos.append(db_curva[idx_f])
            
        angulos_arr = np.array(angulos)
        valores_absolutos = np.array(valores_absolutos)
        
        # Normalización a 0 dB (Cada banda respecto a su propio eje de 0°)
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
            
        # --- EL CAMBIO SOLICITADO AQUÍ: SE USA F_CENTRO NOMINAL EN VEZ DE F_REAL ---
        label_f = f"{f_centro} Hz" if f_centro < 1000 else f"{f_centro/1000:.2f} kHz".replace(".0", "")
        # --------------------------------------------------------------------------
        
        # Graficar esta línea en el plano común
        ax.plot(angulos_rad, valores_ordenados, linewidth=2, label=label_f, zorder=3)

    # --- 4. CONFIGURACIÓN FINAL DE LA ESCALA Y LA GRILLA (POST-BUCLE) ---
    bottom_limit = min(-30, int(np.floor(max_atenuacion_global / 10.0) * 10))
    ticks_radial = list(range(bottom_limit, 10, 10))
    
    ax.set_rlim(bottom=bottom_limit, top=0)
    ax.set_rticks(ticks_radial)
    
    labels_radial = [f"{t} dB" if t != 0 else "0 dB" for t in ticks_radial]
    ax.set_yticklabels(labels_radial, fontsize=9, fontweight='bold', color='dimgray')
    
    ax.set_xticks(np.radians([0, 15, 30, 45, 60, 75, 90, 180, 
                              270, 285, 300, 315, 330]))
    ax.grid(True, ls='-', alpha=0.15, color='gray', zorder=0)
    
    ax.legend(loc="upper left", bbox_to_anchor=(1.08, 1.0), fontsize=10, framealpha=0.9)
    
    if title is None:
        title = "Comparación de Patrones Polares (Tercios de Octava)"
    ax.set_title(title, fontsize=12, pad=25, fontweight='bold')
    
    plt.tight_layout()
    plt.show()