import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- FUNCIONES DE IMPORTACION DE DATOS ---
def importar_datos_smaart(archivo_path):
    """
    Lee un archivo TXT exportado de Smaart y extrae la información 
    en 2 vectores: frecuencias y dbspl.
    """
    # Le asignamos nombres temporales del 0 al 5 (6 columnas potenciales) 
    # para que no se trabe si la línea de títulos tiene menos elementos que los datos.
    df = pd.read_csv(archivo_path, comment=';', header=None, sep=r'\s+', decimal='.', names=range(7))
    
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


def suavizado(f: np.ndarray, amp_db: np.ndarray, oct: int) -> np.ndarray:
    """

    Devuelve una array de amplitudes suavizada en una fracción de octava.

    Parameters
    ----------
    f : numpy.ndarray
        Frecuencias de menor a mayor
    amp_db : numpy.ndarray
        Amplitudes en dB
    oct: int
        Fracción o subdivisión de octava para suavizar
        (ej. OCT=3 -> Suavizado por tercios de octava)
        oct=0 es un caso especial donde no se realiza el suavizado
        y se devuelve la curva intacta

    Returns
    -------
    out : numpy.ndarray
        Amplitudes suavizadas

    """

    if oct == 0:
        return amp_db

    if len(f) != len(amp_db):
        raise Exception("f y amp_db deben ser del mismo largo.")

    ampsmooth_db = np.zeros(np.size(amp_db))
    for i, freq in enumerate(f):
        finf = freq / pow(2, 1 / (2 * oct))  # calcula el corte inferior del promedio
        fsup = freq * pow(2, 1 / (2 * oct))  # calcula el corte superior del promedio

        idx = np.logical_and(
            f >= finf, f <= fsup
        )  # busca los elementos dentro del rango de frecuencias

        # Suma energetica del rango de frecuencias
        amp = pow(10, amp_db[idx] / 10)
        ampsmooth_db[i] = 10 * np.log10(sum(amp) / len(amp))

    return ampsmooth_db


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
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10, framealpha=0.9)
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
    
def comparar_patron_polar_2(frecuencias, lista_db, angulos, lista_f_centro, title=None):
    """
    Grafica MÚLTIPLES patrones polares divididos en el mismo gráfico.
    - Lado Derecho: Primera mitad de las frecuencias (Bajas/Medias).
    - Lado Izquierdo: Segunda mitad de las frecuencias (Altas).
    Todos los datos se normalizan a 0 dB respecto a su propia medición de 0°.
    """
    frecuencias = np.asarray(frecuencias)
    
    # Inicializamos el gráfico polar común
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    
    # Convención acústica estándar (0° arriba, giro horario)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    max_atenuacion_global = -30
    
    # --- DIVISIÓN DE FRECUENCIAS EN DOS MITADES ---
    mitad = len(lista_f_centro) // 2
    
    # --- BUCLE PRINCIPAL ---
    for idx_bucle, f_centro in enumerate(lista_f_centro):
        
        # 1. Encontrar el índice de la frecuencia más cercana
        idx_f = np.argmin(np.abs(frecuencias - f_centro))
        f_real = frecuencias[idx_f]
            
        # Extraer el nivel absoluto para cada ángulo en esta frecuencia
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
        
        # --- 2. ASIGNACIÓN DE HEMISFERIO SEGÚN LA FRECUENCIA ---
        if idx_bucle < mitad:
            angulos_finales = angulos_arr
            lado_str = "Der"
        else:
            angulos_finales = 360 - angulos_arr
            lado_str = "Izq"
        
        # 3. Ordenar los vectores para que Matplotlib dibuje la línea continua
        indices_ordenados = np.argsort(angulos_finales)
        angulos_ordenados = angulos_finales[indices_ordenados]
        valores_ordenados = valores_medidos[indices_ordenados]
        valores_ordenados = np.clip(valores_ordenados, -25, None)
        angulos_rad = np.radians(angulos_ordenados)
        
        # Actualizar la máxima atenuación para la escala radial final
        min_val = np.nanmin(valores_ordenados)
        if min_val < max_atenuacion_global:
            max_atenuacion_global = min_val
            
        # Formatear la etiqueta de la leyenda
        label_f = f"{f_centro} Hz ({lado_str})" if f_centro < 1000 else f"{f_centro/1000:.1f} kHz ({lado_str})".replace(".0", "")
        
        # Graficar la curva en el plano común
        ax.plot(angulos_rad, valores_ordenados, linewidth=2, label=label_f, zorder=3)

    # --- 4. CONFIGURACIÓN DE LA ESCALA RADIAL (dB) ---
    bottom_limit = -25 
    # Añadimos el -25 a los ticks para que se dibuje el círculo perimetral de fondo
    ticks_radial = [-25, -20, -15, -10, -5, 0]
    
    ax.set_rlim(bottom=bottom_limit, top=0)
    ax.set_rticks(ticks_radial)
    
    labels_radial = [f"{t} dB" if t != 0 else "0 dB" for t in ticks_radial]
    ax.set_yticklabels(labels_radial, fontsize=9, fontweight='bold', color='dimgray')
    
    # --- 5. CONFIGURACIÓN DE LOS TICKS ANGULARES SIMÉTRICOS ---
    ticks_angulares = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    ax.set_xticks(np.radians(ticks_angulares))
    
    labels_angulares = [f"{a}°" if a <= 180 else f"{360 - a}°" for a in ticks_angulares]
    ax.set_xticklabels(labels_angulares, fontsize=9.5)
    
    ax.grid(True, ls='-', alpha=0.15, color='gray', zorder=0)
    
    ax.legend(loc="upper left", bbox_to_anchor=(1.12, 1.0), fontsize=10, framealpha=0.9)
    
    if title is None:
        title = "Patrón Polar Comparativo \n(Bajas a la Derecha | Altas a la Izquierda)"
    ax.set_title(title, fontsize=12, pad=25, fontweight='bold')
    
    plt.tight_layout()
    plt.show()