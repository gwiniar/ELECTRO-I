import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def importar_datos_limp(archivo_path):
    """
    Lee el archivo CSV de LIMP, limpia el formato regional de decimales,
    ignora metadatos y devuelve 3 vectores numéricos limpios.
    """
    # Usamos decimal=',' por si LIMP exportó con comas europeas/latinas.
    # Si tus decimales ya eran con punto y el problema era solo el header, decimal=',' no romperá nada.
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

def graficar_ohm(frecuencias, magnitudes):
    """
    Grafica la frecuencia vs Impedancia en escala logarítmica.
    - Muestra los ticks típicos de audio de 20 a 20kHz.
    - Eje Y con divisiones y etiquetas cada 4 Ohms de forma exacta.
    - Detecta la resonancia y la marca con sus coordenadas (fs, Ohms).
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
    
    # --- CONFIGURACIÓN DEL EJE Y (Saltos de a 4 Ohms) ---
    max_ohm = int(np.ceil(magnitudes.max())) # Buscamos el techo entero del valor más alto
    ticks_y = list(range(0, max_ohm + 4, 4))   # Genera la lista [0, 4, 8, 12, 16...]
    labels_y = [f'{val} $\Omega$' for val in ticks_y] # Crea las etiquetas estéticas
    
    plt.ylim(0, max_ohm + 2) # Le damos un pequeño margen arriba para que no corte la curva
    plt.yticks(ticks_y, labels_y)
    plt.ylabel('Impedancia Magnitud (|Z|)', fontsize=11)
    
    # --- DETECTAR Y MARCAR LA RESONANCIA (Coordenadas exactas) ---
    # Buscamos el pico únicamente en graves (f < 300 Hz)
    mascara_graves = frecuencias < 300
    if any(mascara_graves):
        idx_max = magnitudes[mascara_graves].argmax()
        f_res = frecuencias[mascara_graves][idx_max]
        mag_res = magnitudes[mascara_graves][idx_max]
        
        # Dibujar el punto rojo sobre el pico
        plt.scatter(f_res, mag_res, color='red', s=55, zorder=4)
        
        # Texto indicando las coordenadas exactas en formato (fs, valor)
        plt.annotate(f'Resonancia\n({f_res:.1f} Hz, {mag_res:.1f} $\Omega$)', 
                     xy=(f_res, mag_res), 
                     xytext=(f_res * 1.3, mag_res * 0.95),
                     fontsize=10, fontweight='bold', color='red',
                     arrowprops=dict(arrowstyle="->", color='red', lw=1.2))
        
    plt.title('Respuesta de Magnitud (Impedancia)', fontsize=13, pad=15)
    
    # Rejilla de fondo sutil adaptada a las nuevas divisiones
    plt.grid(True, which="both", ls=":", alpha=0.4, color='gray', zorder=2)
    
    plt.tight_layout()
    plt.show()

frecuencias1, magnitud1, fase1 = importar_datos_limp(r'C:\Users\dell_\Downloads\TP1 Electro-20260610T000342Z-3-001\TP1 Electro\Electrico\SIN MASA.csv')

graficar_ohm(frecuencias1, magnitud1)