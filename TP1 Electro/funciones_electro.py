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


def graficar_ohm(frecuencias, magnitudes, title="Respuesta de Magnitud (Impedancia)"):
    """
    Grafica la frecuencia vs Impedancia en escala logarítmica.
    - Muestra los ticks típicos de audio de 20 a 20kHz.
    - Eje Y limitado a un máximo de 40 Ohms con saltos de a 4.
    - Detecta la resonancia en graves y la marca con sus coordenadas (fs, Ohms).
    - Permite personalizar el título del gráfico.
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
    ticks_y = list(range(0, limite_y + 4, 4))   # Genera [0, 4, 8, ..., 40]
    labels_y = [f'{val} $\Omega$' for val in ticks_y]
    
    plt.ylim(0, limite_y) # Forzamos el corte estricto en 40 Ohms
    plt.yticks(ticks_y, labels_y)
    plt.ylabel('Impedancia (|Z|)', fontsize=11)
    
    # --- DETECTAR Y MARCAR LA RESONANCIA (Coordenadas exactas) ---
    mascara_graves = frecuencias < 300
    if any(mascara_graves):
        idx_max = magnitudes[mascara_graves].argmax()
        f_res = frecuencias[mascara_graves][idx_max]
        mag_res = magnitudes[mascara_graves][idx_max]
        
        # Solo dibujamos el marcador si el pico cae dentro de la escala visible (menos de 40 Ohms)
        if mag_res <= limite_y:
            plt.scatter(f_res, mag_res, color='red', s=55, zorder=4)
            
            plt.annotate(f'Resonancia\n({f_res:.1f} Hz, {mag_res:.1f} $\Omega$)', 
                         xy=(f_res, mag_res), 
                         xytext=(f_res * 1.3, mag_res * 0.95),
                         fontsize=10, fontweight='bold', color='red',
                         arrowprops=dict(arrowstyle="->", color='red', lw=1.2))
        
    # Título personalizado por el usuario
    plt.title(title, fontsize=13, pad=15)
    
    # Rejilla de fondo sutil
    plt.grid(True, which="both", ls=":", alpha=0.4, color='gray', zorder=2)
    
    plt.tight_layout()
    plt.show()


def graficar_fase(frecuencias, fases, magnitudes, title="Respuesta de Fase"):
    """
    Grafica la frecuencia vs Fase en escala logarítmica.
    - Muestra los ticks típicos de audio de 20 a 20kHz.
    - Eje Y fijo entre -180° y 180° con saltos cada 45°.
    - Curva de color naranja.
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
        # Usamos color rojo ('red' o 'crimson') para que resalte y mantenga coherencia con el otro gráfico
        plt.axvline(x=f_res, color='crimson', linestyle='--', alpha=0.7, linewidth=1.5, zorder=2)
        
        # Ponemos el texto de fs en la parte superior del gráfico para que no tape la curva
        plt.text(f_res * 1.1, 150, f'$f_s$: {f_res:.1f} Hz', 
                 fontsize=10, fontweight='bold', color='crimson', 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Título personalizado por el usuario
    plt.title(title, fontsize=13, pad=15)
    
    # Rejilla de fondo sutil
    plt.grid(True, which="both", ls=":", alpha=0.4, color='gray', zorder=2)
    
    plt.tight_layout()
    plt.show()
    

def comparar_impedancias(frecuencias, mags_1, mags_2, title="Comparación de Impedancias", label_1="Medición 1", label_2="Medición 2"):
    """
    Grafica DOS curvas de impedancia en el mismo gráfico para compararlas.
    - Curva 1: Azul. Curva 2: Amarillo (Gold para mejor visibilidad).
    - Eje Y limitado a 40 Ohms con saltos de a 4.
    - Detecta y marca de forma independiente la resonancia de ambas curvas.
    - Incluye leyenda automática y título personalizado.
    """
    plt.figure(figsize=(12, 6.5))
    
    # --- GRAFICADO DE AMBAS CURVAS ---
    # Usamos 'gold' en lugar de 'yellow' puro porque el amarillo estándar casi no se ve sobre fondo blanco
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
    labels_y = [f'{val} $\Omega$' for val in ticks_y]
    
    plt.ylim(0, limite_y)
    plt.yticks(ticks_y, labels_y)
    plt.ylabel('Impedancia Magnitud (|Z|)', fontsize=11)
    
    # --- DETECCION Y MARCADO DE RESONANCIAS ---
    mascara_graves = frecuencias < 300
    
    if any(mascara_graves):
        # --- Resonancia Curva 1 (Azul) ---
        idx_1 = mags_1[mascara_graves].argmax()
        f_res1 = frecuencias[mascara_graves][idx_1]
        mag_res1 = mags_1[mascara_graves][idx_1]
        
        if mag_res1 <= limite_y:
            plt.scatter(f_res1, mag_res1, color='black', s=55, zorder=4)
            plt.annotate(f'({f_res1:.1f} Hz, {mag_res1:.1f} $\Omega$)', 
                         xy=(f_res1, mag_res1), 
                         xytext=(f_res1 * 1.3, mag_res1 * 0.95),
                         fontsize=9, fontweight='bold', color='darkblue',
                         arrowprops=dict(arrowstyle="->", color='black', lw=1.2))
            
        # --- Resonancia Curva 2 (Amarillo) ---
        idx_2 = mags_2[mascara_graves].argmax()
        f_res2 = frecuencias[mascara_graves][idx_2]
        mag_res2 = mags_2[mascara_graves][idx_2]
        
        if mag_res2 <= limite_y:
            plt.scatter(f_res2, mag_res2, color='black', s=55, zorder=4)
            # Desplazamos ligeramente el texto hacia abajo (* 0.85) por si los picos están muy cerca y se solapan
            plt.annotate(f'({f_res2:.1f} Hz, {mag_res2:.1f} $\Omega$)', 
                         xy=(f_res2, mag_res2), 
                         xytext=(f_res2 * 1.3, mag_res2 * 0.82),
                         fontsize=9, fontweight='bold', color='red', # Ajustado a un tono oscuro para leer bien
                         arrowprops=dict(arrowstyle="->", color='black', lw=1.2))

    # Título del gráfico
    plt.title(title, fontsize=13, pad=15)
    
    # Leyenda para identificar qué es cada curva (se posiciona automáticamente en el mejor lugar libre)
    plt.legend(loc="upper right", fontsize=10, framealpha=0.9)
    
    # Rejilla de fondo sutil
    plt.grid(True, which="both", ls=":", alpha=0.4, color='gray', zorder=2)
    
    plt.tight_layout()
    plt.show()
    
frecuencias_sinmasa, magnitud_sinmasa, fase_sinmasa = importar_datos_limp(r"C:\Users\dell_\OneDrive\Desktop\ELECTRO I\TP1 Electro\Electrico\SIN MASA.csv")
frecuencias_conmasa, magnitud_conmasa, fase_conmasa = importar_datos_limp(r"C:\Users\dell_\OneDrive\Desktop\ELECTRO I\TP1 Electro\Electrico\CON MASA.csv")

graficar_ohm(frecuencias_sinmasa, magnitud_sinmasa, title="Curva de impedancia: Parlante sin masa agregada")
graficar_fase(frecuencias_sinmasa, fase_sinmasa, magnitud_sinmasa, title="Comportamiento de fase: Parlante sin masa agregada")
graficar_ohm(frecuencias_conmasa, magnitud_conmasa, title='Curva de impedancia: Parlante con masa agregada 17,2 g')
graficar_fase(frecuencias_conmasa, fase_conmasa, magnitud_conmasa, title="Comportamiento de fase: Parlante con masa agregada 17,2 g")

comparar_impedancias(frecuencias_sinmasa, magnitud_sinmasa, magnitud_conmasa,
                     title='Curvas de impedancia', label_1='Sin masa agregada', label_2='Con masa agregada 17,2 g')