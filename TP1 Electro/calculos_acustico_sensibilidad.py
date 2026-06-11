from funciones_electro import importar_datos_smaart, graficar_db, comparar_db
import numpy as np

frecs_filtrado, spl_filtrado = importar_datos_smaart(r"Acústico\fs a 10fs.txt")
spl_filtrado_corregido = np.array(spl_filtrado) - 6

frecs_completo, spl_completo = importar_datos_smaart(r'Acústico\sin filtrar.txt')