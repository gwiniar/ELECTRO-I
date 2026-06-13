from funciones_electro import importar_datos_limp, graficar_ohm, graficar_fase, comparar_impedancias, detectar_resonancia

frecuencias_sinmasa, magnitud_sinmasa, fase_sinmasa = importar_datos_limp(r"Electrico\SIN MASA.csv")
frecuencias_conmasa, magnitud_conmasa, fase_conmasa = importar_datos_limp(r"Electrico\CON MASA.csv")

fs_sinmasa, valorfs_sinmasa = detectar_resonancia(frecuencias_sinmasa, magnitud_sinmasa)
fs_conmasa, valorfs_conmasa = detectar_resonancia(frecuencias_conmasa, magnitud_conmasa)

graficar_ohm(frecuencias_sinmasa, magnitud_sinmasa, title="Curva de impedancia: Parlante sin masa agregada")
graficar_fase(frecuencias_sinmasa, fase_sinmasa, magnitud_sinmasa, title="Comportamiento de fase: Parlante sin masa agregada")
graficar_ohm(frecuencias_conmasa, magnitud_conmasa, title='Curva de impedancia: Parlante con masa agregada 17,2 g')
graficar_fase(frecuencias_conmasa, fase_conmasa, magnitud_conmasa, title="Comportamiento de fase: Parlante con masa agregada 17,2 g")

comparar_impedancias(frecuencias_sinmasa, magnitud_sinmasa, magnitud_conmasa,
                     title='Curvas de impedancia', label_1='Sin masa agregada', label_2='Con masa agregada 17,2 g')

print(f'RESONANCIA sin masa = ({fs_sinmasa} Hz ; {valorfs_sinmasa} Ohm)')
print(f'RESONANCIA con masa = ({fs_conmasa} Hz ; {valorfs_conmasa} Ohm)')

