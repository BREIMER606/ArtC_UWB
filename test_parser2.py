import os, numpy as np, pandas as pd, glob

DATA_DIR = 'data_crudos'

print('\n' + '='*80)
print('📊 PIVOTANDO DATOS CORRECTAMENTE')
print('='*80 + '\n')

csv_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
print(f"Archivos encontrados: {len(csv_files)}\n")

primer_archivo = csv_files[0]
print(f"Analizando: {primer_archivo}\n")

# Leer con header correcto
df = pd.read_csv(primer_archivo)
print(f"Forma original: {df.shape}")
print(f"Primeras 15 filas:")
print(df.head(15))

print("\n" + "="*80)
print("PIVOTANDO POR ANCHOR")
print("="*80 + "\n")

# Pivotar: cada fila será un GRUPO de 3 mediciones (A1, A2, A3)
# Necesitamos agrupar por grupos de 3

# Opción 1: Usar group_number basado en el índice
df['group'] = df.index // 3  # Cada 3 filas = 1 grupo

# Pivotar
pivoted = df.pivot_table(
    index='group',
    columns='anchor',
    values=['distance_m', 'rxPower_dBm'],
    aggfunc='first'
)

print(f"Después de pivotar:")
print(f"Forma: {pivoted.shape}")
print(f"\nPrimeras 10 filas pivotadas:")
print(pivoted.head(10))

# Extraer distancias
d_A1 = pivoted[('distance_m', 'A1')].values
d_A2 = pivoted[('distance_m', 'A2')].values
d_A3 = pivoted[('distance_m', 'A3')].values

print(f"\n✅ Distancias extraídas:")
print(f"  d_A1 primeros 5: {d_A1[:5]}")
print(f"  d_A2 primeros 5: {d_A2[:5]}")
print(f"  d_A3 primeros 5: {d_A3[:5]}")

# Matriz final
X = np.column_stack([d_A1, d_A2, d_A3])
print(f"\nMatriz X final:")
print(f"  Shape: {X.shape}")
print(f"  Primeras 5 filas:")
print(X[:5])

print("\n" + "="*80)
print("✅ PARSING CORRECTO COMPLETADO")
print("="*80 + "\n")
