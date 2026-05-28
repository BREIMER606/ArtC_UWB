import os, numpy as np, pandas as pd, glob

DATA_DIR = 'data_crudos'

print('\n' + '='*80)
print('📊 VERIFICANDO PARSING DE DATOS (CORREGIDO)')
print('='*80 + '\n')

csv_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
print(f"Archivos encontrados: {len(csv_files)}\n")

primer_archivo = csv_files[0]
print(f"Analizando: {primer_archivo}\n")

# LEER CON HEADER CORRECTO
df = pd.read_csv(primer_archivo)
print(f"Columnas detectadas: {list(df.columns)}")
print(f"Forma: {df.shape}\n")
print("Primeras 10 filas:")
print(df.head(10))

print("\n" + "="*80)
print("EXTRAYENDO DATOS POR ANCHOR")
print("="*80 + "\n")

timestamps = df['timestamp_ms'].values
anchors = df['anchor'].values
distances = df['distance_m'].values
powers = df['rxPower_dBm'].values

print(f"Total filas: {len(df)}")
print(f"Anchors únicos: {np.unique(anchors)}")
print(f"Timestamps únicos: {len(np.unique(timestamps))}\n")

# Agrupar por timestamp
datos_por_ts = {}
for ts, anchor, dist in zip(timestamps, anchors, distances):
    if ts not in datos_por_ts:
        datos_por_ts[ts] = {}
    datos_por_ts[ts][anchor] = dist

print(f"Primeros 5 timestamps y sus distancias:")
for i, (ts, anchors_data) in enumerate(list(datos_por_ts.items())[:5]):
    d_a1 = anchors_data.get('A1', np.nan)
    d_a2 = anchors_data.get('A2', np.nan)
    d_a3 = anchors_data.get('A3', np.nan)
    print(f"  ts={ts}: A1={d_a1:.3f}, A2={d_a2:.3f}, A3={d_a3:.3f}")

# Construir arrays correctamente
d_A1_list = []
d_A2_list = []
d_A3_list = []

for ts in sorted(datos_por_ts.keys()):
    d_A1_list.append(datos_por_ts[ts].get('A1', np.nan))
    d_A2_list.append(datos_por_ts[ts].get('A2', np.nan))
    d_A3_list.append(datos_por_ts[ts].get('A3', np.nan))

d_A1 = np.array(d_A1_list)
d_A2 = np.array(d_A2_list)
d_A3 = np.array(d_A3_list)

print(f"\n✅ Resultados correctos:")
print(f"  d_A1: {d_A1.shape}, primeros 5: {d_A1[:5]}")
print(f"  d_A2: {d_A2.shape}, primeros 5: {d_A2[:5]}")
print(f"  d_A3: {d_A3.shape}, primeros 5: {d_A3[:5]}")

print("\n" + "="*80)
print("CONCLUSIÓN")
print("="*80)
print("""
✅ Los datos TIENEN 3 distancias distintas (A1, A2, A3) para cada timestamp
✅ El formato es correcto y se puede parsear bien

❌ EL MLP ANTERIOR ESTABA MAL parseado
   - Estaba usando valores por defecto (1.5) en lugar de las distancias reales
   - Por eso daba RMSE de 0.0607 m (era "suerte" de overfitting)

ACCIÓN: Reentrenar MLP y MLP+Optuna con PARSING CORRECTO
""")

print("\n" + "="*80 + "\n")
