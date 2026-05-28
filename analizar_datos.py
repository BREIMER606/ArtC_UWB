import os, numpy as np, pandas as pd, glob
from sklearn.preprocessing import StandardScaler

DATA_DIR = 'data_crudos'
BASE_RESULTS_DIR = 'results'

print('\n' + '='*80)
print('📊 ANÁLISIS DETALLADO DE DATOS POR SET')
print('='*80 + '\n')

# Cargar datos
print("📂 Cargando datos crudos...\n")
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
X_list, y_list, timestamps_list, filenames_list = [], [], [], []

for csv_file in csv_files:
    filename = os.path.basename(csv_file).replace('.csv', '').replace('uwb_', '')
    coords = filename.split('_')
    if len(coords) == 2:
        x_real, y_real = float(coords[0]), float(coords[1])
        df = pd.read_csv(csv_file)
        
        cols = df.columns.tolist()
        d_a1 = df[[c for c in cols if 'A1' in c or c == 'distance_m']].iloc[:, 0].values
        d_a2 = df[[c for c in cols if 'A2' in c]].iloc[:, 0].values if any('A2' in c for c in cols) else np.full(len(df), 1.5)
        d_a3 = df[[c for c in cols if 'A3' in c]].iloc[:, 0].values if any('A3' in c for c in cols) else np.full(len(df), 1.5)
        
        X_data = np.column_stack([d_a1, d_a2, d_a3])
        y_data = np.full((len(df), 2), [x_real, y_real])
        
        X_list.append(X_data)
        y_list.append(y_data)
        filenames_list.extend([filename] * len(df))
        
        # Obtener timestamps si existen
        if 'timestamp' in cols or 'Timestamp' in cols:
            ts_col = [c for c in cols if 'timestamp' in c.lower()][0]
            timestamps_list.extend(df[ts_col].values)
        else:
            timestamps_list.extend(range(len(df)))

X = np.vstack(X_list)
y = np.vstack(y_list)
timestamps = np.array(timestamps_list)
filenames = np.array(filenames_list)

print(f"✓ Total muestras cargadas: {len(X)}\n")

# Mostrar primeras y últimas muestras
print("="*80)
print("📍 PRIMERAS 10 MUESTRAS")
print("="*80)
for i in range(min(10, len(X))):
    print(f"  [{i:5d}] {filenames[i]:20s} | Pos: ({y[i,0]:.3f}, {y[i,1]:.3f}) | Dist: ({X[i,0]:.3f}, {X[i,1]:.3f}, {X[i,2]:.3f})")

print("\n" + "="*80)
print("📍 ÚLTIMAS 10 MUESTRAS")
print("="*80)
start_idx = max(0, len(X) - 10)
for i in range(start_idx, len(X)):
    print(f"  [{i:5d}] {filenames[i]:20s} | Pos: ({y[i,0]:.3f}, {y[i,1]:.3f}) | Dist: ({X[i,0]:.3f}, {X[i,1]:.3f}, {X[i,2]:.3f})")

# Split 60/20/20
n_train = int(0.6 * len(X))
n_val = int(0.2 * len(X))

print("\n" + "="*80)
print("🔀 SPLIT DE DATOS")
print("="*80)
print(f"\nTRAIN: índices 0 a {n_train-1} ({n_train} muestras)")
print(f"VAL:   índices {n_train} a {n_train+n_val-1} ({n_val} muestras)")
print(f"TEST:  índices {n_train+n_val} a {len(X)-1} ({len(X)-n_train-n_val} muestras)")

# Contar por archivo
print("\n" + "="*80)
print("📁 MUESTRAS POR ARCHIVO EN CADA SET")
print("="*80)

train_idx = list(range(0, n_train))
val_idx = list(range(n_train, n_train+n_val))
test_idx = list(range(n_train+n_val, len(X)))

for set_name, idx_list in [("TRAIN", train_idx), ("VAL", val_idx), ("TEST", test_idx)]:
    print(f"\n{set_name}:")
    files_in_set = filenames[idx_list]
    unique_files, counts = np.unique(files_in_set, return_counts=True)
    for file, count in zip(unique_files, counts):
        print(f"  {file:20s}: {count:5d}")

print("\n" + "="*80)
print("✅ ANÁLISIS COMPLETADO")
print("="*80 + "\n")
