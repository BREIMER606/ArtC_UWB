import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import glob

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

BASE_RESULTS_DIR = 'results'
RESULTS_DIR = os.path.join(BASE_RESULTS_DIR, '08_arboles')
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_DIR = 'data_crudos'

print("\n" + "="*80)
print("🌳 ÁRBOLES DE DECISIÓN vs RANDOM FOREST - LOLO-CV")
print("="*80 + "\n")

def cargar_datos_por_archivo():
    """Carga datos de todos los CSV"""
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
    datos = {}
    
    print(f"📂 Total de archivos: {len(csv_files)}\n")
    
    for f in csv_files:
        name = os.path.basename(f).replace('.csv', '').replace('uwb_', '')
        try:
            x_real, y_real = map(float, name.split('_'))
        except:
            continue
        
        df = pd.read_csv(f)
        df['group'] = df.index // 3
        
        pivoted = df.pivot_table(
            index='group',
            columns='anchor',
            values=['distance_m', 'rxPower_dBm'],
            aggfunc='first'
        )
        
        d_A1 = pivoted[('distance_m', 'A1')].values
        d_A2 = pivoted[('distance_m', 'A2')].values
        d_A3 = pivoted[('distance_m', 'A3')].values
        p_A1 = pivoted[('rxPower_dBm', 'A1')].values
        p_A2 = pivoted[('rxPower_dBm', 'A2')].values
        p_A3 = pivoted[('rxPower_dBm', 'A3')].values
        
        mask = ~(np.isnan(d_A1) | np.isnan(d_A2) | np.isnan(d_A3) |
                 np.isnan(p_A1) | np.isnan(p_A2) | np.isnan(p_A3))
        
        X = np.column_stack([d_A1[mask], d_A2[mask], d_A3[mask],
                             p_A1[mask], p_A2[mask], p_A3[mask]])
        y = np.full((len(X), 2), [x_real, y_real])
        
        datos[name] = {'X': X, 'y': y}
        print(f"  ✓ {name}: {len(X)} muestras")
    
    return datos

def entrenar_fold_arbol(X_train, y_train, X_test, y_test, modelo_tipo='dt'):
    """Entrena un árbol de decisión o Random Forest"""
    
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    
    if modelo_tipo == 'dt':
        # Árbol de Decisión simple
        model_x = DecisionTreeRegressor(
            max_depth=8,              # Limitar profundidad
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        model_y = DecisionTreeRegressor(
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
    else:
        # Random Forest optimizado (menos árboles, menor profundidad)
        model_x = RandomForestRegressor(
            n_estimators=100,          # Reducido de 200
            max_depth=10,              # Reducido de 20
            min_samples_split=10,      # Aumentado de 5
            min_samples_leaf=5,        # Aumentado de 2
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        model_y = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
    
    model_x.fit(X_train_norm, y_train[:, 0])
    model_y.fit(X_train_norm, y_train[:, 1])
    
    y_pred = np.column_stack([
        model_x.predict(X_test_norm),
        model_y.predict(X_test_norm)
    ])
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    errors = np.sqrt((y_test[:, 0] - y_pred[:, 0])**2 + 
                     (y_test[:, 1] - y_pred[:, 1])**2)
    error_promedio = errors.mean()
    
    return rmse, mae, error_promedio

def ejecutar_lolo(datos, archivos, modelo_tipo='dt'):
    """Ejecuta Leave-One-Location-Out CV"""
    
    nombre_modelo = 'Árbol de Decisión' if modelo_tipo == 'dt' else 'Random Forest Optimizado'
    print(f"\n{'='*80}")
    print(f"📊 {nombre_modelo.upper()} - LOLO-CV")
    print(f"{'='*80}\n")
    
    resultados = []
    
    for i, test_archivo in enumerate(archivos, 1):
        train_archivos = [a for a in archivos if a != test_archivo]
        
        X_train = np.vstack([datos[a]['X'] for a in train_archivos])
        y_train = np.vstack([datos[a]['y'] for a in train_archivos])
        
        X_test = datos[test_archivo]['X']
        y_test = datos[test_archivo]['y']
        
        print(f"Fold {i}/{len(archivos)}: TEST={test_archivo}")
        
        rmse, mae, error_promedio = entrenar_fold_arbol(X_train, y_train, X_test, y_test, modelo_tipo)
        
        resultados.append({
            'fold': i,
            'test_archivo': test_archivo,
            'rmse': rmse,
            'mae': mae,
            'error_promedio': error_promedio
        })
        
        print(f"  → RMSE: {rmse:.6f} m\n")
    
    return pd.DataFrame(resultados)

def crear_visualizaciones(df_dt, df_rf, archivos):
    """Crea gráficos comparativos"""
    
    fig, ax = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    
    # Panel 1: Comparación RMSE
    x = np.arange(len(archivos))
    width = 0.35
    ax[0, 0].bar(x - width/2, df_dt['rmse'], width, label='Decision Tree', color='#3498DB', edgecolor='black')
    ax[0, 0].bar(x + width/2, df_rf['rmse'], width, label='Random Forest', color='#E74C3C', edgecolor='black')
    ax[0, 0].axhline(df_dt['rmse'].mean(), color='#3498DB', linestyle='--', linewidth=2, alpha=0.7)
    ax[0, 0].axhline(df_rf['rmse'].mean(), color='#E74C3C', linestyle='--', linewidth=2, alpha=0.7)
    ax[0, 0].set_xlabel('Fold (Ubicación)', fontsize=11, fontweight='bold')
    ax[0, 0].set_ylabel('RMSE (m)', fontsize=11, fontweight='bold')
    ax[0, 0].set_title('RMSE - Comparación de Modelos', fontsize=12, fontweight='bold')
    ax[0, 0].set_xticks(x)
    ax[0, 0].set_xticklabels(range(1, len(archivos)+1))
    ax[0, 0].legend(fontsize=10)
    ax[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Panel 2: Comparación MAE
    ax[0, 1].bar(x - width/2, df_dt['mae'], width, label='Decision Tree', color='#2ECC71', edgecolor='black')
    ax[0, 1].bar(x + width/2, df_rf['mae'], width, label='Random Forest', color='#F39C12', edgecolor='black')
    ax[0, 1].axhline(df_dt['mae'].mean(), color='#2ECC71', linestyle='--', linewidth=2, alpha=0.7)
    ax[0, 1].axhline(df_rf['mae'].mean(), color='#F39C12', linestyle='--', linewidth=2, alpha=0.7)
    ax[0, 1].set_xlabel('Fold (Ubicación)', fontsize=11, fontweight='bold')
    ax[0, 1].set_ylabel('MAE (m)', fontsize=11, fontweight='bold')
    ax[0, 1].set_title('MAE - Comparación de Modelos', fontsize=12, fontweight='bold')
    ax[0, 1].set_xticks(x)
    ax[0, 1].set_xticklabels(range(1, len(archivos)+1))
    ax[0, 1].legend(fontsize=10)
    ax[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Box plots
    data_dt = [df_dt['rmse'], df_dt['mae']]
    data_rf = [df_rf['rmse'], df_rf['mae']]
    positions = [1, 2, 4, 5]
    ax[1, 0].boxplot([df_dt['rmse'], df_rf['rmse'], df_dt['mae'], df_rf['mae']], 
                     positions=positions, labels=['DT\nRMSE', 'RF\nRMSE', 'DT\nMAE', 'RF\nMAE'],
                     patch_artist=True, widths=0.6)
    ax[1, 0].set_ylabel('Error (m)', fontsize=11, fontweight='bold')
    ax[1, 0].set_title('Distribución de Errores', fontsize=12, fontweight='bold')
    ax[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Tabla resumen
    ax[1, 1].axis('off')
    tabla_data = [
        ['Modelo', 'RMSE', 'MAE'],
        ['Decision Tree', f'{df_dt["rmse"].mean():.4f}±{df_dt["rmse"].std():.4f}', 
         f'{df_dt["mae"].mean():.4f}±{df_dt["mae"].std():.4f}'],
        ['Random Forest', f'{df_rf["rmse"].mean():.4f}±{df_rf["rmse"].std():.4f}', 
         f'{df_rf["mae"].mean():.4f}±{df_rf["mae"].std():.4f}'],
    ]
    
    tabla = ax[1, 1].table(cellText=tabla_data, cellLoc='center', loc='center',
                           colWidths=[0.35, 0.325, 0.325])
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1, 3)
    
    for i in range(3):
        tabla[(0, i)].set_facecolor('#3498DB')
        tabla[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'arboles_comparacion.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    datos = cargar_datos_por_archivo()
    archivos = sorted(datos.keys())
    
    # Ejecutar Decision Tree
    df_dt = ejecutar_lolo(datos, archivos, modelo_tipo='dt')
    rmse_dt = df_dt['rmse'].mean()
    std_dt = df_dt['rmse'].std()
    print(f"\n✓ Decision Tree RMSE: {rmse_dt:.6f} ± {std_dt:.6f} m")
    
    # Ejecutar Random Forest optimizado
    df_rf = ejecutar_lolo(datos, archivos, modelo_tipo='rf')
    rmse_rf = df_rf['rmse'].mean()
    std_rf = df_rf['rmse'].std()
    print(f"✓ Random Forest RMSE: {rmse_rf:.6f} ± {std_rf:.6f} m")
    
    # Guardar resultados
    df_dt.to_csv(os.path.join(RESULTS_DIR, 'decision_tree_results.csv'), index=False)
    df_rf.to_csv(os.path.join(RESULTS_DIR, 'random_forest_optimizado_results.csv'), index=False)
    
    # Crear visualizaciones
    crear_visualizaciones(df_dt, df_rf, archivos)
    
    print("\n" + "="*80)
    print("📊 RESUMEN FINAL")
    print("="*80)
    print(f"Decision Tree:        RMSE = {rmse_dt:.6f} ± {std_dt:.6f} m")
    print(f"Random Forest (opt):  RMSE = {rmse_rf:.6f} ± {std_rf:.6f} m")
    
    mejor = "Decision Tree" if rmse_dt < rmse_rf else "Random Forest"
    print(f"\n✅ Mejor modelo: {mejor}")
    print(f"✓ Archivos guardados en {RESULTS_DIR}")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
