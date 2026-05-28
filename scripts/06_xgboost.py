import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import glob
from tqdm import tqdm

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

BASE_RESULTS_DIR = 'results'
RESULTS_DIR = os.path.join(BASE_RESULTS_DIR, '06_xgboost')
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_DIR = 'data_crudos'

print("\n" + "="*80)
print("🌳 XGBOOST - LEAVE-ONE-LOCATION-OUT CROSS-VALIDATION")
print("="*80 + "\n")

def cargar_datos_por_archivo():
    """Carga datos de todos los CSV y devuelve dict con X y y por archivo"""
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
    datos = {}
    
    print(f"📂 Total de archivos: {len(csv_files)}\n")
    
    for f in csv_files:
        # Extraer posición real del nombre del archivo
        name = os.path.basename(f).replace('.csv', '').replace('uwb_', '')
        try:
            x_real, y_real = map(float, name.split('_'))
        except:
            print(f"⚠️  Saltando {f} (nombre inválido)")
            continue
        
        # Leer CSV
        df = pd.read_csv(f)
        
        # Agrupar cada 3 filas (A1, A2, A3)
        df['group'] = df.index // 3
        
        # Pivotar para obtener una fila por grupo
        pivoted = df.pivot_table(
            index='group',
            columns='anchor',
            values=['distance_m', 'rxPower_dBm'],
            aggfunc='first'
        )
        
        # Extraer distancias
        d_A1 = pivoted[('distance_m', 'A1')].values
        d_A2 = pivoted[('distance_m', 'A2')].values
        d_A3 = pivoted[('distance_m', 'A3')].values
        
        # Extraer potencias
        p_A1 = pivoted[('rxPower_dBm', 'A1')].values
        p_A2 = pivoted[('rxPower_dBm', 'A2')].values
        p_A3 = pivoted[('rxPower_dBm', 'A3')].values
        
        # Crear máscara para eliminar NaN
        mask = ~(np.isnan(d_A1) | np.isnan(d_A2) | np.isnan(d_A3) |
                 np.isnan(p_A1) | np.isnan(p_A2) | np.isnan(p_A3))
        
        # Construir matriz de características (6 columnas)
        X = np.column_stack([d_A1[mask], d_A2[mask], d_A3[mask],
                             p_A1[mask], p_A2[mask], p_A3[mask]])
        
        # Construir matriz objetivo (posición real)
        y = np.full((len(X), 2), [x_real, y_real])
        
        datos[name] = {'X': X, 'y': y}
        
        print(f"  ✓ {name}: {len(X)} muestras")
    
    return datos

def entrenar_fold(X_train, y_train, X_test, y_test):
    """Entrena XGBoost en un fold y devuelve métricas"""
    
    # Normalizar datos
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    
    # Parámetros de XGBoost (sin GPU)
    params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbosity': 0,
        'n_jobs': -1  # Usar todos los cores del CPU
    }
    
    # Entrenar modelos separados para X e Y
    model_x = XGBRegressor(**params)
    model_y = XGBRegressor(**params)
    
    model_x.fit(X_train_norm, y_train[:, 0])
    model_y.fit(X_train_norm, y_train[:, 1])
    
    # Predecir
    y_pred = np.column_stack([
        model_x.predict(X_test_norm),
        model_y.predict(X_test_norm)
    ])
    
    # Calcular métricas
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    errors = np.sqrt((y_test[:, 0] - y_pred[:, 0])**2 + 
                     (y_test[:, 1] - y_pred[:, 1])**2)
    error_promedio = errors.mean()
    
    return rmse, mae, error_promedio

def main():
    # Cargar datos
    datos = cargar_datos_por_archivo()
    archivos = sorted(datos.keys())
    
    print(f"\n{'='*80}")
    print(f"📊 LEAVE-ONE-LOCATION-OUT CROSS-VALIDATION ({len(archivos)} folds)")
    print(f"{'='*80}\n")
    
    resultados = []
    
    # Ejecutar LOLO-CV
    for i, test_archivo in enumerate(archivos, 1):
        # Archivos de entrenamiento (todos menos el actual)
        train_archivos = [a for a in archivos if a != test_archivo]
        
        # Concatenar datos de entrenamiento
        X_train = np.vstack([datos[a]['X'] for a in train_archivos])
        y_train = np.vstack([datos[a]['y'] for a in train_archivos])
        
        # Datos de prueba
        X_test = datos[test_archivo]['X']
        y_test = datos[test_archivo]['y']
        
        print(f"Fold {i}/{len(archivos)}: TEST={test_archivo}")
        print(f"  TRAIN: {len(X_train):,} muestras | TEST: {len(X_test):,} muestras")
        
        # Entrenar y evaluar
        rmse, mae, error_promedio = entrenar_fold(X_train, y_train, X_test, y_test)
        
        resultados.append({
            'fold': i,
            'test_archivo': test_archivo,
            'rmse': rmse,
            'mae': mae,
            'error_promedio': error_promedio
        })
        
        print(f"  → RMSE: {rmse:.6f} m | MAE: {mae:.6f} m | Error promedio: {error_promedio:.6f} m\n")
    
    # Crear DataFrame de resultados
    df = pd.DataFrame(resultados)
    
    # Calcular estadísticas
    rmse_mean = df['rmse'].mean()
    rmse_std = df['rmse'].std()
    mae_mean = df['mae'].mean()
    mae_std = df['mae'].std()
    
    print("="*80)
    print("📊 RESULTADOS FINALES")
    print("="*80)
    print(df.to_string(index=False))
    print(f"\nRMSE promedio: {rmse_mean:.6f} ± {rmse_std:.6f} m")
    print(f"MAE promedio: {mae_mean:.6f} ± {mae_std:.6f} m")
    print("="*80 + "\n")
    
    # Guardar CSV de resultados
    df.to_csv(os.path.join(RESULTS_DIR, 'xgboost_results.csv'), index=False)
    
    # Crear visualizaciones
    fig, ax = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    
    # Panel 1: RMSE por fold
    ax[0, 0].bar(range(1, len(archivos)+1), df['rmse'], color='#27AE60', edgecolor='black', linewidth=1.5)
    ax[0, 0].axhline(rmse_mean, color='red', linestyle='--', linewidth=2, label=f'Media: {rmse_mean:.4f} m')
    ax[0, 0].set_xlabel('Fold (Ubicación)', fontsize=11, fontweight='bold')
    ax[0, 0].set_ylabel('RMSE (m)', fontsize=11, fontweight='bold')
    ax[0, 0].set_title('RMSE por Ubicación', fontsize=12, fontweight='bold')
    ax[0, 0].legend(fontsize=10)
    ax[0, 0].grid(True, alpha=0.3, axis='y')
    ax[0, 0].set_xticks(range(1, len(archivos)+1))
    
    # Panel 2: MAE por fold
    ax[0, 1].bar(range(1, len(archivos)+1), df['mae'], color='#E74C3C', edgecolor='black', linewidth=1.5)
    ax[0, 1].axhline(mae_mean, color='blue', linestyle='--', linewidth=2, label=f'Media: {mae_mean:.4f} m')
    ax[0, 1].set_xlabel('Fold (Ubicación)', fontsize=11, fontweight='bold')
    ax[0, 1].set_ylabel('MAE (m)', fontsize=11, fontweight='bold')
    ax[0, 1].set_title('MAE por Ubicación', fontsize=12, fontweight='bold')
    ax[0, 1].legend(fontsize=10)
    ax[0, 1].grid(True, alpha=0.3, axis='y')
    ax[0, 1].set_xticks(range(1, len(archivos)+1))
    
    # Panel 3: Box plot
    ax[1, 0].boxplot([df['rmse'], df['mae']], labels=['RMSE', 'MAE'], patch_artist=True,
                     boxprops=dict(facecolor='#3498DB', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2))
    ax[1, 0].scatter([1]*len(df['rmse']), df['rmse'], color='#27AE60', alpha=0.6, s=100, label='RMSE')
    ax[1, 0].scatter([2]*len(df['mae']), df['mae'], color='#E74C3C', alpha=0.6, s=100, label='MAE')
    ax[1, 0].set_ylabel('Error (m)', fontsize=11, fontweight='bold')
    ax[1, 0].set_title('Distribución de Errores', fontsize=12, fontweight='bold')
    ax[1, 0].grid(True, alpha=0.3, axis='y')
    ax[1, 0].legend(fontsize=10)
    
    # Panel 4: Tabla de resumen
    ax[1, 1].axis('off')
    tabla_data = [
        ['Métrica', 'Valor'],
        ['RMSE promedio', f'{rmse_mean:.6f} ± {rmse_std:.6f} m'],
        ['MAE promedio', f'{mae_mean:.6f} ± {mae_std:.6f} m'],
        ['RMSE mín', f'{df["rmse"].min():.6f} m'],
        ['RMSE máx', f'{df["rmse"].max():.6f} m'],
        ['MAE mín', f'{df["mae"].min():.6f} m'],
        ['MAE máx', f'{df["mae"].max():.6f} m'],
        ['Folds', f'{len(archivos)}'],
        ['Total muestras', f'{sum(len(datos[a]["X"]) for a in archivos):,}']
    ]
    
    tabla = ax[1, 1].table(cellText=tabla_data, cellLoc='left', loc='center',
                           colWidths=[0.4, 0.6])
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1, 2.5)
    
    # Colorear encabezado
    for i in range(2):
        tabla[(0, i)].set_facecolor('#3498DB')
        tabla[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'xgboost_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Crear tabla detallada
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.axis('off')
    
    tabla_detallada = [['Fold', 'Ubicación Test', 'RMSE (m)', 'MAE (m)', 'Error Promedio (m)']]
    for _, row in df.iterrows():
        tabla_detallada.append([
            f"{int(row['fold'])}",
            row['test_archivo'],
            f"{row['rmse']:.6f}",
            f"{row['mae']:.6f}",
            f"{row['error_promedio']:.6f}"
        ])
    
    tabla_detallada.append(['', 'PROMEDIO', f'{rmse_mean:.6f}', f'{mae_mean:.6f}', ''])
    tabla_detallada.append(['', 'DESV. EST.', f'{rmse_std:.6f}', f'{mae_std:.6f}', ''])
    
    tabla_det = ax.table(cellText=tabla_detallada, cellLoc='center', loc='center',
                         colWidths=[0.1, 0.2, 0.2, 0.2, 0.25])
    tabla_det.auto_set_font_size(False)
    tabla_det.set_fontsize(10)
    tabla_det.scale(1, 2)
    
    # Colorear encabezado
    for i in range(5):
        tabla_det[(0, i)].set_facecolor('#27AE60')
        tabla_det[(0, i)].set_text_props(weight='bold', color='white')
    
    # Colorear filas de estadísticas
    for i in range(5):
        tabla_det[(len(tabla_detallada)-2, i)].set_facecolor('#F39C12')
        tabla_det[(len(tabla_detallada)-1, i)].set_facecolor('#F39C12')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'xgboost_tabla_detallada.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Gráficos guardados en {RESULTS_DIR}")
    print(f"✓ xgboost_results.csv")
    print(f"✓ xgboost_results.png")
    print(f"✓ xgboost_tabla_detallada.png")
    print("\n" + "="*80)
    print(f"✅ COMPLETADO | Resultados: {RESULTS_DIR}")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
