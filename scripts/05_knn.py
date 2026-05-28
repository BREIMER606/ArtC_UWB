import os, warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
import glob

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

BASE_RESULTS_DIR = 'results'
RESULTS_DIR = os.path.join(BASE_RESULTS_DIR, '05_knn')
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_DIR = 'data_crudos'

def cargar_datos_por_archivo():
    """Carga datos correctamente parseados por archivo"""
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
    datos_por_archivo = {}
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file).replace('.csv', '').replace('uwb_', '')
        coords = filename.split('_')
        
        if len(coords) == 2:
            x_real, y_real = float(coords[0]), float(coords[1])
            df = pd.read_csv(csv_file)
            
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
            
            valid_mask = ~(np.isnan(d_A1) | np.isnan(d_A2) | np.isnan(d_A3) | 
                          np.isnan(p_A1) | np.isnan(p_A2) | np.isnan(p_A3))
            
            X_data = np.column_stack([d_A1[valid_mask], d_A2[valid_mask], d_A3[valid_mask],
                                     p_A1[valid_mask], p_A2[valid_mask], p_A3[valid_mask]])
            y_data = np.full((len(X_data), 2), [x_real, y_real])
            
            datos_por_archivo[filename] = {
                'X': X_data,
                'y': y_data,
                'n_muestras': len(X_data)
            }
    
    return datos_por_archivo

def entrenar_fold(X_train, y_train, X_test, y_test, k=5):
    """Entrena KNN"""
    
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    
    # KNN multisalida (regresa [x, y] simultáneamente)
    model = KNeighborsRegressor(n_neighbors=k, n_jobs=-1)
    model.fit(X_train_norm, y_train)
    y_pred = model.predict(X_test_norm)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    errors = np.sqrt((y_test[:, 0] - y_pred[:, 0])**2 + (y_test[:, 1] - y_pred[:, 1])**2)
    
    return rmse, mae, errors.mean()

def main():
    print('\n' + '='*80)
    print('🔍 KNN (K-NEAREST NEIGHBORS) - LEAVE-ONE-LOCATION-OUT CV')
    print('='*80 + '\n')
    
    datos_por_archivo = cargar_datos_por_archivo()
    archivos = sorted(datos_por_archivo.keys())
    
    print(f"Total archivos (localizaciones): {len(archivos)}")
    print(f"Features: 6 (d_A1, d_A2, d_A3, power_A1, power_A2, power_A3)")
    print(f"K value: 5\n")
    
    resultados_folds = []
    
    for fold, test_archivo in enumerate(archivos, 1):
        print(f"Fold {fold}/{len(archivos)}: TEST={test_archivo}")
        
        train_archivos = [a for a in archivos if a != test_archivo]
        X_train = np.vstack([datos_por_archivo[a]['X'] for a in train_archivos])
        y_train = np.vstack([datos_por_archivo[a]['y'] for a in train_archivos])
        X_test = datos_por_archivo[test_archivo]['X']
        y_test = datos_por_archivo[test_archivo]['y']
        
        print(f"  TRAIN: {len(X_train)} muestras | TEST: {len(X_test)} muestras")
        
        rmse, mae, error_prom = entrenar_fold(X_train, y_train, X_test, y_test, k=5)
        
        print(f"  → RMSE: {rmse:.6f} m | MAE: {mae:.6f} m\n")
        
        resultados_folds.append({
            'Fold': fold,
            'Test Location': test_archivo,
            'RMSE (m)': rmse,
            'MAE (m)': mae,
            'Error Promedio (m)': error_prom
        })
    
    print("="*80)
    print("📊 RESULTADOS KNN - LOLO-CV")
    print("="*80 + "\n")
    
    resultados_df = pd.DataFrame(resultados_folds)
    print(resultados_df.to_string(index=False))
    print()
    
    rmse_medio = resultados_df['RMSE (m)'].mean()
    rmse_std = resultados_df['RMSE (m)'].std()
    mae_medio = resultados_df['MAE (m)'].mean()
    mae_std = resultados_df['MAE (m)'].std()
    
    print(f"\nRMSE promedio: {rmse_medio:.6f} ± {rmse_std:.6f} m")
    print(f"MAE promedio:  {mae_medio:.6f} ± {mae_std:.6f} m")
    print("="*80 + "\n")
    
    # Guardar CSV
    resultados_df.to_csv(os.path.join(RESULTS_DIR, 'knn_results.csv'), index=False)
    print(f"✓ CSV guardado: {os.path.join(RESULTS_DIR, 'knn_results.csv')}\n")
    
    # GRÁFICO 1: Barras de RMSE
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    
    # Subplot 1: RMSE por fold
    axes[0, 0].bar(range(1, len(archivos)+1), resultados_df['RMSE (m)'], 
                   color='#3498DB', alpha=0.7, edgecolor='black', linewidth=2)
    axes[0, 0].axhline(rmse_medio, color='red', linestyle='--', linewidth=2.5, 
                       label=f'Media: {rmse_medio:.4f} m')
    axes[0, 0].fill_between(range(0, len(archivos)+1), 
                            rmse_medio - rmse_std, rmse_medio + rmse_std, 
                            alpha=0.2, color='red', label=f'±1 Std: ±{rmse_std:.4f} m')
    axes[0, 0].set_xlabel('Fold (Location)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('RMSE (m)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('KNN - RMSE por Location', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_ylim(0, max(resultados_df['RMSE (m)']) * 1.2)
    
    # Subplot 2: MAE por fold
    axes[0, 1].bar(range(1, len(archivos)+1), resultados_df['MAE (m)'], 
                   color='#2ECC71', alpha=0.7, edgecolor='black', linewidth=2)
    axes[0, 1].axhline(mae_medio, color='red', linestyle='--', linewidth=2.5, 
                       label=f'Media: {mae_medio:.4f} m')
    axes[0, 1].set_xlabel('Fold (Location)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('MAE (m)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('KNN - MAE por Location', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_ylim(0, max(resultados_df['MAE (m)']) * 1.2)
    
    # Subplot 3: Box plot
    bp = axes[1, 0].boxplot([resultados_df['RMSE (m)'], resultados_df['MAE (m)']], 
                             labels=['RMSE', 'MAE'], patch_artist=True,
                             widths=0.6, showmeans=True)
    for patch, color in zip(bp['boxes'], ['#3498DB', '#2ECC71']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1, 0].set_ylabel('Error (m)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('KNN - Distribución de Errores', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Subplot 4: Tabla de resultados
    axes[1, 1].axis('off')
    
    # Crear tabla para visualizar
    tabla_data = []
    tabla_data.append(['Métrica', 'Valor'])
    tabla_data.append(['RMSE Promedio', f'{rmse_medio:.6f} m'])
    tabla_data.append(['RMSE Std Dev', f'{rmse_std:.6f} m'])
    tabla_data.append(['MAE Promedio', f'{mae_medio:.6f} m'])
    tabla_data.append(['MAE Std Dev', f'{mae_std:.6f} m'])
    tabla_data.append(['RMSE Mínimo', f'{resultados_df["RMSE (m)"].min():.6f} m'])
    tabla_data.append(['RMSE Máximo', f'{resultados_df["RMSE (m)"].max():.6f} m'])
    tabla_data.append(['Total Folds', f'{len(archivos)}'])
    tabla_data.append(['Tipo Validación', 'LOLO-CV'])
    tabla_data.append(['K Neighbors', '5'])
    tabla_data.append(['Features', '6 (dist+power)'])
    
    tabla = axes[1, 1].table(cellText=tabla_data, cellLoc='left', loc='center',
                             colWidths=[0.5, 0.5])
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(11)
    tabla.scale(1, 2.5)
    
    # Colorear header
    for i in range(2):
        tabla[(0, i)].set_facecolor('#34495E')
        tabla[(0, i)].set_text_props(weight='bold', color='white')
    
    # Colorear filas alternadas
    for i in range(1, len(tabla_data)):
        for j in range(2):
            if i % 2 == 0:
                tabla[(i, j)].set_facecolor('#ECF0F1')
            else:
                tabla[(i, j)].set_facecolor('#FFFFFF')
    
    axes[1, 1].set_title('Resumen de Métricas', fontsize=13, fontweight='bold', pad=20)
    
    plt.suptitle('KNN - Leave-One-Location-Out Cross-Validation', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'knn_results.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico guardado: {os.path.join(RESULTS_DIR, 'knn_results.png')}\n")
    plt.close()
    
    # GRÁFICO 2: Tabla detallada en imagen
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.axis('tight')
    ax.axis('off')
    
    # Preparar datos para tabla
    tabla_folds = []
    tabla_folds.append(['Fold', 'Test Location', 'RMSE (m)', 'MAE (m)', 'Avg Error (m)'])
    
    for idx, row in resultados_df.iterrows():
        tabla_folds.append([
            str(int(row['Fold'])),
            row['Test Location'],
            f"{row['RMSE (m)']:.6f}",
            f"{row['MAE (m)']:.6f}",
            f"{row['Error Promedio (m)']:.6f}"
        ])
    
    # Agregar fila de promedios
    tabla_folds.append(['PROMEDIO', '', 
                        f'{rmse_medio:.6f}',
                        f'{mae_medio:.6f}',
                        f'{resultados_df["Error Promedio (m)"].mean():.6f}'])
    
    tabla_detail = ax.table(cellText=tabla_folds, cellLoc='center', loc='center',
                            colWidths=[0.1, 0.25, 0.2, 0.2, 0.2])
    tabla_detail.auto_set_font_size(False)
    tabla_detail.set_fontsize(11)
    tabla_detail.scale(1, 2.2)
    
    # Colorear header
    for i in range(5):
        tabla_detail[(0, i)].set_facecolor('#2C3E50')
        tabla_detail[(0, i)].set_text_props(weight='bold', color='white')
    
    # Colorear fila de promedio
    for i in range(5):
        tabla_detail[(len(tabla_folds)-1, i)].set_facecolor('#F39C12')
        tabla_detail[(len(tabla_folds)-1, i)].set_text_props(weight='bold')
    
    # Colorear filas alternadas
    for i in range(1, len(tabla_folds)-1):
        for j in range(5):
            if i % 2 == 0:
                tabla_detail[(i, j)].set_facecolor('#ECF0F1')
            else:
                tabla_detail[(i, j)].set_facecolor('#FFFFFF')
    
    plt.suptitle('KNN - Resultados Detallados LOLO-CV', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(os.path.join(RESULTS_DIR, 'knn_tabla_detallada.png'), 
                dpi=300, bbox_inches='tight')
    print(f"✓ Tabla detallada guardada: {os.path.join(RESULTS_DIR, 'knn_tabla_detallada.png')}\n")
    plt.close()
    
    print("="*80)
    print(f"✅ ENTRENAMIENTO COMPLETADO")
    print(f"📁 Resultados en: {RESULTS_DIR}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
