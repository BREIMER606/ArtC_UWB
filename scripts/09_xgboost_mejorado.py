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

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

BASE_RESULTS_DIR = 'results'
RESULTS_DIR = os.path.join(BASE_RESULTS_DIR, '09_xgboost_mejorado')
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_DIR = 'data_crudos'

print("\n" + "="*80)
print("🔍 DIAGNÓSTICO Y XGBOOST MEJORADO")
print("="*80 + "\n")

def cargar_datos_por_archivo():
    """Carga datos de todos los CSV"""
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
    datos = {}
    
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
        
        X_6feat = np.column_stack([d_A1[mask], d_A2[mask], d_A3[mask],
                                   p_A1[mask], p_A2[mask], p_A3[mask]])
        X_3feat = np.column_stack([d_A1[mask], d_A2[mask], d_A3[mask]])
        y = np.full((len(X_6feat), 2), [x_real, y_real])
        
        datos[name] = {'X_6feat': X_6feat, 'X_3feat': X_3feat, 'y': y}
    
    return datos

def entrenar_fold(X_train, y_train, X_test, y_test, config_name=''):
    """Entrena XGBoost optimizado"""
    
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    
    # XGBoost optimizado (parámetros más conservadores)
    params = {
        'n_estimators': 300,
        'max_depth': 4,              # MÁS PEQUEÑO para evitar overfitting
        'learning_rate': 0.05,       # MÁS PEQUEÑO
        'subsample': 0.7,            # MENOS que antes
        'colsample_bytree': 0.7,     # MENOS que antes
        'min_child_weight': 5,       # NUEVO: regularización
        'gamma': 0.1,                # NUEVO: penaliza splits complejos
        'reg_alpha': 0.1,            # NUEVO: L1 regularization
        'reg_lambda': 1.0,           # NUEVO: L2 regularization
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    model_x = XGBRegressor(**params)
    model_y = XGBRegressor(**params)
    
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
    
    return rmse, mae, error_promedio, model_x, model_y

def ejecutar_lolo(datos, archivos, feature_type='6feat'):
    """Ejecuta Leave-One-Location-Out CV"""
    
    tipo_nombre = "6 Features (distancias + potencias)" if feature_type == '6feat' else "3 Features (solo distancias)"
    print(f"\n{'='*80}")
    print(f"📊 XGBOOST MEJORADO - {tipo_nombre}")
    print(f"{'='*80}\n")
    
    resultados = []
    
    for i, test_archivo in enumerate(archivos, 1):
        train_archivos = [a for a in archivos if a != test_archivo]
        
        X_key = f'X_{feature_type}'
        X_train = np.vstack([datos[a][X_key] for a in train_archivos])
        y_train = np.vstack([datos[a]['y'] for a in train_archivos])
        
        X_test = datos[test_archivo][X_key]
        y_test = datos[test_archivo]['y']
        
        print(f"Fold {i}/{len(archivos)}: TEST={test_archivo}")
        
        rmse, mae, error_promedio, _, _ = entrenar_fold(X_train, y_train, X_test, y_test, tipo_nombre)
        
        resultados.append({
            'fold': i,
            'test_archivo': test_archivo,
            'rmse': rmse,
            'mae': mae,
            'error_promedio': error_promedio
        })
        
        print(f"  → RMSE: {rmse:.6f} m\n")
    
    return pd.DataFrame(resultados)

def main():
    datos = cargar_datos_por_archivo()
    archivos = sorted(datos.keys())
    
    print("\n📋 DIAGNOSTICANDO CONFIGURACIONES:\n")
    
    # Ejecutar con 6 features
    print("Probando con 6 features (distancias + potencias)...")
    df_6feat = ejecutar_lolo(datos, archivos, feature_type='6feat')
    rmse_6feat = df_6feat['rmse'].mean()
    std_6feat = df_6feat['rmse'].std()
    
    # Ejecutar con 3 features
    print("Probando con 3 features (solo distancias)...")
    df_3feat = ejecutar_lolo(datos, archivos, feature_type='3feat')
    rmse_3feat = df_3feat['rmse'].mean()
    std_3feat = df_3feat['rmse'].std()
    
    # Guardar resultados
    df_6feat.to_csv(os.path.join(RESULTS_DIR, 'xgboost_6feat_results.csv'), index=False)
    df_3feat.to_csv(os.path.join(RESULTS_DIR, 'xgboost_3feat_results.csv'), index=False)
    
    # Comparación
    print("\n" + "="*80)
    print("📊 RESULTADOS COMPARATIVOS")
    print("="*80)
    print(f"\nXGBoost con 6 features: RMSE = {rmse_6feat:.6f} ± {std_6feat:.6f} m")
    print(f"XGBoost con 3 features: RMSE = {rmse_3feat:.6f} ± {std_3feat:.6f} m")
    print(f"MLP (baseline):          RMSE = 0.199800 ± 0.089500 m")
    
    print(f"\nMejora XGBoost 6feat vs MLP: {((0.1998 - rmse_6feat)/0.1998)*100:.1f}%")
    print(f"Mejora XGBoost 3feat vs MLP: {((0.1998 - rmse_3feat)/0.1998)*100:.1f}%")
    
    mejor_config = "3 features" if rmse_3feat < rmse_6feat else "6 features"
    print(f"\n✅ Mejor configuración: XGBoost con {mejor_config}")
    
    # Visualización
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # Comparación RMSE
    x = np.arange(len(archivos))
    width = 0.25
    ax[0].bar(x - width, df_6feat['rmse'], width, label='6 features', color='#3498DB', edgecolor='black')
    ax[0].bar(x, df_3feat['rmse'], width, label='3 features', color='#2ECC71', edgecolor='black')
    ax[0].axhline(0.1998, color='red', linestyle='--', linewidth=2, alpha=0.7, label='MLP (0.1998 m)')
    ax[0].axhline(rmse_6feat, color='#3498DB', linestyle=':', linewidth=1.5, alpha=0.5)
    ax[0].axhline(rmse_3feat, color='#2ECC71', linestyle=':', linewidth=1.5, alpha=0.5)
    ax[0].set_xlabel('Fold (Ubicación)', fontsize=11, fontweight='bold')
    ax[0].set_ylabel('RMSE (m)', fontsize=11, fontweight='bold')
    ax[0].set_title('XGBoost Mejorado - Comparación de Configuraciones', fontsize=12, fontweight='bold')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(range(1, len(archivos)+1))
    ax[0].legend(fontsize=10)
    ax[0].grid(True, alpha=0.3, axis='y')
    
    # Tabla resumen
    ax[1].axis('off')
    tabla_data = [
        ['Modelo', 'RMSE (m)', 'Desv. Est.', 'vs MLP'],
        ['XGBoost 6feat', f'{rmse_6feat:.6f}', f'{std_6feat:.6f}', 
         f'{((0.1998 - rmse_6feat)/0.1998)*100:+.1f}%'],
        ['XGBoost 3feat', f'{rmse_3feat:.6f}', f'{std_3feat:.6f}', 
         f'{((0.1998 - rmse_3feat)/0.1998)*100:+.1f}%'],
        ['MLP (ref)', '0.1998', '0.0895', '0.0%']
    ]
    
    tabla = ax[1].table(cellText=tabla_data, cellLoc='center', loc='center',
                       colWidths=[0.3, 0.25, 0.25, 0.2])
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1, 2.5)
    
    for i in range(4):
        tabla[(0, i)].set_facecolor('#34495E')
        tabla[(0, i)].set_text_props(weight='bold', color='white')
    
    tabla[(4, 0)].set_facecolor('#FFD700')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'xgboost_mejorado_comparacion.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Gráfico guardado: {RESULTS_DIR}/xgboost_mejorado_comparacion.png")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
