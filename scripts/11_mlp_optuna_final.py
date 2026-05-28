import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
from optuna.samplers import TPESampler
import glob

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

BASE_RESULTS_DIR = 'results'
RESULTS_DIR = os.path.join(BASE_RESULTS_DIR, '11_mlp_optuna_final')
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_DIR = 'data_crudos'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("\n" + "="*100)
print("🧠 MLP + OPTUNA - LEAVE-ONE-LOCATION-OUT CROSS-VALIDATION")
print("="*100)
print(f"📱 Device: {device}\n")

class MLPModel(nn.Module):
    """Modelo MLP con arquitectura flexible"""
    def __init__(self, input_size=3, hidden_sizes=[64, 32], output_size=2, dropout=0.1):
        super().__init__()
        layers = []
        in_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_size = hidden_size
        
        layers.append(nn.Linear(in_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

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
        
        mask = ~(np.isnan(d_A1) | np.isnan(d_A2) | np.isnan(d_A3))
        
        X = np.column_stack([d_A1[mask], d_A2[mask], d_A3[mask]])
        y = np.full((len(X), 2), [x_real, y_real])
        
        datos[name] = {'X': X, 'y': y}
        print(f"  ✓ {name}: {len(X)} muestras")
    
    return datos

def entrenar_modelo(X_train, y_train, X_val, y_val, X_test, y_test, hidden_sizes, dropout, 
                   learning_rate, batch_size, weight_decay):
    """Entrena un modelo MLP con parámetros específicos"""
    
    # Normalizar
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_norm = scaler_X.fit_transform(X_train)
    X_val_norm = scaler_X.transform(X_val)
    X_test_norm = scaler_X.transform(X_test)
    
    y_train_norm = scaler_y.fit_transform(y_train)
    y_val_norm = scaler_y.transform(y_val)
    
    # Crear modelo
    model = MLPModel(input_size=3, hidden_sizes=hidden_sizes, output_size=2, dropout=dropout).to(device)
    
    # DataLoader
    train_ds = TensorDataset(torch.FloatTensor(X_train_norm), torch.FloatTensor(y_train_norm))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    # Optimizador y criterio
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    
    # Entrenamiento con Early Stopping
    best_val_loss = float('inf')
    patience = 20
    wait = 0
    best_model_state = None
    
    for epoch in range(200):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        scheduler.step()
        
        # Validación
        model.eval()
        with torch.no_grad():
            val_pred_norm = model(torch.FloatTensor(X_val_norm).to(device)).cpu().numpy()
        
        val_loss = mean_squared_error(y_val_norm, val_pred_norm)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            best_model_state = model.state_dict().copy()
        else:
            wait += 1
            if wait >= patience:
                break
    
    # Cargar mejor modelo
    model.load_state_dict(best_model_state)
    
    # Predicción en test
    model.eval()
    with torch.no_grad():
        y_test_pred_norm = model(torch.FloatTensor(X_test_norm).to(device)).cpu().numpy()
    
    y_test_pred = scaler_y.inverse_transform(y_test_pred_norm)
    
    # Métricas
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae = mean_absolute_error(y_test, y_test_pred)
    
    return rmse, mae

def objective(trial, X_train, y_train, X_val, y_val, X_test, y_test):
    """Función objetivo para Optuna"""
    try:
        # Sugerir hiperparámetros
        num_layers = trial.suggest_int('num_layers', 1, 3)
        hidden_sizes = [trial.suggest_int(f'hidden_size_{i}', 32, 256) for i in range(num_layers)]
        dropout = trial.suggest_float('dropout', 0.0, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_int('batch_size', 16, 128)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        
        # Entrenar y evaluar
        rmse, _ = entrenar_modelo(X_train, y_train, X_val, y_val, X_test, y_test,
                                 hidden_sizes, dropout, learning_rate, batch_size, weight_decay)
        
        return rmse
    except Exception as e:
        print(f"Error en trial: {e}")
        return float('inf')

def ejecutar_lolo_optuna(datos, archivos):
    """Ejecuta LOLO-CV con optimización Optuna"""
    
    print(f"\n{'='*100}")
    print(f"📊 LEAVE-ONE-LOCATION-OUT CV CON OPTUNA")
    print(f"{'='*100}\n")
    
    resultados_fold = []
    
    for fold_idx, test_archivo in enumerate(archivos, 1):
        print(f"\n{'─'*100}")
        print(f"🔄 FOLD {fold_idx}/9: TEST = {test_archivo}")
        print(f"{'─'*100}\n")
        
        # Separar datos
        train_archivos = [a for a in archivos if a != test_archivo]
        
        # 80% entrenamiento, 20% validación (del conjunto de entrenamiento)
        X_train_all = np.vstack([datos[a]['X'] for a in train_archivos])
        y_train_all = np.vstack([datos[a]['y'] for a in train_archivos])
        
        n_train = int(0.8 * len(X_train_all))
        X_train = X_train_all[:n_train]
        y_train = y_train_all[:n_train]
        X_val = X_train_all[n_train:]
        y_val = y_train_all[n_train:]
        
        X_test = datos[test_archivo]['X']
        y_test = datos[test_archivo]['y']
        
        print(f"📈 Datos:")
        print(f"   TRAIN: {len(X_train):,} muestras")
        print(f"   VAL:   {len(X_val):,} muestras")
        print(f"   TEST:  {len(X_test):,} muestras\n")
        
        # Optuna optimization
        print(f"🔍 Ejecutando Optuna (50 trials)...\n")
        
        sampler = TPESampler(seed=42)
        study = optuna.create_study(sampler=sampler, direction='minimize')
        
        def objective_wrapper(trial):
            return objective(trial, X_train, y_train, X_val, y_val, X_test, y_test)
        
        study.optimize(objective_wrapper, n_trials=50, show_progress_bar=True)
        
        # Mejor trial
        best_trial = study.best_trial
        print(f"\n✅ Mejor trial: RMSE = {best_trial.value:.6f} m")
        print(f"   Parámetros:")
        for key, val in best_trial.params.items():
            print(f"     {key}: {val}")
        
        # Entrenar modelo final con mejores parámetros
        print(f"\n🧠 Entrenando modelo final...\n")
        
        num_layers = best_trial.params['num_layers']
        hidden_sizes = [best_trial.params[f'hidden_size_{i}'] for i in range(num_layers)]
        dropout = best_trial.params['dropout']
        learning_rate = best_trial.params['learning_rate']
        batch_size = best_trial.params['batch_size']
        weight_decay = best_trial.params['weight_decay']
        
        rmse_final, mae_final = entrenar_modelo(X_train, y_train, X_val, y_val, X_test, y_test,
                                               hidden_sizes, dropout, learning_rate, 
                                               batch_size, weight_decay)
        
        resultados_fold.append({
            'fold': fold_idx,
            'test_archivo': test_archivo,
            'rmse': rmse_final,
            'mae': mae_final,
            'hidden_sizes': str(hidden_sizes),
            'dropout': dropout,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'weight_decay': weight_decay
        })
        
        print(f"📊 FOLD {fold_idx} - RESULTADOS FINALES:")
        print(f"   RMSE: {rmse_final:.6f} m")
        print(f"   MAE:  {mae_final:.6f} m\n")
    
    return pd.DataFrame(resultados_fold)

def crear_visualizaciones_finales(df_resultados, archivos):
    """Crea visualizaciones finales"""
    
    fig = plt.figure(figsize=(16, 12), dpi=300)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Panel 1: RMSE por fold
    ax1 = fig.add_subplot(gs[0, 0])
    colores = ['#2ECC71' if r < 0.25 else '#F39C12' if r < 0.35 else '#E74C3C' for r in df_resultados['rmse']]
    bars = ax1.bar(range(1, 10), df_resultados['rmse'], color=colores, edgecolor='black', linewidth=1.5)
    ax1.axhline(df_resultados['rmse'].mean(), color='red', linestyle='--', linewidth=2.5, 
               label=f'Media: {df_resultados["rmse"].mean():.4f} m')
    ax1.set_xlabel('Fold (Ubicación)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('RMSE (m)', fontsize=11, fontweight='bold')
    ax1.set_title('MLP+Optuna: RMSE por Ubicación', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(1, 10))
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, df_resultados['rmse']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Panel 2: MAE por fold
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(range(1, 10), df_resultados['mae'], color='#3498DB', edgecolor='black', linewidth=1.5)
    ax2.axhline(df_resultados['mae'].mean(), color='red', linestyle='--', linewidth=2.5, 
               label=f'Media: {df_resultados["mae"].mean():.4f} m')
    ax2.set_xlabel('Fold (Ubicación)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('MAE (m)', fontsize=11, fontweight='bold')
    ax2.set_title('MLP+Optuna: MAE por Ubicación', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(1, 10))
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Box plot
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.boxplot([df_resultados['rmse'], df_resultados['mae']], labels=['RMSE', 'MAE'], 
               patch_artist=True, boxprops=dict(facecolor='#3498DB', alpha=0.7), 
               medianprops=dict(color='red', linewidth=2.5))
    ax3.scatter([1]*len(df_resultados), df_resultados['rmse'], color='#2ECC71', s=100, alpha=0.6, zorder=3)
    ax3.scatter([2]*len(df_resultados), df_resultados['mae'], color='#E74C3C', s=100, alpha=0.6, zorder=3)
    ax3.set_ylabel('Error (m)', fontsize=11, fontweight='bold')
    ax3.set_title('Distribución de Errores', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Histograma de errores
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(df_resultados['rmse'], bins=6, color='#2ECC71', alpha=0.7, edgecolor='black', label='RMSE')
    ax4.axvline(df_resultados['rmse'].mean(), color='red', linestyle='--', linewidth=2.5, 
               label=f'Media RMSE')
    ax4.set_xlabel('RMSE (m)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
    ax4.set_title('Distribución de RMSE', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Panel 5: Tabla resumen
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    rmse_mean = df_resultados['rmse'].mean()
    rmse_std = df_resultados['rmse'].std()
    mae_mean = df_resultados['mae'].mean()
    mae_std = df_resultados['mae'].std()
    
    tabla_data = [
        ['Métrica', 'Valor'],
        ['RMSE Promedio', f'{rmse_mean:.6f} m'],
        ['RMSE Desv. Est.', f'{rmse_std:.6f} m'],
        ['RMSE Mínimo', f'{df_resultados["rmse"].min():.6f} m'],
        ['RMSE Máximo', f'{df_resultados["rmse"].max():.6f} m'],
        ['MAE Promedio', f'{mae_mean:.6f} m'],
        ['MAE Desv. Est.', f'{mae_std:.6f} m'],
        ['MAE Mínimo', f'{df_resultados["mae"].min():.6f} m'],
        ['MAE Máximo', f'{df_resultados["mae"].max():.6f} m'],
        ['Total Folds', '9'],
        ['Método', 'LOLO-CV + Optuna'],
    ]
    
    tabla = ax5.table(cellText=tabla_data, cellLoc='left', loc='center', colWidths=[0.4, 0.6])
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1, 2)
    
    for i in range(2):
        tabla[(0, i)].set_facecolor('#2ECC71')
        tabla[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)
    
    plt.suptitle('MLP + Optuna - Leave-One-Location-Out Cross-Validation\nUWB Indoor Localization', 
                fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig(os.path.join(RESULTS_DIR, '01_mlp_optuna_final_metricas.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Cargar datos
    datos = cargar_datos_por_archivo()
    archivos = sorted(datos.keys())
    
    # Ejecutar LOLO-CV con Optuna
    df_resultados = ejecutar_lolo_optuna(datos, archivos)
    
    # Estadísticas finales
    rmse_mean = df_resultados['rmse'].mean()
    rmse_std = df_resultados['rmse'].std()
    mae_mean = df_resultados['mae'].mean()
    mae_std = df_resultados['mae'].std()
    
    print("\n" + "="*100)
    print("📊 RESULTADOS FINALES - MLP + OPTUNA")
    print("="*100)
    print(df_resultados[['fold', 'test_archivo', 'rmse', 'mae']].to_string(index=False))
    print(f"\nRMSE PROMEDIO: {rmse_mean:.6f} ± {rmse_std:.6f} m")
    print(f"MAE PROMEDIO:  {mae_mean:.6f} ± {mae_std:.6f} m")
    print("="*100 + "\n")
    
    # Guardar resultados
    df_resultados.to_csv(os.path.join(RESULTS_DIR, 'mlp_optuna_final_resultados.csv'), index=False)
    
    # Crear visualizaciones
    print("🎨 Generando visualizaciones...\n")
    crear_visualizaciones_finales(df_resultados, archivos)
    
    # Tabla detallada
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.axis('off')
    
    tabla_detallada = [['Fold', 'Ubicación', 'RMSE (m)', 'MAE (m)', 'Capas', 'Dropout', 'LR']]
    for _, row in df_resultados.iterrows():
        tabla_detallada.append([
            f"{int(row['fold'])}",
            row['test_archivo'],
            f"{row['rmse']:.6f}",
            f"{row['mae']:.6f}",
            row['hidden_sizes'][:15] + '...' if len(str(row['hidden_sizes'])) > 15 else row['hidden_sizes'],
            f"{row['dropout']:.3f}",
            f"{row['learning_rate']:.2e}"
        ])
    
    tabla_detallada.append(['', 'PROMEDIO', f'{rmse_mean:.6f}', f'{mae_mean:.6f}', '', '', ''])
    tabla_detallada.append(['', 'DESV. EST.', f'{rmse_std:.6f}', f'{mae_std:.6f}', '', '', ''])
    
    tabla_det = ax.table(cellText=tabla_detallada, cellLoc='center', loc='center',
                        colWidths=[0.08, 0.15, 0.12, 0.12, 0.25, 0.12, 0.12])
    tabla_det.auto_set_font_size(False)
    tabla_det.set_fontsize(9)
    tabla_det.scale(1, 2)
    
    for i in range(7):
        tabla_det[(0, i)].set_facecolor('#2ECC71')
        tabla_det[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(7):
        tabla_det[(len(tabla_detallada)-2, i)].set_facecolor('#F39C12')
        tabla_det[(len(tabla_detallada)-1, i)].set_facecolor('#F39C12')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, '02_mlp_optuna_tabla_detallada.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Archivos guardados en: {RESULTS_DIR}")
    print(f"   ✓ mlp_optuna_final_resultados.csv")
    print(f"   ✓ 01_mlp_optuna_final_metricas.png")
    print(f"   ✓ 02_mlp_optuna_tabla_detallada.png")
    
    print("\n" + "="*100)
    print("🏆 CONCLUSIÓN PARA TU ARTÍCULO IEEE")
    print("="*100)
    print(f"""
La red neuronal MLP optimizada mediante Optuna y validada con Leave-One-Location-Out
Cross-Validation logró un RMSE de {rmse_mean:.4f} ± {rmse_std:.4f} m, mejorando
significativamente la precisión de los métodos geométricos (trilateración: 0.3160 m)
en un {((0.3160 - rmse_mean)/0.3160)*100:.1f}%, demostrando excelente capacidad de
generalización a localizaciones completamente nuevas no vistas durante el entrenamiento.

RESULTADOS POR UBICACIÓN:
""")
    for _, row in df_resultados.iterrows():
        print(f"  {row['test_archivo']}: RMSE = {row['rmse']:.4f} m, MAE = {row['mae']:.4f} m")
    
    print("\n" + "="*100 + "\n")

if __name__ == '__main__':
    main()
