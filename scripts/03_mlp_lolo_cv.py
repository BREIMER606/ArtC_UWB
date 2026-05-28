import os, time, warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import glob
from tqdm import tqdm
import pickle

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

BASE_RESULTS_DIR = 'results'
RESULTS_DIR = os.path.join(BASE_RESULTS_DIR, '03_mlp_lolo_cv')
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_DIR = 'data_crudos'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n📱 Device: {device}\n")

class MLPModel(nn.Module):
    def __init__(self, input_size=3, hidden_sizes=[64, 32], output_size=2, dropout=0.2):
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
            
            valid_mask = ~(np.isnan(d_A1) | np.isnan(d_A2) | np.isnan(d_A3))
            d_A1 = d_A1[valid_mask]
            d_A2 = d_A2[valid_mask]
            d_A3 = d_A3[valid_mask]
            
            X_data = np.column_stack([d_A1, d_A2, d_A3])
            y_data = np.full((len(X_data), 2), [x_real, y_real])
            
            datos_por_archivo[filename] = {
                'X': X_data,
                'y': y_data,
                'n_muestras': len(X_data),
                'posicion': (x_real, y_real)
            }
    
    return datos_por_archivo

def entrenar_fold(X_train, y_train, X_test, y_test, fold_num):
    """Entrena un fold del CV"""
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_norm = scaler_X.fit_transform(X_train)
    y_train_norm = scaler_y.fit_transform(y_train)
    X_test_norm = scaler_X.transform(X_test)
    y_test_norm = scaler_y.transform(y_test)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train_norm), torch.FloatTensor(y_train_norm))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = MLPModel(input_size=3, hidden_sizes=[64, 32], output_size=2, dropout=0.2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(150):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        scheduler.step()
        
        if train_loss < best_val_loss:
            best_val_loss = train_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_norm).to(device)
        y_pred_norm = model(X_test_tensor).cpu().numpy()
    
    y_pred = scaler_y.inverse_transform(y_pred_norm)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    errors = np.sqrt((y_test[:, 0] - y_pred[:, 0])**2 + (y_test[:, 1] - y_pred[:, 1])**2)
    
    return rmse, mae, errors.mean(), y_test, y_pred

def main():
    print('\n' + '='*80)
    print('🧠 MLP - LEAVE-ONE-LOCATION-OUT CROSS-VALIDATION')
    print('='*80 + '\n')
    
    datos_por_archivo = cargar_datos_por_archivo()
    archivos = sorted(datos_por_archivo.keys())
    
    print(f"Total archivos (localizaciones): {len(archivos)}\n")
    
    resultados_folds = []
    
    for fold, test_archivo in enumerate(archivos, 1):
        print(f"Fold {fold}/{len(archivos)}: TEST={test_archivo}")
        
        train_archivos = [a for a in archivos if a != test_archivo]
        X_train = np.vstack([datos_por_archivo[a]['X'] for a in train_archivos])
        y_train = np.vstack([datos_por_archivo[a]['y'] for a in train_archivos])
        X_test = datos_por_archivo[test_archivo]['X']
        y_test = datos_por_archivo[test_archivo]['y']
        
        print(f"  TRAIN: {len(train_archivos)} archivos, {len(X_train)} muestras")
        print(f"  TEST:  1 archivo ({test_archivo}), {len(X_test)} muestras")
        
        rmse, mae, error_promedio, y_test_fold, y_pred_fold = entrenar_fold(
            X_train, y_train, X_test, y_test, fold
        )
        
        print(f"  → RMSE: {rmse:.6f} m\n")
        
        resultados_folds.append({
            'fold': fold,
            'test_archivo': test_archivo,
            'rmse': rmse,
            'mae': mae,
            'error_promedio': error_promedio
        })
    
    print("="*80)
    print("📊 RESULTADOS LEAVE-ONE-LOCATION-OUT CV")
    print("="*80)
    
    resultados_df = pd.DataFrame(resultados_folds)
    print(f"\n{resultados_df.to_string(index=False)}\n")
    
    rmse_medio = resultados_df['rmse'].mean()
    rmse_std = resultados_df['rmse'].std()
    mae_medio = resultados_df['mae'].mean()
    mae_std = resultados_df['mae'].std()
    
    print(f"RMSE promedio: {rmse_medio:.6f} ± {rmse_std:.6f} m")
    print(f"MAE promedio:  {mae_medio:.6f} ± {mae_std:.6f} m")
    print("="*80 + "\n")
    
    resultados_df.to_csv(os.path.join(RESULTS_DIR, 'lolo_cv_results.csv'), index=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    axes[0].bar(range(1, len(archivos)+1), resultados_df['rmse'], color='#3498DB', alpha=0.7, edgecolor='black')
    axes[0].axhline(rmse_medio, color='red', linestyle='--', linewidth=2, label=f'Media: {rmse_medio:.4f}')
    axes[0].set_xlabel('Fold (Location)')
    axes[0].set_ylabel('RMSE (m)')
    axes[0].set_title('RMSE por Location')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].boxplot([resultados_df['rmse']], labels=['LOLO-CV'], patch_artist=True)
    axes[1].scatter([1]*len(resultados_df['rmse']), resultados_df['rmse'], alpha=0.5, s=100, color='#E74C3C')
    axes[1].set_ylabel('RMSE (m)')
    axes[1].set_title('Distribución RMSE')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'lolo_cv_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Completado | {RESULTS_DIR}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
