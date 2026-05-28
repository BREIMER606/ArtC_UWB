import os, warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import glob
from tqdm import tqdm

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

BASE_RESULTS_DIR = 'results'
RESULTS_DIR = os.path.join(BASE_RESULTS_DIR, '04_rnn_simple')
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_DIR = 'data_crudos'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n📱 Device: {device}\n")

class SimpleRNN(nn.Module):
    def __init__(self, input_size=3, hidden_size=16, output_size=2):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, num_layers=1)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # Último timestep
        return self.fc(out)

def cargar_datos_por_archivo():
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
                'n_muestras': len(X_data)
            }
    
    return datos_por_archivo

def crear_secuencias(X, seq_len=3):
    """Crea secuencias de longitud seq_len"""
    X_seq = []
    for i in range(len(X) - seq_len + 1):
        X_seq.append(X[i:i+seq_len])
    return np.array(X_seq)

def entrenar_fold(X_train, y_train, X_test, y_test, seq_len=3):
    """Entrena un fold"""
    
    # Crear secuencias
    X_train_seq = crear_secuencias(X_train, seq_len)
    y_train_seq = y_train[seq_len-1:]
    
    X_test_seq = crear_secuencias(X_test, seq_len)
    y_test_seq = y_test[seq_len-1:]
    
    # Normalizar
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_seq_flat = X_train_seq.reshape(-1, X_train_seq.shape[-1])
    X_train_seq_norm = scaler_X.fit_transform(X_train_seq_flat).reshape(X_train_seq.shape)
    X_test_seq_norm = scaler_X.transform(X_test_seq.reshape(-1, X_test_seq.shape[-1])).reshape(X_test_seq.shape)
    y_train_seq_norm = scaler_y.fit_transform(y_train_seq)
    y_test_seq_norm = scaler_y.transform(y_test_seq)
    
    # Dataset
    train_dataset = TensorDataset(torch.FloatTensor(X_train_seq_norm), torch.FloatTensor(y_train_seq_norm))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Modelo
    model = SimpleRNN(input_size=3, hidden_size=16, output_size=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(100):
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
        
        if train_loss < best_loss:
            best_loss = train_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Test
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_seq_norm).to(device)
        y_pred_norm = model(X_test_tensor).cpu().numpy()
    
    y_pred = scaler_y.inverse_transform(y_pred_norm)
    
    rmse = np.sqrt(mean_squared_error(y_test_seq, y_pred))
    mae = mean_absolute_error(y_test_seq, y_pred)
    errors = np.sqrt((y_test_seq[:, 0] - y_pred[:, 0])**2 + (y_test_seq[:, 1] - y_pred[:, 1])**2)
    
    return rmse, mae, errors.mean()

def main():
    print('\n' + '='*80)
    print('🧠 RNN SIMPLE - LEAVE-ONE-LOCATION-OUT CV (BASELINE)')
    print('='*80 + '\n')
    
    datos_por_archivo = cargar_datos_por_archivo()
    archivos = sorted(datos_por_archivo.keys())
    
    print(f"Total archivos: {len(archivos)}\n")
    
    resultados_folds = []
    
    for fold, test_archivo in enumerate(archivos, 1):
        print(f"Fold {fold}/{len(archivos)}: TEST={test_archivo}")
        
        train_archivos = [a for a in archivos if a != test_archivo]
        X_train = np.vstack([datos_por_archivo[a]['X'] for a in train_archivos])
        y_train = np.vstack([datos_por_archivo[a]['y'] for a in train_archivos])
        X_test = datos_por_archivo[test_archivo]['X']
        y_test = datos_por_archivo[test_archivo]['y']
        
        print(f"  TRAIN: {len(X_train)} muestras | TEST: {len(X_test)} muestras")
        
        rmse, mae, error_prom = entrenar_fold(X_train, y_train, X_test, y_test, seq_len=3)
        
        print(f"  → RMSE: {rmse:.6f} m\n")
        
        resultados_folds.append({
            'fold': fold,
            'test_archivo': test_archivo,
            'rmse': rmse,
            'mae': mae
        })
    
    print("="*80)
    print("📊 RESULTADOS RNN SIMPLE")
    print("="*80)
    
    resultados_df = pd.DataFrame(resultados_folds)
    print(f"\n{resultados_df.to_string(index=False)}\n")
    
    rmse_medio = resultados_df['rmse'].mean()
    rmse_std = resultados_df['rmse'].std()
    
    print(f"RMSE promedio: {rmse_medio:.6f} ± {rmse_std:.6f} m")
    print("="*80 + "\n")
    
    resultados_df.to_csv(os.path.join(RESULTS_DIR, 'rnn_results.csv'), index=False)
    
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    ax.bar(range(1, len(archivos)+1), resultados_df['rmse'], color='#9B59B6', alpha=0.7, edgecolor='black')
    ax.axhline(rmse_medio, color='red', linestyle='--', linewidth=2, label=f'Media: {rmse_medio:.4f}')
    ax.set_xlabel('Fold (Location)')
    ax.set_ylabel('RMSE (m)')
    ax.set_title('RNN Simple - RMSE por Location (BASELINE)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'rnn_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Completado | {RESULTS_DIR}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
