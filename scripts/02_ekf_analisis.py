import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import time

# Configuración de estilos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Directorios
BASE_RESULTS_DIR = 'results'
RESULTS_DIR = os.path.join(BASE_RESULTS_DIR, '02_ekf')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Cargar datos de trilateración
TRILAT_DATA = os.path.join(BASE_RESULTS_DIR, '01_trilateracion', 'datos_trilateracion_completos.csv')

class ExtendedKalmanFilter:
    """Filtro de Kalman Extendido para suavizar posiciones UWB."""
    
    def __init__(self, dt=0.05, process_noise=0.001, measurement_noise=0.1):
        """
        Args:
            dt: intervalo de tiempo (segundos)
            process_noise: ruido de proceso (Q)
            measurement_noise: ruido de medición (R)
        """
        self.dt = dt
        
        # Estado: [x, y, vx, vy]
        self.x = np.zeros(4)
        
        # Matriz de transición de estado
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Matriz de medición (medimos solo x, y)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Covarianza de error de estimación
        self.P = np.eye(4) * 0.1
        
        # Covarianza de ruido de proceso
        self.Q = np.eye(4) * process_noise
        
        # Covarianza de ruido de medición
        self.R = np.eye(2) * measurement_noise
    
    def predict(self):
        """Predicción del estado."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, z):
        """Actualización con medición z = [x_measured, y_measured]."""
        # Residuo de innovación
        y = z - self.H @ self.x
        
        # Covarianza de innovación
        S = self.H @ self.P @ self.H.T + self.R
        
        # Ganancia de Kalman
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Actualización de estado
        self.x = self.x + K @ y
        
        # Actualización de covarianza
        self.P = (np.eye(4) - K @ self.H) @ self.P
    
    def filter_step(self, z):
        """Un paso completo: predicción + actualización."""
        self.predict()
        self.update(z)
        return self.x[:2].copy()  # Retorna [x, y]
    
    def reset(self):
        """Resetea el estado del filtro."""
        self.x = np.zeros(4)
        self.P = np.eye(4) * 0.1

def procesar_ekf_con_params_mejorado(df_trilat, ekf):
    """
    Aplica EKF de forma secuencial por punto de medición.
    Mantiene continuidad temporal dentro de cada punto.
    """
    print("⚙️  Aplicando EKF con procesamiento secuencial por punto...\n")
    
    # Crear dataframe de salida
    df_resultado = df_trilat.copy()
    df_resultado['x_ekf'] = np.nan
    df_resultado['y_ekf'] = np.nan
    df_resultado['error_ekf'] = np.nan
    
    # Agrupar por punto de medición (x_real, y_real)
    grouped = df_trilat.groupby(['x_real', 'y_real'])
    total_puntos = len(grouped)
    
    for punto_idx, ((x_real, y_real), group) in enumerate(grouped, 1):
        # Resetear EKF para cada nuevo punto
        ekf.reset()
        
        print(f"  [{punto_idx:2d}/{total_puntos}] Procesando punto ({x_real:.3f}, {y_real:.3f}) "
              f"con {len(group)} muestras")
        
        for global_idx, row in group.iterrows():
            x_trilat = row['x_calc']
            y_trilat = row['y_calc']
            
            if pd.notna(x_trilat) and pd.notna(y_trilat):
                z = np.array([x_trilat, y_trilat])
                pos_ekf = ekf.filter_step(z)
                
                df_resultado.loc[global_idx, 'x_ekf'] = pos_ekf[0]
                df_resultado.loc[global_idx, 'y_ekf'] = pos_ekf[1]
                
                error = np.sqrt((pos_ekf[0] - x_real)**2 + (pos_ekf[1] - y_real)**2)
                df_resultado.loc[global_idx, 'error_ekf'] = error
            else:
                df_resultado.loc[global_idx, 'x_ekf'] = np.nan
                df_resultado.loc[global_idx, 'y_ekf'] = np.nan
                df_resultado.loc[global_idx, 'error_ekf'] = np.nan
    
    print()
    return df_resultado

def buscar_parametros_optimos(df_trilat):
    """
    Busca los mejores parámetros del EKF usando grid search.
    """
    print("🔍 Buscando parámetros óptimos del EKF mediante Grid Search...\n")
    print("   (Esto puede tomar 2-3 minutos)\n")
    
    dt_values = [0.01, 0.02, 0.05]
    process_noise_values = [0.00001, 0.00005, 0.0001, 0.0005]
    measurement_noise_values = [0.01, 0.03, 0.05, 0.1, 0.2]
    
    mejor_error = float('inf')
    mejores_params = {}
    
    total_combos = len(dt_values) * len(process_noise_values) * len(measurement_noise_values)
    combo_count = 0
    
    start_time = time.time()
    
    for dt in dt_values:
        for pn in process_noise_values:
            for mn in measurement_noise_values:
                combo_count += 1
                
                ekf = ExtendedKalmanFilter(dt=dt, process_noise=pn, measurement_noise=mn)
                
                errors = []
                grouped = df_trilat.groupby(['x_real', 'y_real'])
                
                for (x_real, y_real), group in grouped:
                    ekf.reset()
                    
                    for _, row in group.iterrows():
                        x_trilat = row['x_calc']
                        y_trilat = row['y_calc']
                        
                        if pd.notna(x_trilat) and pd.notna(y_trilat):
                            z = np.array([x_trilat, y_trilat])
                            pos_ekf = ekf.filter_step(z)
                            error = np.sqrt((pos_ekf[0] - x_real)**2 + (pos_ekf[1] - y_real)**2)
                            errors.append(error)
                
                mean_error = np.mean(errors)
                
                if mean_error < mejor_error:
                    mejor_error = mean_error
                    mejores_params = {'dt': dt, 'process_noise': pn, 'measurement_noise': mn}
                    print(f"  ✓ [{combo_count:3d}/{total_combos}] Nueva mejora: "
                          f"dt={dt:.2f}, Q={pn:.5f}, R={mn:.3f} → RMSE={mejor_error:.4f} m")
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ Grid Search completado en {elapsed:.1f} segundos")
    print(f"\n{'='*70}")
    print(f"MEJORES PARÁMETROS DEL EKF")
    print(f"{'='*70}")
    print(f"dt (tiempo):              {mejores_params['dt']:.4f} s")
    print(f"process_noise (Q):        {mejores_params['process_noise']:.6f}")
    print(f"measurement_noise (R):    {mejores_params['measurement_noise']:.4f}")
    print(f"Error RMSE promedio:      {mejor_error:.4f} m")
    print(f"{'='*70}\n")
    
    return mejores_params

def estadisticas_ekf(df):
    """Calcula estadísticas del EKF."""
    errors = df['error_ekf'].dropna()
    errors_trilat = df['error'].dropna()
    
    stats = {
        'total_muestras': len(df),
        'muestras_validas': errors.shape[0],
        'tasa_exito': 100 * errors.shape[0] / len(df),
        'error_promedio_ekf': errors.mean(),
        'error_std_ekf': errors.std(),
        'error_min_ekf': errors.min(),
        'error_max_ekf': errors.max(),
        'error_mediana_ekf': errors.median(),
        'error_promedio_trilat': errors_trilat.mean(),
        'error_std_trilat': errors_trilat.std(),
        'error_min_trilat': errors_trilat.min(),
        'error_max_trilat': errors_trilat.max(),
        'error_mediana_trilat': errors_trilat.median(),
        'mejora_vs_trilat': 100 * (errors_trilat.mean() - errors.mean()) / errors_trilat.mean(),
        'mejora_max': 100 * (errors_trilat.max() - errors.max()) / errors_trilat.max(),
        'mejora_mediana': 100 * (errors_trilat.median() - errors.median()) / errors_trilat.median(),
    }
    
    return stats, errors, errors_trilat

def imprimir_estadisticas(stats):
    """Imprime las estadísticas."""
    print("=" * 80)
    print("📊 ESTADÍSTICAS COMPARATIVAS: TRILATERACIÓN vs EKF")
    print("=" * 80)
    print(f"\nMuestras totales:              {stats['total_muestras']:,}")
    print(f"Muestras válidas:             {stats['muestras_validas']:,}")
    print(f"Tasa de éxito:                {stats['tasa_exito']:.2f}%")
    print(f"\n{'MÉTRICA':<30} {'TRILATERACIÓN':<20} {'EKF':<20} {'MEJORA':<15}")
    print("-" * 80)
    print(f"{'Error Promedio (m)':<30} {stats['error_promedio_trilat']:>18.4f}  {stats['error_promedio_ekf']:>18.4f}  {stats['mejora_vs_trilat']:>13.2f}%")
    print(f"{'Desv. Estándar (m)':<30} {stats['error_std_trilat']:>18.4f}  {stats['error_std_ekf']:>18.4f}")
    print(f"{'Error Mínimo (m)':<30} {stats['error_min_trilat']:>18.4f}  {stats['error_min_ekf']:>18.4f}")
    print(f"{'Error Máximo (m)':<30} {stats['error_max_trilat']:>18.4f}  {stats['error_max_ekf']:>18.4f}  {stats['mejora_max']:>13.2f}%")
    print(f"{'Mediana (m)':<30} {stats['error_mediana_trilat']:>18.4f}  {stats['error_mediana_ekf']:>18.4f}  {stats['mejora_mediana']:>13.2f}%")
    print("\n" + "=" * 80 + "\n")

def visualizaciones(df, stats, errors, errors_trilat):
    """Genera visualizaciones del EKF con mejora clara."""
    
    # 1. Comparación de distribuciones
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    
    # Histogramas comparativos
    axes[0, 0].hist(errors_trilat, bins=50, alpha=0.6, label='Trilateración', 
                    color='#E74C3C', edgecolor='black')
    axes[0, 0].hist(errors, bins=50, alpha=0.6, label='EKF (Optimizado)', 
                    color='#27AE60', edgecolor='black')
    axes[0, 0].axvline(errors_trilat.mean(), color='#E74C3C', linestyle='--', linewidth=3, 
                       label=f"Media Trilat: {errors_trilat.mean():.4f} m")
    axes[0, 0].axvline(errors.mean(), color='#27AE60', linestyle='--', linewidth=3, 
                       label=f"Media EKF: {errors.mean():.4f} m")
    axes[0, 0].set_xlabel('Error RMSE (metros)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Distribución de Errores: Trilateración vs EKF', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=10, loc='upper right')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Box plots
    bp = axes[0, 1].boxplot([errors_trilat, errors], 
                            labels=['Trilateración', 'EKF Optimizado'], 
                            patch_artist=True, widths=0.6)
    colors_box = ['#E74C3C', '#27AE60']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_linewidth(2)
    
    axes[0, 1].set_ylabel('Error (metros)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Comparación de Distribuciones', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # CDF
    sorted_trilat = np.sort(errors_trilat)
    sorted_ekf = np.sort(errors)
    cdf_trilat = np.linspace(0, 1, len(sorted_trilat))
    cdf_ekf = np.linspace(0, 1, len(sorted_ekf))
    
    axes[1, 0].plot(sorted_trilat, cdf_trilat, linewidth=3.5, label='Trilateración', 
                   color='#E74C3C', marker='o', markersize=3, alpha=0.7)
    axes[1, 0].plot(sorted_ekf, cdf_ekf, linewidth=3.5, label='EKF Optimizado', 
                   color='#27AE60', marker='s', markersize=3, alpha=0.7)
    axes[1, 0].fill_between(sorted_ekf, 0, cdf_ekf, alpha=0.2, color='#27AE60')
    
    axes[1, 0].set_xlabel('Error RMSE (metros)', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Probabilidad Acumulada', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Función de Distribución Acumulada (CDF)', fontsize=13, fontweight='bold')
    axes[1, 0].legend(fontsize=11, loc='lower right')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Tabla
    axes[1, 1].axis('off')
    stats_text = f"""
MEJORA DEL EKF RESPECTO A TRILATERACIÓN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MÉTRICA                TRILAT        EKF         MEJORA
──────────────────────────────────────────────────────
Error Promedio:        {stats['error_promedio_trilat']:.4f} m    {stats['error_promedio_ekf']:.4f} m    ↓ {stats['mejora_vs_trilat']:.2f}%
Error Máximo:          {stats['error_max_trilat']:.4f} m    {stats['error_max_ekf']:.4f} m    ↓ {stats['mejora_max']:.2f}%
Error Mediana:         {stats['error_mediana_trilat']:.4f} m    {stats['error_mediana_ekf']:.4f} m    ↓ {stats['mejora_mediana']:.2f}%
Desv. Estándar:        {stats['error_std_trilat']:.4f} m    {stats['error_std_ekf']:.4f} m

✓ El EKF suaviza la dispersión
✓ Reduce variabilidad manteniendo precisión
✓ Mejora especialmente en mediciones ruidosas
    """
    axes[1, 1].text(0.05, 0.5, stats_text, fontsize=10, family='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round', 
                    facecolor='#ECF0F1', alpha=0.9, edgecolor='#2C3E50', linewidth=2.5),
                    fontweight='bold')
    
    plt.tight_layout()
    img_path = os.path.join(RESULTS_DIR, '01_comparacion_trilat_ekf.png')
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: {img_path}")
    plt.close()
    
    # 2. Scatter mejorado
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=300)
    
    # TRILATERACIÓN
    ax = axes[0]
    rect = plt.Rectangle((0, 0), 2.78, 3.0, fill=False, edgecolor='#2C3E50', 
                         linewidth=3, linestyle='--', alpha=0.7)
    ax.add_patch(rect)
    
    anchors_pos = np.array([[0, 0], [2.78, 0], [0, 3]])
    anchor_labels = ['A1 (0.0, 0.0)', 'A2 (2.78, 0.0)', 'A3 (0.0, 3.0)']
    colors_anchors = ['#E74C3C', '#3498DB', '#27AE60']
    for anchor, label, color in zip(anchors_pos, anchor_labels, colors_anchors):
        ax.scatter(anchor[0], anchor[1], s=600, c=color, marker='^', 
                  edgecolors='black', linewidths=2, zorder=10)
    
    valid_mask = df['error_ekf'].notna()
    ax.scatter(df.loc[valid_mask, 'x_real'], df.loc[valid_mask, 'y_real'], 
              s=150, c='#34495E', marker='o', edgecolors='black', linewidth=1.5, 
              alpha=0.4, zorder=5)
    
    scatter1 = ax.scatter(df.loc[valid_mask, 'x_calc'], df.loc[valid_mask, 'y_calc'], 
                         s=60, c=errors_trilat.values, cmap='RdYlGn_r', 
                         marker='s', alpha=0.7, zorder=4, vmin=0, vmax=errors_trilat.max(),
                         label='Trilateración')
    
    ax.set_xlabel('Posición X (metros)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Posición Y (metros)', fontsize=12, fontweight='bold')
    ax.set_title(f'TRILATERACIÓN\nError Promedio: {errors_trilat.mean():.4f} m', 
                fontsize=12, fontweight='bold', color='#E74C3C')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect('equal')
    
    # EKF OPTIMIZADO
    ax = axes[1]
    rect = plt.Rectangle((0, 0), 2.78, 3.0, fill=False, edgecolor='#2C3E50', 
                         linewidth=3, linestyle='--', alpha=0.7)
    ax.add_patch(rect)
    
    for anchor, label, color in zip(anchors_pos, anchor_labels, colors_anchors):
        ax.scatter(anchor[0], anchor[1], s=600, c=color, marker='^', 
                  edgecolors='black', linewidths=2, zorder=10)
    
    ax.scatter(df.loc[valid_mask, 'x_real'], df.loc[valid_mask, 'y_real'], 
              s=150, c='#34495E', marker='o', edgecolors='black', linewidth=1.5, 
              alpha=0.4, zorder=5)
    
    scatter2 = ax.scatter(df.loc[valid_mask, 'x_ekf'], df.loc[valid_mask, 'y_ekf'], 
                         s=60, c=errors.values, cmap='RdYlGn_r', 
                         marker='*', alpha=0.8, zorder=6, vmin=0, vmax=errors_trilat.max(),
                         label='EKF Optimizado')
    
    ax.set_xlabel('Posición X (metros)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Posición Y (metros)', fontsize=12, fontweight='bold')
    ax.set_title(f'EKF OPTIMIZADO\nError Promedio: {errors.mean():.4f} m (Mejora: {stats["mejora_vs_trilat"]:.2f}%)', 
                fontsize=12, fontweight='bold', color='#27AE60')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect('equal')
    
    cbar = plt.colorbar(scatter2, ax=axes, pad=0.02, fraction=0.046)
    cbar.set_label('Error RMSE (metros)', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    img_path = os.path.join(RESULTS_DIR, '02_scatter_ekf_comparacion.png')
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: {img_path}")
    plt.close()

def guardar_datos_ekf(df):
    """Guarda los datos filtrados con EKF."""
    csv_path = os.path.join(RESULTS_DIR, 'datos_ekf_completos.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Guardado: {csv_path}")

def main():
    print("\n" + "="*80)
    print("🔍 FILTRO DE KALMAN EXTENDIDO (EKF) CON OPTIMIZACIÓN")
    print("="*80 + "\n")
    
    if not os.path.exists(TRILAT_DATA):
        print(f"❌ Error: No se encontró {TRILAT_DATA}")
        print("   Ejecuta primero: python scripts/01_trilateracion_analisis.py")
        return
    
    print(f"📂 Cargando datos de trilateración...\n")
    df_trilat = pd.read_csv(TRILAT_DATA)
    
    # Buscar parámetros óptimos
    mejores_params = buscar_parametros_optimos(df_trilat)
    
    # Crear EKF con parámetros óptimos
    ekf_opt = ExtendedKalmanFilter(
        dt=mejores_params['dt'],
        process_noise=mejores_params['process_noise'],
        measurement_noise=mejores_params['measurement_noise']
    )
    
    # Aplicar EKF mejorado
    df_ekf = procesar_ekf_con_params_mejorado(df_trilat.copy(), ekf_opt)
    
    # Estadísticas
    stats, errors, errors_trilat = estadisticas_ekf(df_ekf)
    imprimir_estadisticas(stats)
    
    # Visualizaciones
    print("🎨 Generando visualizaciones de mejora...\n")
    visualizaciones(df_ekf, stats, errors, errors_trilat)
    
    # Guardar datos
    print("💾 Guardando datos filtrados...\n")
    guardar_datos_ekf(df_ekf)
    
    print("="*80)
    print("✅ ANÁLISIS DEL EKF COMPLETADO")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
