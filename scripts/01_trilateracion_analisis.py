import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Configuración de estilos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Directorios
BASE_RESULTS_DIR = 'results'
RESULTS_DIR = os.path.join(BASE_RESULTS_DIR, '01_trilateracion')
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_DIR = 'data_crudos'

# Posiciones de los anclajes UWB (metros)
ANCHORS = {
    'A1': np.array([0.0, 0.0]),
    'A2': np.array([2.78, 0.0]),
    'A3': np.array([0.0, 3.0])
}

def trilateracion(d1, d2, d3, anchor_pos=None):
    """
    Calcula posición (x, y) usando trilateración con 3 distancias.
    
    Args:
        d1, d2, d3: distancias a A1, A2, A3 (metros)
        anchor_pos: dict con posiciones de anclajes
    
    Returns:
        (x, y): coordenadas estimadas
    """
    if anchor_pos is None:
        anchor_pos = ANCHORS
    
    A1 = anchor_pos['A1']
    A2 = anchor_pos['A2']
    A3 = anchor_pos['A3']
    
    # Sistema lineal: 2 ecuaciones, 2 incógnitas
    # (x - x1)² + (y - y1)² = d1²
    # (x - x2)² + (y - y2)² = d2²
    # Resta ecuaciones para obtener sistema lineal
    
    A = np.array([
        [2*(A2[0] - A1[0]), 2*(A2[1] - A1[1])],
        [2*(A3[0] - A1[0]), 2*(A3[1] - A1[1])]
    ])
    
    b = np.array([
        d1**2 - d2**2 + np.linalg.norm(A2)**2 - np.linalg.norm(A1)**2,
        d1**2 - d3**2 + np.linalg.norm(A3)**2 - np.linalg.norm(A1)**2
    ])
    
    try:
        pos = np.linalg.solve(A, b)
        return pos[0], pos[1]
    except np.linalg.LinAlgError:
        return np.nan, np.nan

def procesar_csv(filepath, x_real, y_real):
    """
    Procesa un archivo CSV y aplica trilateración.
    
    Returns:
        DataFrame con columnas: timestamp, anchor, distance_m, x_calc, y_calc, error
    """
    df = pd.read_csv(filepath)
    
    # Pivotar para tener una fila por timestamp con d_A1, d_A2, d_A3
    pivot_df = df.pivot_table(
        index='timestamp_ms',
        columns='anchor',
        values='distance_m',
        aggfunc='first'
    ).reset_index()
    
    # Llenar NaNs con el último valor válido (forward fill, backward fill)
    pivot_df = pivot_df.ffill()
    pivot_df = pivot_df.bfill()
    
    # Aplicar trilateración
    x_calc = []
    y_calc = []
    errors = []
    
    for _, row in pivot_df.iterrows():
        d1 = row.get('A1', np.nan)
        d2 = row.get('A2', np.nan)
        d3 = row.get('A3', np.nan)
        
        if pd.notna(d1) and pd.notna(d2) and pd.notna(d3):
            x, y = trilateracion(d1, d2, d3)
            if not np.isnan(x) and not np.isnan(y):
                x_calc.append(x)
                y_calc.append(y)
                error = np.sqrt((x - x_real)**2 + (y - y_real)**2)
                errors.append(error)
            else:
                x_calc.append(np.nan)
                y_calc.append(np.nan)
                errors.append(np.nan)
        else:
            x_calc.append(np.nan)
            y_calc.append(np.nan)
            errors.append(np.nan)
    
    pivot_df['x_calc'] = x_calc
    pivot_df['y_calc'] = y_calc
    pivot_df['error'] = errors
    pivot_df['x_real'] = x_real
    pivot_df['y_real'] = y_real
    
    return pivot_df

def cargar_todos_datos():
    """
    Carga todos los CSVs de data_crudos y aplica trilateración.
    
    Returns:
        DataFrame consolidado con todos los puntos
    """
    print("📂 Cargando datos crudos...\n")
    
    all_data = []
    
    for filename in sorted(os.listdir(DATA_DIR)):
        if filename.endswith('.csv'):
            # Extraer coordenadas reales del nombre: uwb_X.XXX_Y.YYY.csv
            try:
                name_parts = filename.replace('uwb_', '').replace('.csv', '').split('_')
                x_real = float(name_parts[0])
                y_real = float(name_parts[1])
            except ValueError:
                print(f"⚠️  No se pudo extraer coordenadas de {filename}")
                continue
            
            filepath = os.path.join(DATA_DIR, filename)
            print(f"  ✓ Procesando: {filename} (real: {x_real:.3f}, {y_real:.3f})")
            
            df = procesar_csv(filepath, x_real, y_real)
            all_data.append(df)
    
    consolidated = pd.concat(all_data, ignore_index=True)
    print(f"\n✅ Total de muestras procesadas: {len(consolidated)}\n")
    
    return consolidated

def estadisticas_trilateracion(df):
    """
    Calcula estadísticas de error de trilateración.
    
    Returns:
        dict con estadísticas
    """
    errors = df['error'].dropna()
    
    stats = {
        'total_muestras': len(df),
        'muestras_validas': errors.shape[0],
        'tasa_exito': 100 * errors.shape[0] / len(df),
        'error_promedio': errors.mean(),
        'error_std': errors.std(),
        'error_min': errors.min(),
        'error_max': errors.max(),
        'error_mediana': errors.median(),
        'error_p25': errors.quantile(0.25),
        'error_p75': errors.quantile(0.75),
        'error_p95': errors.quantile(0.95),
    }
    
    return stats, errors

def imprimir_estadisticas(stats):
    """Imprime las estadísticas en la consola."""
    print("=" * 60)
    print("📊 ESTADÍSTICAS DE TRILATERACIÓN")
    print("=" * 60)
    print(f"\nMuestras totales:              {stats['total_muestras']:,}")
    print(f"Muestras válidas:             {stats['muestras_validas']:,}")
    print(f"Tasa de éxito:                {stats['tasa_exito']:.2f}%")
    print(f"\nError RMSE promedio:          {stats['error_promedio']:.4f} m")
    print(f"Desv. estándar:               {stats['error_std']:.4f} m")
    print(f"Error mínimo:                 {stats['error_min']:.4f} m")
    print(f"Error máximo:                 {stats['error_max']:.4f} m")
    print(f"Error mediana:                {stats['error_mediana']:.4f} m")
    print(f"Error percentil 25:           {stats['error_p25']:.4f} m")
    print(f"Error percentil 75:           {stats['error_p75']:.4f} m")
    print(f"Error percentil 95:           {stats['error_p95']:.4f} m")
    print("\n" + "=" * 60 + "\n")

def visualizaciones(df, stats, errors):
    """Genera todas las visualizaciones."""
    
    # 1. Distribución de errores (histograma + KDE)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    
    # Histograma
    axes[0, 0].hist(errors, bins=50, alpha=0.7, color='#3498DB', edgecolor='black')
    axes[0, 0].axvline(stats['error_promedio'], color='red', linestyle='--', linewidth=2.5, 
                       label=f"Media: {stats['error_promedio']:.4f} m")
    axes[0, 0].axvline(stats['error_mediana'], color='green', linestyle='--', linewidth=2.5, 
                       label=f"Mediana: {stats['error_mediana']:.4f} m")
    axes[0, 0].set_xlabel('Error RMSE (metros)', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Distribución de Errores de Trilateración', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # KDE
    errors.plot(kind='density', ax=axes[0, 1], color='#E74C3C', linewidth=2.5)
    axes[0, 1].fill_between(axes[0, 1].get_lines()[0].get_xdata(), 
                             axes[0, 1].get_lines()[0].get_ydata(), alpha=0.3, color='#E74C3C')
    axes[0, 1].set_xlabel('Error RMSE (metros)', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Densidad de Probabilidad del Error', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Box plot
    box_data = [errors]
    bp = axes[1, 0].boxplot(box_data, labels=['Error RMSE'], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#27AE60')
        patch.set_alpha(0.7)
    axes[1, 0].set_ylabel('Error (metros)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Box Plot del Error', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Tabla de estadísticas
    axes[1, 1].axis('off')
    stats_text = f"""
ESTADÍSTICAS DE TRILATERACIÓN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Muestras válidas:      {stats['muestras_validas']:,} / {stats['total_muestras']:,}
Tasa de éxito:         {stats['tasa_exito']:.2f}%

Error Promedio:        {stats['error_promedio']:.4f} m
Desv. Estándar:        {stats['error_std']:.4f} m
Mediana:               {stats['error_mediana']:.4f} m

Rango:                 {stats['error_min']:.4f} — {stats['error_max']:.4f} m
P25:                   {stats['error_p25']:.4f} m
P75:                   {stats['error_p75']:.4f} m
P95:                   {stats['error_p95']:.4f} m
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round', 
                    facecolor='#ECF0F1', alpha=0.8, edgecolor='#2C3E50', linewidth=2))
    
    plt.tight_layout()
    img_path = os.path.join(RESULTS_DIR, '01_distribucion_errores.png')
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: {img_path}")
    plt.close()
    
    # 2. Scatter plot: posiciones reales vs calculadas
    fig, ax = plt.subplots(figsize=(14, 12), dpi=300)
    
    # Área de cobertura
    rect = plt.Rectangle((0, 0), 2.78, 3.0, fill=False, edgecolor='#2C3E50', 
                         linewidth=3, linestyle='--', alpha=0.7, label='Área de cobertura')
    ax.add_patch(rect)
    
    # Anclajes
    anchors_pos = np.array([[0, 0], [2.78, 0], [0, 3]])
    anchor_labels = ['A1 (0.0, 0.0)', 'A2 (2.78, 0.0)', 'A3 (0.0, 3.0)']
    colors_anchors = ['#E74C3C', '#3498DB', '#27AE60']
    for anchor, label, color in zip(anchors_pos, anchor_labels, colors_anchors):
        ax.scatter(anchor[0], anchor[1], s=600, c=color, marker='^', 
                  edgecolors='black', linewidths=2, zorder=10, label=label)
    
    # Puntos reales y calculados
    valid_mask = df['error'].notna()
    scatter1 = ax.scatter(df.loc[valid_mask, 'x_real'], df.loc[valid_mask, 'y_real'], 
                         s=50, c=df.loc[valid_mask, 'error'], cmap='RdYlGn_r', 
                         marker='o', edgecolors='black', linewidth=0.5, alpha=0.6, 
                         label='Posiciones reales', vmin=errors.min(), vmax=errors.max())
    
    scatter2 = ax.scatter(df.loc[valid_mask, 'x_calc'], df.loc[valid_mask, 'y_calc'], 
                         s=30, c=df.loc[valid_mask, 'error'], cmap='RdYlGn_r', 
                         marker='x', linewidth=1, alpha=0.5, label='Posiciones calculadas',
                         vmin=errors.min(), vmax=errors.max())
    
    # Líneas de conexión
    for idx in valid_mask[valid_mask].index[::100]:  # cada 100 puntos para no saturar
        ax.plot([df.loc[idx, 'x_real'], df.loc[idx, 'x_calc']], 
               [df.loc[idx, 'y_real'], df.loc[idx, 'y_calc']], 
               'k-', alpha=0.1, linewidth=0.5)
    
    cbar = plt.colorbar(scatter1, ax=ax, pad=0.02)
    cbar.set_label('Error RMSE (metros)', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Posición X (metros)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Posición Y (metros)', fontsize=12, fontweight='bold')
    ax.set_title('Trilateración: Posiciones Reales vs Calculadas', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    
    plt.tight_layout()
    img_path = os.path.join(RESULTS_DIR, '02_scatter_trilateracion.png')
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: {img_path}")
    plt.close()
    
    # 3. Error vs puntos de medición
    puntos_unicos = df.groupby(['x_real', 'y_real']).agg({
        'error': ['mean', 'std', 'count']
    }).reset_index()
    puntos_unicos.columns = ['x_real', 'y_real', 'error_mean', 'error_std', 'count']
    
    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
    
    scatter = ax.scatter(puntos_unicos['x_real'], puntos_unicos['y_real'], 
                        s=200, c=puntos_unicos['error_mean'], cmap='RdYlGn_r',
                        edgecolors='black', linewidth=2, alpha=0.8, 
                        vmin=puntos_unicos['error_mean'].min(), 
                        vmax=puntos_unicos['error_mean'].max())
    
    # Anotaciones
    for idx, row in puntos_unicos.iterrows():
        ax.annotate(f"{row['error_mean']:.3f}m", 
                   xy=(row['x_real'], row['y_real']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, fontweight='bold', alpha=0.8)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Error Promedio RMSE (metros)', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Posición X (metros)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Posición Y (metros)', fontsize=12, fontweight='bold')
    ax.set_title('Error Promedio por Punto de Medición', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    img_path = os.path.join(RESULTS_DIR, '03_error_por_punto.png')
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: {img_path}")
    plt.close()

def guardar_datos_procesados(df):
    """Guarda los datos procesados en CSV."""
    csv_path = os.path.join(RESULTS_DIR, 'datos_trilateracion_completos.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Guardado: {csv_path}")

def main():
    print("\n" + "="*60)
    print("🔍 ANÁLISIS DE TRILATERACIÓN - UWB")
    print("="*60 + "\n")
    
    # Cargar datos
    df = cargar_todos_datos()
    
    # Estadísticas
    stats, errors = estadisticas_trilateracion(df)
    imprimir_estadisticas(stats)
    
    # Visualizaciones
    print("🎨 Generando visualizaciones...\n")
    visualizaciones(df, stats, errors)
    
    # Guardar datos
    print("\n💾 Guardando datos procesados...\n")
    guardar_datos_procesados(df)
    
    print("="*60)
    print("✅ ANÁLISIS COMPLETADO")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
