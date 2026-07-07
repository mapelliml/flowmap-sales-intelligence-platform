# =============================================================================
# EDA_STORES: Análisis de Estructura de Tiendas
# =============================================================================
# Análisis de tiendas para identificación de oportunidades de optimización
# operativa, segmentación comercial e inventario.
#
# Autor: Senior Data Scientist - Retail & Supply Chain Analytics
# Versión: 1.0 - Optimizado para datasets de 125+ millones de registros
# =============================================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN DE ESTILO PARA GRÁFICOS EJECUTIVOS
# =============================================================================

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('Set2')
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'

# Colores corporativos
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'tertiary': '#F18F01',
    'success': '#2ecc71',
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'info': '#3498db'
}

# =============================================================================
# CONFIGURACIÓN DE RUTAS
# =============================================================================

def setup_paths():
    """Configurar rutas relativas del proyecto"""
    script_path = Path(__file__).parent
    project_root = script_path.parent.parent
    data_raw_path = project_root / 'data' / 'raw'
    reports_path = project_root / 'reports' / 'eda_stores'
    
    # Crear directorio de reports
    reports_path.mkdir(parents=True, exist_ok=True)
    
    return {
        'project_root': project_root,
        'data_raw': data_raw_path,
        'reports': reports_path,
        'train': data_raw_path / 'train.csv',
        'stores': data_raw_path / 'stores.csv'
    }

paths = setup_paths()

print('=' * 80)
print('ANÁLISIS DE ESTRUCTURA DE TIENDAS')
print('=' * 80)
print(f'\n📁 Proyecto: {paths["project_root"]}')
print(f'📁 Datos: {paths["data_raw"]}')
print(f'📄 Train: {paths["train"].name} ({paths["train"].stat().st_size / 1024**3:.2f} GB)')
print(f'📄 Stores: {paths["stores"].name}')
print()

# =============================================================================
# CARGA DE STORES.CSV (archivo pequeño)
# =============================================================================

print('Cargando datos de tiendas...')
stores_df = pd.read_csv(paths['stores'])

print(f'\n✅ Stores cargadas: {len(stores_df):,} tiendas')
print(f'📊 Columnas: {list(stores_df.columns)}')
print(f'\n🔍 Estructura de tiendas:')
print(f'  - Ciudades únicas: {stores_df["city"].nunique()}')
print(f'  - Estados únicos: {stores_df["state"].nunique()}')
print(f'  - Tipos de tienda: {stores_df["type"].nunique()} ({sorted(stores_df["type"].unique())})')
print(f'  - Clusters únicos: {stores_df["cluster"].nunique()}')
print()

# =============================================================================
# ANÁLISIS 1: ESTRUCTURA BÁSICA DE TIENDAS
# =============================================================================

print('=' * 80)
print('1. ESTRUCTURA BÁSICA DE TIENDAS')
print('=' * 80)

total_stores = len(stores_df)
total_cities = stores_df['city'].nunique()
total_clusters = stores_df['cluster'].nunique()
total_types = stores_df['type'].nunique()

print(f'\n📊 MÉTRICAS DE ESTRUCTURA:')
print(f'  • Total de tiendas: {total_stores}')
print(f'  • Ciudades representadas: {total_cities}')
print(f'  • Clusters existentes: {total_clusters}')
print(f'  • Tipos de tienda: {total_types}')

# Distribución por tipo
print(f'\n📊 DISTRIBUCIÓN POR TIPO DE TIENDA:')
tipo_counts = stores_df['type'].value_counts().sort_index()
for tipo, count in tipo_counts.items():
    pct = (count / total_stores * 100)
    print(f'  • Tipo {tipo}: {count} tiendas ({pct:.1f}%)')

# Distribución por cluster
print(f'\n📊 DISTRIBUCIÓN POR CLUSTER:')
cluster_counts = stores_df['cluster'].value_counts().sort_index()
for cluster, count in cluster_counts.items():
    pct = (count / total_stores * 100)
    print(f'  • Cluster {cluster}: {count} tiendas ({pct:.1f}%)')

# Top 5 ciudades con más tiendas
print(f'\n🏙️  TOP 5 CIUDADES POR NÚMERO DE TIENDAS:')
top_cities_stores = stores_df['city'].value_counts().head(5)
for idx, (city, count) in enumerate(top_cities_stores.items(), 1):
    pct = (count / total_stores * 100)
    print(f'  {idx}. {city}: {count} tiendas ({pct:.1f}%)')

# =============================================================================
# PROCESAMIENTO OPTIMIZADO DE TRAIN.CSV EN CHUNKS
# =============================================================================

print('\n' + '=' * 80)
print('PROCESAMIENTO DE TRAIN.CSV EN CHUNKS')
print('=' * 80)

# Estructuras para acumular resultados
ventas_por_store = defaultdict(float)
ventas_por_city = defaultdict(float)
ventas_por_cluster = defaultdict(float)
ventas_por_type = defaultdict(float)
registros_por_store = defaultdict(int)
dias_activos_por_store = defaultdict(set)
total_ventas = 0.0
total_registros = 0

chunk_size = 500_000
chunk_count = 0

print(f'\n📊 Configuración:')
print(f'  - Chunk size: {chunk_size:,} registros')
print(f'  - Columnas procesadas: date, store_nbr, unit_sales')
print(f'  - Merge con stores para: city, type, cluster')
print()

# Leer en chunks con merge optimizado
for chunk in pd.read_csv(
    paths['train'],
    chunksize=chunk_size,
    usecols=['date', 'store_nbr', 'unit_sales'],
    engine='python',
    on_bad_lines='skip'
):
    chunk_count += 1
    
    # Limpieza de datos
    chunk = chunk.dropna(subset=['store_nbr', 'date'])
    chunk['store_nbr'] = pd.to_numeric(chunk['store_nbr'], errors='coerce').fillna(0).astype('int64')
    chunk['unit_sales'] = pd.to_numeric(chunk['unit_sales'], errors='coerce').fillna(0)
    
    total_registros += len(chunk)
    
    # Ventas por tienda
    ventas_store_chunk = chunk.groupby('store_nbr')['unit_sales'].sum()
    for store, ventas in ventas_store_chunk.items():
        ventas_por_store[store] += ventas
        registros_por_store[store] += chunk[chunk['store_nbr'] == store].shape[0]
    
    # Días activos por tienda (para cálculo de promedio diario)
    dias_store = chunk.groupby('store_nbr')['date'].unique()
    for store, dias in dias_store.items():
        dias_activos_por_store[store].update(dias)
    
    # Ventas por ciudad, tipo y cluster (usando merge vectorizado - MÁS RÁPIDO)
    chunk_with_stores = chunk.merge(stores_df[['store_nbr', 'city', 'type', 'cluster']], 
                                     on='store_nbr', how='left')
    
    # Ventas por ciudad
    ventas_city_chunk = chunk_with_stores.groupby('city')['unit_sales'].sum()
    for city, ventas in ventas_city_chunk.items():
        ventas_por_city[city] += ventas
    
    # Ventas por tipo
    ventas_type_chunk = chunk_with_stores.groupby('type')['unit_sales'].sum()
    for store_type, ventas in ventas_type_chunk.items():
        ventas_por_type[store_type] += ventas
    
    # Ventas por cluster
    ventas_cluster_chunk = chunk_with_stores.groupby('cluster')['unit_sales'].sum()
    for cluster, ventas in ventas_cluster_chunk.items():
        ventas_por_cluster[cluster] += ventas
    
    total_ventas += chunk['unit_sales'].sum()
    
    # Mostrar progreso cada 4 chunks
    if chunk_count % 4 == 0:
        print(f'  ✓ Chunk {chunk_count}: {len(chunk):,} filas | Total: {total_registros:,} | Ventas: ${total_ventas:,.0f}')

print(f'\n✅ Procesamiento completado: {chunk_count} chunks, {total_registros:,} registros')
print(f'💰 Ventas totales: ${total_ventas:,.2f}')

# =============================================================================
# CREAR DATAFRAMES PARA ANÁLISIS
# =============================================================================

# Ventas por tienda
ventas_stores_df = pd.DataFrame([
    {
        'store_nbr': store,
        'ventas_totales': ventas,
        'registros': registros_por_store[store],
        'dias_activos': len(dias_activos_por_store[store]),
        'ventas_promedio_diario': ventas / len(dias_activos_por_store[store]) if len(dias_activos_por_store[store]) > 0 else 0
    }
    for store, ventas in ventas_por_store.items()
]).merge(stores_df, on='store_nbr', how='left')

# Calcular participación porcentual
ventas_stores_df['participacion_pct'] = (ventas_stores_df['ventas_totales'] / total_ventas * 100)

# Ventas por ciudad
ventas_ciudad_df = pd.DataFrame([
    {'city': city, 'ventas_totales': ventas}
    for city, ventas in ventas_por_city.items()
]).sort_values('ventas_totales', ascending=False).reset_index(drop=True)
ventas_ciudad_df['participacion_pct'] = (ventas_ciudad_df['ventas_totales'] / total_ventas * 100)

# Ventas por cluster
ventas_cluster_df = pd.DataFrame([
    {'cluster': cluster, 'ventas_totales': ventas}
    for cluster, ventas in ventas_por_cluster.items()
]).sort_values('ventas_totales', ascending=False).reset_index(drop=True)
ventas_cluster_df['participacion_pct'] = (ventas_cluster_df['ventas_totales'] / total_ventas * 100)

# Ventas por tipo
ventas_tipo_df = pd.DataFrame([
    {'type': store_type, 'ventas_totales': ventas}
    for store_type, ventas in ventas_por_type.items()
]).sort_values('ventas_totales', ascending=False).reset_index(drop=True)
ventas_tipo_df['participacion_pct'] = (ventas_tipo_df['ventas_totales'] / total_ventas * 100)

# =============================================================================
# ANÁLISIS 2: RANKING DE TIENDAS
# =============================================================================

print('\n' + '=' * 80)
print('2. RANKING DE TIENDAS POR VENTAS')
print('=' * 80)

# Top 10 tiendas con mayores ventas
print('\n🏆 TOP 10 TIENDAS CON MAYORES VENTAS:')
top_10_ventas = ventas_stores_df.nlargest(10, 'ventas_totales')
for idx, (_, row) in enumerate(top_10_ventas.iterrows(), 1):
    print(f'  {idx}. Tienda {int(row["store_nbr"])} ({row["city"]}, {row["type"]}): ${row["ventas_totales"]:,.0f} ({row["participacion_pct"]:.2f}%)')

# Top 10 tiendas con menores ventas
print('\n📉 TOP 10 TIENDAS CON MENORES VENTAS:')
bottom_10_ventas = ventas_stores_df.nsmallest(10, 'ventas_totales')
for idx, (_, row) in enumerate(bottom_10_ventas.iterrows(), 1):
    print(f'  {idx}. Tienda {int(row["store_nbr"])} ({row["city"]}, {row["type"]}): ${row["ventas_totales"]:,.0f} ({row["participacion_pct"]:.2f}%)')

# =============================================================================
# ANÁLISIS 3: CONCENTRACIÓN DE VENTAS
# =============================================================================

print('\n' + '=' * 80)
print('3. CONCENTRACIÓN DE VENTAS')
print('=' * 80)

# Calcular concentración
ventas_stores_ordenado = ventas_stores_df.sort_values('ventas_totales', ascending=False).reset_index(drop=True)
ventas_stores_ordenado['ventas_acumuladas'] = ventas_stores_ordenado['ventas_totales'].cumsum()
ventas_stores_ordenado['pct_ventas_acum'] = (ventas_stores_ordenado['ventas_acumuladas'] / total_ventas * 100)
ventas_stores_ordenado['pct_stores_acum'] = (np.arange(1, len(ventas_stores_ordenado) + 1) / len(ventas_stores_ordenado) * 100)

# Punto de Pareto (80/20)
punto_80 = ventas_stores_ordenado[ventas_stores_ordenado['pct_ventas_acum'] >= 80].iloc[0]
punto_90 = ventas_stores_ordenado[ventas_stores_ordenado['pct_ventas_acum'] >= 90].iloc[0]

print(f'\n📊 CONCENTRACIÓN DE VENTAS:')
print(f'  • El {punto_80["pct_stores_acum"]:.1f}% de tiendas genera el 80% de ventas')
print(f'  • El {punto_90["pct_stores_acum"]:.1f}% de tiendas genera el 90% de ventas')

# Coeficiente de Gini aproximado
n = len(ventas_stores_ordenado)
cumulative_share = ventas_stores_ordenado['pct_ventas_acum'].values / 100
gini = (2 * np.sum((np.arange(1, n + 1) / n) * cumulative_share)) - (n + 1) / n
print(f'  • Coeficiente de Gini: {gini:.3f} ({'Alta concentración' if gini > 0.5 else 'Concentración moderada' if gini > 0.3 else 'Baja concentración'})')

# =============================================================================
# ANÁLISIS 4: VENTAS POR CIUDAD
# =============================================================================

print('\n' + '=' * 80)
print('4. VENTAS POR CIUDAD')
print('=' * 80)

print(f'\n🏙️  TOP 10 CIUDADES POR VENTAS:')
top_10_ciudades = ventas_ciudad_df.head(10)
for idx, (_, row) in enumerate(top_10_ciudades.iterrows(), 1):
    print(f'  {idx}. {row["city"]}: ${row["ventas_totales"]:,.0f} ({row["participacion_pct"]:.2f}%)')

# Ciudades con menor desempeño
print(f'\n📉 BOTTOM 5 CIUDADES POR VENTAS:')
bottom_5_ciudades = ventas_ciudad_df.tail(5)
for idx, (_, row) in enumerate(bottom_5_ciudades.iterrows(), 1):
    print(f'  {idx}. {row["city"]}: ${row["ventas_totales"]:,.0f} ({row["participacion_pct"]:.2f}%)')

# =============================================================================
# ANÁLISIS 5: VENTAS POR CLUSTER
# =============================================================================

print('\n' + '=' * 80)
print('5. VENTAS POR CLUSTER')
print('=' * 80)

print(f'\n📊 RANKING DE CLUSTERS POR VENTAS:')
clusters_ordenado = ventas_cluster_df.sort_values('ventas_totales', ascending=False)
for idx, (_, row) in enumerate(clusters_ordenado.iterrows(), 1):
    n_stores = len(stores_df[stores_df['cluster'] == row['cluster']])
    print(f'  {idx}. Cluster {row["cluster"]}: ${row["ventas_totales"]:,.0f} ({row["participacion_pct"]:.2f}%) - {n_stores} tiendas')

# =============================================================================
# ANÁLISIS 6: VENTAS POR TIPO DE TIENDA
# =============================================================================

print('\n' + '=' * 80)
print('6. VENTAS POR TIPO DE TIENDA')
print('=' * 80)

print(f'\n🏪 VENTAS POR TIPO DE TIENDA:')
for _, row in ventas_tipo_df.iterrows():
    n_stores = len(stores_df[stores_df['type'] == row['type']])
    print(f'  • Tipo {row["type"]}: ${row["ventas_totales"]:,.0f} ({row["participacion_pct"]:.2f}%) - {n_stores} tiendas')

# =============================================================================
# ANÁLISIS 7: VENTAS PROMEDIO DIARIAS POR TIENDA
# =============================================================================

print('\n' + '=' * 80)
print('7. VENTAS PROMEDIO DIARIAS POR TIENDA')
print('=' * 80)

# Top 10 tiendas por ventas promedio diario
print(f'\n📈 TOP 10 TIENDAS POR VENTAS PROMEDIO DIARIO:')
top_10_diario = ventas_stores_df.nlargest(10, 'ventas_promedio_diario')
for idx, (_, row) in enumerate(top_10_diario.iterrows(), 1):
    print(f'  {idx}. Tienda {int(row["store_nbr"])} ({row["city"]}): ${row["ventas_promedio_diario"]:,.0f}/día')

# Bottom 10 tiendas por ventas promedio diario
print(f'\n📉 BOTTOM 10 TIENDAS POR VENTAS PROMEDIO DIARIO:')
bottom_10_diario = ventas_stores_df.nsmallest(10, 'ventas_promedio_diario')
for idx, (_, row) in enumerate(bottom_10_diario.iterrows(), 1):
    print(f'  {idx}. Tienda {int(row["store_nbr"])} ({row["city"]}): ${row["ventas_promedio_diario"]:,.0f}/día')

# =============================================================================
# ANÁLISIS 8: IDENTIFICACIÓN DE TIENDAS ESTRATÉGICAS
# =============================================================================

print('\n' + '=' * 80)
print('8. IDENTIFICACIÓN DE TIENDAS ESTRATÉGICAS')
print('=' * 80)

# Clasificación de tiendas por importancia
def clasificar_tienda(row):
    if row['participacion_pct'] >= 5:
        return 'CRÍTICA'
    elif row['participacion_pct'] >= 2:
        return 'IMPORTANTE'
    elif row['participacion_pct'] >= 1:
        return 'ESTÁNDAR'
    else:
        return 'BAJO VOLUMEN'

ventas_stores_df['categoria'] = ventas_stores_df.apply(clasificar_tienda, axis=1)

categoria_resumen = ventas_stores_df.groupby('categoria').agg({
    'store_nbr': 'count',
    'ventas_totales': 'sum',
    'ventas_promedio_diario': 'mean'
}).reset_index()
categoria_resumen.columns = ['Categoría', 'Cantidad_Tiendas', 'Ventas_Totales', 'Ventas_Promedio_Diario']
categoria_resumen['Porcentaje_Ventas'] = (categoria_resumen['Ventas_Totales'] / total_ventas * 100)

print(f'\n🎯 CLASIFICACIÓN DE TIENDAS POR IMPORTANCIA:')
print(f'{"Categoría":<15} {"Tiendas":>10} {"% Tiendas":>12} {"Ventas ($)":>18} {"% Ventas":>10}')
print('-' * 70)
for _, row in categoria_resumen.iterrows():
    pct_tiendas = (row['Cantidad_Tiendas'] / total_stores * 100)
    print(f'{row["Categoría"]:<15} {row["Cantidad_Tiendas"]:>10,} {pct_tiendas:>11.2f}% ${row["Ventas_Totales"]:>15,.0f} {row["Porcentaje_Ventas"]:>9.2f}%')

# Tiendas críticas (top 5%)
tiendas_criticas = ventas_stores_df[ventas_stores_df['categoria'] == 'CRÍTICA']
print(f'\n⚠️  TIENDAS CRÍTICAS (>{5}% de participación):')
for _, row in tiendas_criticas.iterrows():
    print(f'  • Tienda {int(row["store_nbr"])} ({row["city"]}, Tipo {row["type"]}): {row["participacion_pct"]:.2f}% de ventas totales')

# =============================================================================
# ANÁLISIS 9: ANÁLISIS PARETO DE TIENDAS
# =============================================================================

print('\n' + '=' * 80)
print('9. ANÁLISIS PARETO DE TIENDAS')
print('=' * 80)

# Calcular acumulado para Pareto
pareto_stores = ventas_stores_df.sort_values('ventas_totales', ascending=False).reset_index(drop=True)
pareto_stores['ventas_acumuladas'] = pareto_stores['ventas_totales'].cumsum()
pareto_stores['pct_acumulado'] = (pareto_stores['ventas_acumuladas'] / total_ventas * 100)
pareto_stores['tiendas_acumuladas'] = np.arange(1, len(pareto_stores) + 1)
pareto_stores['pct_tiendas_acum'] = (pareto_stores['tiendas_acumuladas'] / len(pareto_stores) * 100)

# Puntos clave de Pareto
punto_50 = pareto_stores[pareto_stores['pct_acumulado'] >= 50].iloc[0]
punto_80 = pareto_stores[pareto_stores['pct_acumulado'] >= 80].iloc[0]
punto_95 = pareto_stores[pareto_stores['pct_acumulado'] >= 95].iloc[0]

print(f'\n📊 PUNTOS CLAVE DE PARETO:')
print(f'  • 50% de ventas: {punto_50["tiendas_acumuladas"]} tiendas ({punto_50["pct_tiendas_acum"]:.1f}%)')
print(f'  • 80% de ventas: {punto_80["tiendas_acumuladas"]} tiendas ({punto_80["pct_tiendas_acum"]:.1f}%)')
print(f'  • 95% de ventas: {punto_95["tiendas_acumuladas"]} tiendas ({punto_95["pct_tiendas_acum"]:.1f}%)')

# =============================================================================
# VISUALIZACIONES
# =============================================================================

print('\n' + '=' * 80)
print('GENERANDO VISUALIZACIONES')
print('=' * 80)

# 1. Top 15 tiendas por ventas
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

ax1 = axes[0, 0]
top_15_stores = ventas_stores_df.nlargest(15, 'ventas_totales')
sns.barplot(data=top_15_stores, x='ventas_totales', y='store_nbr', ax=ax1, palette='viridis')
ax1.set_title('Top 15 Tiendas por Ventas Totales', fontsize=14, fontweight='bold')
ax1.set_xlabel('Ventas Totales ($)')
ax1.set_ylabel('Tienda')
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# 2. Ventas por ciudad (top 10)
ax2 = axes[0, 1]
top_10_ciudades_plot = ventas_ciudad_df.head(10)
sns.barplot(data=top_10_ciudades_plot, x='ventas_totales', y='city', ax=ax2, palette='rocket')
ax2.set_title('Top 10 Ciudades por Ventas Totales', fontsize=14, fontweight='bold')
ax2.set_xlabel('Ventas Totales ($)')
ax2.set_ylabel('Ciudad')
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# 3. Ventas por cluster
ax3 = axes[1, 0]
clusters_plot = ventas_cluster_df.sort_values('cluster')
sns.barplot(data=clusters_plot, x='cluster', y='ventas_totales', ax=ax3, palette='coolwarm')
ax3.set_title('Ventas por Cluster', fontsize=14, fontweight='bold')
ax3.set_xlabel('Cluster')
ax3.set_ylabel('Ventas Totales ($)')
ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# 4. Ventas por tipo
ax4 = axes[1, 1]
tipos_plot = ventas_tipo_df.sort_values('type')
sns.barplot(data=tipos_plot, x='type', y='ventas_totales', ax=ax4, palette='Set1')
ax4.set_title('Ventas por Tipo de Tienda', fontsize=14, fontweight='bold')
ax4.set_xlabel('Tipo')
ax4.set_ylabel('Ventas Totales ($)')
ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

plt.tight_layout()
plt.savefig(paths['reports'] / '01_ranking_ventas.png', dpi=150, bbox_inches='tight')
plt.show()

# 5. Curva de Pareto
fig, ax1 = plt.subplots(figsize=(14, 7))

# Mostrar solo las primeras 50 tiendas para claridad
pareto_limit = min(50, len(pareto_stores))
pareto_subset = pareto_stores.head(pareto_limit)

ax1.bar(range(len(pareto_subset)), pareto_subset['ventas_totales'], color=COLORS['primary'], alpha=0.6)
ax1.set_title('Curva de Pareto - Ventas por Tienda', fontsize=14, fontweight='bold')
ax1.set_xlabel('Tiendas (ordenadas por ventas)')
ax1.set_ylabel('Ventas ($)', color=COLORS['primary'])

ax2 = ax1.twinx()
ax2.plot(pareto_subset['pct_tiendas_acum'], pareto_subset['pct_acumulado'], color=COLORS['danger'], linewidth=2, marker='o', markersize=3)
ax2.axhline(y=80, color='green', linestyle='--', alpha=0.7, label='80% ventas')
ax2.axvline(x=punto_80['pct_tiendas_acum'], color='orange', linestyle='--', alpha=0.7, label=f'{punto_80["pct_tiendas_acum"]:.1f}% tiendas')
ax2.set_ylabel('Ventas Acumuladas (%)', color=COLORS['danger'])
ax2.legend()

plt.tight_layout()
plt.savefig(paths['reports'] / '02_pareto_tiendas.png', dpi=150, bbox_inches='tight')
plt.show()

# 6. Distribución de categorías
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax1 = axes[0]
categoria_counts = ventas_stores_df['categoria'].value_counts()
colors_cat = {'CRÍTICA': COLORS['danger'], 'IMPORTANTE': COLORS['warning'], 'ESTÁNDAR': COLORS['info'], 'BAJO VOLUMEN': COLORS['success']}
wedges, texts, autotexts = ax1.pie(categoria_counts.values, labels=categoria_counts.index, autopct='%1.1f%%', 
                                     colors=[colors_cat[cat] for cat in categoria_counts.index])
ax1.set_title('Distribución de Tiendas por Categoría', fontsize=14, fontweight='bold')

ax2 = axes[1]
categoria_ventas = categoria_resumen.sort_values('Porcentaje_Ventas', ascending=False)
sns.barplot(data=categoria_ventas, x='Categoría', y='Porcentaje_Ventas', ax=ax2, 
            palette=[colors_cat[cat] for cat in categoria_ventas['Categoría']])
ax2.set_title('Participación de Ventas por Categoría', fontsize=14, fontweight='bold')
ax2.set_ylabel('Porcentaje de Ventas (%)')
ax2.set_xlabel('Categoría')

plt.tight_layout()
plt.savefig(paths['reports'] / '03_distribucion_categorias.png', dpi=150, bbox_inches='tight')
plt.show()

# 7. Mapa de calor de ventas por ciudad y tipo
fig, ax = plt.subplots(figsize=(12, 8))

# Crear matriz ciudad x tipo
ciudad_tipo_matrix = ventas_stores_df.groupby(['city', 'type'])['ventas_totales'].sum().unstack(fill_value=0)
sns.heatmap(ciudad_tipo_matrix, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax, linewidths=0.5)
ax.set_title('Mapa de Calor: Ventas por Ciudad y Tipo de Tienda', fontsize=14, fontweight='bold')
ax.set_ylabel('Ciudad')
ax.set_xlabel('Tipo de Tienda')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

plt.tight_layout()
plt.savefig(paths['reports'] / '04_heatmap_ciudad_tipo.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# EXPORTAR RESULTADOS
# =============================================================================

print('\n' + '=' * 80)
print('EXPORTANDO RESULTADOS')
print('=' * 80)

# 1. Ventas por tienda
ventas_stores_df[['store_nbr', 'city', 'state', 'type', 'cluster', 'ventas_totales', 
                   'participacion_pct', 'ventas_promedio_diario', 'categoria']].to_csv(
    paths['reports'] / 'ventas_por_tienda.csv', index=False
)
print('✅ ventas_por_tienda.csv exportado')

# 2. Ventas por ciudad
ventas_ciudad_df.to_csv(paths['reports'] / 'ventas_por_ciudad.csv', index=False)
print('✅ ventas_por_ciudad.csv exportado')

# 3. Ventas por cluster
ventas_cluster_df.to_csv(paths['reports'] / 'ventas_por_cluster.csv', index=False)
print('✅ ventas_por_cluster.csv exportado')

# 4. Ventas por tipo
ventas_tipo_df.to_csv(paths['reports'] / 'ventas_por_tipo.csv', index=False)
print('✅ ventas_por_tipo.csv exportado')

# 5. Resumen ejecutivo
resumen_ejecutivo = {
    'fecha_analisis': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'metricas_globales': {
        'total_tiendas': int(total_stores),
        'total_cities': int(total_cities),
        'total_clusters': int(total_clusters),
        'total_types': int(total_types),
        'total_registros_procesados': int(total_registros),
        'total_ventas': float(total_ventas)
    },
    'concentracion': {
        'punto_80_tiendas': int(punto_80['tiendas_acumuladas']),
        'punto_80_porcentaje': float(punto_80['pct_tiendas_acum']),
        'coeficiente_gini': float(gini)
    },
    'top_5_tiendas': ventas_stores_df.nlargest(5, 'ventas_totales')[['store_nbr', 'city', 'ventas_totales', 'participacion_pct']].to_dict('records'),
    'top_5_ciudades': ventas_ciudad_df.head(5).to_dict('records'),
    'top_5_clusters': ventas_cluster_df.head(5).to_dict('records'),
    'clasificacion_tiendas': categoria_resumen.to_dict('records')
}

import json
with open(paths['reports'] / 'resumen_ejecutivo.json', 'w') as f:
    json.dump(resumen_ejecutivo, f, indent=2, default=str)
print('✅ resumen_ejecutivo.json exportado')

# =============================================================================
# CONCLUSIONES EJECUTIVAS Y RECOMENDACIONES
# =============================================================================

print('\n' + '=' * 80)
print('CONCLUSIONES EJECUTIVAS Y RECOMENDACIONES')
print('=' * 80)

# Tienda con más ventas
top_store = ventas_stores_df.iloc[0]
# Tienda con menos ventas
bottom_store = ventas_stores_df.iloc[-1]

print(f'''
📊 RESUMEN EJECUTIVO:

1. ESTRUCTURA DE RED:
   • Total de tiendas: {total_stores}
   • Cobertura geográfica: {total_cities} ciudades en {stores_df['state'].nunique()} estados
   • Segmentación: {total_types} tipos de tienda, {total_clusters} clusters
   • Tienda más grande: Tienda {int(top_store['store_nbr'])} en {top_store['city']} (${top_store['ventas_totales']:,.0f})
   • Tienda más pequeña: Tienda {int(bottom_store['store_nbr'])} en {bottom_store['city']} (${bottom_store['ventas_totales']:,.0f})

2. CONCENTRACIÓN DE VENTAS:
   • El {punto_80['tiendas_acumuladas']} ({punto_80['pct_tiendas_acum']:.1f}%) de tiendas genera el 80% de ventas
   • Coeficiente de Gini: {gini:.3f} ({'Alta concentración - Riesgo operativo' if gini > 0.5 else 'Concentración moderada' if gini > 0.3 else 'Distribución equilibrada'})
   • Tiendas críticas ({len(tiendas_criticas)}): generan {tiendas_criticas['participacion_pct'].sum():.1f}% de ventas

3. OPORTUNIDADES GEOGRÁFICAS:
   • Ciudad líder: {ventas_ciudad_df.iloc[0]['city']} (${ventas_ciudad_df.iloc[0]['ventas_totales']:,.0f})
   • Cluster más rentable: Cluster {ventas_cluster_df.iloc[0]['cluster']} (${ventas_cluster_df.iloc[0]['ventas_totales']:,.0f})
   • Tipo dominante: Tipo {ventas_tipo_df.iloc[0]['type']} (${ventas_tipo_df.iloc[0]['ventas_totales']:,.0f})

💡 RECOMENDACIONES ESTRATÉGICAS:

1. GESTIÓN DE INVENTARIO:
   • Tiendas CRÍTICAS: Implementar sistema de reposición continua (VMI)
   • Tiendas IMPORTANTES: Revisión semanal de stock, niveles de seguridad optimizados
   • Tiendas ESTÁNDAR/BAJO VOLUMEN: Evaluar consolidación de pedidos, reducir frecuencia
   • Priorizar {punto_80['tiendas_acumuladas']} tiendas que generan 80% de ingresos

2. FORECASTING:
   • Modelos diferenciados por cluster (patrones de demanda distintos)
   • Incorporar estacionalidad local por ciudad
   • Tiendas críticas: forecast diario con horizonte corto
   • Tiendas estándar: forecast semanal con horizonte medio

3. OPTIMIZACIÓN OPERATIVA:
   • Analizar viabilidad de tiendas de BAJO VOLUMEN (posible cierre/reattachment)
   • Replicar mejores prácticas de tiendas CRÍTICAS en otras categorías
   • Evaluar expansión en ciudades con alta densidad de tiendas importantes
   • Considerar cambios de formato para tiendas bajo-performantes

4. ESTRATEGIA COMERCIAL:
   • Enfocar inversiones de marketing en tiendas con mayor potencial de crecimiento
   • Desarrollar programas de lealtad diferenciados por segmento de tienda
   • Alinear surtido con perfil demográfico de cada ciudad/cluster
   • Implementar pricing dinámico basado en elasticidad por segmento

5. GESTIÓN DE RIESGOS:
   • Monitorear dependencia de tiendas críticas (riesgo de concentración)
   • Desarrollar planes de contingencia para tiendas estratégicas
   • Diversificar presencia geográfica para reducir riesgos locales
''')

print('=' * 80)
print('✅ ANÁLISIS COMPLETADO - REPORTES GUARDADOS EN:')
print(f'   {paths["reports"]}')
print('=' * 80)