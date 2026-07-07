# =============================================================================
# EDA_PRODUCTOS: Análisis de Estructura Comercial
# =============================================================================
# Versión Python script para evitar problemas con el notebook

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo para gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('Set2')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

# =============================================================================
# CONFIGURACIÓN DE RUTAS
# =============================================================================

script_path = Path(__file__).parent
project_root = script_path.parent.parent
data_raw_path = project_root / 'data' / 'raw'

train_file = data_raw_path / 'train.csv'
items_file = data_raw_path / 'items.csv'

print('=' * 80)
print('CONFIGURACIÓN DEL ENTORNO')
print('=' * 80)
print(f'\n📁 Root del proyecto: {project_root}')
print(f'📁 Datos raw: {data_raw_path}')
print(f'📄 Train: {train_file.name} ({train_file.stat().st_size / 1024**3:.2f} GB)')
print(f'📄 Items: {items_file.name} ({items_file.stat().st_size / 1024:.1f} KB)')
print()

# =============================================================================
# CARGA DE ITEMS.CSV
# =============================================================================

print('Cargando catálogo de productos...')
items_df = pd.read_csv(items_file)

print(f'\n✅ Items cargados: {len(items_df):,}')
print(f'📊 Columnas: {list(items_df.columns)}')
print(f'\nPrimeros 5 productos:')
print(items_df.head())

total_productos = len(items_df)
total_perecederos = (items_df['perishable'] == 1).sum()
total_no_perecederos = (items_df['perishable'] == 0).sum()
pct_perecederos = (total_perecederos / total_productos * 100)
pct_no_perecederos = (total_no_perecederos / total_productos * 100)

print(f'\n🔍 Valores únicos:')
print(f'  - Familias: {items_df["family"].nunique()}')
print(f'  - Clases: {items_df["class"].nunique()}')
print(f'  - Perecederos (1): {total_perecederos} ({pct_perecederos:.2f}%)')
print(f'  - No perecederos (0): {total_no_perecederos} ({pct_no_perecederos:.2f}%)')

# =============================================================================
# PROCESAMIENTO OPTIMIZADO DE TRAIN.CSV EN CHUNKS
# =============================================================================

print('\n' + '=' * 80)
print('PROCESAMIENTO DE TRAIN.CSV EN CHUNKS')
print('=' * 80)

# Estructuras para acumular resultados
ventas_por_item = defaultdict(float)
registros_por_item = defaultdict(int)
total_ventas = 0.0
total_registros = 0

chunk_size = 500_000
chunk_count = 0

print(f'\n📊 Configuración:')
print(f'  - Chunk size: {chunk_size:,} registros')
print(f'  - Columnas a procesar: item_nbr, unit_sales, onpromotion')
print()

# Leer en chunks sin especificar dtype para evitar errores con NA
for chunk in pd.read_csv(
    train_file,
    chunksize=chunk_size,
    usecols=['item_nbr', 'unit_sales', 'onpromotion'],
    engine='python',
    on_bad_lines='skip'
):
    chunk_count += 1
    
    # Limpieza de datos
    chunk = chunk.dropna(subset=['item_nbr'])
    chunk['onpromotion'] = chunk['onpromotion'].fillna(0)
    chunk['item_nbr'] = pd.to_numeric(chunk['item_nbr'], errors='coerce').fillna(0).astype('int64')
    chunk['unit_sales'] = pd.to_numeric(chunk['unit_sales'], errors='coerce').fillna(0)
    chunk['onpromotion'] = pd.to_numeric(chunk['onpromotion'], errors='coerce').fillna(0)
    
    total_registros += len(chunk)
    
    # Agrupar por item_nbr y sumar ventas
    ventas_por_item_chunk = chunk.groupby('item_nbr')['unit_sales'].sum()
    for item_nbr, ventas in ventas_por_item_chunk.items():
        ventas_por_item[item_nbr] += ventas
    
    # Contar registros por item
    registros_por_item_chunk = chunk.groupby('item_nbr').size()
    for item_nbr, count in registros_por_item_chunk.items():
        registros_por_item[item_nbr] += count
    
    # Sumar ventas totales del chunk
    total_ventas += chunk['unit_sales'].sum()
    
    # Mostrar progreso cada 4 chunks
    if chunk_count % 4 == 0:
        print(f'  ✓ Chunk {chunk_count}: {len(chunk):,} filas | Total: {total_registros:,} filas | Ventas: ${total_ventas:,.0f}')

print(f'\n✅ Procesamiento completado: {chunk_count} chunks, {total_registros:,} registros')
print(f'💰 Ventas totales: ${total_ventas:,.2f}')

# Convertir a DataFrame
ventas_items_df = pd.DataFrame([
    {'item_nbr': item, 'ventas_totales': ventas, 'registros': registros_por_item[item]}
    for item, ventas in ventas_por_item.items()
])

# Merge con items_df
ventas_items_df = ventas_items_df.merge(items_df, on='item_nbr', how='left')

# Calcular ventas por familia
ventas_family_df = ventas_items_df.groupby('family').agg({
    'ventas_totales': 'sum',
    'item_nbr': 'nunique',
    'registros': 'sum'
}).reset_index()
ventas_family_df.columns = ['family', 'ventas_totales', 'num_skus', 'total_registros']
ventas_family_df['porcentaje_ventas'] = (ventas_family_df['ventas_totales'] / total_ventas * 100)

# Calcular ventas por perecederos vs no perecederos
ventas_perishable_df = ventas_items_df.groupby('perishable').agg({
    'ventas_totales': 'sum',
    'item_nbr': 'nunique'
}).reset_index()
ventas_perishable_df.columns = ['perishable', 'ventas_totales', 'num_skus']
ventas_perishable_df['porcentaje_ventas'] = (ventas_perishable_df['ventas_totales'] / total_ventas * 100)

# =============================================================================
# RESULTADOS PRINCIPALES
# =============================================================================

print('\n' + '=' * 80)
print('RESULTADOS PRINCIPALES')
print('=' * 80)

# Top 5 familias
print('\n🏆 TOP 5 FAMILIAS POR VENTAS:')
top_familias = ventas_family_df.nlargest(5, 'ventas_totales')
for _, row in top_familias.iterrows():
    print(f'  {row["family"]}: ${row["ventas_totales"]:,.0f} ({row["porcentaje_ventas"]:.2f}%)')

# Top 5 productos
print('\n📦 TOP 5 PRODUCTOS MÁS VENDIDOS:')
top_productos = ventas_items_df.nlargest(5, 'ventas_totales')
for idx, (_, row) in enumerate(top_productos.iterrows(), 1):
    pct = (row['ventas_totales'] / total_ventas * 100)
    print(f'  {idx}. Item {row["item_nbr"]} ({row["family"]}): ${row["ventas_totales"]:,.0f} ({pct:.3f}%)')

# Clasificación ABC
ventas_items_df['participacion'] = (ventas_items_df['ventas_totales'] / total_ventas * 100)
pareto_df = ventas_items_df.sort_values('ventas_totales', ascending=False).reset_index(drop=True)
pareto_df['ventas_acumuladas'] = pareto_df['ventas_totales'].cumsum()
pareto_df['pct_acumulado'] = (pareto_df['ventas_acumuladas'] / total_ventas * 100)

def clasificar_abc(pct_acum):
    if pct_acum <= 80:
        return 'A'
    elif pct_acum <= 95:
        return 'B'
    else:
        return 'C'

pareto_df['clase_abc'] = pareto_df['pct_acumulado'].apply(clasificar_abc)

productos_a = len(pareto_df[pareto_df['clase_abc'] == 'A'])
productos_b = len(pareto_df[pareto_df['clase_abc'] == 'B'])
productos_c = len(pareto_df[pareto_df['clase_abc'] == 'C'])

print(f'\n📊 CLASIFICACIÓN ABC:')
print(f'  Clase A: {productos_a:,} productos')
print(f'  Clase B: {productos_b:,} productos')
print(f'  Clase C: {productos_c:,} productos')

# Exportar resultados
results_dir = project_root / 'reports' / 'eda_productos'
results_dir.mkdir(parents=True, exist_ok=True)

ventas_family_df.to_csv(results_dir / 'ventas_por_familia.csv', index=False)
top_productos.to_csv(results_dir / 'top_20_productos.csv', index=False)
pareto_df[['item_nbr', 'family', 'ventas_totales', 'clase_abc']].to_csv(results_dir / 'clasificacion_abc.csv', index=False)

print(f'\n✅ Resultados exportados a: {results_dir}')
print('=' * 80)
print('✅ ANÁLISIS COMPLETADO')
print('=' * 80)