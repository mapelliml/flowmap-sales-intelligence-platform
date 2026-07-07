# =============================================================================
# OPTIMIZACION_INVENTARIO_QTY: Sistema de Optimización de Inventario por Producto
# =============================================================================
# Sistema de optimización de inventario basado en cantidades de producto
# (unidades vendidas), NO en importe monetario. Trabaja exclusivamente con
# productos Clase A obtenidos mediante análisis ABC.
#
# Autor: Lead Data Scientist - Inventory Optimization & Supply Chain
# Versión: 1.0 - Optimizado para datasets grandes con procesamiento en chunks
# =============================================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN DE ESTILO Y COLORES CORPORATIVOS
# =============================================================================

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('Set2')
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11

COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'tertiary': '#F18F01',
    'success': '#2ecc71',
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'info': '#3498db',
    'purple': '#8E44AD',
    'dark_green': '#1E8449'
}

# =============================================================================
# CONFIGURACIÓN DE RUTAS
# =============================================================================

def setup_paths():
    """Configurar rutas del proyecto"""
    script_path = Path(__file__).parent
    project_root = script_path.parent.parent
    
    # Rutas de entrada
    data_raw_path = project_root / 'data' / 'raw'
    
    # Rutas de salida
    inventory_reports = project_root / 'reports' / 'inventory'
    inventory_reports.mkdir(parents=True, exist_ok=True)
    
    visuals_dir = project_root / 'visuals' / 'inventory_qty'
    visuals_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        'project_root': project_root,
        'data_raw': data_raw_path,
        'inventory_reports': inventory_reports,
        'visuals': visuals_dir,
        'train': data_raw_path / 'train.csv',
        'items': data_raw_path / 'items.csv'
    }

paths = setup_paths()

print('=' * 80)
print('SISTEMA DE OPTIMIZACIÓN DE INVENTARIO POR PRODUCTO (UNIDADES)')
print('=' * 80)
print(f'\n📁 Proyecto: {paths["project_root"]}')
print(f'📁 Reports de salida: {paths["inventory_reports"]}')
print(f'📁 Visualizaciones: {paths["visuals"]}')
print()

# =============================================================================
# PARÁMETROS CONFIGURABLES DEL SISTEMA
# =============================================================================

class InventoryParameters:
    """Clase para gestionar parámetros configurables del sistema de inventario"""
    
    def __init__(self):
        # Nivel de servicio (Z-score para 95% = 1.645)
        self.service_level = 0.95
        self.z_score = 1.65  # Aproximación para 95%
        
        # Lead times a simular (días)
        self.lead_times = [2, 5, 7]
        
        # Umbrales de clasificación ABC
        self.abc_thresholds = {
            'A': 0.80,  # 80% del valor acumulado
            'B': 0.95,  # 95% del valor acumulado
            'C': 1.00   # 100% del valor acumulado
        }
        
        # Umbral de riesgo de rotura
        self.stockout_threshold = 0.20  # 20% del ROP
        
    def get_config_summary(self):
        """Retornar resumen de configuración"""
        return {
            'Nivel de servicio': f'{self.service_level * 100:.1f}%',
            'Z-score': f'{self.z_score:.3f}',
            'Lead times': f'{self.lead_times} días',
            'Umbral Clase A': f'{self.abc_thresholds["A"] * 100:.0f}%',
            'Umbral Clase B': f'{self.abc_thresholds["B"] * 100:.0f}%',
        }

params = InventoryParameters()

print('CONFIGURACIÓN DEL SISTEMA:')
for key, value in params.get_config_summary().items():
    print(f'  • {key}: {value}')
print()

# =============================================================================
# CARGA DE DATOS - ITEMS
# =============================================================================

print('=' * 80)
print('CARGANDO DATOS - ITEMS')
print('=' * 80)

items_df = pd.read_csv(paths['items'])
print(f'✅ Items cargados: {len(items_df):,} productos')
print(f'📊 Familias únicas: {items_df["family"].nunique()}')
print()

# =============================================================================
# PROCESAMIENTO DE TRAIN.CSV EN CHUNKS - VENTAS TOTALES POR PRODUCTO
# =============================================================================

print('=' * 80)
print('PROCESANDO TRAIN.CSV - VENTAS TOTALES POR PRODUCTO (ABC)')
print('=' * 80)

# Procesar train.csv en chunks para obtener ventas totales por producto
ventas_por_producto = defaultdict(float)
chunk_size = 1_000_000
total_ventas_global = 0.0
total_registros = 0

print(f'📊 Procesando train.csv en chunks de {chunk_size:,} registros...')

for chunk in pd.read_csv(
    paths['train'],
    chunksize=chunk_size,
    usecols=['item_nbr', 'unit_sales'],
    engine='python',
    on_bad_lines='skip'
):
    chunk['item_nbr'] = pd.to_numeric(chunk['item_nbr'], errors='coerce')
    chunk['unit_sales'] = pd.to_numeric(chunk['unit_sales'], errors='coerce').fillna(0)
    
    # Acumular ventas por producto
    ventas_item = chunk.groupby('item_nbr')['unit_sales'].sum()
    for item_nbr, ventas in ventas_item.items():
        ventas_por_producto[item_nbr] += ventas
    
    total_ventas_global += chunk['unit_sales'].sum()
    total_registros += len(chunk)

print(f'✅ Procesamiento completado: {total_registros:,} registros')
print(f'📊 Productos únicos: {len(ventas_por_producto):,}')
print(f'💰 Ventas totales: {total_ventas_global:,.0f} unidades')
print()

# =============================================================================
# CLASIFICACIÓN ABC
# =============================================================================

print('=' * 80)
print('CLASIFICACIÓN ABC DE PRODUCTOS')
print('=' * 80)

# Crear DataFrame de ventas por producto
abc_df = pd.DataFrame([
    {'item_nbr': item_nbr, 'ventas_totales': ventas}
    for item_nbr, ventas in ventas_por_producto.items()
]).sort_values('ventas_totales', ascending=False).reset_index(drop=True)

# Calcular porcentaje acumulado
abc_df['porcentaje_ventas'] = abc_df['ventas_totales'] / abc_df['ventas_totales'].sum() * 100
abc_df['porcentaje_acumulado'] = abc_df['porcentaje_ventas'].cumsum()

# Clasificar ABC
def clasificar_abc(porcentaje_acumulado):
    if porcentaje_acumulado <= params.abc_thresholds['A'] * 100:
        return 'A'
    elif porcentaje_acumulado <= params.abc_thresholds['B'] * 100:
        return 'B'
    else:
        return 'C'

abc_df['clase'] = abc_df['porcentaje_acumulado'].apply(clasificar_abc)

# Resumen de clasificación ABC
resumen_abc = abc_df['clase'].value_counts().sort_index()
print('\n📊 RESUMEN CLASIFICACIÓN ABC:')
for clase in ['A', 'B', 'C']:
    count = resumen_abc.get(clase, 0)
    pct_productos = count / len(abc_df) * 100
    ventas_clase = abc_df[abc_df['clase'] == clase]['ventas_totales'].sum()
    pct_ventas = ventas_clase / abc_df['ventas_totales'].sum() * 100
    print(f'  Clase {clase}: {count} productos ({pct_productos:.1f}%) - {pct_ventas:.1f}% de ventas')

# Seleccionar productos Clase A
productos_clase_a = abc_df[abc_df['clase'] == 'A']['item_nbr'].tolist()
print(f'\n🎯 Productos Clase A seleccionados: {len(productos_clase_a)}')
print()

# =============================================================================
# PROCESAMIENTO DE TRAIN.CSV - SERIE TEMPORAL POR PRODUCTO (CLASE A)
# =============================================================================

print('=' * 80)
print('PROCESANDO TRAIN.CSV - SERIE TEMPORAL POR PRODUCTO (CLASE A)')
print('=' * 80)

# Estructuras para acumular datos por producto
producto_stats = defaultdict(lambda: {
    'ventas_totales': 0,
    'dias_con_venta': 0,
    'ventas_diarias': [],
    'fechas': set()
})

# Conjunto de productos clase A para filtrado rápido
productos_clase_a_set = set(productos_clase_a)

chunk_size = 500_000
chunk_count = 0

print(f'📊 Procesando serie temporal para {len(productos_clase_a_set):,} productos Clase A...')

for chunk in pd.read_csv(
    paths['train'],
    chunksize=chunk_size,
    usecols=['date', 'item_nbr', 'unit_sales'],
    parse_dates=['date'],
    engine='python',
    on_bad_lines='skip'
):
    chunk_count += 1
    
    # Limpieza
    chunk['item_nbr'] = pd.to_numeric(chunk['item_nbr'], errors='coerce')
    chunk['unit_sales'] = pd.to_numeric(chunk['unit_sales'], errors='coerce').fillna(0)
    
    # Filtrar solo productos Clase A
    chunk = chunk[chunk['item_nbr'].isin(productos_clase_a_set)]
    
    if chunk.empty:
        continue
    
    # Agrupar por producto y fecha
    for (item_nbr, fecha), group in chunk.groupby(['item_nbr', 'date']):
        ventas_dia = group['unit_sales'].sum()
        producto_stats[item_nbr]['ventas_totales'] += ventas_dia
        producto_stats[item_nbr]['dias_con_venta'] += 1
        producto_stats[item_nbr]['ventas_diarias'].append(ventas_dia)
        producto_stats[item_nbr]['fechas'].add(fecha)
    
    if chunk_count % 20 == 0:
        print(f'  ✓ Chunk {chunk_count}')

print(f'✅ Procesamiento completado: {chunk_count} chunks')
print()

# =============================================================================
# CALCULAR ESTADÍSTICAS POR PRODUCTO
# =============================================================================

print('=' * 80)
print('CALCULANDO ESTADÍSTICAS POR PRODUCTO')
print('=' * 80)

# Obtener rango de fechas total
todas_fechas = set()
for item_nbr in productos_clase_a:
    if item_nbr in producto_stats:
        todas_fechas.update(producto_stats[item_nbr]['fechas'])

if todas_fechas:
    fecha_min = min(todas_fechas)
    fecha_max = max(todas_fechas)
    total_dias = (fecha_max - fecha_min).days + 1
else:
    total_dias = 1

print(f'📅 Período analizado: {fecha_min.date()} a {fecha_max.date()} ({total_dias} días)')
print()

# Calcular estadísticas para cada producto
producto_detalle = []

for item_nbr in productos_clase_a:
    if item_nbr not in producto_stats:
        continue
    
    stats = producto_stats[item_nbr]
    ventas_totales = stats['ventas_totales']
    dias_con_venta = stats['dias_con_venta']
    ventas_diarias = stats['ventas_diarias']
    
    # Demanda promedio diaria (sobre todos los días del período)
    demanda_promedio_diaria = ventas_totales / total_dias
    
    # Desviación estándar de demanda diaria
    if len(ventas_diarias) > 1:
        std_demanda = np.std(ventas_diarias, ddof=1)
    else:
        std_demanda = 0
    
    # Coeficiente de variación
    cv = (std_demanda / demanda_promedio_diaria * 100) if demanda_promedio_diaria > 0 else 0
    
    # Obtener información del producto
    item_info = items_df[items_df['item_nbr'] == item_nbr]
    familia = item_info['family'].values[0] if len(item_info) > 0 else 'Unknown'
    clase_producto = item_info['class'].values[0] if len(item_info) > 0 else 0
    perishable = item_info['perishable'].values[0] if len(item_info) > 0 else 0
    
    producto_detalle.append({
        'item_nbr': item_nbr,
        'familia': familia,
        'clase_producto': clase_producto,
        'perishable': perishable,
        'ventas_totales': ventas_totales,
        'dias_con_venta': dias_con_venta,
        'demanda_promedio_diaria': demanda_promedio_diaria,
        'std_demanda': std_demanda,
        'cv': cv,
        'clase_abc': 'A'
    })

productos_df = pd.DataFrame(producto_detalle)
print(f'✅ Estadísticas calculadas para {len(productos_df):,} productos Clase A')
print()

# Mostrar resumen
print('📊 RESUMEN ESTADÍSTICAS PRODUCTOS CLASE A:')
print(f'  • Ventas totales promedio: {productos_df["ventas_totales"].mean():,.0f} unidades')
print(f'  • Demanda diaria promedio: {productos_df["demanda_promedio_diaria"].mean():,.2f} unidades')
print(f'  • Desviación estándar promedio: {productos_df["std_demanda"].mean():,.2f} unidades')
print(f'  • CV promedio: {productos_df["cv"].mean():.2f}%')
print()

# =============================================================================
# CALCULAR STOCK DE SEGURIDAD Y PUNTO DE PEDIDO PARA CADA LEAD TIME
# =============================================================================

print('=' * 80)
print('CALCULANDO STOCK DE SEGURIDAD Y PUNTO DE PEDIDO')
print('=' * 80)

# Estructura para almacenar resultados
inventario_resultados = []

for _, row in productos_df.iterrows():
    item_nbr = row['item_nbr']
    demanda_promedio = row['demanda_promedio_diaria']
    std_demanda = row['std_demanda']
    
    for lead_time in params.lead_times:
        # Stock de Seguridad: SS = Z × std_demanda × √(lead_time)
        stock_seguridad = params.z_score * std_demanda * np.sqrt(lead_time)
        
        # Punto de Pedido: ROP = demanda_promedio × lead_time + stock_seguridad
        punto_pedido = demanda_promedio * lead_time + stock_seguridad
        
        inventario_resultados.append({
            'item_nbr': item_nbr,
            'familia': row['familia'],
            'clase_abc': row['clase_abc'],
            'demanda_promedio_diaria': demanda_promedio,
            'std_demanda': std_demanda,
            'cv': row['cv'],
            'ventas_totales': row['ventas_totales'],
            'dias_con_venta': row['dias_con_venta'],
            'lead_time': lead_time,
            'stock_seguridad': stock_seguridad,
            'punto_pedido': punto_pedido,
            'demanda_during_leadtime': demanda_promedio * lead_time
        })

inventario_df = pd.DataFrame(inventario_resultados)
print(f'✅ Cálculos completados para {len(inventario_df):,} combinaciones producto-lead_time')
print()

# =============================================================================
# SIMULAR STOCK ACTUAL PARA CADA PRODUCTO
# =============================================================================

print('=' * 80)
print('SIMULANDO STOCK ACTUAL')
print('=' * 80)

# Simular stock actual basado en una distribución razonable
# Usamos una distribución normal truncada centrada alrededor del ROP
np.random.seed(42)  # Para reproducibilidad

# Seleccionar lead time principal (7 días) para simulación de stock
lead_time_principal = 7

# Filtrar datos para lead time principal
inventario_principal = inventario_df[inventario_df['lead_time'] == lead_time_principal].copy()

# Simular stock actual
# Estrategia: stock actual varía alrededor del punto de pedido
# Algunos productos tendrán stock bajo (riesgo), otros tendrán exceso
for idx, row in inventario_principal.iterrows():
    rop = row['punto_pedido']
    ss = row['stock_seguridad']
    
    # Simular stock con variabilidad
    # 70% de productos con stock cerca del ROP
    # 15% de productos con stock bajo (riesgo de rotura)
    # 15% de productos con stock alto (exceso)
    random_factor = np.random.random()
    
    if random_factor < 0.15:
        # Stock bajo - riesgo de rotura
        stock_actual = max(0, rop * np.random.uniform(0.3, 0.9))
    elif random_factor < 0.30:
        # Stock moderadamente bajo
        stock_actual = rop * np.random.uniform(0.9, 1.1)
    else:
        # Stock normal o alto
        stock_actual = rop * np.random.uniform(0.8, 1.5)
    
    inventario_principal.at[idx, 'stock_actual'] = round(stock_actual)

print(f'✅ Stock actual simulado para {len(inventario_principal):,} productos')
print()

# =============================================================================
# DETECTAR RIESGO DE ROTURA
# =============================================================================

print('=' * 80)
print('DETECTANDO RIESGO DE ROTURA')
print('=' * 80)

# Riesgo de rotura: stock_actual < ROP
inventario_principal['riesgo_rotura'] = (inventario_principal['stock_actual'] < inventario_principal['punto_pedido']).astype(int)
inventario_principal['stock_faltante'] = np.maximum(0, inventario_principal['punto_pedido'] - inventario_principal['stock_actual'])
inventario_principal['dias_cobertura'] = np.where(
    inventario_principal['demanda_promedio_diaria'] > 0,
    inventario_principal['stock_actual'] / inventario_principal['demanda_promedio_diaria'],
    0
)

# Nivel de riesgo (porcentaje de stock faltante respecto al ROP)
inventario_principal['nivel_riesgo_pct'] = np.where(
    inventario_principal['punto_pedido'] > 0,
    (inventario_principal['stock_faltante'] / inventario_principal['punto_pedido']) * 100,
    0
)

# Clasificar nivel de riesgo
def clasificar_nivel_riesgo(pct):
    if pct == 0:
        return 'Sin Riesgo'
    elif pct < 10:
        return 'Riesgo Bajo'
    elif pct < 25:
        return 'Riesgo Medio'
    else:
        return 'Riesgo Alto'

inventario_principal['nivel_riesgo'] = inventario_principal['nivel_riesgo_pct'].apply(clasificar_nivel_riesgo)

# Resumen de riesgo
riesgo_resumen = inventario_principal['nivel_riesgo'].value_counts()
print('\n📊 RESUMEN DE RIESGO DE ROTURA:')
for nivel in ['Sin Riesgo', 'Riesgo Bajo', 'Riesgo Medio', 'Riesgo Alto']:
    count = riesgo_resumen.get(nivel, 0)
    pct = count / len(inventario_principal) * 100
    print(f'  {nivel}: {count} productos ({pct:.1f}%)')

print(f'\n⚠️  Productos con riesgo de rotura: {inventario_principal["riesgo_rotura"].sum():,}')
print()

# =============================================================================
# CALCULAR CANTIDAD RECOMENDADA DE COMPRA
# =============================================================================

print('=' * 80)
print('CALCULANDO CANTIDADES RECOMENDADAS DE COMPRA')
print('=' * 80)

# compra_recomendada = max(ROP + stock_seguridad - stock_actual, 0)
inventario_principal['compra_recomendada'] = np.maximum(
    0,
    inventario_principal['punto_pedido'] + inventario_principal['stock_seguridad'] - inventario_principal['stock_actual']
)

# Resumen de compras
total_compras = inventario_principal['compra_recomendada'].sum()
productos_a_comprar = (inventario_principal['compra_recomendada'] > 0).sum()

print(f'📦 Total unidades a comprar: {total_compras:,.0f}')
print(f'📦 Productos que requieren compra: {productos_a_comprar:,} ({productos_a_comprar/len(inventario_principal)*100:.1f}%)')
print()

# =============================================================================
# CREAR RANKING DE PRODUCTOS CON MAYOR RIESGO DE ROTURA
# =============================================================================

print('=' * 80)
print('RANKING DE PRODUCTOS CON MAYOR RIESGO DE ROTURA')
print('=' * 80)

# Crear score de riesgo combinado
# Score = nivel_riesgo_pct * peso_ventas (productos con más ventas tienen mayor impacto)
max_ventas = inventario_principal['ventas_totales'].max()
inventario_principal['peso_ventas'] = inventario_principal['ventas_totales'] / max_ventas
inventario_principal['score_riesgo'] = inventario_principal['nivel_riesgo_pct'] * inventario_principal['peso_ventas']

# Ordenar por score de riesgo
ranking_riesgo = inventario_principal.sort_values('score_riesgo', ascending=False).reset_index(drop=True)
ranking_riesgo['ranking'] = ranking_riesgo.index + 1

print('\n🏆 TOP 10 PRODUCTOS CON MAYOR RIESGO:')
for _, row in ranking_riesgo.head(10).iterrows():
    print(f'  {row["ranking"]}. Item {row["item_nbr"]} ({row["familia"]}): '
          f'Riesgo={row["nivel_riesgo_pct"]:.1f}%, '
          f'Faltante={row["stock_faltante"]:,.0f} unidades')
print()

# =============================================================================
# EXPORTAR RESULTADOS
# =============================================================================

print('=' * 80)
print('EXPORTANDO RESULTADOS')
print('=' * 80)

# 1. Inventario de Productos
inventario_export = inventario_principal[[
    'item_nbr', 'familia', 'clase_abc', 'demanda_promedio_diaria', 'std_demanda',
    'cv', 'ventas_totales', 'dias_con_venta', 'lead_time', 'stock_seguridad',
    'punto_pedido', 'stock_actual', 'dias_cobertura'
]].copy()
inventario_export.to_csv(paths['inventory_reports'] / 'inventario_productos.csv', index=False)
print('✅ inventario_productos.csv exportado')

# 2. Riesgo de Rotura
riesgo_export = inventario_principal[[
    'item_nbr', 'familia', 'clase_abc', 'ventas_totales', 'demanda_promedio_diaria',
    'stock_seguridad', 'punto_pedido', 'stock_actual', 'dias_cobertura',
    'stock_faltante', 'nivel_riesgo_pct', 'nivel_riesgo', 'riesgo_rotura', 'score_riesgo'
]].copy()
riesgo_export.to_csv(paths['inventory_reports'] / 'riesgo_rotura.csv', index=False)
print('✅ riesgo_rotura.csv exportado')

# 3. Compras Recomendadas
compras_export = inventario_principal[[
    'item_nbr', 'familia', 'clase_abc', 'ventas_totales', 'demanda_promedio_diaria',
    'stock_seguridad', 'punto_pedido', 'stock_actual', 'stock_faltante',
    'nivel_riesgo', 'compra_recomendada'
]].copy()
compras_export['compra_recomendada'] = compras_export['compra_recomendada'].round(0).astype(int)
compras_export.to_csv(paths['inventory_reports'] / 'compras_recomendadas.csv', index=False)
print('✅ compras_recomendadas.csv exportado')

# 4. Resumen Ejecutivo
resumen_data = []
for familia in inventario_principal['familia'].unique():
    fam_data = inventario_principal[inventario_principal['familia'] == familia]
    resumen_data.append({
        'familia': familia,
        'total_productos': len(fam_data),
        'productos_con_riesgo': int(fam_data['riesgo_rotura'].sum()),
        'pct_productos_riesgo': round(fam_data['riesgo_rotura'].mean() * 100, 2),
        'ventas_totales': round(fam_data['ventas_totales'].sum(), 0),
        'demanda_diaria_total': round(fam_data['demanda_promedio_diaria'].sum(), 2),
        'stock_seguridad_total': round(fam_data['stock_seguridad'].sum(), 2),
        'stock_actual_total': round(fam_data['stock_actual'].sum(), 2),
        'compras_recomendadas_total': round(fam_data['compra_recomendada'].sum(), 0),
        'unidades_faltantes_total': round(fam_data['stock_faltante'].sum(), 0)
    })

resumen_df = pd.DataFrame(resumen_data).sort_values('compras_recomendadas_total', ascending=False)
resumen_df.to_csv(paths['inventory_reports'] / 'resumen_ejecutivo.csv', index=False)
print('✅ resumen_ejecutivo.csv exportado')

print(f'\n📁 Todos los reportes guardados en: {paths["inventory_reports"]}')
print()

# =============================================================================
# GENERAR VISUALIZACIONES
# =============================================================================

print('=' * 80)
print('GENERANDO VISUALIZACIONES')
print('=' * 80)

# 1. Top Productos con Mayor Riesgo de Rotura
fig, ax = plt.subplots(figsize=(14, 8))
top_15_riesgo = ranking_riesgo.head(15)
colores_riesgo = []
for _, row in top_15_riesgo.iterrows():
    if row['nivel_riesgo'] == 'Riesgo Alto':
        colores_riesgo.append(COLORS['danger'])
    elif row['nivel_riesgo'] == 'Riesgo Medio':
        colores_riesgo.append(COLORS['warning'])
    else:
        colores_riesgo.append(COLORS['info'])

barras = ax.barh(
    top_15_riesgo['item_nbr'].astype(str),
    top_15_riesgo['nivel_riesgo_pct'],
    color=colores_riesgo
)
ax.set_xlabel('Nivel de Riesgo (%)', fontsize=12)
ax.set_ylabel('Producto (item_nbr)', fontsize=12)
ax.set_title('Top 15 Productos con Mayor Riesgo de Rotura', fontsize=14, fontweight='bold')
ax.invert_yaxis()
for barra, valor in zip(barras, top_15_riesgo['nivel_riesgo_pct']):
    ax.text(barra.get_width() + 0.5, barra.get_y() + barra.get_height()/2,
            f'{valor:.1f}%', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(paths['visuals'] / '01_top_riesgo_rotura.png', dpi=150, bbox_inches='tight')
plt.close()
print('  ✅ 01_top_riesgo_rotura.png')

# 2. Histograma de Stock de Seguridad
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(inventario_principal['stock_seguridad'], bins=30, color=COLORS['primary'], edgecolor='white', alpha=0.7)
ax.set_xlabel('Stock de Seguridad (unidades)', fontsize=12)
ax.set_ylabel('Frecuencia', fontsize=12)
ax.set_title('Distribución de Stock de Seguridad por Producto', fontsize=14, fontweight='bold')
ax.axvline(x=inventario_principal['stock_seguridad'].mean(), color=COLORS['danger'], linestyle='--', 
           label=f'Media: {inventario_principal["stock_seguridad"].mean():.0f} unidades')
ax.legend()
plt.tight_layout()
plt.savefig(paths['visuals'] / '02_histograma_stock_seguridad.png', dpi=150, bbox_inches='tight')
plt.close()
print('  ✅ 02_histograma_stock_seguridad.png')

# 3. Histograma de Puntos de Pedido
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(inventario_principal['punto_pedido'], bins=30, color=COLORS['secondary'], edgecolor='white', alpha=0.7)
ax.set_xlabel('Punto de Pedido (unidades)', fontsize=12)
ax.set_ylabel('Frecuencia', fontsize=12)
ax.set_title('Distribución de Puntos de Pedido por Producto', fontsize=14, fontweight='bold')
ax.axvline(x=inventario_principal['punto_pedido'].mean(), color=COLORS['danger'], linestyle='--',
           label=f'Media: {inventario_principal["punto_pedido"].mean():.0f} unidades')
ax.legend()
plt.tight_layout()
plt.savefig(paths['visuals'] / '03_histograma_punto_pedido.png', dpi=150, bbox_inches='tight')
plt.close()
print('  ✅ 03_histograma_punto_pedido.png')

# 4. Distribución de Compras Recomendadas
fig, ax = plt.subplots(figsize=(12, 6))
compras_data = inventario_principal[inventario_principal['compra_recomendada'] > 0]['compra_recomendada']
ax.hist(compras_data, bins=30, color=COLORS['tertiary'], edgecolor='white', alpha=0.7)
ax.set_xlabel('Cantidad Recomendada de Compra (unidades)', fontsize=12)
ax.set_ylabel('Frecuencia', fontsize=12)
ax.set_title('Distribución de Compras Recomendadas', fontsize=14, fontweight='bold')
if len(compras_data) > 0:
    ax.axvline(x=compras_data.mean(), color=COLORS['danger'], linestyle='--',
               label=f'Media: {compras_data.mean():.0f} unidades')
    ax.legend()
plt.tight_layout()
plt.savefig(paths['visuals'] / '04_distribucion_compras.png', dpi=150, bbox_inches='tight')
plt.close()
print('  ✅ 04_distribucion_compras.png')

# 5. ABC vs Riesgo de Rotura
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Análisis ABC vs Riesgo de Rotura', fontsize=16, fontweight='bold')

# Gráfico 1: Distribución de riesgo por clase ABC
ax = axes[0]
riesgo_por_clase = inventario_principal.groupby('clase_abc')['riesgo_rotura'].mean() * 100
barras = ax.bar(riesgo_por_clase.index, riesgo_por_clase.values, 
               color=[COLORS['danger'], COLORS['warning'], COLORS['success']])
ax.set_xlabel('Clase ABC', fontsize=12)
ax.set_ylabel('Productos con Riesgo (%)', fontsize=12)
ax.set_title('Porcentaje de Productos con Riesgo por Clase ABC', fontsize=12, fontweight='bold')
for barra, valor in zip(barras, riesgo_por_clase.values):
    ax.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 1,
            f'{valor:.1f}%', ha='center', fontsize=10)

# Gráfico 2: Nivel de riesgo promedio por familia
ax = axes[1]
riesgo_por_familia = inventario_principal.groupby('familia')['nivel_riesgo_pct'].mean().sort_values(ascending=False).head(10)
barras = ax.barh(riesgo_por_familia.index, riesgo_por_familia.values, color=COLORS['primary'])
ax.set_xlabel('Nivel de Riesgo Promedio (%)', fontsize=12)
ax.set_ylabel('Familia', fontsize=12)
ax.set_title('Nivel de Riesgo Promedio por Familia (Top 10)', fontsize=12, fontweight='bold')
for barra, valor in zip(barras, riesgo_por_familia.values):
    ax.text(barra.get_width() + 0.5, barra.get_y() + barra.get_height()/2,
            f'{valor:.1f}%', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(paths['visuals'] / '05_abc_vs_riesgo.png', dpi=150, bbox_inches='tight')
plt.close()
print('  ✅ 05_abc_vs_riesgo.png')

# 6. Dashboard de Resumen
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Dashboard de Inventario - Productos Clase A', fontsize=16, fontweight='bold')

# Gráfico 1: Distribución de días de cobertura
ax = axes[0, 0]
ax.hist(inventario_principal['dias_cobertura'], bins=30, color=COLORS['info'], edgecolor='white', alpha=0.7)
ax.axvline(x=lead_time_principal, color=COLORS['danger'], linestyle='--', label=f'Lead Time ({lead_time_principal} días)')
ax.set_xlabel('Días de Cobertura', fontsize=11)
ax.set_ylabel('Frecuencia', fontsize=11)
ax.set_title('Distribución de Días de Cobertura', fontsize=12, fontweight='bold')
ax.legend()

# Gráfico 2: Stock Actual vs Punto de Pedido
ax = axes[0, 1]
scatter_data = inventario_principal.head(100)  # Mostrar solo 100 para claridad
ax.scatter(scatter_data['punto_pedido'], scatter_data['stock_actual'], alpha=0.6, color=COLORS['primary'])
ax.plot([0, scatter_data['punto_pedido'].max()], [0, scatter_data['punto_pedido'].max()], 
        'r--', label='ROP = Stock Actual')
ax.set_xlabel('Punto de Pedido (unidades)', fontsize=11)
ax.set_ylabel('Stock Actual (unidades)', fontsize=11)
ax.set_title('Stock Actual vs Punto de Pedido', fontsize=12, fontweight='bold')
ax.legend()

# Gráfico 3: Compras recomendadas por familia
ax = axes[1, 0]
compras_por_familia = inventario_principal.groupby('familia')['compra_recomendada'].sum().sort_values(ascending=False).head(10)
barras = ax.barh(compras_por_familia.index, compras_por_familia.values, color=COLORS['tertiary'])
ax.set_xlabel('Compras Recomendadas (unidades)', fontsize=11)
ax.set_ylabel('Familia', fontsize=11)
ax.set_title('Compras Recomendadas por Familia (Top 10)', fontsize=12, fontweight='bold')

# Gráfico 4: Resumen de riesgo
ax = axes[1, 1]
riesgo_counts = inventario_principal['nivel_riesgo'].value_counts()
colores_pie = [COLORS['success'], COLORS['info'], COLORS['warning'], COLORS['danger']]
ax.pie(riesgo_counts.values, labels=riesgo_counts.index, autopct='%1.1f%%', 
       colors=colores_pie[:len(riesgo_counts)], startangle=90)
ax.set_title('Distribución de Niveles de Riesgo', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(paths['visuals'] / '06_dashboard_resumen.png', dpi=150, bbox_inches='tight')
plt.close()
print('  ✅ 06_dashboard_resumen.png')

print(f'\n📁 Visualizaciones guardadas en: {paths["visuals"]}')
print()

# =============================================================================
# CONCLUSIONES EJECUTIVAS Y RECOMENDACIONES
# =============================================================================

print('=' * 80)
print('CONCLUSIONES EJECUTIVAS Y RECOMENDACIONES')
print('=' * 80)

# Calcular métricas globales
total_productos = len(inventario_principal)
productos_con_riesgo = int(inventario_principal['riesgo_rotura'].sum())
pct_riesgo = productos_con_riesgo / total_productos * 100
total_unidades_comprar = inventario_principal['compra_recomendada'].sum()
total_unidades_faltantes = inventario_principal['stock_faltante'].sum()

# Productos con exceso de stock (stock > 1.5 * ROP)
productos_exceso = (inventario_principal['stock_actual'] > inventario_principal['punto_pedido'] * 1.5).sum()
pct_exceso = productos_exceso / total_productos * 100

# Productos críticos (riesgo alto)
productos_criticos = (inventario_principal['nivel_riesgo'] == 'Riesgo Alto').sum()
pct_criticos = productos_criticos / total_productos * 100

print(f'''
╔══════════════════════════════════════════════════════════════════════════════╗
║                           RESUMEN EJECUTIVO                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 MÉTRICAS GLOBALES:
   • Total productos Clase A analizados: {total_productos:,}
   • Productos con riesgo de rotura: {productos_con_riesgo:,} ({pct_riesgo:.1f}%)
   • Productos críticos (riesgo alto): {productos_criticos:,} ({pct_criticos:.1f}%)
   • Productos con exceso de stock: {productos_exceso:,} ({pct_exceso:.1f}%)

📦 UNIDADES:
   • Total unidades a comprar: {total_unidades_comprar:,.0f}
   • Total unidades faltantes: {total_unidades_faltantes:,.0f}
   • Lead time principal: {lead_time_principal} días

═══════════════════════════════════════════════════════════════════════════════

💡 CONCLUSIONES DE NEGOCIO:

1️⃣  PRODUCTOS CRÍTICOS (Riesgo Alto de Rotura):
''')

# Listar productos críticos
productos_criticos_df = inventario_principal[inventario_principal['nivel_riesgo'] == 'Riesgo Alto']
if len(productos_criticos_df) > 0:
    print(f'   ⚠️  Se identificaron {len(productos_criticos_df)} productos con riesgo ALTO:')
    for _, row in productos_criticos_df.head(5).iterrows():
        print(f'      • Item {row["item_nbr"]} ({row["familia"]}): '
              f'Faltan {row["stock_faltante"]:,.0f} unidades')
    if len(productos_criticos_df) > 5:
        print(f'      ... y {len(productos_criticos_df) - 5} productos más')
else:
    print('   ✅ No se identificaron productos con riesgo crítico')

print(f'''
2️⃣  UNIDADES A COMPRAR:
   • Total recomendado: {total_unidades_comprar:,.0f} unidades
   • Productos que requieren compra: {productos_a_comprar:,}
   • Compra promedio por producto: {total_unidades_comprar/productos_a_comprar:,.0f} unidades (cuando aplica)

3️⃣  PRODUCTOS CON EXCESO DE STOCK:
   • {productos_exceso} productos tienen stock > 150% del punto de pedido
   • Esto representa capital inmovilizado innecesario
   • Recomendación: Revisar políticas de compra para estos productos

4️⃣  PRODUCTOS CON RIESGO DE ROTURA:
   • {productos_con_riesgo} productos ({pct_riesgo:.1f}%) tienen stock < punto de pedido
   • Impacto operativo: Posibles pérdidas de ventas y clientes insatisfechos
   • Acción inmediata requerida para {productos_criticos} productos críticos

═══════════════════════════════════════════════════════════════════════════════

📈 IMPACTO OPERATIVO:

''')

# Impacto por familia
print('   IMPACTO POR FAMILIA DE PRODUCTOS:')
for _, row in resumen_df.head(5).iterrows():
    print(f'   • {row["familia"]}:')
    print(f'      - Productos con riesgo: {row["productos_con_riesgo"]} de {row["total_productos"]}')
    print(f'      - Compras recomendadas: {row["compras_recomendadas_total"]:,.0f} unidades')
    print(f'      - Unidades faltantes: {row["unidades_faltantes_total"]:,.0f}')

print(f'''
═══════════════════════════════════════════════════════════════════════════════

🔄 PRÓXIMOS PASOS:

   1. Ejecutar compras urgentes para productos críticos
   2. Revisar políticas de reorden para productos con riesgo medio
   3. Analizar exceso de stock y ajustar niveles objetivo
   4. Implementar monitoreo continuo de niveles de inventario
   5. Expandir análisis a productos Clase B
   6. Integrar con sistema de gestión de almacén (WMS)

═══════════════════════════════════════════════════════════════════════════════
''')

# Tabla resumen final
print('''
╔══════════════════════════════════════════════════════════════════════════════╗
║                         TABLA RESUMEN POR FAMILIA                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
''')

print(f'{"Familia":<20} {"Productos":>10} {"Con Riesgo":>12} {"Compras Rec.":>14} {"Faltantes":>12}')
print('─' * 70)

for _, row in resumen_df.iterrows():
    print(f'{row["familia"]:<20} {row["total_productos"]:>10} {row["productos_con_riesgo"]:>12} '
          f'{row["compras_recomendadas_total"]:>14,.0f} {row["unidades_faltantes_total"]:>12,.0f}')

print()
print('=' * 80)
print('✅ ANÁLISIS DE OPTIMIZACIÓN DE INVENTARIO POR PRODUCTO COMPLETADO')
print('=' * 80)
print(f'\n📁 Reports exportados: {paths["inventory_reports"]}')
print(f'📁 Visualizaciones: {paths["visuals"]}')
print()
print('📊 Archivos generados:')
print('   • inventario_productos.csv')
print('   • riesgo_rotura.csv')
print('   • compras_recomendadas.csv')
print('   • resumen_ejecutivo.csv')
print()
print('📈 Gráficos generados:')
print('   • 01_top_riesgo_rotura.png')
print('   • 02_histograma_stock_seguridad.png')
print('   • 03_histograma_punto_pedido.png')
print('   • 04_distribucion_compras.png')
print('   • 05_abc_vs_riesgo.png')
print('   • 06_dashboard_resumen.png')
print('=' * 80)