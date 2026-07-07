# =============================================================================
# EDA_DEMANDA: Análisis de Demanda y Estructura Comercial
# =============================================================================
# Versión Python script - Análisis completo para forecasting y gestión de stock
# Autor: Senior Data Scientist - Retail & Supply Chain Analytics
# =============================================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
    reports_path = project_root / 'reports' / 'eda_demanda'
    
    # Crear directorio de reports
    reports_path.mkdir(parents=True, exist_ok=True)
    
    return {
        'project_root': project_root,
        'data_raw': data_raw_path,
        'reports': reports_path,
        'train': data_raw_path / 'train.csv',
        'items': data_raw_path / 'items.csv',
        'stores': data_raw_path / 'stores.csv',
        'holidays': data_raw_path / 'holidays_events.csv',
        'transactions': data_raw_path / 'transactions.csv'
    }

paths = setup_paths()

print('=' * 80)
print('ANÁLISIS DE DEMANDA Y ESTRUCTURA COMERCIAL')
print('=' * 80)
print(f'\n📁 Proyecto: {paths["project_root"]}')
print(f'📁 Datos: {paths["data_raw"]}')
print(f'📄 Train: {paths["train"].name} ({paths["train"].stat().st_size / 1024**3:.2f} GB)')
print(f'📄 Items: {paths["items"].name}')
print(f'📄 Stores: {paths["stores"].name}')
print(f'📄 Holidays: {paths["holidays"].name}')
print(f'📄 Transactions: {paths["transactions"].name}')
print()

# =============================================================================
# CARGA DE DATOS AUXILIARES (archivos pequeños)
# =============================================================================

print('Cargando datos auxiliares...')

# Items
items_df = pd.read_csv(paths['items'])
print(f'✅ Items: {len(items_df):,} productos, {items_df["family"].nunique()} familias')

# Stores
stores_df = pd.read_csv(paths['stores'])
print(f'✅ Stores: {len(stores_df):,} tiendas')

# Holidays
holidays_df = pd.read_csv(paths['holidays'], parse_dates=['date'])
print(f'✅ Holidays: {len(holidays_df):,} eventos')

# Transactions
transactions_df = pd.read_csv(paths['transactions'], parse_dates=['date'])
print(f'✅ Transactions: {len(transactions_df):,} registros')

total_productos = len(items_df)

# =============================================================================
# PROCESAMIENTO OPTIMIZADO DE TRAIN.CSV EN CHUNKS
# =============================================================================

print('\n' + '=' * 80)
print('PROCESAMIENTO DE TRAIN.CSV')
print('=' * 80)

# Estructuras para acumular resultados
ventas_por_fecha = defaultdict(float)
ventas_por_year_month = defaultdict(float)
ventas_por_weekday = defaultdict(float)
ventas_por_item = defaultdict(float)
ventas_por_family = defaultdict(float)
ventas_por_store = defaultdict(float)
registros_por_item = defaultdict(int)
promociones_por_fecha = defaultdict(int)
total_ventas = 0.0
total_registros = 0
total_promociones = 0

chunk_size = 500_000
chunk_count = 0

print(f'\n📊 Configuración:')
print(f'  - Chunk size: {chunk_size:,} registros')
print(f'  - Procesando: date, store_nbr, item_nbr, unit_sales, onpromotion')
print()

# Leer en chunks
for chunk in pd.read_csv(
    paths['train'],
    chunksize=chunk_size,
    usecols=['date', 'store_nbr', 'item_nbr', 'unit_sales', 'onpromotion'],
    engine='python',
    on_bad_lines='skip'
):
    chunk_count += 1
    
    # Limpieza de datos
    chunk = chunk.dropna(subset=['item_nbr', 'date'])
    chunk['item_nbr'] = pd.to_numeric(chunk['item_nbr'], errors='coerce').fillna(0).astype('int64')
    chunk['unit_sales'] = pd.to_numeric(chunk['unit_sales'], errors='coerce').fillna(0)
    chunk['onpromotion'] = pd.to_numeric(chunk['onpromotion'], errors='coerce').fillna(0).astype('int8')
    
    # Convertir fecha
    chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
    chunk = chunk.dropna(subset=['date'])
    chunk['year_month'] = chunk['date'].dt.to_period('M')
    chunk['weekday'] = chunk['date'].dt.weekday  # 0=Lunes, 6=Domingo
    
    total_registros += len(chunk)
    
    # Ventas por fecha
    ventas_fecha_chunk = chunk.groupby('date')['unit_sales'].sum()
    for fecha, ventas in ventas_fecha_chunk.items():
        ventas_por_fecha[fecha] += ventas
    
    # Ventas por año-mes
    ventas_ym_chunk = chunk.groupby('year_month')['unit_sales'].sum()
    for ym, ventas in ventas_ym_chunk.items():
        ventas_por_year_month[ym] += ventas
    
    # Ventas por día de la semana
    ventas_wd_chunk = chunk.groupby('weekday')['unit_sales'].sum()
    for wd, ventas in ventas_wd_chunk.items():
        ventas_por_weekday[wd] += ventas
    
    # Ventas por producto
    ventas_item_chunk = chunk.groupby('item_nbr')['unit_sales'].sum()
    for item, ventas in ventas_item_chunk.items():
        ventas_por_item[item] += ventas
        registros_por_item[item] += chunk[chunk['item_nbr'] == item].shape[0]
    
    # Ventas por familia (merge parcial)
    chunk_families = chunk.merge(items_df[['item_nbr', 'family']], on='item_nbr', how='left')
    ventas_family_chunk = chunk_families.groupby('family')['unit_sales'].sum()
    for family, ventas in ventas_family_chunk.items():
        ventas_por_family[family] += ventas
    
    # Ventas por tienda
    ventas_store_chunk = chunk.groupby('store_nbr')['unit_sales'].sum()
    for store, ventas in ventas_store_chunk.items():
        ventas_por_store[store] += ventas
    
    # Promociones por fecha
    promo_fecha_chunk = chunk.groupby('date')['onpromotion'].sum()
    for fecha, promo in promo_fecha_chunk.items():
        promociones_por_fecha[fecha] += promo
    
    total_ventas += chunk['unit_sales'].sum()
    total_promociones += chunk['onpromotion'].sum()
    
    # Mostrar progreso
    if chunk_count % 4 == 0:
        print(f'  ✓ Chunk {chunk_count}: {len(chunk):,} filas | Total: {total_registros:,} | Ventas: ${total_ventas:,.0f}')

print(f'\n✅ Procesamiento completado: {chunk_count} chunks, {total_registros:,} registros')
print(f'💰 Ventas totales: ${total_ventas:,.2f}')
print(f'🎯 Promociones totales: {total_promociones:,}')

# =============================================================================
# CREAR DATAFRAMES PARA ANÁLISIS
# =============================================================================

# Ventas diarias
ventas_diarias_df = pd.DataFrame(list(ventas_por_fecha.items()), columns=['date', 'ventas'])
ventas_diarias_df = ventas_diarias_df.sort_values('date').reset_index(drop=True)
ventas_diarias_df['year_month'] = ventas_diarias_df['date'].dt.to_period('M')
ventas_diarias_df['weekday'] = ventas_diarias_df['date'].dt.weekday
ventas_diarias_df['month'] = ventas_diarias_df['date'].dt.month
ventas_diarias_df['year'] = ventas_diarias_df['date'].dt.year
ventas_diarias_df['week'] = ventas_diarias_df['date'].dt.isocalendar().week

# Ventas mensuales
ventas_mensuales_df = pd.DataFrame([
    {'year_month': ym, 'ventas': ventas}
    for ym, ventas in ventas_por_year_month.items()
]).sort_values('year_month').reset_index(drop=True)

# Ventas por producto
ventas_items_df = pd.DataFrame([
    {'item_nbr': item, 'ventas_totales': ventas, 'registros': registros_por_item[item]}
    for item, ventas in ventas_por_item.items()
]).merge(items_df, on='item_nbr', how='left')

# Ventas por familia
ventas_family_df = pd.DataFrame([
    {'family': family, 'ventas_totales': ventas}
    for family, ventas in ventas_por_family.items()
]).sort_values('ventas_totales', ascending=False).reset_index(drop=True)
ventas_family_df['porcentaje_ventas'] = (ventas_family_df['ventas_totales'] / total_ventas * 100)

# =============================================================================
# ANÁLISIS 1: TENDENCIA DE VENTAS
# =============================================================================

print('\n' + '=' * 80)
print('1. TENDENCIA DE VENTAS')
print('=' * 80)

# Calcular media móvil de 7 días
ventas_diarias_df['mm_7d'] = ventas_diarias_df['ventas'].rolling(window=7, min_periods=1).mean()
ventas_diarias_df['mm_30d'] = ventas_diarias_df['ventas'].rolling(window=30, min_periods=1).mean()

# Tendencia lineal
ventas_diarias_df['dias_desde_inicio'] = (ventas_diarias_df['date'] - ventas_diarias_df['date'].min()).dt.days
slope = np.polyfit(ventas_diarias_df['dias_desde_inicio'], ventas_diarias_df['ventas'], 1)[0]
tendencia_diaria = slope * ventas_diarias_df['dias_desde_inicio'] + np.polyfit(ventas_diarias_df['dias_desde_inicio'], ventas_diarias_df['ventas'], 1)[1]

print(f'\n📈 Tendencia de ventas:')
print(f'  - Pendiente diaria: ${slope:,.2f}/día')
print(f'  - Tendencia: {"CRECIENTE" if slope > 0 else "DECRECIENTE"}')

# Gráfico de tendencia
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Gráfico 1: Ventas diarias con media móvil
axes[0].plot(ventas_diarias_df['date'], ventas_diarias_df['ventas'], alpha=0.3, color='gray', label='Ventas diarias')
axes[0].plot(ventas_diarias_df['date'], ventas_diarias_df['mm_7d'], color=COLORS['primary'], linewidth=2, label='Media 7 días')
axes[0].plot(ventas_diarias_df['date'], ventas_diarias_df['mm_30d'], color=COLORS['secondary'], linewidth=2, label='Media 30 días')
axes[0].set_title('Tendencia de Ventas Diarias', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Ventas ($)')
axes[0].legend()
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)

# Gráfico 2: Ventas mensuales con tendencia
ventas_mensuales_df['ventas_acum'] = ventas_mensuales_df['ventas'].cumsum()
axes[1].bar(range(len(ventas_mensuales_df)), ventas_mensuales_df['ventas'], color=COLORS['primary'], alpha=0.7)
axes[1].plot(range(len(ventas_mensuales_df)), ventas_mensuales_df['ventas'].rolling(3, min_periods=1).mean(), 
            color=COLORS['danger'], linewidth=2, marker='o', label='Media móvil 3 meses')
axes[1].set_title('Ventas Mensuales', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Mes')
axes[1].set_ylabel('Ventas ($)')
axes[1].legend()

plt.tight_layout()
plt.savefig(paths['reports'] / '01_tendencia_ventas.png', dpi=150, bbox_inches='tight')
plt.show()

print('\n💡 CONCLUSIÓN DE NEGOCIO:')
if slope > 0:
    print(f'  Las ventas muestran una tendencia CRECIENTE de ${slope:,.0f} por día.')
    print('  Esto sugiere expansión del negocio o mejora en la demanda.')
else:
    print(f'  Las ventas muestran una tendencia DECRECIENTE de ${abs(slope):,.0f} por día.')
    print('  Se recomienda investigar causas y tomar acciones correctivas.')

# =============================================================================
# ANÁLISIS 2: ESTACIONALIDAD
# =============================================================================

print('\n' + '=' * 80)
print('2. ESTACIONALIDAD')
print('=' * 80)

# Estacionalidad mensual
ventas_por_mes = ventas_diarias_df.groupby('month')['ventas'].mean()
meses_nombres = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']

print(f'\n📊 Estacionalidad mensual (ventas promedio por día):')
for mes, nombre in enumerate(meses_nombres, 1):
    if mes in ventas_por_mes.index:
        print(f'  {nombre}: ${ventas_por_mes[mes]:,.0f}')

# Estacionalidad semanal
ventas_por_weekday = ventas_diarias_df.groupby('weekday')['ventas'].mean()
dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

print(f'\n📊 Estacionalidad semanal (ventas promedio por día):')
for dia, nombre in enumerate(dias_semana):
    if dia in ventas_por_weekday.index:
        print(f'  {nombre}: ${ventas_por_weekday[dia]:,.0f}')

# Gráfico de estacionalidad
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Gráfico 1: Estacionalidad mensual
axes[0].bar(meses_nombres, [ventas_por_mes.get(i, 0) for i in range(1, 13)], color=COLORS['primary'], alpha=0.7)
axes[0].set_title('Estacionalidad Mensual', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Ventas Promedio Diario ($)')
axes[0].set_xlabel('Mes')

# Gráfico 2: Estacionalidad semanal
axes[1].bar(dias_semana, [ventas_por_weekday.get(i, 0) for i in range(7)], color=COLORS['secondary'], alpha=0.7)
axes[1].set_title('Estacionalidad Semanal', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Ventas Promedio Diario ($)')
axes[1].set_xlabel('Día de la Semana')

plt.tight_layout()
plt.savefig(paths['reports'] / '02_estacionalidad.png', dpi=150, bbox_inches='tight')
plt.show()

print('\n💡 CONCLUSIÓN DE NEGOCIO:')
mejor_mes = meses_nombres[ventas_por_mes.idxmax() - 1]
peor_mes = meses_nombres[ventas_por_mes.idxmin() - 1]
mejor_dia = dias_semana[ventas_por_weekday.idxmax()]
peor_dia = dias_semana[ventas_por_weekday.idxmin()]

print(f'  Mejor mes: {mejor_mes} (${ventas_por_mes.max():,.0f}/día)')
print(f'  Peor mes: {peor_mes} (${ventas_por_mes.min():,.0f}/día)')
print(f'  Mejor día: {mejor_dia} (${ventas_por_weekday.max():,.0f})')
print(f'  Peor día: {peor_dia} (${ventas_por_weekday.min():,.0f})')
print(f'  La estacionalidad debe considerarse en:')
print('    - Planificación de inventario')
print('    - Estrategias de marketing')
print('    - Gestión de personal')

# =============================================================================
# ANÁLISIS 3: IMPACTO DE FESTIVOS
# =============================================================================

print('\n' + '=' * 80)
print('3. IMPACTO DE FESTIVOS')
print('=' * 80)

# Merge con holidays
ventas_con_holidays = ventas_diarias_df.merge(holidays_df, on='date', how='left')
ventas_con_holidays['es_festivo'] = ventas_con_holidays['date'].isin(holidays_df['date'])

# Comparar ventas en festivos vs no festivos
ventas_festivos = ventas_con_holidays[ventas_con_holidays['es_festivo'] == True]['ventas'].mean()
ventas_no_festivos = ventas_con_holidays[ventas_con_holidays['es_festivo'] == False]['ventas'].mean()
impacto_festivo = ((ventas_festivos - ventas_no_festivos) / ventas_no_festivos) * 100

print(f'\n🎄 Impacto de festivos:')
print(f'  - Ventas promedio en festivos: ${ventas_festivos:,.0f}')
print(f'  - Ventas promedio en no festivos: ${ventas_no_festivos:,.0f}')
print(f'  - Impacto: {impacto_festivo:+.2f}%')

# Top 5 festivos por ventas
festivos_ventas = ventas_con_holidays[ventas_con_holidays['es_festivo']].nlargest(5, 'ventas')
print(f'\n🏆 Top 5 festivos por ventas:')
for _, row in festivos_ventas.iterrows():
    print(f'  {row["date"].date()} ({row["description"] if pd.notna(row["description"]) else "N/A"}): ${row["ventas"]:,.0f}')

# Gráfico
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Gráfico 1: Comparativa festivos vs no festivos
categorias = ['No Festivos', 'Festivos']
valores = [ventas_no_festivos, ventas_festivos]
colores = [COLORS['info'], COLORS['warning']]
bars = axes[0].bar(categorias, valores, color=colores, alpha=0.7)
axes[0].set_title('Ventas Promedio: Festivos vs No Festivos', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Ventas Promedio ($)')

for bar, val in zip(bars, valores):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height,
                f'${val:,.0f}', ha='center', va='bottom', fontweight='bold')

# Gráfico 2: Ventas en festivos a lo largo del tiempo
festivos_df = ventas_con_holidays[ventas_con_holidays['es_festivo']]
axes[1].scatter(festivos_df['date'], festivos_df['ventas'], color=COLORS['warning'], alpha=0.7, s=50)
axes[1].axhline(y=ventas_no_festivos, color='gray', linestyle='--', label=f'Promedio no festivos (${ventas_no_festivos:,.0f})')
axes[1].set_title('Ventas en Días Festivos', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Ventas ($)')
axes[1].legend()
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig(paths['reports'] / '03_festivos.png', dpi=150, bbox_inches='tight')
plt.show()

print('\n💡 CONCLUSIÓN DE NEGOCIO:')
if impacto_festivo > 0:
    print(f'  Los festivos generan un {impacto_festivo:.1f}% MÁS de ventas que días normales.')
    print('  Oportunidad: Preparar inventario extra y promociones específicas.')
else:
    print(f'  Los festivos generan un {abs(impacto_festivo):.1f}% MENOS de ventas que días normales.')
    print('  Riesgo: Ajustar inventario y personal para evitar sobrecostos.')

# =============================================================================
# ANÁLISIS 4: IMPACTO DE PROMOCIONES
# =============================================================================

print('\n' + '=' * 80)
print('4. IMPACTO DE PROMOCIONES')
print('=' * 80)

# Agregar promociones a ventas diarias
ventas_diarias_df['promociones'] = [promociones_por_fecha.get(fecha, 0) for fecha in ventas_diarias_df['date']]
ventas_diarias_df['es_promocion'] = ventas_diarias_df['promociones'] > 0

# Comparar ventas con y sin promoción
ventas_con_promo = ventas_diarias_df[ventas_diarias_df['es_promocion'] == True]['ventas'].mean()
ventas_sin_promo = ventas_diarias_df[ventas_diarias_df['es_promocion'] == False]['ventas'].mean()
impacto_promo = ((ventas_con_promo - ventas_sin_promo) / ventas_sin_promo) * 100

print(f'\n🎯 Impacto de promociones:')
print(f'  - Ventas promedio CON promoción: ${ventas_con_promo:,.0f}')
print(f'  - Ventas promedio SIN promoción: ${ventas_sin_promo:,.0f}')
print(f'  - Impacto: {impacto_promo:+.2f}%')

# Correlación entre promociones y ventas
correlacion = ventas_diarias_df[['ventas', 'promociones']].corr().iloc[0, 1]
print(f'  - Correlación promociones-ventas: {correlacion:.3f}')

# Gráfico
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Gráfico 1: Comparativa con/sin promoción
categorias = ['Sin Promo', 'Con Promo']
valores = [ventas_sin_promo, ventas_con_promo]
colores = [COLORS['info'], COLORS['success']]
bars = axes[0].bar(categorias, valores, color=colores, alpha=0.7)
axes[0].set_title('Ventas Promedio: Con vs Sin Promoción', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Ventas Promedio ($)')

for bar, val in zip(bars, valores):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height,
                f'${val:,.0f}', ha='center', va='bottom', fontweight='bold')

# Gráfico 2: Scatter promociones vs ventas
axes[1].scatter(ventas_diarias_df['promociones'], ventas_diarias_df['ventas'], alpha=0.5, color=COLORS['primary'])
axes[1].set_title('Relación Promociones vs Ventas', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Número de Promociones')
axes[1].set_ylabel('Ventas ($)')

# Línea de tendencia
z = np.polyfit(ventas_diarias_df['promociones'], ventas_diarias_df['ventas'], 1)
p = np.poly1d(z)
axes[1].plot(ventas_diarias_df['promociones'].sort_values(), p(ventas_diarias_df['promociones'].sort_values()), 
            color=COLORS['danger'], linewidth=2, label=f'Tendencia (r={correlacion:.3f})')
axes[1].legend()

plt.tight_layout()
plt.savefig(paths['reports'] / '04_promociones.png', dpi=150, bbox_inches='tight')
plt.show()

print('\n💡 CONCLUSIÓN DE NEGOCIO:')
if impacto_promo > 0:
    print(f'  Las promociones incrementan las ventas en un {impacto_promo:.1f}%.')
    print('  Recomendación: Evaluar ROI de promociones y optimizar frecuencia.')
else:
    print(f'  Las promociones NO están generando incremento significativo de ventas.')
    print('  Recomendación: Revisar estrategia promocional y segmentación.')

# =============================================================================
# ANÁLISIS 5: VENTAS SEMANALES Y MENSUALES
# =============================================================================

print('\n' + '=' * 80)
print('5. VENTAS SEMANALES Y MENSUALES')
print('=' * 80)

# Ventas por semana
ventas_semanales_df = ventas_diarias_df.groupby('week')['ventas'].sum().reset_index()
ventas_semanales_df.columns = ['semana', 'ventas']

# Ventas por mes (ya calculado)
print(f'\n📊 Top 10 semanas por ventas:')
top_semanas = ventas_semanales_df.nlargest(10, 'ventas')
for _, row in top_semanas.iterrows():
    print(f'  Semana {row["semana"]}: ${row["ventas"]:,.0f}')

print(f'\n📊 Top 10 meses por ventas:')
top_meses = ventas_mensuales_df.nlargest(10, 'ventas')
for _, row in top_meses.iterrows():
    print(f'  {row["year_month"]}: ${row["ventas"]:,.0f}')

# Gráfico
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Gráfico 1: Ventas semanales (últimas 52 semanas)
ventas_semanales_52 = ventas_semanales_df.tail(52)
axes[0].plot(ventas_semanales_52['semana'], ventas_semanales_52['ventas'], color=COLORS['primary'], linewidth=2, marker='o', markersize=3)
axes[0].set_title('Ventas Semanales (Últimas 52 Semanas)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Ventas ($)')
axes[0].set_xlabel('Semana')

# Gráfico 2: Ventas mensuales
axes[1].bar(range(len(ventas_mensuales_df)), ventas_mensuales_df['ventas'], color=COLORS['secondary'], alpha=0.7)
axes[1].set_title('Ventas Mensuales Totales', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Ventas ($)')
axes[1].set_xlabel('Mes')

plt.tight_layout()
plt.savefig(paths['reports'] / '05_ventas_semanales_mensuales.png', dpi=150, bbox_inches='tight')
plt.show()

print('\n💡 CONCLUSIÓN DE NEGOCIO:')
semana_max = top_semanas.iloc[0]
mes_max = top_meses.iloc[0]
print(f'  Mejor semana: Semana {semana_max["semana"]} (${semana_max["ventas"]:,.0f}])')
print(f'  Mejor mes: {mes_max["year_month"]} (${mes_max["ventas"]:,.0f}])')
print('  Estas métricas son clave para:')
print('    - Planificación de capacidad')
print('    - Gestión de inventario estacional')
print('    - Presupuestación anual')

# =============================================================================
# ANÁLISIS 6: ABC ANALYSIS + PARETO + FAMILIAS (VALOR PARA OPERACIONES)
# =============================================================================

print('\n' + '=' * 80)
print('6. ABC ANALYSIS + PARETO + FAMILIAS DE PRODUCTO')
print('=' * 80)
print('\n🎯 ANÁLISIS CRÍTICO PARA GESTIÓN DE INVENTARIO Y STOCK')

# Calcular participación de cada producto
ventas_items_df['participacion'] = (ventas_items_df['ventas_totales'] / total_ventas * 100)

# Ordenar para Pareto
pareto_df = ventas_items_df.sort_values('ventas_totales', ascending=False).reset_index(drop=True)
pareto_df['ventas_acumuladas'] = pareto_df['ventas_totales'].cumsum()
pareto_df['pct_acumulado'] = (pareto_df['ventas_acumuladas'] / total_ventas * 100)
pareto_df['productos_acumulados'] = np.arange(1, len(pareto_df) + 1)
pareto_df['pct_productos_acum'] = (pareto_df['productos_acumulados'] / len(pareto_df) * 100)

# Clasificación ABC
def clasificar_abc(pct_acum):
    if pct_acum <= 80:
        return 'A'
    elif pct_acum <= 95:
        return 'B'
    else:
        return 'C'

pareto_df['clase_abc'] = pareto_df['pct_acumulado'].apply(clasificar_abc)

# Resumen ABC
abc_summary = pareto_df.groupby('clase_abc').agg({
    'item_nbr': 'count',
    'ventas_totales': 'sum'
}).reset_index()
abc_summary.columns = ['Clase', 'Cantidad_Productos', 'Ventas_Totales']
abc_summary['Porcentaje_Productos'] = (abc_summary['Cantidad_Productos'] / len(pareto_df) * 100)
abc_summary['Porcentaje_Ventas'] = (abc_summary['Ventas_Totales'] / total_ventas * 100)

print(f'\n📊 CLASIFICACIÓN ABC:')
print(f'{"Clase":<8} {"Productos":>10} {"% Productos":>12} {"Ventas ($)":>15} {"% Ventas":>10}')
print('-' * 60)
for _, row in abc_summary.iterrows():
    print(f'{row["Clase"]:<8} {row["Cantidad_Productos"]:>10,} {row["Porcentaje_Productos"]:>11.2f}% ${row["Ventas_Totales"]:>13,.0f} {row["Porcentaje_Ventas"]:>9.2f}%')

# Punto de Pareto
punto_80 = pareto_df[pareto_df['pct_acumulado'] >= 80].iloc[0]
print(f'\n🎯 PUNTO DE PARETO (80/20):')
print(f'  - {punto_80["productos_acumulados"]:,.0f} productos ({punto_80["pct_productos_acum"]:.2f}%) generan el 80% de ventas')

# Top familias
print(f'\n🏆 TOP 10 FAMILIAS POR VENTAS:')
top_familias = ventas_family_df.head(10)
for idx, (_, row) in enumerate(top_familias.iterrows(), 1):
    print(f'  {idx}. {row["family"]}: ${row["ventas_totales"]:,.0f} ({row["porcentaje_ventas"]:.2f}%)')

# Gráficos ABC + Pareto
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Gráfico 1: Curva de Pareto
ax1 = axes[0, 0]
ax1.bar(range(len(pareto_df)), pareto_df['ventas_totales'], color=COLORS['primary'], alpha=0.6)
ax1_twin = ax1.twinx()
ax1_twin.plot(pareto_df['pct_productos_acum'], pareto_df['pct_acumulado'], color=COLORS['danger'], linewidth=2, marker='o', markersize=3)
ax1_twin.axhline(y=80, color='green', linestyle='--', alpha=0.7)
ax1_twin.axvline(x=punto_80['pct_productos_acum'], color='orange', linestyle='--', alpha=0.7)
ax1.set_title('Curva de Pareto', fontsize=14, fontweight='bold')
ax1.set_xlabel('Productos (ordenados por ventas)')
ax1.set_ylabel('Ventas ($)', color=COLORS['primary'])
ax1_twin.set_ylabel('Ventas Acumuladas (%)', color=COLORS['danger'])
ax1.set_xlim(0, min(500, len(pareto_df)))  # Mostrar solo los primeros 500 productos

# Gráfico 2: Distribución ABC
ax2 = axes[0, 1]
colors_abc = {'A': COLORS['success'], 'B': COLORS['warning'], 'C': COLORS['danger']}
for clase in ['A', 'B', 'C']:
    clase_data = abc_summary[abc_summary['Clase'] == clase]
    if not clase_data.empty:
        ax2.bar(clase, clase_data['Porcentaje_Productos'].values[0], color=colors_abc[clase], alpha=0.7)
ax2.set_title('Distribución de Productos por Clase ABC', fontsize=14, fontweight='bold')
ax2.set_ylabel('Porcentaje de Productos (%)')

# Gráfico 3: Top 15 familias
ax3 = axes[1, 0]
top_15_familias = ventas_family_df.head(15)
sns.barplot(data=top_15_familias, x='ventas_totales', y='family', ax=ax3, palette='viridis')
ax3.set_title('Top 15 Familias por Ventas', fontsize=14, fontweight='bold')
ax3.set_xlabel('Ventas Totales ($)')
ax3.set_ylabel('Familia')
ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Gráfico 4: Ventas acumuladas por familia
ax4 = axes[1, 1]
ventas_family_df['ventas_acum'] = ventas_family_df['ventas_totales'].cumsum()
ventas_family_df['pct_acum'] = (ventas_family_df['ventas_acum'] / total_ventas * 100)
ax4.plot(range(len(ventas_family_df)), ventas_family_df['pct_acum'], color=COLORS['primary'], linewidth=2)
ax4.axhline(y=80, color='green', linestyle='--', alpha=0.7, label='80% ventas')
ax4.set_title('Ventas Acumuladas por Familia', fontsize=14, fontweight='bold')
ax4.set_xlabel('Familias (ordenadas por ventas)')
ax4.set_ylabel('Ventas Acumuladas (%)')
ax4.legend()

plt.tight_layout()
plt.savefig(paths['reports'] / '06_abc_pareto_familias.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# EXPORTAR RESULTADOS
# =============================================================================

print('\n' + '=' * 80)
print('EXPORTANDO RESULTADOS')
print('=' * 80)

# 1. Clasificación ABC completa
pareto_df[['item_nbr', 'family', 'class', 'ventas_totales', 'participacion', 'clase_abc']].to_csv(
    paths['reports'] / 'clasificacion_abc.csv', index=False
)
print('✅ Clasificación ABC exportada')

# 2. Ventas por familia
ventas_family_df.to_csv(paths['reports'] / 'ventas_por_familia.csv', index=False)
print('✅ Ventas por familia exportadas')

# 3. Ventas diarias
ventas_diarias_df[['date', 'ventas', 'mm_7d', 'mm_30d', 'es_promocion', 'promociones']].to_csv(
    paths['reports'] / 'ventas_diarias.csv', index=False
)
print('✅ Ventas diarias exportadas')

# 4. Ventas mensuales
ventas_mensuales_df.to_csv(paths['reports'] / 'ventas_mensuales.csv', index=False)
print('✅ Ventas mensuales exportadas')

# 5. Resumen ejecutivo en JSON
import json

resumen_ejecutivo = {
    'fecha_analisis': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'metricas_globales': {
        'total_registros': int(total_registros),
        'total_ventas': float(total_ventas),
        'total_productos': int(len(ventas_items_df)),
        'total_familias': int(len(ventas_family_df)),
        'total_promociones': int(total_promociones)
    },
    'tendencia': {
        'pendiente_diaria': float(slope),
        'direccion': 'CRECIENTE' if slope > 0 else 'DECRECIENTE'
    },
    'estacionalidad': {
        'mejor_mes': mejor_mes,
        'peor_mes': peor_mes,
        'mejor_dia': mejor_dia,
        'peor_dia': peor_dia
    },
    'festivos': {
        'ventas_promedio_festivos': float(ventas_festivos),
        'ventas_promedio_no_festivos': float(ventas_no_festivos),
        'impacto_porcentaje': float(impacto_festivo)
    },
    'promociones': {
        'ventas_con_promo': float(ventas_con_promo),
        'ventas_sin_promo': float(ventas_sin_promo),
        'impacto_porcentaje': float(impacto_promo),
        'correlacion': float(correlacion)
    },
    'abc_analysis': {
        'productos_a': int(abc_summary[abc_summary['Clase'] == 'A']['Cantidad_Productos'].values[0]),
        'productos_b': int(abc_summary[abc_summary['Clase'] == 'B']['Cantidad_Productos'].values[0]),
        'productos_c': int(abc_summary[abc_summary['Clase'] == 'C']['Cantidad_Productos'].values[0]),
        'punto_pareto_productos': int(punto_80['productos_acumulados']),
        'punto_pareto_porcentaje': float(punto_80['pct_productos_acum'])
    },
    'top_5_familias': ventas_family_df.head(5)[['family', 'ventas_totales', 'porcentaje_ventas']].to_dict('records')
}

with open(paths['reports'] / 'resumen_ejecutivo.json', 'w') as f:
    json.dump(resumen_ejecutivo, f, indent=2, default=str)
print('✅ Resumen ejecutivo exportado')

# =============================================================================
# RESUMEN FINAL Y RECOMENDACIONES
# =============================================================================

print('\n' + '=' * 80)
print('RESUMEN EJECUTIVO Y RECOMENDACIONES')
print('=' * 80)

print(f'''
📊 MÉTRICAS CLAVE:
   • Total registros: {total_registros:,}
   • Ventas totales: ${total_ventas:,.2f}
   • Productos únicos: {len(ventas_items_df):,}
   • Familias: {len(ventas_family_df)}
   • Período: {ventas_diarias_df["date"].min().date()} a {ventas_diarias_df["date"].max().date()}

📈 TENDENCIA:
   • Pendiente diaria: ${slope:,.2f}/día ({'↑ CRECIENTE' if slope > 0 else '↓ DECRECIENTE'})

🎯 CLASIFICACIÓN ABC:
   • Clase A: {abc_summary[abc_summary["Clase"] == "A"]["Cantidad_Productos"].values[0]:,.0f} productos ({abc_summary[abc_summary["Clase"] == "A"]["Porcentaje_Productos"].values[0]:.1f}%) → {abc_summary[abc_summary["Clase"] == "A"]["Porcentaje_Ventas"].values[0]:.1f}% ventas
   • Clase B: {abc_summary[abc_summary["Clase"] == "B"]["Cantidad_Productos"].values[0]:,.0f} productos ({abc_summary[abc_summary["Clase"] == "B"]["Porcentaje_Productos"].values[0]:.1f}%) → {abc_summary[abc_summary["Clase"] == "B"]["Porcentaje_Ventas"].values[0]:.1f}% ventas
   • Clase C: {abc_summary[abc_summary["Clase"] == "C"]["Cantidad_Productos"].values[0]:,.0f} productos ({abc_summary[abc_summary["Clase"] == "C"]["Porcentaje_Productos"].values[0]:.1f}%) → {abc_summary[abc_summary["Clase"] == "C"]["Porcentaje_Ventas"].values[0]:.1f}% ventas

💡 RECOMENDACIONES ESTRATÉGICAS:

1. GESTIÓN DE INVENTARIO:
   • Productos Clase A: Revisión diaria, stock de seguridad alto
   • Productos Clase B: Revisión semanal, niveles optimizados
   • Productos Clase C: Revisión mensual, evaluar eliminación

2. FORECASTING:
   • Incorporar estacionalidad mensual y semanal
   • Ajustar por festivos ({impacto_festivo:+.1f}% impacto)
   • Considerar efecto promociones ({impacto_promo:+.1f}% impacto)

3. OPERACIONES:
   • Enfocar {punto_80["productos_acumulados"]:,.0f} productos que generan 80% de ingresos
   • Optimizar espacio en almacén según clasificación ABC
   • Implementar políticas de reposición diferenciadas

4. ESTRATEGIA COMERCIAL:
   • Fortalecer top {len(top_familias)} familias que generan mayor valor
   • Evaluar ROI de promociones (correlación: {correlacion:.3f})
   • Preparar inventario extra para festivos de alta demanda
''')

print('=' * 80)
print('✅ ANÁLISIS COMPLETADO - REPORTES GUARDADOS EN:')
print(f'   {paths["reports"]}')
print('=' * 80)