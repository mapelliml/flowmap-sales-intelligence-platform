# ==============================================================================
# ANÁLISIS OPTIMIZADO DE COBERTURA TEMPORAL (VERSIÓN CHUNKED CON GROUPBY)
# ==============================================================================
# Versión más eficiente usando groupby en cada chunk en lugar de iterar

from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURACIÓN Y RUTAS
# ==============================================================================

script_path = Path(__file__).parent
project_root = script_path.parent.parent
data_raw_path = project_root / "data" / "raw"
train_file = data_raw_path / "train.csv"

print("=" * 80)
print("ANÁLISIS OPTIMIZADO DE COBERTURA TEMPORAL")
print("=" * 80)
print(f"\nArchivo: {train_file}")
print(f"Tamaño: {train_file.stat().st_size / 1024**3:.2f} GB\n")

# ==============================================================================
# CONVERTSOR PARA ONPROMOTION
# ==============================================================================

def convert_onpromotion(value):
    if isinstance(value, str):
        return 1 if value.strip().lower() == 'true' else 0
    return int(value) if pd.notna(value) else 0

# ==============================================================================
# LEER EN CHUNKS Y ACUMULAR CON GROUPBY (MÁS EFICIENTE)
# ==============================================================================

print("Leyendo archivo en chunks y agregando con groupby...")
print("-" * 80)

fecha_min = None
fecha_max = None
registros_por_year_month = defaultdict(int)
registros_por_year = defaultdict(int)
calidad_datos = defaultdict(lambda: {
    'total': 0,
    'nulos': 0,
    'negativos': 0,
    'ceros': 0,
    'con_promo': 0,
    'unit_sales_mean': 0,
    'unit_sales_sum': 0,
    'unit_sales_count': 0
})

chunk_size = 500_000
chunk_count = 0
total_rows = 0
promo_primer_mes = None

#añadido por mi#
tiendas_unicas = set()
productos_unicos = set()

ventas_por_year = defaultdict(float)

venta_min = float('inf')
venta_max = float('-inf')

suma_ventas = 0
conteo_ventas = 0

total_promociones = 0
#fin de añadido por mi#


# Leer en chunks
for chunk in pd.read_csv(
    train_file,
    chunksize=chunk_size,
    usecols=['date',  'store_nbr','item_nbr','unit_sales', 'onpromotion'],
    converters={'onpromotion': convert_onpromotion},
    dtype={'unit_sales': 'float32', 'onpromotion': 'uint8'}
):
    chunk_count += 1
    total_rows += len(chunk)
    
    # Convertir fecha
    chunk['date'] = pd.to_datetime(chunk['date'], format='%Y-%m-%d', errors='coerce')
    chunk['year_month'] = chunk['date'].dt.to_period('M')
    chunk['year'] = chunk['date'].dt.year

    #añadido por mi#
    tiendas_unicas.update(chunk['store_nbr'].unique())  
    productos_unicos.update(chunk['item_nbr'].unique())
    ventas_year_chunk=chunk.groupby('year')['unit_sales'].sum()
    for year, ventas in ventas_year_chunk.items():
        ventas_por_year[year] += ventas

        venta_min = min(venta_min, chunk['unit_sales'].min())
        venta_max = max(venta_max, chunk['unit_sales'].max())
        suma_ventas += chunk['unit_sales'].sum()
        conteo_ventas += chunk['unit_sales'].count()

        total_promociones += chunk['onpromotion'].sum()
    #fin de añadido por mi#

    
    # Actualizar fechas mín/máx
    valid_dates = chunk['date'].dropna()
    if len(valid_dates) > 0:
        chunk_min = valid_dates.min()
        chunk_max = valid_dates.max()
        if fecha_min is None or chunk_min < fecha_min:
            fecha_min = chunk_min
        if fecha_max is None or chunk_max > fecha_max:
            fecha_max = chunk_max
    
    # Agrupar por año-mes y calcular estadísticas
    grouped_month = chunk.groupby('year_month').agg({
        'unit_sales': ['count', 'sum', 'mean'],
        'onpromotion': 'sum'
    }).reset_index()
    
    for _, row in grouped_month.iterrows():
        year_month_str = str(row[('year_month', '')])
        registros_por_year_month[year_month_str] += row[('unit_sales', 'count')]
        
        stats = calidad_datos[year_month_str]
        stats['total'] += row[('unit_sales', 'count')]
        stats['unit_sales_sum'] += row[('unit_sales', 'sum')]
        stats['unit_sales_count'] += row[('unit_sales', 'count')]
        stats['con_promo'] += row[('onpromotion', 'sum')]
    
    # Agrupar por año
    grouped_year = chunk.groupby('year').size()
    for year, count in grouped_year.items():
        registros_por_year[year] += count
    
    # Analizar nulos, negativos, ceros
    for year_month in chunk['year_month'].unique():
        if pd.isna(year_month):
            continue
        year_month_str = str(year_month)
        chunk_subset = chunk[chunk['year_month'] == year_month]
        
        stats = calidad_datos[year_month_str]
        stats['nulos'] += chunk_subset['unit_sales'].isna().sum()
        stats['negativos'] += (chunk_subset['unit_sales'] < 0).sum()
        stats['ceros'] += (chunk_subset['unit_sales'] == 0).sum()
    
    # Detectar primer mes con promociones
    if promo_primer_mes is None:
        promo_por_mes = chunk.groupby('year_month')['onpromotion'].sum()
        promos = promo_por_mes[promo_por_mes > 0]
        if len(promos) > 0:
            promo_primer_mes = str(promos.index.min())
    
    # Mostrar progreso
    if chunk_count % 4 == 0:
        print(f"  ✓ Chunk {chunk_count}: {len(chunk):,} filas | Total: {total_rows:,}")

print(f"\n✓ Lectura completada: {chunk_count} chunks, {total_rows:,} filas\n")

# Calcular totales globales desde calidad_datos
total_nulos_global = sum(stats['nulos'] for stats in calidad_datos.values())
total_negativos_global = sum(stats['negativos'] for stats in calidad_datos.values())
total_ceros_global = sum(stats['ceros'] for stats in calidad_datos.values())

# ==============================================================================
# RESULTADOS: ANÁLISIS 1 - RANGO TEMPORAL
# ==============================================================================

print("=" * 80)
print("1. RANGO TEMPORAL DEL DATASET COMPLETO")
print("=" * 80)
print(f"\nFecha mínima: {fecha_min}")
print(f"Fecha máxima: {fecha_max}")

duracion_dias = (fecha_max - fecha_min).days
duracion_anos = duracion_dias / 365.25
print(f"Duración total: {duracion_dias:,} días ({duracion_anos:.2f} años)")

# ==============================================================================
# RESULTADOS: ANÁLISIS 2 - AÑOS
# ==============================================================================

print("\n" + "=" * 80)
print("2. COBERTURA EN AÑOS")
print("=" * 80)

años = sorted(registros_por_year.keys())
print(f"\nAños cubiertos: {len(años)} ({min(años)} - {max(años)})")

# ==============================================================================
# RESULTADOS: ANÁLISIS 3 - REGISTROS POR AÑO
# ==============================================================================

print("\n" + "=" * 80)
print("3. REGISTROS POR AÑO")
print("=" * 80)

total_reg = sum(registros_por_year.values())
print("\nAño    | Registros        | % del Total")
print("-" * 45)
for año in sorted(registros_por_year.keys()):
    count = registros_por_year[año]
    pct = (count / total_reg) * 100
    print(f"{año}   | {count:>16,} | {pct:>6.2f}%")

print("-" * 45)
print(f"Total  | {total_reg:>16,} | 100.00%")

# ==============================================================================
# RESULTADOS: ANÁLISIS 4 - REGISTROS POR MES
# ==============================================================================

print("\n" + "=" * 80)
print("4. DISTRIBUCIÓN POR MES (Top 10 y Bottom 10)")
print("=" * 80)

# Top 10 meses
print("\nTop 10 MESES CON MÁS REGISTROS:")
print("Mes         | Registros        | % del Total")
print("-" * 48)
top_months = sorted(registros_por_year_month.items(), key=lambda x: x[1], reverse=True)[:10]
for month, count in top_months:
    pct = (count / total_reg) * 100
    print(f"{month}    | {count:>16,} | {pct:>6.2f}%")

# Bottom 10 meses
print("\nBottom 10 MESES CON MENOS REGISTROS:")
print("Mes         | Registros        | % del Total")
print("-" * 48)
bottom_months = sorted(registros_por_year_month.items(), key=lambda x: x[1])[:10]
for month, count in bottom_months:
    pct = (count / total_reg) * 100
    print(f"{month}    | {count:>16,} | {pct:>6.2f}%")

# ==============================================================================
# RESULTADOS: ANÁLISIS 5 - PROMOCIONES
# ==============================================================================

print("\n" + "=" * 80)
print("5. APARICIÓN DE PROMOCIONES")
print("=" * 80)

if promo_primer_mes:
    print(f"\nPrimer mes con promociones: {promo_primer_mes}")
else:
    print("\n⚠️  No se encontraron promociones")

print("\nDistribución de promociones por mes:")
print("Mes         | Total Reg    | Con Promo    | % Promo")
print("-" * 52)
for month in sorted(registros_por_year_month.keys()):
    total = registros_por_year_month[month]
    con_promo = calidad_datos[month].get('con_promo', 0)
    pct_promo = (con_promo / total * 100) if total > 0 else 0
    print(f"{month}    | {total:>12,} | {con_promo:>12,} | {pct_promo:>6.2f}%")

# ==============================================================================
# RESULTADOS: ANÁLISIS 7 - ESTRUCTURA DEL NEGOCIO
# ==============================================================================

print("\n" + "=" * 80)
print("7. ESTRUCTURA DEL NEGOCIO")
print("=" * 80)

print(f"\nTiendas únicas: {len(tiendas_unicas):,}")
print(f"Productos únicos: {len(productos_unicos):,}")
print(f"Meses únicos: {len(registros_por_year_month):,}")

venta_promedio = suma_ventas / conteo_ventas

print("\nESTADÍSTICAS DE VENTAS")
print("-" * 40)

print(f"Venta mínima: {venta_min:,.2f}")
print(f"Venta máxima: {venta_max:,.2f}")
print(f"Venta promedio: {venta_promedio:,.2f}")

porcentaje_promos = (total_promociones / total_reg) * 100

print("\nPROMOCIONES")
print("-" * 40)

print(f"Registros en promoción: {total_promociones:,}")
print(f"Porcentaje promociones: {porcentaje_promos:.2f}%")

# ==============================================================================
# RESULTADOS: ANÁLISIS 8 - EVOLUCIÓN DE VENTAS POR AÑO
# ==============================================================================
print("\n" + "=" * 80)
print("8. EVOLUCIÓN DEL VOLUMEN DE VENTAS")
print("=" * 80)

print("\nAño    | Ventas Totales")
print("-" * 40)

for year in sorted(ventas_por_year.keys()):
    print(f"{year}   | {ventas_por_year[year]:>18,.2f}")

# ==============================================================================
# RESULTADOS: ANÁLISIS 9 - PERIODO RECOMENDADO PARA EL CASO DE ESTUDIO
# ==============================================================================

print("\n" + "=" * 80)
print("9. RECOMENDACIÓN DEL CASO DE ESTUDIO")
print("=" * 80)

print(f"""
Cobertura temporal detectada:
• Inicio: {fecha_min.date()}
• Fin: {fecha_max.date()}

Recomendación:

1. Utilizar al menos los últimos 2 años completos.
2. Incluir períodos con promociones.
3. Incluir festivos nacionales.
4. Incluir estacionalidad anual completa.
5. Incluir eventos extraordinarios relevantes.

Periodo sugerido:
• 2015-01-01 a 2017-07-31

Motivos:
✓ Suficiente histórico para forecasting.
✓ Presencia de promociones.
✓ Festivos y estacionalidad anual.
✓ Incluye el terremoto de 2016.
✓ Adecuado para simulaciones de inventario.
""")


# ==============================================================================
# DASHBOARD EJECUTIVO DEL DATASET
# ==============================================================================    
print("\n" + "=" * 80)
print("RESUMEN EJECUTIVO")
print("=" * 80)

print(f"""
📅 Cobertura temporal: {duracion_anos:.2f} años
🏪 Tiendas únicas: {len(tiendas_unicas):,}
📦 Productos únicos: {len(productos_unicos):,}
📊 Registros totales: {total_reg:,}
📈 Venta promedio: {venta_promedio:.2f}
🎯 Registros promocionados: {porcentaje_promos:.2f}%
↩️ Devoluciones: {total_negativos_global:,}
⚠️ Valores nulos: {total_nulos_global:,}
""")
# ==============================================================================
# RESUMEN FINAL
# ==============================================================================

print("\n" + "=" * 80)
print("RESUMEN FINAL")
print("=" * 80)

print(f"""
📊 COBERTURA TEMPORAL:
   • Período: {fecha_min.date()} a {fecha_max.date()}
   • Duración: {duracion_anos:.2f} años ({duracion_dias:,} días)
   • Años cubiertos: {len(años)}
   • Meses únicos: {len(registros_por_year_month)}
   
📈 VOLUMEN DE DATOS:
   • Total de registros: {total_reg:,}
   • Registros promedio/mes: {total_reg / len(registros_por_year_month):,.0f}
   • Máximo en un mes: {max(registros_por_year_month.values()):,}
   • Mínimo en un mes: {min(registros_por_year_month.values()):,}

⚠️  PROBLEMAS DETECTADOS:
   • Valores nulos: {total_nulos_global:,} ({total_nulos_global/total_reg*100:.4f}%)
   • Devoluciones (negativos): {total_negativos_global:,} ({total_negativos_global/total_reg*100:.4f}%)
   • Unit_sales = 0: {total_ceros_global:,} ({total_ceros_global/total_reg*100:.2f}%)
   • Primer mes con promociones: {promo_primer_mes if promo_primer_mes else 'Sin datos'}

✅ CONCLUSIONES:
   - Dataset de alta calidad con muy pocos valores nulos
   - Cobertura temporal completa de {duracion_anos:.1f} años
   - Devoluciones bajas, sugiere buena calidad de datos
   - Promociones ausentes en primeros períodos (cambio estructural importante)
""")

print("=" * 80)
print("✓ ANÁLISIS TEMPORAL COMPLETADO")
print("=" * 80)
