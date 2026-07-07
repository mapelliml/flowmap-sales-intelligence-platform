# ==============================================================================
# ANÁLISIS DE TAMAÑO Y ESTRUCTURA DEL DATASET FAVORITA TRAIN
# ==============================================================================

from pathlib import Path
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CONFIGURACIÓN DE RUTAS
# ==============================================================================

script_path = Path(__file__).parent
project_root = script_path.parent.parent
data_raw_path = project_root / "data" / "raw"
train_file = data_raw_path / "train.csv"

print("=" * 80)
print("ANÁLISIS DE TAMAÑO Y ESTRUCTURA - DATASET FAVORITA TRAIN")
print("=" * 80)

# ==============================================================================
# 2. ESTIMAR TAMAÑO EN MEMORIA DEL ARCHIVO COMPLETO
# ==============================================================================

print("\n📊 PASO 1: ESTIMACIÓN DE TAMAÑO EN MEMORIA")
print("-" * 80)

# Tamaño en disco
file_size_bytes = train_file.stat().st_size
file_size_mb = file_size_bytes / 1024**2
file_size_gb = file_size_bytes / 1024**3

print(f"\nTamaño en disco:")
print(f"  {file_size_mb:,.2f} MB ({file_size_gb:.2f} GB)")

# Estimar tamaño en memoria (típicamente 5-10x más grande que en disco para CSV)
print(f"\nEstimación de tamaño en memoria (sin optimización):")
print(f"  ~ {file_size_mb * 5:,.2f} MB (estimado 5x el tamaño en disco)")
print(f"  ~ {file_size_gb * 5:.2f} GB")
print(f"\n  ⚠️  ADVERTENCIA: Este dataset EXCEDE la memoria típica disponible")
print(f"      → Requiere carga en chunks o submuestreo")

# ==============================================================================
# 3. LEER MUESTRA INICIAL PARA ANÁLISIS ESTRUCTURAL
# ==============================================================================

print("\n\n📖 PASO 2: LEER MUESTRA DE 1 MILLÓN DE FILAS")
print("-" * 80)

nrows_sample = 1_000_000

print(f"\nCargando primeras {nrows_sample:,} filas...")

try:
    # Leer muestra sin especificar tipos (dejar que pandas infiera)
    df_sample = pd.read_csv(
        train_file,
        nrows=nrows_sample,
        low_memory=False
    )
    print(f"✓ Muestra cargada exitosamente")
    print(f"  Shape: {df_sample.shape}")
    
except Exception as e:
    print(f"❌ Error al cargar: {e}")
    # Intentar con tipos específicos más simples
    print("\n🔄 Intentando carga alternativa con conversor personalizado...")
    
    def convert_bool(val):
        """Convertir valores booleanos a 0/1"""
        if isinstance(val, str):
            return 1 if val.lower() == 'true' else 0
        return val
    
    df_sample = pd.read_csv(
        train_file,
        nrows=nrows_sample,
        converters={'onpromotion': convert_bool}
    )
    print(f"✓ Muestra cargada exitosamente")
    print(f"  Shape: {df_sample.shape}")

# ==============================================================================
# 4. ANALIZAR ESTRUCTURA DEL DATASET
# ==============================================================================

print("\n\n🔍 PASO 3: ANÁLISIS ESTRUCTURAL")
print("-" * 80)

# Información básica
print(f"\n1. COLUMNAS Y TIPOS:")
print(f"   Total de columnas: {len(df_sample.columns)}")
for col in df_sample.columns:
    print(f"   - {col:<15} {str(df_sample[col].dtype):<15} {df_sample[col].nunique():>8,} únicos")

# Consumo de memoria de la muestra
print(f"\n2. CONSUMO DE MEMORIA (muestra de {nrows_sample:,} filas):")
memoria_muestra_mb = df_sample.memory_usage(deep=False).sum() / 1024**2
print(f"   Total: {memoria_muestra_mb:.2f} MB")
print(f"   Por columna:")
for col in df_sample.columns:
    mem = df_sample[col].memory_usage(deep=False) / 1024**2
    print(f"   - {col:<15} {mem:>8.4f} MB")

# Proyectar tamaño completo
print(f"\n3. PROYECCIÓN PARA DATASET COMPLETO:")
# Contar número total de filas en el archivo
print(f"   Contando filas totales (puede tomar tiempo)...")
total_rows = sum(1 for _ in open(train_file)) - 1  # -1 para header
print(f"   Total de filas: {total_rows:,}")

memoria_proyectada_mb = (total_rows / len(df_sample)) * memoria_muestra_mb
print(f"   Memoria proyectada: {memoria_proyectada_mb:,.2f} MB ({memoria_proyectada_mb/1024:.2f} GB)")

# Análisis de valores nulos
print(f"\n4. VALORES NULOS:")
nulos = df_sample.isnull().sum()
porcentaje = (nulos / len(df_sample)) * 100
for col in df_sample.columns:
    if nulos[col] > 0:
        print(f"   - {col:<15} {nulos[col]:>10,} ({porcentaje[col]:>6.2f}%)")
if nulos.sum() == 0:
    print(f"   ✓ No hay valores nulos en la muestra")

# Rango temporal
if 'date' in df_sample.columns:
    df_sample['date'] = pd.to_datetime(df_sample['date'])
    print(f"\n5. RANGO TEMPORAL:")
    print(f"   Mínimo: {df_sample['date'].min()}")
    print(f"   Máximo: {df_sample['date'].max()}")
    print(f"   Días únicos: {df_sample['date'].nunique():,}")
    print(f"   Duración: {(df_sample['date'].max() - df_sample['date'].min()).days} días")

# Análisis de valores en onpromotion
if 'onpromotion' in df_sample.columns:
    print(f"\n6. VALORES EN COLUMNA 'onpromotion':")
    print(f"   Valores únicos: {df_sample['onpromotion'].unique()}")
    print(f"   Tipo de dato: {df_sample['onpromotion'].dtype}")

# ==============================================================================
# 5. PROPONER ESTRATEGIA EFICIENTE
# ==============================================================================

print("\n\n💡 PASO 4: ESTRATEGIA RECOMENDADA PARA TRABAJAR CON EL DATASET")
print("-" * 80)

print(f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                      ESTRATEGIA OPTIMIZADA RECOMENDADA                    ║
╚═══════════════════════════════════════════════════════════════════════════╝

1️⃣  OPTIMIZAR TIPOS DE DATOS
   • store_nbr: uint8 (en lugar de int64)
   • product_id: uint32 (en lugar de int64)
   • sales: float32 (en lugar de float64)
   • onpromotion: uint8 (en lugar de int64)
   • date: datetime64[ns] (optimizado)
   
   → Reducción esperada: 40-50% del tamaño actual

2️⃣  ESTRATEGIA DE CARGA Y PROCESAMIENTO
   ✅ OPCIÓN A: Carga en Chunks (Recomendado para análisis)
      - Leer en bloques de 500k-1M filas
      - Procesar cada chunk
      - Concatenar resultados al final
      - Ideal para: agregaciones, estadísticas, transformaciones
   
   ✅ OPCIÓN B: Usar Dask (Para análisis distribuido)
      - Paralelizar lectura y procesamiento
      - Trabaja con memoria limitada automáticamente
      - Ideal para: ML, grandes agregaciones, operaciones complejas
   
   ✅ OPCIÓN C: Guardar en Parquet
      - Exportar con tipos optimizados
      - Compresión: snappy (rápido) o gzip (máxima compresión)
      - Reutilizar en análisis posteriores
      - Ideal para: reproducibilidad, compartir datos

3️⃣  SUBMUESTREO ESTRATIFICADO (Si necesitas muestra representativa)
   • Mantener proporción de tiendas, productos, fechas
   • Usar 1-2M filas (10-25% del total)
   • Validar resultados en subset antes de correr en dataset completo

4️⃣  PIPELINE RECOMENDADO
   Paso 1: Optimizar tipos con carga en chunks
   Paso 2: Guardar en Parquet (data/processed/train_optimized.parquet)
   Paso 3: Cargar desde Parquet para análisis posteriores
   Paso 4: Usar chunks solo si el Parquet es aún muy grande

5️⃣  CONFIGURACIÓN DE MEMORIA RECOMENDADA
   • Para esta máquina:
     - Chunk size: 500,000 - 1,000,000 filas
     - Memory limit (si usas Dask): 2-4 GB
     - Deixar buffer para el SO: 2-3 GB

╔═══════════════════════════════════════════════════════════════════════════╗
║                            ACCIÓN INMEDIATA                              ║
╚═══════════════════════════════════════════════════════════════════════════╝

Ejecuta el notebook: 00_Data_Loading_Chunked.ipynb
→ Cargará, limpiará y exportará a Parquet en un solo paso
→ Generará archivo optimizado listo para análisis

Tamaño esperado del Parquet:
  • Original en disco: {file_size_mb:.2f} MB
  • Parquet con tipos opt.: ~{file_size_mb * 0.3:.2f} MB (30-40%)
  • CSV.gz: ~{file_size_mb * 0.5:.2f} MB (50-60%)

""")

print("=" * 80)
print("✓ ANÁLISIS COMPLETADO")
print("=" * 80)
