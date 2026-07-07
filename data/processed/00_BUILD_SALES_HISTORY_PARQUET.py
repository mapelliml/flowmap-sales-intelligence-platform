"""
================================================================================
00_BUILD_SALES_HISTORY_PARQUET.py
================================================================================

SCRIPT DE PRODUCCIÓN PARA CONSTRUCCIÓN DE DATASET OPTIMIZADO DE VENTAS

Objetivo:
---------
Crear un archivo Parquet ligero y optimizado a partir del dataset Favorita,
específicamente diseñado para ser consumido por aplicaciones Streamlit.

Características:
- Lectura por chunks de train.csv (archivo gigante que no cabe en memoria)
- Procesamiento eficiente en memoria con optimización de tipos
- Agregación por chunks para reducir volumen de datos
- Merge con tablas auxiliares (items, stores)
- Limpieza y estandarización de datos
- Salida en formato Parquet optimizado para Streamlit

Autor: Senior Data Engineer
Fecha: 2026
================================================================================
"""

# =============================================================================
# IMPORTACIONES
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import warnings

# Suprimir warnings no críticos para limpieza de output
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# =============================================================================
# CONFIGURACIÓN DE RUTAS
# =============================================================================

# Ruta base del proyecto (usando pathlib para compatibilidad multiplataforma)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Rutas de entrada (datos raw)
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_PATH = RAW_DIR / "train.csv"
ITEMS_PATH = RAW_DIR / "items.csv"
STORES_PATH = RAW_DIR / "stores.csv"

# Ruta de salida (datos procesados)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "sales_history_real.parquet"

# Configuración de chunks
# NOTA: El tamaño del chunk es un balance entre uso de memoria y rendimiento.
# Un chunk muy pequeño hace muchas iteraciones, uno muy grande puede saturar memoria.
CHUNK_SIZE = 500_000  # 500k filas por chunk - ajustable según memoria disponible

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def log_message(message: str, level: str = "INFO"):
    """
    Imprime mensajes de progreso con timestamp.
    
    Args:
        message: Mensaje a imprimir
        level: Nivel del mensaje (INFO, WARNING, ERROR, SUCCESS)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def optimize_dataframe_memory(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Optimiza el uso de memoria de un DataFrame convirtiendo los tipos de datos
    a versiones más ligeras cuando es posible.
    
    Estrategia:
    - Enteros: int64 -> int32 o int16 según rango
    - Flotantes: float64 -> float32 según precisión necesaria
    - Strings: object -> category cuando hay pocos valores únicos
    
    Args:
        df: DataFrame a optimizar
        verbose: Si True, imprime información de optimización
    
    Returns:
        DataFrame optimizado
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        # Optimización de enteros
        if col_type in [np.int64, np.int32, np.int16, np.int8]:
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        
        # Optimización de flotantes
        elif col_type in [np.float64, np.float32]:
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
        
        # Optimización de strings a category (solo si hay pocos valores únicos)
        elif col_type == object:
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:  # Menos del 50% de valores únicos
                df[col] = df[col].astype('category')
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    if verbose:
        reduction = ((start_mem - end_mem) / start_mem) * 100
        log_message(f"Memoria: {start_mem:.2f} MB -> {end_mem:.2f} MB (reducción: {reduction:.1f}%)")
    
    return df


def clean_and_transform_chunk(chunk_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y transforma un chunk de datos del train.csv.
    
    Procesos:
    1. Renombrar unit_sales a sales
    2. Merge con items para obtener family
    3. Eliminación de filas con valores nulos críticos
    4. Conversión de tipos de datos
    5. Agregación por grupo para reducir cardinalidad
    
    Args:
        chunk_df: Chunk del DataFrame original
        items_df: DataFrame de items para obtener family
    
    Returns:
        Chunk procesado y agregado
    """
    # Guardar conteo inicial
    initial_rows = len(chunk_df)
    
    # 1. Renombrar unit_sales a sales (nombre estandarizado)
    chunk_df = chunk_df.rename(columns={'unit_sales': 'sales'})
    
    # 2. Merge con items para obtener la familia del producto
    chunk_df = chunk_df.merge(
        items_df[['item_nbr', 'family']],
        on='item_nbr',
        how='left'
    )
    
    # 3. Limpieza de valores nulos críticos
    # Eliminamos filas donde faltan datos esenciales
    chunk_df = chunk_df.dropna(subset=['date', 'store_nbr', 'family', 'sales'])
    
    # 4. Eliminación de filas con ventas negativas o cero (no tienen sentido para análisis)
    # NOTA: Si se necesitan incluir, cambiar la condición
    chunk_df = chunk_df[chunk_df['sales'] > 0]
    
    # 5. Conversión de tipos de datos
    chunk_df['date'] = pd.to_datetime(chunk_df['date'], format='%Y-%m-%d', errors='coerce')
    chunk_df['store_nbr'] = pd.to_numeric(chunk_df['store_nbr'], errors='coerce').astype('int32')
    chunk_df['item_nbr'] = pd.to_numeric(chunk_df['item_nbr'], errors='coerce').astype('int32')
    chunk_df['sales'] = pd.to_numeric(chunk_df['sales'], errors='coerce').astype('float32')
    chunk_df['onpromotion'] = pd.to_numeric(chunk_df['onpromotion'], errors='coerce').fillna(0).astype('int16')
    
    # 6. Eliminación de filas con fechas inválidas después de la conversión
    chunk_df = chunk_df.dropna(subset=['date'])
    
    # 7. Agregación por chunk para reducir cardinalidad
    # Agrupamos por todas las columnas relevantes y sumamos ventas
    # Esto reduce significativamente el número de filas si hay múltiples
    # registros para el mismo producto/tienda/día
    # NOTA: No incluimos item_nbr porque queremos consolidar por familia, no por producto individual
    aggregation_dict = {
        'sales': 'sum',
        'onpromotion': 'mean'  # Promedio de promoción para el grupo
    }
    
    group_cols = ['date', 'store_nbr', 'family']
    
    # Verificamos que existan las columnas de agrupación
    missing_cols = [col for col in group_cols if col not in chunk_df.columns]
    if missing_cols:
        raise ValueError(f"Columnas faltantes para agregación: {missing_cols}")
    
    chunk_aggregated = chunk_df.groupby(group_cols).agg(aggregation_dict).reset_index()
    
    # Redondeamos ventas a 2 decimales
    chunk_aggregated['sales'] = chunk_aggregated['sales'].round(2)
    chunk_aggregated['onpromotion'] = chunk_aggregated['onpromotion'].round(0).astype('int16')
    
    # Log de reducción
    final_rows = len(chunk_aggregated)
    reduction_pct = ((initial_rows - final_rows) / initial_rows) * 100 if initial_rows > 0 else 0
    
    log_message(f"Chunk procesado: {initial_rows:,} -> {final_rows:,} filas (reducción: {reduction_pct:.1f}%)")
    
    return chunk_aggregated


# =============================================================================
# CARGA DE TABLAS AUXILIARES (ITEMS Y STORES)
# =============================================================================

def load_auxiliary_tables() -> tuple:
    """
    Carga las tablas auxiliares de items y stores.
    Estas tablas son pequeñas y pueden cargarse completas en memoria.
    
    Returns:
        Tupla (items_df, stores_df)
    """
    log_message("Cargando tablas auxiliares (items.csv, stores.csv)...")
    
    # Cargar items.csv - solo columnas necesarias
    items_df = pd.read_csv(
        ITEMS_PATH,
        usecols=['item_nbr', 'family'],
        dtype={
            'item_nbr': 'int32',
            'family': 'object'
        }
    )
    
    # Eliminar duplicados en items (por si acaso)
    items_df = items_df.drop_duplicates(subset=['item_nbr'])
    
    log_message(f"Items cargados: {len(items_df):,} productos únicos")
    
    # Cargar stores.csv - solo columnas necesarias
    stores_df = pd.read_csv(
        STORES_PATH,
        usecols=['store_nbr', 'city', 'state', 'type', 'cluster'],
        dtype={
            'store_nbr': 'int32',
            'city': 'object',
            'state': 'object',
            'type': 'object',
            'cluster': 'int8'
        }
    )
    
    # Eliminar duplicados en stores (por si acaso)
    stores_df = stores_df.drop_duplicates(subset=['store_nbr'])
    
    log_message(f"Tiendas cargadas: {len(stores_df):,} tiendas únicas")
    
    return items_df, stores_df


# =============================================================================
# PROCESAMIENTO PRINCIPAL POR CHUNKS
# =============================================================================

def process_train_data(items_df: pd.DataFrame, stores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Procesa el archivo train.csv por chunks y realiza los merges necesarios.
    
    Estrategia:
    1. Leer train.csv en chunks de tamaño configurable
    2. Procesar cada chunk individualmente (limpieza, transformación, agregación)
    3. Acumular resultados en una lista
    4. Concatenar todos los chunks procesados
    5. Realizar agregación final para consolidar datos
    6. Hacer merge con tablas auxiliares
    
    Args:
        items_df: DataFrame de items
        stores_df: DataFrame de stores
    
    Returns:
        DataFrame final procesado
    """
    
    # Verificar que el archivo train.csv existe
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"El archivo train.csv no existe en: {TRAIN_PATH}")
    
    log_message(f"Iniciando procesamiento de train.csv (chunksize={CHUNK_SIZE:,})")
    log_message(f"Archivo: {TRAIN_PATH}")
    
    # Lista para almacenar chunks procesados
    chunks_processed = []
    chunk_counter = 0
    total_rows_processed = 0
    
    # Tiempo de inicio
    start_time = time.time()
    
    # Lectura por chunks del train.csv
    # usecols: Solo cargamos las columnas necesarias para reducir memoria
    # dtype: Especificamos tipos iniciales para optimizar carga
    # NOTA: El train.csv original tiene 'unit_sales' en lugar de 'sales'
    # y no tiene 'family' (esa viene en items.csv)
    chunk_reader = pd.read_csv(
        TRAIN_PATH,
        chunksize=CHUNK_SIZE,
        usecols=['id', 'date', 'store_nbr', 'item_nbr', 'unit_sales', 'onpromotion'],
        dtype={
            'id': 'int64',
            'store_nbr': 'int32',
            'item_nbr': 'int32',
            'unit_sales': 'float32',
            'onpromotion': 'float32'
        },
        parse_dates=['date'],
        low_memory=False  # Evita warnings de tipos mixtos
    )
    
    for chunk in chunk_reader:
        chunk_counter += 1
        total_rows_processed += len(chunk)
        
        log_message(f"Procesando chunk {chunk_counter} ({len(chunk):,} filas) - Total acumulado: {total_rows_processed:,} filas")
        
        # Limpiar y transformar el chunk (pasando items_df para obtener family)
        chunk_clean = clean_and_transform_chunk(chunk, items_df)
        
        # Almacenar chunk procesado
        chunks_processed.append(chunk_clean)
        
        # Liberar memoria del chunk original
        del chunk
        
        # Opcional: Forzar garbage collection cada cierto número de chunks
        if chunk_counter % 10 == 0:
            import gc
            gc.collect()
    
    # Tiempo de procesamiento de chunks
    chunk_processing_time = time.time() - start_time
    log_message(f"Procesamiento de chunks completado en {chunk_processing_time:.2f} segundos")
    log_message(f"Total de chunks procesados: {chunk_counter}")
    log_message(f"Total de filas procesadas (antes de concatenar): {total_rows_processed:,}")
    
    # Concatenar todos los chunks procesados
    log_message("Concatenando chunks procesados...")
    concat_start = time.time()
    
    df_combined = pd.concat(chunks_processed, ignore_index=True)
    
    # Liberar memoria de la lista de chunks
    del chunks_processed
    
    concat_time = time.time() - concat_start
    log_message(f"Concatenación completada en {concat_time:.2f} segundos")
    log_message(f"Filas después de concatenar: {len(df_combined):,}")
    
    # Agregación final para consolidar datos duplicados entre chunks
    # (puede haber registros iguales en differentes chunks)
    log_message("Realizando agregación final...")
    agg_start = time.time()
    
    final_aggregation = {
        'sales': 'sum',
        'onpromotion': 'mean'
    }
    
    df_aggregated = df_combined.groupby(['date', 'store_nbr', 'family']).agg(final_aggregation).reset_index()
    df_aggregated['sales'] = df_aggregated['sales'].round(2)
    df_aggregated['onpromotion'] = df_aggregated['onpromotion'].round(0).astype('int16')
    
    agg_time = time.time() - agg_start
    log_message(f"Agregación final completada en {agg_time:.2f} segundos")
    log_message(f"Filas después de agregación final: {len(df_aggregated):,}")
    
    # Liberar memoria
    del df_combined
    
    # =============================================================================
    # MERGE CON TABLAS AUXILIARES
    # =============================================================================
    
    log_message("Realizando merge con tabla de tiendas (stores)...")
    
    # Merge con stores para obtener city, state, type, cluster
    df_merged = df_aggregated.merge(
        stores_df,
        on='store_nbr',
        how='left'
    )
    
    log_message(f"Filas después del merge con stores: {len(df_merged):,}")
    
    # Verificar si hay tiendas sin información
    missing_stores = df_merged['city'].isna().sum()
    if missing_stores > 0:
        log_message(f"ADVERTENCIA: {missing_stores} filas sin información de tienda", level="WARNING")
    
    # =============================================================================
    # RENOMBRAR COLUMNAS A ESPAÑOL (para app Streamlit)
    # =============================================================================
    
    log_message("Renombrando columnas a español...")
    
    # Mapeo de nombres de columnas
    column_mapping = {
        'date': 'fecha',
        'family': 'familia',
        'store_nbr': 'tienda',
        'sales': 'ventas',
        'onpromotion': 'onpromotion',  # Se mantiene en inglés por ser término técnico
        'city': 'city',                # Se mantiene en inglés por ser término estándar
        'state': 'state',              # Se mantiene en inglés por ser término estándar
        'type': 'type',                # Se mantiene en inglés por ser término estándar
        'cluster': 'cluster'           # Se mantiene en inglés por ser término estándar
    }
    
    # Renombrar solo las columnas que existen
    existing_cols = {k: v for k, v in column_mapping.items() if k in df_merged.columns}
    df_final = df_merged.rename(columns=existing_cols)
    
    # =============================================================================
    # OPTIMIZACIÓN FINAL DE TIPOS
    # =============================================================================
    
    log_message("Optimizando tipos de datos finales...")
    df_final = optimize_dataframe_memory(df_final, verbose=True)
    
    # Asegurar tipos específicos para columnas clave
    df_final['ventas'] = df_final['ventas'].astype('float32')
    df_final['tienda'] = df_final['tienda'].astype('int32')
    df_final['fecha'] = pd.to_datetime(df_final['fecha'])
    
    # Ordenar por fecha y tienda para mejor rendimiento en consultas
    log_message("Ordenando datos por fecha y tienda...")
    df_final = df_final.sort_values(['fecha', 'tienda', 'familia']).reset_index(drop=True)
    
    return df_final


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """
    Función principal que orquesta todo el proceso de construcción del dataset.
    """
    
    print("=" * 80)
    print("CONSTRUCCIÓN DE SALES_HISTORY_REAL.PARQUET")
    print("=" * 80)
    print()
    
    # Tiempo total de ejecución
    total_start = time.time()
    
    # 1. Verificar que los archivos de entrada existen
    log_message("Verificando archivos de entrada...")
    
    required_files = [TRAIN_PATH, ITEMS_PATH, STORES_PATH]
    for file_path in required_files:
        if not file_path.exists():
            log_message(f"ERROR: Archivo no encontrado: {file_path}", level="ERROR")
            raise FileNotFoundError(f"Archivo requerido no encontrado: {file_path}")
        else:
            log_message(f"✓ Archivo encontrado: {file_path.name}")
    
    print()
    
    # 2. Crear directorio de salida si no existe
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    log_message(f"Directorio de salida verificado: {PROCESSED_DIR}")
    
    # 3. Cargar tablas auxiliares
    print()
    log_message("FASE 1: Carga de tablas auxiliares")
    print("-" * 40)
    items_df, stores_df = load_auxiliary_tables()
    
    # 4. Procesar datos principales
    print()
    log_message("FASE 2: Procesamiento de datos de ventas")
    print("-" * 40)
    df_final = process_train_data(items_df, stores_df)
    
    # 5. Guardar como Parquet
    print()
    log_message("FASE 3: Guardando archivo Parquet...")
    print("-" * 40)
    
    # Guardar con compresión snappy (buen balance entre tamaño y velocidad)
    # compression='snappy' es rápido y reduce bien el tamaño
    df_final.to_parquet(
        OUTPUT_PATH,
        engine='pyarrow',
        compression='snappy',
        index=False
    )
    
    log_message(f"✓ Archivo guardado exitosamente: {OUTPUT_PATH}")
    
    # 6. Verificar archivo guardado
    file_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    log_message(f"Tamaño del archivo: {file_size_mb:.2f} MB")
    
    # =============================================================================
    # INFORME FINAL
    # =============================================================================
    
    total_time = time.time() - total_start
    
    print()
    print("=" * 80)
    print("INFORME FINAL")
    print("=" * 80)
    print(f"✓ Shape final del dataset: {df_final.shape[0]:,} filas × {df_final.shape[1]} columnas")
    print(f"✓ Columnas finales: {list(df_final.columns)}")
    print(f"✓ Ruta del archivo generado: {OUTPUT_PATH}")
    print(f"✓ Tamaño del archivo: {file_size_mb:.2f} MB")
    print(f"✓ Tiempo total de procesamiento: {total_time:.2f} segundos ({total_time/60:.2f} minutos)")
    print()
    
    # Estadísticas básicas del dataset
    print("ESTADÍSTICAS BÁSICAS:")
    print(f"  - Período cubierto: {df_final['fecha'].min().date()} a {df_final['fecha'].max().date()}")
    print(f"  - Total de tiendas únicas: {df_final['tienda'].nunique():,}")
    print(f"  - Total de familias únicas: {df_final['familia'].nunique():,}")
    print(f"  - Total de ventas (suma): {df_final['ventas'].sum():,.2f}")
    print(f"  - Promedio de ventas por registro: {df_final['ventas'].mean():.2f}")
    print(f"  - Memoria usada en DataFrame: {df_final.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print()
    
    # Información de tipos de datos
    print("TIPOS DE DATOS:")
    for col, dtype in df_final.dtypes.items():
        print(f"  - {col}: {dtype}")
    print()
    
    # Muestra de datos
    print("MUESTRA DE DATOS (primeras 5 filas):")
    print(df_final.head().to_string())
    print()
    
    print("=" * 80)
    log_message("PROCESO COMPLETADO EXITOSAMENTE", level="SUCCESS")
    print("=" * 80)


# =============================================================================
# EJECUCIÓN DEL SCRIPT
# =============================================================================

if __name__ == "__main__":
    main()