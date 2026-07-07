# ==============================================================================
# CARGAR DATASET DE ENTRENAMIENTO - ANÁLISIS EXPLORATORIO INICIAL
# ==============================================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CONSTRUIR RUTAS RELATIVAS ROBUSTAS CON PATHLIB
# ==============================================================================

# Obtener la ruta absoluta del script actual y navegar hacia la carpeta raíz del proyecto
# __file__ proporciona la ruta completa del script
# Path(__file__).parent = notebooks/01_eda
# Path(__file__).parent.parent = notebooks
# Path(__file__).parent.parent.parent = segundo proyecto (raíz del proyecto)
script_path = Path(__file__).parent
project_root = script_path.parent.parent  # Sube desde notebooks/01_eda hasta raíz

# Definir rutas a carpetas clave
data_raw_path = project_root / "data" / "raw"
data_processed_path = project_root / "data" / "processed"

# Definir rutas a archivos específicos
train_file = data_raw_path / "train.csv"

# Validar que el archivo existe antes de cargar
assert train_file.exists(), f"Archivo no encontrado: {train_file}"

print(f"✓ Ruta del proyecto: {project_root}")
print(f"✓ Archivo a cargar: {train_file}\n")

# ==============================================================================
# 2. CARGAR DATASET DE ENTRENAMIENTO EN CHUNKS (Por memoria limitada)
# ==============================================================================

# El dataset es muy grande (~3GB), usar nrows para cargar muestra representativa
# Cargar últimas 2 millones de filas (últimos 6 meses aproximadamente)
nrows_to_load = 2_000_000

print(f"⚠️  Dataset muy grande. Cargando muestra de {nrows_to_load:,} filas...")

# Función para convertir valores booleanos representados como strings a números
def convert_onpromotion(value):
    """Convierte 'False'/'True' o 0/1 a entero"""
    if isinstance(value, str):
        return 1 if value.strip().lower() == 'true' else 0
    return int(value) if pd.notna(value) else 0

# Leer archivo CSV con conversor y límite de filas
df_train = pd.read_csv(
    train_file,
    converters={'onpromotion': convert_onpromotion},
    nrows=nrows_to_load,
    skiprows=lambda x: x > 0 and x < (3650871 - nrows_to_load)  # Saltar filas iniciales, cargar últimas
)

print("=" * 70)
print("INFORMACIÓN DEL DATASET DE ENTRENAMIENTO (Muestra)")
print("=" * 70)

# ==============================================================================
# 3. CONVERTIR COLUMNA DATE A DATETIME
# ==============================================================================

# Convertir columna 'date' a tipo datetime para operaciones de series temporales
df_train['date'] = pd.to_datetime(df_train['date'], format='%Y-%m-%d')

# Establecer 'date' como índice para trabajar con series temporales
df_train = df_train.sort_values('date').reset_index(drop=True)

print(f"\n✓ Columna 'date' convertida a datetime")
print(f"  Rango temporal: {df_train['date'].min()} a {df_train['date'].max()}")

# ==============================================================================
# 4. MOSTRAR FORMA DEL DATASET
# ==============================================================================

print(f"\n{'Shape del dataset:':<30} {df_train.shape}")
print(f"{'Filas:':<30} {df_train.shape[0]:,}")
print(f"{'Columnas:':<30} {df_train.shape[1]}")

# ==============================================================================
# 5. MOSTRAR TIPOS DE DATOS
# ==============================================================================

print(f"\n{'Tipos de datos:':<30}")
print("-" * 70)
print(df_train.dtypes)

# ==============================================================================
# 6. MOSTRAR CONSUMO DE MEMORIA
# ==============================================================================

# Calcular uso de memoria en MB sin usar deep=True (más rápido para datasets grandes)
memoria_mb = df_train.memory_usage(deep=False).sum() / 1024**2
memoria_por_columna = df_train.memory_usage(deep=False) / 1024**2

print(f"\n{'Consumo de memoria total:':<30} {memoria_mb:.2f} MB")
print(f"\n{'Memoria por columna:':<30}")
print("-" * 70)
for col, mem in memoria_por_columna.items():
    print(f"  {col:<25} {mem:.4f} MB")

# ==============================================================================
# 7. MOSTRAR PRIMERAS FILAS
# ==============================================================================

print(f"\n{'Primeras 5 filas del dataset:':<30}")
print("-" * 70)
print(df_train.head())

# ==============================================================================
# 8. MOSTRAR ESTADÍSTICAS DESCRIPTIVAS
# ==============================================================================

print(f"\n{'Estadísticas descriptivas:':<30}")
print("-" * 70)
print(df_train.describe())

# ==============================================================================
# 9. ANALIZAR VALORES NULOS (MISSING VALUES)
# ==============================================================================

# Calcular cantidad y porcentaje de valores nulos por columna
valores_nulos = df_train.isnull().sum()
porcentaje_nulos = (df_train.isnull().sum() / len(df_train)) * 100

print(f"\n{'Análisis de valores nulos:':<30}")
print("-" * 70)

tabla_nulos = pd.DataFrame({
    'Columna': df_train.columns,
    'Valores Nulos': valores_nulos.values,
    'Porcentaje (%)': porcentaje_nulos.values
}).sort_values('Porcentaje (%)', ascending=False)

print(tabla_nulos)

# Resumen de completitud
completitud = (1 - (valores_nulos.sum() / (df_train.shape[0] * df_train.shape[1]))) * 100
print(f"\n✓ Completitud del dataset: {completitud:.2f}%")

# ==============================================================================
# 10. INFORMACIÓN ADICIONAL SOBRE CARACTERÍSTICAS CLAVE
# ==============================================================================

print(f"\n{'Información de variables clave:':<30}")
print("-" * 70)
print(f"  Tiendas únicas:          {df_train['store_nbr'].nunique()}")
print(f"  Productos únicos:        {df_train['item_nbr'].nunique()}")
print(f"  Períodos de tiempo:      {df_train['date'].nunique()}")
print(f"  Rango de ventas:         {df_train['unit_sales'].min():.2f} - {df_train['unit_sales'].max():.2f}")
print(f"  Venta promedio:          {df_train['unit_sales'].mean():.2f}")
print(f"  Porcentaje en promoción: {(df_train['onpromotion'].sum() / len(df_train) * 100):.2f}%")

print("\n" + "=" * 70)
print("✓ CARGA Y EXPLORACIÓN INICIAL COMPLETADA")
print("=" * 70)

# --------------------------------------------------------------------------
# 0. VALIDACIÓN Y LIMPIEZA INICIAL
# --------------------------------------------------------------------------
# Asegurar que el dataset ya está cargado en df_train
assert 'df_train' in globals() or 'df_train' in locals(), "El dataframe df_train debe estar cargado."

# Convertir fechas y columnas críticas a tipos correctos
if df_train['date'].dtype == object:
    df_train['date'] = pd.to_datetime(df_train['date'], errors='coerce')

if df_train['unit_sales'].dtype == object:
    df_train['unit_sales'] = pd.to_numeric(df_train['unit_sales'], errors='coerce')

if df_train['onpromotion'].dtype == object:
    df_train['onpromotion'] = df_train['onpromotion'].replace(
        {'False': 0, 'True': 1, 'false': 0, 'true': 1}
    )
df_train['onpromotion'] = pd.to_numeric(df_train['onpromotion'], errors='coerce').fillna(0).astype(int)

# --------------------------------------------------------------------------
# 1. CUÁNTOS REGISTROS EXISTEN
# --------------------------------------------------------------------------
n_records = len(df_train)
print(f"1. Registros totales: {n_records:,}")

# --------------------------------------------------------------------------
# 2. RANGO TEMPORAL DEL DATASET
# --------------------------------------------------------------------------
date_min = df_train['date'].min()
date_max = df_train['date'].max()
print(f"2. Rango temporal: {date_min} a {date_max}")

# --------------------------------------------------------------------------
# 3. NÚMERO DE TIENDAS ÚNICAS
# --------------------------------------------------------------------------
n_stores = df_train['store_nbr'].nunique()
print(f"3. Tiendas únicas: {n_stores:,}")

# --------------------------------------------------------------------------
# 4. NÚMERO DE PRODUCTOS ÚNICOS
# --------------------------------------------------------------------------
n_products = df_train['item_nbr'].nunique()
print(f"4. Productos únicos: {n_products:,}")

# --------------------------------------------------------------------------
# 5. DISTRIBUCIÓN DE UNIT_SALES
# --------------------------------------------------------------------------
print("5. Distribución de unit_sales:")
print(df_train['unit_sales'].describe())

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
sns.histplot(df_train['unit_sales'].clip(lower=-10, upper=50), bins=80, kde=False, color='steelblue')
plt.title('Histograma unit_sales (truncado a [-10, 50])')
plt.xlabel('unit_sales')
plt.ylabel('Frecuencia')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
sns.boxplot(x=df_train['unit_sales'].clip(lower=df_train['unit_sales'].quantile(0.01),
                                          upper=df_train['unit_sales'].quantile(0.99)),
            color='darkorange')
plt.title('Boxplot unit_sales (1%-99%)')
plt.xlabel('unit_sales')

plt.tight_layout()
plt.show()

# Interpretación:
# La mayoría de las ventas se concentran en valores bajos y hay cola larga de pedidos grandes.
# La visualización truncada ayuda a ver la base del negocio sin ser dominada por outliers.

# --------------------------------------------------------------------------
# 6. NÚMERO DE DEVOLUCIONES (unit_sales negativas)
# --------------------------------------------------------------------------
n_returns = (df_train['unit_sales'] < 0).sum()
print(f"6. Devoluciones (unit_sales < 0): {n_returns:,}")

returns_pct = n_returns / n_records * 100
print(f"   Porcentaje de devoluciones: {returns_pct:.4f}%")

# Interpretación:
# Las devoluciones son un indicador clave de riesgo comercial y control de inventario.

# --------------------------------------------------------------------------
# 7. PORCENTAJE DE PROMOCIONES
# --------------------------------------------------------------------------
promo_count = df_train['onpromotion'].sum()
promo_pct = promo_count / n_records * 100
print(f"7. Registros en promoción: {promo_count:,}")
print(f"   Porcentaje de promociones: {promo_pct:.2f}%")

promo_dist = df_train['onpromotion'].value_counts(normalize=True) * 100
print(f"   Distribución onpromotion:\n{promo_dist.round(2)}")

# Interpretación:
# El porcentaje de promociones ayuda a entender el impacto del marketing en la demanda.

# --------------------------------------------------------------------------
# 8. VALORES NULOS POR COLUMNA
# --------------------------------------------------------------------------
null_counts = df_train.isnull().sum()
null_pct = (null_counts / n_records) * 100
null_summary = pd.DataFrame({
    'nulos': null_counts,
    'porcentaje': null_pct
}).sort_values('porcentaje', ascending=False)

print("8. Valores nulos por columna:")
print(null_summary)

# --------------------------------------------------------------------------
# 9. ESTADÍSTICAS DESCRIPTIVAS
# --------------------------------------------------------------------------
print("9. Estadísticas descriptivas para variables numéricas:")
print(df_train[['unit_sales', 'onpromotion']].describe())

print("\n9b. Estadísticas adicionales para variables categóricas:")
print(f"  Tiendas más frecuentes:\n{df_train['store_nbr'].value_counts().head(10)}")
print(f"\n  Productos con más registros:\n{df_train['item_nbr'].value_counts().head(10)}")

# Interpretación final:
# Este dataset muestra un negocio con muchas transacciones pequeñas,
# algunas devoluciones y una proporción de promociones que debe ser evaluada
# junto con la estacionalidad para entender el comportamiento de demanda.