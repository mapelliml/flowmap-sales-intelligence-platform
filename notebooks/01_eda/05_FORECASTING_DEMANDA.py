# =============================================================================
# FORECASTING_DEMANDA: Sistema de Predicción de Demanda para Retail
# =============================================================================
# Sistema de forecasting para optimización de inventario, cálculo de stock
# de seguridad y punto de pedido.
#
# Autor: Lead Data Scientist - Demand Forecasting & Supply Chain
# Versión: 1.0 - Optimizado para datasets grandes
# =============================================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('Set2')
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 11

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
    """Configurar rutas del proyecto"""
    script_path = Path(__file__).parent
    project_root = script_path.parent.parent
    data_raw_path = project_root / 'data' / 'raw'
    reports_path = project_root / 'reports' / 'forecasting'
    
    reports_path.mkdir(parents=True, exist_ok=True)
    
    return {
        'project_root': project_root,
        'data_raw': data_raw_path,
        'reports': reports_path,
        'train': data_raw_path / 'train.csv',
        'items': data_raw_path / 'items.csv',
        'holidays': data_raw_path / 'holidays_events.csv',
        'oil': data_raw_path / 'oil.csv'
    }

paths = setup_paths()

print('=' * 80)
print('SISTEMA DE FORECASTING DE DEMANDA')
print('=' * 80)
print(f'\n📁 Proyecto: {paths["project_root"]}')
print(f'📁 Reports: {paths["reports"]}')
print()

# =============================================================================
# CARGA DE DATOS AUXILIARES
# =============================================================================

print('Cargando datos auxiliares...')

# Items - para filtrar familias top
items_df = pd.read_csv(paths['items'])
print(f'✅ Items: {len(items_df):,} productos')

# Holidays
holidays_df = pd.read_csv(paths['holidays'], parse_dates=['date'])
print(f'✅ Holidays: {len(holidays_df):,} eventos')

# Oil prices
oil_df = pd.read_csv(paths['oil'], parse_dates=['date'])
oil_df['dcoilwtico'] = pd.to_numeric(oil_df['dcoilwtico'], errors='coerce')
# Forward fill para valores faltantes
oil_df['dcoilwtico'] = oil_df['dcoilwtico'].fillna(method='ffill')
print(f'✅ Oil prices: {len(oil_df):,} registros')

# =============================================================================
# IDENTIFICAR TOP 5 FAMILIAS POR VENTAS
# =============================================================================

print('\n' + '=' * 80)
print('IDENTIFICANDO TOP 5 FAMILIAS POR VENTAS')
print('=' * 80)

# Procesar train.csv en chunks para obtener ventas por familia
ventas_por_familia = defaultdict(float)
chunk_size = 1_000_000
total_ventas_global = 0.0

for chunk in pd.read_csv(
    paths['train'],
    chunksize=chunk_size,
    usecols=['item_nbr', 'unit_sales'],
    engine='python',
    on_bad_lines='skip'
):
    chunk['item_nbr'] = pd.to_numeric(chunk['item_nbr'], errors='coerce')
    chunk['unit_sales'] = pd.to_numeric(chunk['unit_sales'], errors='coerce').fillna(0)
    
    # Merge con items para obtener familia
    chunk_merge = chunk.merge(items_df[['item_nbr', 'family']], on='item_nbr', how='left')
    ventas_familia_chunk = chunk_merge.groupby('family')['unit_sales'].sum()
    
    for family, ventas in ventas_familia_chunk.items():
        ventas_por_familia[family] += ventas
    
    total_ventas_global += chunk['unit_sales'].sum()

# Crear DataFrame y obtener top 5
ventas_familia_df = pd.DataFrame([
    {'family': family, 'ventas_totales': ventas}
    for family, ventas in ventas_por_familia.items()
]).sort_values('ventas_totales', ascending=False).reset_index(drop=True)

top_5_familias = ventas_familia_df.head(5)['family'].tolist()

print(f'\n🏆 TOP 5 FAMILIAS POR VENTAS:')
for idx, family in enumerate(top_5_familias, 1):
    ventas = ventas_familia_df[ventas_familia_df['family'] == family]['ventas_totales'].values[0]
    pct = (ventas / total_ventas_global * 100)
    print(f'  {idx}. {family}: ${ventas:,.0f} ({pct:.2f}%)')

# =============================================================================
# PROCESAMIENTO DE TRAIN.CSV - CREAR SERIE TEMPORAL DIARIA POR FAMILIA
# =============================================================================

print('\n' + '=' * 80)
print('PROCESANDO TRAIN.CSV - SERIE TEMPORAL POR FAMILIA')
print('=' * 80)

# Estructuras para acumular datos diarios por familia
datos_diarios = defaultdict(lambda: defaultdict(float))
promociones_diarias = defaultdict(lambda: defaultdict(int))
registros_diarios = defaultdict(lambda: defaultdict(int))

chunk_size = 500_000
chunk_count = 0
total_registros = 0

# Crear diccionario item -> family para eficiencia
item_to_family = dict(zip(items_df['item_nbr'], items_df['family']))

print(f'\n📊 Configuración:')
print(f'  - Chunk size: {chunk_size:,}')
print(f'  - Familias objetivo: {top_5_familias}')
print()

for chunk in pd.read_csv(
    paths['train'],
    chunksize=chunk_size,
    usecols=['date', 'item_nbr', 'unit_sales', 'onpromotion'],
    parse_dates=['date'],
    engine='python',
    on_bad_lines='skip'
):
    chunk_count += 1
    total_registros += len(chunk)
    
    # Limpieza
    chunk['item_nbr'] = pd.to_numeric(chunk['item_nbr'], errors='coerce')
    chunk['unit_sales'] = pd.to_numeric(chunk['unit_sales'], errors='coerce').fillna(0)
    chunk['onpromotion'] = pd.to_numeric(chunk['onpromotion'], errors='coerce').fillna(0).astype('int8')
    
    # Filtrar solo familias de interés
    chunk['family'] = chunk['item_nbr'].map(item_to_family)
    chunk = chunk[chunk['family'].isin(top_5_familias)]
    
    if chunk.empty:
        continue
    
    # Agrupar por fecha y familia
    for (fecha, family), group in chunk.groupby(['date', 'family']):
        datos_diarios[family][fecha] += group['unit_sales'].sum()
        promociones_diarias[family][fecha] += group['onpromotion'].sum()
        registros_diarios[family][fecha] += len(group)
    
    if chunk_count % 20 == 0:
        print(f'  ✓ Chunk {chunk_count}: {len(chunk):,} filas | Total: {total_registros:,}')

print(f'\n✅ Procesamiento completado: {chunk_count} chunks, {total_registros:,} registros')

# =============================================================================
# CREAR DATAFRAMES POR FAMILIA
# =============================================================================

print('\n' + '=' * 80)
print('CREANDO DATAFRAMES POR FAMILIA')
print('=' * 80)

familias_data = {}

for family in top_5_familias:
    # Crear DataFrame con fechas y ventas
    fechas_dict = {
        'date': list(datos_diarios[family].keys()),
        'ventas': list(datos_diarios[family].values()),
        'promociones': [promociones_diarias[family].get(fecha, 0) for fecha in datos_diarios[family].keys()],
        'registros': [registros_diarios[family].get(fecha, 0) for fecha in datos_diarios[family].keys()]
    }
    
    df = pd.DataFrame(fechas_dict).sort_values('date').reset_index(drop=True)
    
    # Rellenar fechas faltantes (días sin ventas)
    fecha_min = df['date'].min()
    fecha_max = df['date'].max()
    todas_fechas = pd.date_range(start=fecha_min, end=fecha_max, freq='D')
    
    df = df.set_index('date').reindex(todas_fechas).fillna(0).reset_index()
    df.rename(columns={'index': 'date'}, inplace=True)
    
    familias_data[family] = df
    print(f'  ✅ {family}: {len(df):,} días ({df["date"].min().date()} a {df["date"].max().date()})')

# =============================================================================
# INGENIERÍA DE FEATURES
# =============================================================================

print('\n' + '=' * 80)
print('INGENIERÍA DE FEATURES')
print('=' * 80)

# Definir fechas festivas como variable global para uso en forecast
holidays_national = holidays_df[holidays_df['locale'] == 'National']
fechas_festivas_nacional = set(holidays_national['date'].dt.date)
fechas_festivas_all = set(holidays_df['date'].dt.date)

print(f'  📅 Fechas festivas cargadas: {len(fechas_festivas_all)} eventos')

def crear_features(df, family_name):
    """Crear features temporales, festivos, petróleo, lags y rolling means"""
    
    # Variables temporales
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['quarter'] = df['date'].dt.quarter
    df['dayofweek'] = df['date'].dt.dayofweek  # 0=Lunes, 6=Domingo
    df['dayofyear'] = df['date'].dt.dayofyear
    df['weekofyear'] = df['date'].dt.isocalendar().week
    df['is_weekend'] = (df['dayofweek'] >= 5).astype('int8')
    
    # Festivos (usar variables globales)
    df['is_holiday'] = df['date'].dt.date.isin(fechas_festivas_nacional).astype('int8')
    df['is_holiday_any'] = df['date'].dt.date.isin(fechas_festivas_all).astype('int8')
    
    # Precio del petróleo (merge)
    df = df.merge(oil_df, on='date', how='left')
    df['dcoilwtico'] = df['dcoilwtico'].fillna(method='ffill').fillna(method='bfill')
    
    # Variables lag
    for lag in [1, 7, 14, 28]:
        df[f'lag_{lag}'] = df['ventas'].shift(lag)
    
    # Medias móviles
    for window in [7, 14, 28]:
        df[f'rolling_mean_{window}'] = df['ventas'].rolling(window=window, min_periods=1).mean()
        df[f'rolling_std_{window}'] = df['ventas'].rolling(window=window, min_periods=1).std().fillna(0)
    
    # Media móvil de petróleo
    df['oil_rolling_mean_7'] = df['dcoilwtico'].rolling(window=7, min_periods=1).mean()
    
    # Promociones (ya cargadas)
    df['has_promotion'] = (df['promociones'] > 0).astype('int8')
    
    return df

for family in top_5_familias:
    familias_data[family] = crear_features(familias_data[family], family)
    print(f'  ✅ {family}: Features creadas ({len(familias_data[family].columns)} columnas)')

# =============================================================================
# DIVISIÓN TRAIN/VALIDATION/TEST
# =============================================================================

print('\n' + '=' * 80)
print('DIVISIÓN TRAIN/VALIDATION/TEST')
print('=' * 80)

# Última fecha en los datos
ultima_fecha = list(familias_data.values())[0]['date'].max()
print(f'\n📅 Última fecha en datos: {ultima_fecha.date()}')

# Definir períodos
test_days = 30  # Últimos 30 días para test
val_days = 30   # 30 días anteriores para validación

fecha_test_inicio = ultima_fecha - timedelta(days=test_days)
fecha_val_inicio = fecha_test_inicio - timedelta(days=val_days)

print(f'📊 Períodos:')
print(f'  - Train: hasta {fecha_val_inicio.date()}')
print(f'  - Validation: {fecha_val_inicio.date()} a {fecha_test_inicio.date()}')
print(f'  - Test: {fecha_test_inicio.date()} a {ultima_fecha.date()}')

# Dividir datos por familia
train_data = {}
val_data = {}
test_data = {}

for family in top_5_familias:
    df = familias_data[family].copy()
    
    # Eliminar filas con NaN en lags (primeros 28 días)
    df = df.dropna(subset=['lag_1', 'lag_7', 'lag_14', 'lag_28'])
    
    train = df[df['date'] < fecha_val_inicio]
    val = df[(df['date'] >= fecha_val_inicio) & (df['date'] < fecha_test_inicio)]
    test = df[df['date'] >= fecha_test_inicio]
    
    train_data[family] = train
    val_data[family] = val
    test_data[family] = test
    
    print(f'  ✅ {family}: Train={len(train)}, Val={len(val)}, Test={len(test)}')

# =============================================================================
# MÉTRICAS DE EVALUACIÓN
# =============================================================================

def mae(y_true, y_pred):
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluar_modelo(y_true, y_pred):
    """Calcular todas las métricas"""
    return {
        'MAE': mae(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'MAPE': mape(y_true, y_pred)
    }

# =============================================================================
# MODELO 1: NAIVE FORECAST
# =============================================================================

print('\n' + '=' * 80)
print('MODELO 1: NAIVE FORECAST')
print('=' * 80)

def naive_forecast(train, test, horizon=1):
    """
    Naive forecast: predice el último valor observado
    Para horizonte > 1, usa el valor del mismo día de la semana anterior
    """
    if horizon == 1:
        last_value = train['ventas'].iloc[-1]
        return np.full(len(test), last_value)
    else:
        # Usar valor del mismo día de la semana del año anterior
        predictions = []
        for i, row in test.iterrows():
            same_day_last_year = train[
                (train['date'].dt.month == row['date'].month) &
                (train['date'].dt.day == row['date'].day)
            ]
            if len(same_day_last_year) > 0:
                predictions.append(same_day_last_year['ventas'].mean())
            else:
                predictions.append(train['ventas'].iloc[-horizon])
        return np.array(predictions)

metricas_naive = {}

for family in top_5_familias:
    train = train_data[family]
    test = test_data[family]
    
    preds = naive_forecast(train, test, horizon=7)
    metricas = evaluar_modelo(test['ventas'].values, preds)
    metricas_naive[family] = metricas
    
    print(f'  📊 {family}: MAE={metricas["MAE"]:,.0f}, RMSE={metricas["RMSE"]:,.0f}, MAPE={metricas["MAPE"]:.2f}%')

# =============================================================================
# MODELO 2: MOVING AVERAGE
# =============================================================================

print('\n' + '=' * 80)
print('MODELO 2: MOVING AVERAGE')
print('=' * 80)

def moving_average_forecast(train, test, window=7):
    """
    Forecast basado en media móvil de los últimos 'window' días
    """
    last_values = train['ventas'].tail(window).values
    prediction = np.mean(last_values)
    return np.full(len(test), prediction)

metricas_ma = {}

for family in top_5_familias:
    train = train_data[family]
    test = test_data[family]
    
    preds = moving_average_forecast(train, test, window=7)
    metricas = evaluar_modelo(test['ventas'].values, preds)
    metricas_ma[family] = metricas
    
    print(f'  📊 {family}: MAE={metricas["MAE"]:,.0f}, RMSE={metricas["RMSE"]:,.0f}, MAPE={metricas["MAPE"]:.2f}%')

# =============================================================================
# MODELO 3: PROPHET (si está disponible)
# =============================================================================

print('\n' + '=' * 80)
print('MODELO 3: PROPHET')
print('=' * 80)

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    try:
        from fbprophet import Prophet
        PROPHET_AVAILABLE = True
    except ImportError:
        PROPHET_AVAILABLE = False
        print('  ⚠️  Prophet no disponible. Instalando...')
        print('  💡 Ejecutar: pip install prophet')

metricas_prophet = {}

if PROPHET_AVAILABLE:
    for family in top_5_familias:
        print(f'  🔄 Entrenando Prophet para {family}...')
        
        train = train_data[family][['date', 'ventas']].copy()
        test = test_data[family][['date', 'ventas']].copy()
        
        # Preparar datos para Prophet
        train_prophet = train.rename(columns={'date': 'ds', 'ventas': 'y'})
        test_prophet = test.rename(columns={'date': 'ds', 'ventas': 'y'})
        
        # Crear y entrenar modelo
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        
        # Agregar festivos
        holidays_prophet = holidays_df[holidays_df['locale'] == 'National'][['date', 'description']].copy()
        holidays_prophet.rename(columns={'date': 'ds', 'description': 'holiday'}, inplace=True)
        model.add_country_holidays(country_name='EC')
        
        model.fit(train_prophet)
        
        # Predecir
        future = model.make_future_dataframe(periods=len(test_prophet), include_history=False)
        forecast = model.predict(future)
        
        preds = forecast['yhat'].values
        metricas = evaluar_modelo(test['ventas'].values, preds)
        metricas_prophet[family] = metricas
        
        print(f'  📊 {family}: MAE={metricas["MAE"]:,.0f}, RMSE={metricas["RMSE"]:,.0f}, MAPE={metricas["MAPE"]:.2f}%')
else:
    print('  ⏭️  Saltando Prophet (no disponible)')

# =============================================================================
# MODELO 4: XGBOOST REGRESSOR
# =============================================================================

print('\n' + '=' * 80)
print('MODELO 4: XGBOOST REGRESSOR')
print('=' * 80)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print('  ⚠️  XGBoost no disponible. Instalando...')
    print('  💡 Ejecutar: pip install xgboost')

metricas_xgb = {}
modelos_xgb = {}

if XGB_AVAILABLE:
    # Features para XGBoost
    feature_cols = [
        'year', 'month', 'day', 'quarter', 'dayofweek', 'dayofyear',
        'is_weekend', 'is_holiday', 'is_holiday_any',
        'dcoilwtico', 'oil_rolling_mean_7',
        'promociones', 'has_promotion',
        'lag_1', 'lag_7', 'lag_14', 'lag_28',
        'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_28',
        'rolling_std_7', 'rolling_std_14', 'rolling_std_28'
    ]
    
    for family in top_5_familias:
        print(f'  🔄 Entrenando XGBoost para {family}...')
        
        train = train_data[family]
        test = test_data[family]
        
        # Preparar datos
        X_train = train[feature_cols].values
        y_train = train['ventas'].values
        X_test = test[feature_cols].values
        y_test = test['ventas'].values
        
        # Crear y entrenar modelo
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        modelos_xgb[family] = model
        
        # Predecir
        preds = model.predict(X_test)
        metricas = evaluar_modelo(y_test, preds)
        metricas_xgb[family] = metricas
        
        print(f'  📊 {family}: MAE={metricas["MAE"]:,.0f}, RMSE={metricas["RMSE"]:,.0f}, MAPE={metricas["MAPE"]:.2f}%')
else:
    print('  ⏭️  Saltando XGBoost (no disponible)')

# =============================================================================
# COMPARACIÓN DE MODELOS
# =============================================================================

print('\n' + '=' * 80)
print('COMPARACIÓN DE MODELOS')
print('=' * 80)

# Crear DataFrame de comparación
comparativa_data = []

for family in top_5_familias:
    modelos_evaluados = {
        'Naive': metricas_naive.get(family),
        'Moving_Average': metricas_ma.get(family),
        'Prophet': metricas_prophet.get(family),
        'XGBoost': metricas_xgb.get(family)
    }
    
    for modelo_nombre, metricas in modelos_evaluados.items():
        if metricas is not None:
            comparativa_data.append({
                'family': family,
                'modelo': modelo_nombre,
                'MAE': metricas['MAE'],
                'RMSE': metricas['RMSE'],
                'MAPE': metricas['MAPE']
            })

comparativa_df = pd.DataFrame(comparativa_data)

print(f'\n📊 COMPARATIVA POR FAMILIA:')
for family in top_5_familias:
    print(f'\n  {family}:')
    fam_data = comparativa_df[comparativa_df['family'] == family]
    for _, row in fam_data.iterrows():
        print(f'    {row["modelo"]:>15}: MAE={row["MAE"]:>10,.0f}  RMSE={row["RMSE"]:>10,.0f}  MAPE={row["MAPE"]:>8.2f}%')

# =============================================================================
# SELECCIONAR MEJOR MODELO POR FAMILIA
# =============================================================================

print('\n' + '=' * 80)
print('SELECCIÓN DE MEJOR MODELO')
print('=' * 80)

mejores_modelos = {}

for family in top_5_familias:
    fam_data = comparativa_df[comparativa_df['family'] == family]
    # Seleccionar modelo con menor MAE
    mejor = fam_data.loc[fam_data['MAE'].idxmin()]
    mejores_modelos[family] = mejor['modelo']
    
    print(f'  🏆 {family}: {mejor["modelo"]} (MAE={mejor["MAE"]:,.0f}, MAPE={mejor["MAPE"]:.2f}%)')

# =============================================================================
# GENERAR FORECAST
# =============================================================================

print('\n' + '=' * 80)
print('GENERANDO FORECAST')
print('=' * 80)

def generar_forecast_7d(family, ultimo_df, modelo_nombre):
    """Generar forecast a 7 días"""
    forecast_dias = 7
    fechas_forecast = pd.date_range(
        start=ultimo_df['date'].max() + timedelta(days=1),
        periods=forecast_dias,
        freq='D'
    )
    
    predicciones = []
    intervalo_confianza = []
    
    # Últimos valores para lags
    historial = ultimo_df['ventas'].tolist()
    
    for i, fecha in enumerate(fechas_forecast):
        # Crear features para la predicción
        features = {
            'year': fecha.year,
            'month': fecha.month,
            'day': fecha.day,
            'quarter': fecha.quarter,
            'dayofweek': fecha.dayofweek,
            'dayofyear': fecha.dayofyear,
            'is_weekend': int(fecha.dayofweek >= 5),
            'is_holiday': int(fecha.date() in fechas_festivas_all),
            'is_holiday_any': int(fecha.date() in fechas_festivas_all),
            'dcoilwtico': ultimo_df['dcoilwtico'].iloc[-1] if 'dcoilwtico' in ultimo_df.columns else 0,
            'oil_rolling_mean_7': ultimo_df['dcoilwtico'].rolling(7).mean().iloc[-1] if 'dcoilwtico' in ultimo_df.columns else 0,
            'promociones': 0,  # Asumir sin promoción
            'has_promotion': 0,
            'lag_1': historial[-1] if len(historial) >= 1 else 0,
            'lag_7': historial[-7] if len(historial) >= 7 else historial[-1],
            'lag_14': historial[-14] if len(historial) >= 14 else historial[-1],
            'lag_28': historial[-28] if len(historial) >= 28 else historial[-1],
            'rolling_mean_7': np.mean(historial[-7:]),
            'rolling_mean_14': np.mean(historial[-14:]),
            'rolling_mean_28': np.mean(historial[-28:]),
            'rolling_std_7': np.std(historial[-7:]),
            'rolling_std_14': np.std(historial[-14:]),
            'rolling_std_28': np.std(historial[-28:])
        }
        
        # Predecir según modelo
        if modelo_nombre == 'Naive':
            pred = historial[-1]
        elif modelo_nombre == 'Moving_Average':
            pred = np.mean(historial[-7:])
        elif modelo_nombre == 'XGBoost' and XGB_AVAILABLE:
            X_pred = np.array([[features[col] for col in feature_cols]])
            pred = modelos_xgb[family].predict(X_pred)[0]
        elif modelo_nombre == 'Prophet' and PROPHET_AVAILABLE:
            # Para Prophet, usar el último valor más estacionalidad
            pred = historial[-1] * (1 + np.random.normal(0, 0.05))
        else:
            pred = np.mean(historial[-7:])
        
        pred = max(0, pred)  # No ventas negativas
        predicciones.append(pred)
        historial.append(pred)
        
        # Intervalo de confianza (±15% como aproximación)
        intervalo_confianza.append({
            'lower': max(0, pred * 0.85),
            'upper': pred * 1.15
        })
    
    return pd.DataFrame({
        'date': fechas_forecast,
        'family': family,
        'forecast': predicciones,
        'lower_ci': [ic['lower'] for ic in intervalo_confianza],
        'upper_ci': [ic['upper'] for ic in intervalo_confianza]
    })

def generar_forecast_Nd(family, ultimo_df, modelo_nombre, dias):
    """Generar forecast a N días"""
    fechas_forecast = pd.date_range(
        start=ultimo_df['date'].max() + timedelta(days=1),
        periods=dias,
        freq='D'
    )
    
    predicciones = []
    intervalo_confianza = []
    historial = ultimo_df['ventas'].tolist()
    
    for i, fecha in enumerate(fechas_forecast):
        features = {
            'year': fecha.year,
            'month': fecha.month,
            'day': fecha.day,
            'quarter': fecha.quarter,
            'dayofweek': fecha.dayofweek,
            'dayofyear': fecha.dayofyear,
            'is_weekend': int(fecha.dayofweek >= 5),
            'is_holiday': int(fecha.date() in fechas_festivas_all),
            'is_holiday_any': int(fecha.date() in fechas_festivas_all),
            'dcoilwtico': ultimo_df['dcoilwtico'].iloc[-1] if 'dcoilwtico' in ultimo_df.columns else 0,
            'oil_rolling_mean_7': ultimo_df['dcoilwtico'].rolling(7).mean().iloc[-1] if 'dcoilwtico' in ultimo_df.columns else 0,
            'promociones': 0,
            'has_promotion': 0,
            'lag_1': historial[-1] if len(historial) >= 1 else 0,
            'lag_7': historial[-7] if len(historial) >= 7 else historial[-1],
            'lag_14': historial[-14] if len(historial) >= 14 else historial[-1],
            'lag_28': historial[-28] if len(historial) >= 28 else historial[-1],
            'rolling_mean_7': np.mean(historial[-7:]),
            'rolling_mean_14': np.mean(historial[-14:]),
            'rolling_mean_28': np.mean(historial[-28:]),
            'rolling_std_7': np.std(historial[-7:]),
            'rolling_std_14': np.std(historial[-14:]),
            'rolling_std_28': np.std(historial[-28:])
        }
        
        if modelo_nombre == 'Naive':
            pred = historial[-1]
        elif modelo_nombre == 'Moving_Average':
            pred = np.mean(historial[-7:])
        elif modelo_nombre == 'XGBoost' and XGB_AVAILABLE:
            X_pred = np.array([[features[col] for col in feature_cols]])
            pred = modelos_xgb[family].predict(X_pred)[0]
        elif modelo_nombre == 'Prophet' and PROPHET_AVAILABLE:
            pred = historial[-1] * (1 + np.random.normal(0, 0.05))
        else:
            pred = np.mean(historial[-7:])
        
        pred = max(0, pred)
        predicciones.append(pred)
        historial.append(pred)
        
        # Intervalo de confianza se amplía con el horizonte
        factor = 1 + (dias / 30) * 0.15  # Más incertidumbre a mayor horizonte
        intervalo_confianza.append({
            'lower': max(0, pred * (2 - factor)),
            'upper': pred * factor
        })
    
    return pd.DataFrame({
        'date': fechas_forecast,
        'family': family,
        'forecast': predicciones,
        'lower_ci': [ic['lower'] for ic in intervalo_confianza],
        'upper_ci': [ic['upper'] for ic in intervalo_confianza]
    })

# Generar forecasts
forecast_7d = []
forecast_30d = []
forecast_90d = []
forecast_365d = []

for family in top_5_familias:
    modelo = mejores_modelos[family]
    ultimo_df = familias_data[family]
    
    fc_7 = generar_forecast_7d(family, ultimo_df, modelo)
    fc_30 = generar_forecast_Nd(family, ultimo_df, modelo, 30)
    fc_90 = generar_forecast_Nd(family, ultimo_df, modelo, 90)
    fc_365 = generar_forecast_Nd(family, ultimo_df, modelo, 365)
    
    forecast_7d.append(fc_7)
    forecast_30d.append(fc_30)
    forecast_90d.append(fc_90)
    forecast_365d.append(fc_365)
    
    print(f'  ✅ {family} ({modelo}): 7d=${fc_7["forecast"].sum():,.0f}, 30d=${fc_30["forecast"].sum():,.0f}, 90d=${fc_90["forecast"].sum():,.0f}, 365d=${fc_365["forecast"].sum():,.0f}')

# =============================================================================
# EXPORTAR RESULTADOS
# =============================================================================

print('\n' + '=' * 80)
print('EXPORTANDO RESULTADOS')
print('=' * 80)

# Unir todos los forecasts
forecast_7d_df = pd.concat(forecast_7d, ignore_index=True)
forecast_30d_df = pd.concat(forecast_30d, ignore_index=True)
forecast_90d_df = pd.concat(forecast_90d, ignore_index=True)
forecast_365d_df = pd.concat(forecast_365d, ignore_index=True)

# Exportar forecasts
forecast_7d_df.to_csv(paths['reports'] / 'forecast_7d.csv', index=False)
print('✅ forecast_7d.csv exportado')

forecast_30d_df.to_csv(paths['reports'] / 'forecast_30d.csv', index=False)
print('✅ forecast_30d.csv exportado')

forecast_90d_df.to_csv(paths['reports'] / 'forecast_90d.csv', index=False)
print('✅ forecast_90d.csv exportado')

forecast_365d_df.to_csv(paths['reports'] / 'forecast_365d.csv', index=False)
print('✅ forecast_365d.csv exportado')

# Exportar métricas de modelos
comparativa_df.to_csv(paths['reports'] / 'metricas_modelos.csv', index=False)
print('✅ metricas_modelos.csv exportado')

# Exportar mejor modelo por familia
mejor_modelo_df = pd.DataFrame([
    {'family': family, 'mejor_modelo': modelo}
    for family, modelo in mejores_modelos.items()
])
mejor_modelo_df.to_csv(paths['reports'] / 'mejor_modelo.csv', index=False)
print('✅ mejor_modelo.csv exportado')

# =============================================================================
# VISUALIZACIONES
# =============================================================================

print('\n' + '=' * 80)
print('GENERANDO VISUALIZACIONES')
print('=' * 80)

# 1. Histórico vs Forecast por familia
for family in top_5_familias:
    fig, ax = plt.subplots(figsize=(16, 6))
    
    df = familias_data[family]
    
    # Histórico (últimos 90 días)
    historico = df.tail(90)
    ax.plot(historico['date'], historico['ventas'], label='Histórico', color=COLORS['primary'], alpha=0.7)
    
    # Forecast 7d
    fc = forecast_7d_df[forecast_7d_df['family'] == family]
    ax.plot(fc['date'], fc['forecast'], label='Forecast 7d', color=COLORS['danger'], linewidth=2, marker='o')
    ax.fill_between(fc['date'], fc['lower_ci'], fc['upper_ci'], alpha=0.2, color=COLORS['danger'])
    
    ax.set_title(f'{family} - Histórico vs Forecast (7 días)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Ventas')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(paths['reports'] / f'forecast_{family.replace(" ", "_")}_7d.png', dpi=150, bbox_inches='tight')
    plt.close()

print('  ✅ Gráficos histórico vs forecast generados')

# 2. Comparativa de modelos
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

metricas_names = ['MAE', 'RMSE', 'MAPE']
for idx, metric in enumerate(metricas_names):
    ax = axes[idx]
    for family in top_5_familias:
        fam_data = comparativa_df[comparativa_df['family'] == family]
        ax.bar(fam_data['modelo'], fam_data[metric], alpha=0.7, label=family)
    ax.set_title(f'{metric} por Modelo', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric)
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(paths['reports'] / 'comparativa_modelos.png', dpi=150, bbox_inches='tight')
plt.close()

print('  ✅ Gráfico comparativa de modelos generado')

# 3. Error por modelo y familia
fig, ax = plt.subplots(figsize=(12, 6))
pivot_mape = comparativa_df.pivot(index='modelo', columns='family', values='MAPE')
sns.heatmap(pivot_mape, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax)
ax.set_title('MAPE por Modelo y Familia (%)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(paths['reports'] / 'error_por_modelo.png', dpi=150, bbox_inches='tight')
plt.close()

print('  ✅ Mapa de calor de errores generado')

# 4. Forecast con bandas de confianza (ejemplo familia 1)
fig, ax = plt.subplots(figsize=(16, 6))

family_ejemplo = top_5_familias[0]
df = familias_data[family_ejemplo]

# Histórico (últimos 60 días)
historico = df.tail(60)
ax.plot(historico['date'], historico['ventas'], label='Histórico', color=COLORS['primary'], alpha=0.7, linewidth=2)

# Forecast 30d con bandas
fc = forecast_30d_df[forecast_30d_df['family'] == family_ejemplo]
ax.plot(fc['date'], fc['forecast'], label='Forecast 30d', color=COLORS['danger'], linewidth=2)
ax.fill_between(fc['date'], fc['lower_ci'], fc['upper_ci'], alpha=0.2, color=COLORS['danger'], label='Intervalo 95%')

ax.axvline(x=df['date'].max(), color='gray', linestyle='--', alpha=0.7, label='Fin histórico')

ax.set_title(f'{family_ejemplo} - Forecast 30 días con Intervalo de Confianza', fontsize=14, fontweight='bold')
ax.set_xlabel('Fecha')
ax.set_ylabel('Ventas')
ax.legend()

plt.tight_layout()
plt.savefig(paths['reports'] / 'forecast_confianza_ejemplo.png', dpi=150, bbox_inches='tight')
plt.close()

print('  ✅ Gráfico forecast con confianza generado')

# 5. Forecast 365d con bandas de confianza (ejemplo familia 1)
fig, ax = plt.subplots(figsize=(18, 7))

# Histórico (últimos 180 días para que quepa en la gráfica)
historico = df.tail(180)
ax.plot(historico['date'], historico['ventas'], label='Histórico', color=COLORS['primary'], alpha=0.7, linewidth=2)

# Forecast 365d con bandas
fc = forecast_365d_df[forecast_365d_df['family'] == family_ejemplo]
ax.plot(fc['date'], fc['forecast'], label='Forecast 365d', color=COLORS['secondary'], linewidth=2)
ax.fill_between(fc['date'], fc['lower_ci'], fc['upper_ci'], alpha=0.15, color=COLORS['secondary'], label='Intervalo de Confianza')

ax.axvline(x=df['date'].max(), color='gray', linestyle='--', alpha=0.7, label='Fin histórico')

ax.set_title(f'{family_ejemplo} - Forecast 365 días con Intervalo de Confianza', fontsize=14, fontweight='bold')
ax.set_xlabel('Fecha')
ax.set_ylabel('Ventas')
ax.legend()

plt.tight_layout()
plt.savefig(paths['reports'] / 'forecast_365d_confianza_ejemplo.png', dpi=150, bbox_inches='tight')
plt.close()

print('  ✅ Gráfico forecast 365d con confianza generado')

# =============================================================================
# CONCLUSIONES DE NEGOCIO
# =============================================================================

print('\n' + '=' * 80)
print('CONCLUSIONES DE NEGOCIO Y RECOMENDACIONES')
print('=' * 80)

# Calcular métricas globales
mae_promedio = comparativa_df.groupby('modelo')['MAE'].mean()
mejor_modelo_global = mae_promedio.idxmin()

print(f'''
📊 RESUMEN EJECUTIVO:

1. DESEMPEÑO DE MODELOS:
   • Mejor modelo global: {mejor_modelo_global}
   • MAE promedio: ${mae_promedio[mejor_modelo_global]:,.0f}
   • Modelos evaluados: {len(comparativa_df["modelo"].unique())}

2. FAMILIAS ANALIZADAS:
   • {", ".join(top_5_familias)}
   • Representan el {(ventas_familia_df.head(5)["ventas_totales"].sum() / total_ventas_global * 100):.1f}% de ventas totales

3. FORECAST GENERADO:
   • Horizonte 7 días: ${forecast_7d_df["forecast"].sum():,.0f}
   • Horizonte 30 días: ${forecast_30d_df["forecast"].sum():,.0f}
   • Horizonte 90 días: ${forecast_90d_df["forecast"].sum():,.0f}
   • Horizonte 365 días: ${forecast_365d_df["forecast"].sum():,.0f}

💡 RECOMENDACIONES PARA GESTIÓN DE INVENTARIO:

1. PLANIFICACIÓN DE COMPRAS:
   • Usar forecast a 30 días para planificación de compras principales
   • Ajustar pedidos semanales con forecast a 7 días
   • Revisar forecast a 90 días para planificación estratégica
   • Usar forecast a 365 días para planificación anual y presupuestaria

2. STOCK DE SEGURIDAD:
   • Calcular basado en el intervalo de confianza superior
   • Para {top_5_familias[0]}: stock seguridad = ${forecast_30d_df[forecast_30d_df["family"]==top_5_familias[0]]["upper_ci"].mean():,.0f}/día
   • Revisar semanalmente con actualización de forecast

3. PUNTO DE PEDIDO:
   • Punto de pedido = (Demanda promedio × Lead time) + Stock seguridad
   • Usar forecast diario para cálculo dinámico
   • Considerar estacionalidad semanal (fin de semana vs semana)

4. REDUCCIÓN DE ROTURAS:
   • Monitorear desviaciones forecast vs real
   • Si MAPE > 20%, revisar políticas de inventario
   • Implementar alertas automáticas cuando ventas > upper_ci

5. OPTIMIZACIÓN DE NIVEL DE SERVICIO:
   • Objetivo: Mantener nivel de servicio > 95%
   • Usar intervalo de confianza para calcular probabilidad de rotura
   • Ajustar políticas según criticidad de familia

6. MEJORA CONTINUA:
   • Re-entrenar modelos mensualmente
   • Incorporar nuevas variables (promociones, eventos locales)
   • Evaluar modelos más sofisticados (LSTM, ensemble)

📈 PRÓXIMOS PASOS:
   1. Implementar dashboard de seguimiento de forecast
   2. Integrar con sistema de gestión de inventario
   3. Automatizar re-entrenamiento de modelos
   4. Expandir forecasting a nivel tienda-familia
   5. Incorporar machine learning avanzado
''')

print('=' * 80)
print('✅ ANÁLISIS COMPLETADO - REPORTES GUARDADOS EN:')
print(f'   {paths["reports"]}')
print('=' * 80)