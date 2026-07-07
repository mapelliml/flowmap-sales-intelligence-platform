# =============================================================================
# 10_SIMULACION_ESCENARIOS.py
# Sistema de Simulación Avanzada de Demanda e Inventario por Escenarios
# =============================================================================
# Este script genera simulaciones avanzadas de demanda e inventario basadas en:
# - Históricos de ventas
# - Festivos
# - Tiendas
# - Transacciones
# - Productos
# - Parámetros de negocio
#
# El resultado final es un archivo simulacion_final_escenarios.csv que puede ser
# utilizado por una aplicación Streamlit para análisis interactivo de escenarios.
#
# Autor: Lead Data Scientist - Inventory Optimization & Supply Chain
# Versión: 1.0 - Simulación de Escenarios Multi-paramétricos
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
import logging
import gc
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN DE LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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
    'dark_green': '#1E8449',
    'light_blue': '#5DADE2',
    'orange': '#E67E22',
    'red': '#C0392B'
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
    data_output_path = project_root / 'data'
    data_output_path.mkdir(parents=True, exist_ok=True)
    
    reports_path = project_root / 'reports' / 'scenarios'
    reports_path.mkdir(parents=True, exist_ok=True)
    
    visuals_dir = project_root / 'visuals' / 'scenarios'
    visuals_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        'project_root': project_root,
        'data_raw': data_raw_path,
        'data_output': data_output_path,
        'reports': reports_path,
        'visuals': visuals_dir,
        'train': data_raw_path / 'train.csv',
        'items': data_raw_path / 'items.csv',
        'holidays': data_raw_path / 'holidays_events.csv',
        'stores': data_raw_path / 'stores.csv',
        'transactions': data_raw_path / 'transactions.csv'
    }

paths = setup_paths()

logger.info('=' * 80)
logger.info('SISTEMA DE SIMULACIÓN DE ESCENARIOS DE DEMANDA E INVENTARIO')
logger.info('=' * 80)
logger.info(f'\n📁 Proyecto: {paths["project_root"]}')
logger.info(f'📁 Reports de salida: {paths["reports"]}')
logger.info(f'📁 Visualizaciones: {paths["visuals"]}')
logger.info(f'📁 Datos de salida: {paths["data_output"]}')
logger.info('')

# =============================================================================
# PARÁMETROS CONFIGURABLES DEL SISTEMA
# =============================================================================

class SimulationParameters:
    """Clase para gestionar parámetros configurables del sistema de simulación"""
    
    def __init__(self):
        # Niveles de servicio (Z-score)
        self.service_levels = {
            0.90: 1.28,  # 90% -> Z = 1.28
            0.95: 1.65,  # 95% -> Z = 1.65
            0.99: 2.33   # 99% -> Z = 2.33
        }
        
        # Lead times a simular (días)
        self.lead_times = [2, 5, 7, 14]
        
        # Umbrales de clasificación ABC
        self.abc_thresholds = {
            'A': 0.80,  # 80% del valor acumulado
            'B': 0.95,  # 95% del valor acumulado
            'C': 1.00   # 100% del valor acumulado
        }
        
        # Escenarios de simulación
        self.scenarios = {
            'BASELINE': {
                'description': 'Escenario base sin cambios',
                'holiday_effect': 0.0,
                'promotion_effect': 0.0,
                'demand_multiplier': 1.0,
                'lead_time_multiplier': 1.0
            },
            'FESTIVO': {
                'description': 'Efecto de festivo histórico',
                'holiday_effect': None,  # Se calculará dinámicamente
                'promotion_effect': 0.0,
                'demand_multiplier': 1.0,
                'lead_time_multiplier': 1.0
            },
            'PROMOCION_SUAVE': {
                'description': 'Promoción 10% descuento',
                'holiday_effect': 0.0,
                'promotion_effect': None,  # Se calculará dinámicamente
                'demand_multiplier': 1.0,
                'lead_time_multiplier': 1.0
            },
            'PROMOCION_MEDIA': {
                'description': 'Promoción 20% descuento',
                'holiday_effect': 0.0,
                'promotion_effect': None,
                'demand_multiplier': 1.0,
                'lead_time_multiplier': 1.0
            },
            'PROMOCION_AGRESIVA': {
                'description': 'Promoción 30% descuento',
                'holiday_effect': 0.0,
                'promotion_effect': None,
                'demand_multiplier': 1.0,
                'lead_time_multiplier': 1.0
            },
            'FESTIVO_MAS_PROMOCION': {
                'description': 'Festivo + Promoción 20%',
                'description': 'Festivo + Promoción 20%',
                'holiday_effect': None,
                'promotion_effect': None,
                'demand_multiplier': 1.0,
                'lead_time_multiplier': 1.0
            },
            'ALTA_DEMANDA': {
                'description': 'Incremento de demanda +50%',
                'holiday_effect': 0.0,
                'promotion_effect': 0.0,
                'demand_multiplier': 1.5,
                'lead_time_multiplier': 1.0
            },
            'CRISIS_PROVEEDOR': {
                'description': 'Lead Time duplicado',
                'holiday_effect': 0.0,
                'promotion_effect': 0.0,
                'demand_multiplier': 1.0,
                'lead_time_multiplier': 2.0
            }
        }
        
        # Semilla para reproducibilidad
        np.random.seed(42)
        
    def get_config_summary(self):
        """Retornar resumen de configuración"""
        return {
            'Niveles de servicio': list(self.service_levels.keys()),
            'Lead times': f'{self.lead_times} días',
            'Escenarios': list(self.scenarios.keys()),
            'Umbral Clase A': f'{self.abc_thresholds["A"] * 100:.0f}%',
            'Umbral Clase B': f'{self.abc_thresholds["B"] * 100:.0f}%'
        }

params = SimulationParameters()

logger.info('CONFIGURACIÓN DEL SISTEMA:')
for key, value in params.get_config_summary().items():
    logger.info(f'  • {key}: {value}')
logger.info('')

# =============================================================================
# CARGA DE DATOS
# =============================================================================

class DataLoader:
    """Clase para cargar y preprocesar datos"""
    
    def __init__(self, paths):
        self.paths = paths
        
    def load_items(self):
        """Cargar datos de items"""
        logger.info('Cargando items.csv...')
        items_df = pd.read_csv(self.paths['items'])
        logger.info(f'✅ Items cargados: {len(items_df):,} productos')
        logger.info(f'📊 Familias únicas: {items_df["family"].nunique()}')
        return items_df
    
    def load_holidays(self):
        """Cargar datos de festivos"""
        logger.info('Cargando holidays_events.csv...')
        holidays_df = pd.read_csv(self.paths['holidays'])
        holidays_df['date'] = pd.to_datetime(holidays_df['date'], cache=False)
        logger.info(f'✅ Festivos cargados: {len(holidays_df):,} eventos')
        return holidays_df
    
    def load_stores(self):
        """Cargar datos de tiendas"""
        logger.info('Cargando stores.csv...')
        stores_df = pd.read_csv(self.paths['stores'])
        logger.info(f'✅ Tiendas cargadas: {len(stores_df):,} tiendas')
        return stores_df
    
    def load_transactions(self):
        """Cargar datos de transacciones"""
        logger.info('Cargando transactions.csv...')
        transactions_df = pd.read_csv(self.paths['transactions'])
        transactions_df['date'] = pd.to_datetime(transactions_df['date'], cache=False)
        logger.info(f'✅ Transacciones cargadas: {len(transactions_df):,} registros')
        return transactions_df
    
    def load_train_chunked(self, usecols=None):
        """Cargar train.csv en chunks optimizados para datasets masivos"""
        if usecols is None:
            usecols = ['date', 'store_nbr', 'item_nbr', 'unit_sales', 'onpromotion']
        
        logger.info(f'Procesando train.csv en chunks optimizados...')
        chunks = []
        chunk_size = 100_000  # Reducido para evitar problemas de memoria
        total_rows = 0
        
        for chunk in pd.read_csv(
            self.paths['train'],
            chunksize=chunk_size,
            usecols=usecols,
            engine='python',
            on_bad_lines='skip',
            dtype={
                'store_nbr': 'uint16',
                'item_nbr': 'uint32',
                'unit_sales': 'float32',
                'onpromotion': 'object'
            }
        ):
            # Procesamiento optimizado de fechas sin cache
            chunk['date'] = pd.to_datetime(chunk['date'], cache=False)
            chunk['item_nbr'] = pd.to_numeric(chunk['item_nbr'], errors='coerce').astype('uint32')
            chunk['unit_sales'] = pd.to_numeric(chunk['unit_sales'], errors='coerce').fillna(0).astype('float32')
            chunk['onpromotion'] = chunk['onpromotion'].fillna(False).astype('bool')
            
            chunks.append(chunk)
            total_rows += len(chunk)
            
            if total_rows % 500_000 == 0:
                logger.info(f'  Procesados {total_rows:,} registros...')
            
            # Liberar memoria periódicamente si hay muchos chunks
            if len(chunks) % 10 == 0:
                temp_df = pd.concat(chunks, ignore_index=True)
                chunks = [temp_df]
                gc.collect()
        
        if len(chunks) == 1:
            train_df = chunks[0]
        else:
            train_df = pd.concat(chunks, ignore_index=True)
        
        logger.info(f'✅ Train cargado: {total_rows:,} registros')
        return train_df

# =============================================================================
# ANÁLISIS HISTÓRICO (OPTIMIZADO PARA GRANDES DATOS)
# =============================================================================

class HistoricalAnalyzer:
    """Clase para realizar análisis histórico de patrones de demanda (optimizado para grandes datasets)"""
    
    def __init__(self, paths, items_df, holidays_df, stores_df, transactions_df):
        self.paths = paths
        self.items_df = items_df
        self.holidays_df = holidays_df
        self.stores_df = stores_df
        self.transactions_df = transactions_df
        
        # Crear diccionario de festivos para búsqueda rápida
        self.holiday_dict = {}
        for _, row in holidays_df.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])[:10]
            self.holiday_dict[date_str] = {
                'type': row['type'],
                'locale_name': row['locale_name']
            }
        
    def analyze_holiday_effect_chunked(self):
        """Calcular incremento medio de ventas en festivos (procesamiento en chunks)"""
        logger.info('Analizando efecto de festivos (procesamiento en chunks)...')
        
        # Acumuladores
        normal_sales_sum = 0.0
        normal_count = 0
        holiday_sales_sum = 0.0
        holiday_count = 0
        
        holiday_type_sums = defaultdict(float)
        holiday_type_counts = defaultdict(int)
        
        chunk_size = 100_000  # Reducido para evitar problemas de memoria
        total_rows = 0
        
        for chunk in pd.read_csv(
            self.paths['train'],
            chunksize=chunk_size,
            usecols=['date', 'unit_sales'],
            engine='python',
            on_bad_lines='skip',
            dtype={
                'unit_sales': 'float32'
            }
        ):
            # Procesar fechas sin cache
            chunk['date'] = pd.to_datetime(chunk['date'], cache=False)
            chunk['unit_sales'] = pd.to_numeric(chunk['unit_sales'], errors='coerce').fillna(0)
            
            for _, row in chunk.iterrows():
                date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])[:10]
                sales = row['unit_sales']
                
                if date_str in self.holiday_dict:
                    holiday_sales_sum += sales
                    holiday_count += 1
                    
                    htype = self.holiday_dict[date_str]['type']
                    if htype in ['Holiday', 'Event', 'Additional']:
                        holiday_type_sums[htype] += sales
                        holiday_type_counts[htype] += 1
                else:
                    normal_sales_sum += sales
                    normal_count += 1
            
            total_rows += len(chunk)
            if total_rows % 500_000 == 0:
                logger.info(f'  Procesados {total_rows:,} registros...')
        
        # Calcular efectos
        ventas_normales = normal_sales_sum / normal_count if normal_count > 0 else 0
        ventas_festivo = holiday_sales_sum / holiday_count if holiday_count > 0 else 0
        
        holiday_effect = (ventas_festivo - ventas_normales) / ventas_normales if ventas_normales > 0 else 0
        
        # Por tipo de festivo
        holiday_effects_by_type = {}
        for htype in ['Holiday', 'Event', 'Additional']:
            if holiday_type_counts[htype] > 0:
                avg_type = holiday_type_sums[htype] / holiday_type_counts[htype]
                effect = (avg_type - ventas_normales) / ventas_normales if ventas_normales > 0 else 0
                holiday_effects_by_type[htype] = effect
        
        logger.info(f'📊 Efecto festivo general: {holiday_effect*100:.2f}%')
        logger.info(f'📊 Efecto por tipo: {holiday_effects_by_type}')
        
        return {
            'general': holiday_effect,
            'by_type': holiday_effects_by_type,
            'by_locale': {}
        }
    
    def analyze_promotion_effect_chunked(self):
        """Calcular incremento medio de ventas con promociones (procesamiento en chunks)"""
        logger.info('Analizando efecto de promociones (procesamiento en chunks)...')
        
        # Acumuladores globales
        no_promo_sales_sum = 0.0
        no_promo_count = 0
        promo_sales_sum = 0.0
        promo_count = 0
        
        # Por familia
        family_no_promo = defaultdict(lambda: {'sum': 0.0, 'count': 0})
        family_promo = defaultdict(lambda: {'sum': 0.0, 'count': 0})
        
        chunk_size = 100_000  # Reducido para evitar problemas de memoria
        total_rows = 0
        
        for chunk in pd.read_csv(
            self.paths['train'],
            chunksize=chunk_size,
            usecols=['item_nbr', 'unit_sales', 'onpromotion'],
            engine='python',
            on_bad_lines='skip',
            dtype={
                'item_nbr': 'uint32',
                'unit_sales': 'float32',
                'onpromotion': 'object'
            }
        ):
            chunk['item_nbr'] = pd.to_numeric(chunk['item_nbr'], errors='coerce').astype('uint32')
            chunk['unit_sales'] = pd.to_numeric(chunk['unit_sales'], errors='coerce').fillna(0).astype('float32')
            chunk['onpromotion'] = chunk['onpromotion'].fillna(False).astype('bool')
            
            # Merge con items para obtener familia
            chunk_merged = chunk.merge(self.items_df[['item_nbr', 'family']], on='item_nbr', how='left')
            
            for _, row in chunk_merged.iterrows():
                sales = row['unit_sales']
                family = row['family']
                is_promo = row['onpromotion']
                
                if is_promo:
                    promo_sales_sum += sales
                    promo_count += 1
                    family_promo[family]['sum'] += sales
                    family_promo[family]['count'] += 1
                else:
                    no_promo_sales_sum += sales
                    no_promo_count += 1
                    family_no_promo[family]['sum'] += sales
                    family_no_promo[family]['count'] += 1
            
            total_rows += len(chunk)
            if total_rows % 500_000 == 0:
                logger.info(f'  Procesados {total_rows:,} registros...')
        
        # Calcular efectos
        ventas_sin_promo = no_promo_sales_sum / no_promo_count if no_promo_count > 0 else 0
        ventas_con_promo = promo_sales_sum / promo_count if promo_count > 0 else 0
        
        promotion_effect = (ventas_con_promo - ventas_sin_promo) / ventas_sin_promo if ventas_sin_promo > 0 else 0
        
        # Por familia
        promotion_by_family = {}
        for family in family_no_promo:
            if family_no_promo[family]['count'] > 0 and family_promo[family]['count'] > 0:
                avg_no_promo = family_no_promo[family]['sum'] / family_no_promo[family]['count']
                avg_promo = family_promo[family]['sum'] / family_promo[family]['count']
                if avg_no_promo > 0:
                    effect = (avg_promo - avg_no_promo) / avg_no_promo
                    promotion_by_family[family] = effect
        
        logger.info(f'📊 Efecto promoción general: {promotion_effect*100:.2f}%')
        
        return {
            'general': promotion_effect,
            'by_family': promotion_by_family
        }
    
    def analyze_transaction_effect(self):
        """Calcular correlación entre transacciones y ventas"""
        logger.info('Analizando efecto de transacciones...')
        
        # Muestrear para cálculo de correlación (más eficiente)
        sample_dates = self.transactions_df['date'].unique()[:100]
        
        # Calcular ventas por tienda y fecha (solo para muestra)
        chunk_size = 100_000  # Reducido
        ventas_agregadas = defaultdict(lambda: {'sales': 0.0})
        
        for chunk in pd.read_csv(
            self.paths['train'],
            chunksize=chunk_size,
            usecols=['date', 'store_nbr', 'unit_sales'],
            engine='python',
            on_bad_lines='skip',
            dtype={
                'store_nbr': 'uint16',
                'unit_sales': 'float32'
            }
        ):
            chunk['date'] = pd.to_datetime(chunk['date'], cache=False)
            chunk['unit_sales'] = pd.to_numeric(chunk['unit_sales'], errors='coerce').fillna(0)
            
            # Filtrar solo fechas de muestra
            chunk = chunk[chunk['date'].isin(sample_dates)]
            
            for (date, store), group in chunk.groupby(['date', 'store_nbr']):
                key = (str(date)[:10], store)
                ventas_agregadas[key]['sales'] += group['unit_sales'].sum()
        
        # Crear DataFrame para correlación
        corr_data = []
        for key, val in ventas_agregadas.items():
            date_str, store = key
            trans_row = self.transactions_df[(self.transactions_df['date'].astype(str).str[:10] == date_str) & 
                                             (self.transactions_df['store_nbr'] == store)]
            if len(trans_row) > 0:
                corr_data.append({
                    'unit_sales': val['sales'],
                    'transactions': trans_row['transactions'].values[0]
                })
        
        if len(corr_data) > 10:
            corr_df = pd.DataFrame(corr_data)
            correlation = corr_df['unit_sales'].corr(corr_df['transactions'])
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                corr_df['transactions'], corr_df['unit_sales']
            )
        else:
            correlation = 0
            slope = 0
            intercept = 0
        
        logger.info(f'📊 Correlación transacciones-ventas: {correlation:.3f}')
        
        return {
            'correlation': correlation,
            'slope': slope,
            'intercept': intercept
        }
    
    def analyze_demand_variability_chunked(self):
        """Calcular variabilidad histórica de demanda (procesamiento en chunks)"""
        logger.info('Analizando variabilidad de demanda (procesamiento en chunks)...')
        
        # Acumular ventas diarias por producto
        daily_sales = defaultdict(list)
        
        chunk_size = 100_000  # Reducido
        total_rows = 0
        
        for chunk in pd.read_csv(
            self.paths['train'],
            chunksize=chunk_size,
            usecols=['date', 'item_nbr', 'unit_sales'],
            engine='python',
            on_bad_lines='skip',
            dtype={
                'item_nbr': 'uint32',
                'unit_sales': 'float32'
            }
        ):
            chunk['date'] = pd.to_datetime(chunk['date'], cache=False)
            chunk['item_nbr'] = pd.to_numeric(chunk['item_nbr'], errors='coerce').astype('uint32')
            chunk['unit_sales'] = pd.to_numeric(chunk['unit_sales'], errors='coerce').fillna(0).astype('float32')
            
            # Agrupar por producto y fecha
            for (item_nbr, date), group in chunk.groupby(['item_nbr', 'date']):
                date_str = str(date)[:10]
                daily_sales[(item_nbr, date_str)].append(group['unit_sales'].sum())
            
            total_rows += len(chunk)
            if total_rows % 10_000_000 == 0:
                logger.info(f'  Procesados {total_rows:,} registros...')
        
        # Calcular estadísticas por producto
        variability_data = []
        for (item_nbr, date_str), sales_list in daily_sales.items():
            variability_data.append({
                'item_nbr': item_nbr,
                'date': date_str,
                'unit_sales': sum(sales_list)
            })
        
        variability_df = pd.DataFrame(variability_data)
        
        # Calcular estadísticas por producto
        variability_stats = variability_df.groupby('item_nbr').agg({
            'unit_sales': ['mean', 'std', 'min', 'max', 'count']
        }).round(4)
        variability_stats.columns = ['mean_demand', 'std_demand', 'min_demand', 'max_demand', 'days_with_sales']
        variability_stats = variability_stats.reset_index()
        
        # Coeficiente de variación
        variability_stats['cv'] = np.where(
            variability_stats['mean_demand'] > 0,
            variability_stats['std_demand'] / variability_stats['mean_demand'],
            0
        )
        
        logger.info(f'📊 CV promedio: {variability_stats["cv"].mean():.3f}')
        
        return variability_stats
    
    def classify_abc_chunked(self, items_df):
        """Clasificación ABC de productos (procesamiento en chunks)"""
        logger.info('Realizando clasificación ABC (procesamiento en chunks)...')
        
        # Calcular ventas totales por producto en chunks
        total_sales = defaultdict(float)
        
        chunk_size = 100_000  # Reducido
        total_rows = 0
        
        for chunk in pd.read_csv(
            self.paths['train'],
            chunksize=chunk_size,
            usecols=['item_nbr', 'unit_sales'],
            engine='python',
            on_bad_lines='skip',
            dtype={
                'item_nbr': 'uint32',
                'unit_sales': 'float32'
            }
        ):
            chunk['item_nbr'] = pd.to_numeric(chunk['item_nbr'], errors='coerce').astype('uint32')
            chunk['unit_sales'] = pd.to_numeric(chunk['unit_sales'], errors='coerce').fillna(0).astype('float32')
            
            for item_nbr, sales in chunk.groupby('item_nbr')['unit_sales'].sum().items():
                total_sales[item_nbr] += sales
            
            total_rows += len(chunk)
            if total_rows % 10_000_000 == 0:
                logger.info(f'  Procesados {total_rows:,} registros...')
        
        # Crear DataFrame
        abc_df = pd.DataFrame([
            {'item_nbr': item_nbr, 'total_sales': sales}
            for item_nbr, sales in total_sales.items()
        ])
        
        # Merge con items
        abc_df = abc_df.merge(items_df[['item_nbr', 'family', 'class', 'perishable']], on='item_nbr', how='left')
        
        # Ordenar y calcular porcentaje acumulado
        abc_df = abc_df.sort_values('total_sales', ascending=False).reset_index(drop=True)
        abc_df['pct_sales'] = abc_df['total_sales'] / abc_df['total_sales'].sum() * 100
        abc_df['cumulative_pct'] = abc_df['pct_sales'].cumsum()
        
        # Clasificar ABC
        def classify(row):
            if row['cumulative_pct'] <= params.abc_thresholds['A'] * 100:
                return 'A'
            elif row['cumulative_pct'] <= params.abc_thresholds['B'] * 100:
                return 'B'
            else:
                return 'C'
        
        abc_df['abc_class'] = abc_df.apply(classify, axis=1)
        
        # Resumen
        for cls in ['A', 'B', 'C']:
            count = len(abc_df[abc_df['abc_class'] == cls])
            pct_products = count / len(abc_df) * 100
            pct_sales = abc_df[abc_df['abc_class'] == cls]['total_sales'].sum() / abc_df['total_sales'].sum() * 100
            logger.info(f'  Clase {cls}: {count} productos ({pct_products:.1f}%) - {pct_sales:.1f}% de ventas')
        
        return abc_df

# =============================================================================
# SIMULADOR DE ESCENARIOS
# =============================================================================

class ScenarioSimulator:
    """Clase principal para simulación de escenarios"""
    
    def __init__(self, abc_df, variability_stats, historical_analysis, items_df, stores_df):
        self.abc_df = abc_df
        self.variability_stats = variability_stats
        self.historical_analysis = historical_analysis
        self.items_df = items_df
        self.stores_df = stores_df
        
        # Seleccionar productos Clase A para simulación detallada
        self.products_a = abc_df[abc_df['abc_class'] == 'A']['item_nbr'].tolist()
        logger.info(f'🎯 Productos Clase A para simulación: {len(self.products_a)}')
        
    def simulate_scenario(self, scenario_name, lead_time, service_level):
        """Simular un escenario específico para un lead time y nivel de servicio"""
        
        scenario_config = params.scenarios[scenario_name]
        z_score = params.service_levels[service_level]
        
        results = []
        
        for item_nbr in self.products_a:
            # Obtener datos del producto
            item_variability = self.variability_stats[self.variability_stats['item_nbr'] == item_nbr]
            if item_variability.empty:
                continue
                
            item_var = item_variability.iloc[0]
            mean_demand = item_var['mean_demand']
            std_demand = item_var['std_demand']
            cv = item_var['cv']
            
            # Obtener familia
            item_info = self.items_df[self.items_df['item_nbr'] == item_nbr]
            family = item_info['family'].values[0] if len(item_info) > 0 else 'Unknown'
            
            # Calcular demanda base
            base_demand = mean_demand
            
            # Aplicar efectos del escenario
            holiday_effect = scenario_config['holiday_effect']
            promotion_effect = scenario_config['promotion_effect']
            demand_multiplier = scenario_config['demand_multiplier']
            lead_time_multiplier = scenario_config['lead_time_multiplier']
            
            # Si el efecto es None, usar el histórico calculado
            if holiday_effect is None:
                holiday_effect = self.historical_analysis['holiday']['general']
            if promotion_effect is None:
                promotion_effect = self.historical_analysis['promotion']['general']
            
            # Calcular demanda ajustada
            adjusted_demand = base_demand * (1 + holiday_effect) * (1 + promotion_effect) * demand_multiplier
            
            # Lead time ajustado
            adjusted_lead_time = lead_time * lead_time_multiplier
            
            # Stock de seguridad: SS = Z × std_demanda × √(lead_time)
            stock_seguridad = z_score * std_demand * np.sqrt(adjusted_lead_time)
            
            # Punto de pedido: ROP = demanda × lead_time + stock_seguridad
            punto_pedido = adjusted_demand * adjusted_lead_time + stock_seguridad
            
            # Simular stock actual (distribución alrededor del ROP)
            random_factor = np.random.random()
            if random_factor < 0.15:
                stock_actual = max(0, punto_pedido * np.random.uniform(0.3, 0.9))
            elif random_factor < 0.30:
                stock_actual = punto_pedido * np.random.uniform(0.9, 1.1)
            else:
                stock_actual = punto_pedido * np.random.uniform(0.8, 1.5)
            
            # Riesgo de rotura
            riesgo_rotura = 1 if stock_actual < punto_pedido else 0
            stock_faltante = max(0, punto_pedido - stock_actual)
            
            # Fill rate (tasa de cumplimiento)
            if adjusted_demand > 0:
                fill_rate = min(1.0, stock_actual / (adjusted_demand * adjusted_lead_time))
            else:
                fill_rate = 1.0
            
            # Unidades faltantes
            unidades_faltantes = stock_faltante
            
            # Días de cobertura
            dias_cobertura = stock_actual / adjusted_demand if adjusted_demand > 0 else 0
            
            # Compra recomendada
            compra_recomendada = max(0, punto_pedido + stock_seguridad - stock_actual)
            
            # Probabilidad de rotura (usando distribución normal)
            if std_demand > 0:
                z_calc = (stock_actual - adjusted_demand * adjusted_lead_time) / (std_demand * np.sqrt(adjusted_lead_time))
                prob_rotura = 1 - stats.norm.cdf(z_calc)
            else:
                prob_rotura = 0 if stock_actual >= adjusted_demand * adjusted_lead_time else 1
            
            results.append({
                'item_nbr': item_nbr,
                'family': family,
                'escenario': scenario_name,
                'descripcion_escenario': scenario_config['description'],
                'lead_time': adjusted_lead_time,
                'lead_time_original': lead_time,
                'nivel_servicio': service_level,
                'z_score': z_score,
                'demanda_base': round(base_demand, 4),
                'demanda_ajustada': round(adjusted_demand, 4),
                'impacto_festivo_pct': round(holiday_effect * 100, 2),
                'impacto_promocion_pct': round(promotion_effect * 100, 2),
                'demand_multiplier': demand_multiplier,
                'lead_time_multiplier': lead_time_multiplier,
                'stock_seguridad': round(stock_seguridad, 2),
                'punto_pedido': round(punto_pedido, 2),
                'stock_actual': round(stock_actual, 2),
                'compra_recomendada': round(compra_recomendada, 2),
                'riesgo_rotura': riesgo_rotura,
                'probabilidad_rotura': round(prob_rotura, 4),
                'fill_rate': round(fill_rate, 4),
                'unidades_faltantes': round(unidades_faltantes, 2),
                'dias_cobertura': round(dias_cobertura, 2),
                'cv': round(cv, 4),
                'std_demanda': round(std_demand, 4)
            })
        
        return results
    
    def run_all_scenarios(self):
        """Ejecutar todas las simulaciones de escenarios"""
        logger.info('')
        logger.info('=' * 80)
        logger.info('EJECUTANDO SIMULACIONES DE ESCENARIOS')
        logger.info('=' * 80)
        
        all_results = []
        total_combinations = len(params.scenarios) * len(params.lead_times) * len(params.service_levels)
        current = 0
        
        for scenario_name in params.scenarios:
            for lead_time in params.lead_times:
                for service_level in params.service_levels:
                    current += 1
                    logger.info(f'  [{current}/{total_combinations}] Simulando: {scenario_name} | LT={lead_time}d | SL={service_level*100:.0f}%')
                    
                    results = self.simulate_scenario(scenario_name, lead_time, service_level)
                    all_results.extend(results)
        
        results_df = pd.DataFrame(all_results)
        logger.info(f'✅ Simulación completada: {len(results_df):,} combinaciones')
        
        return results_df

# =============================================================================
# GENERADOR DE REPORTES
# =============================================================================

class ReportGenerator:
    """Clase para generar reportes y resúmenes"""
    
    def __init__(self, results_df, paths):
        self.results_df = results_df
        self.paths = paths
    
    def export_simulation_results(self):
        """Exportar resultados completos de simulación"""
        logger.info('')
        logger.info('=' * 80)
        logger.info('EXPORTANDO RESULTADOS DE SIMULACIÓN')
        logger.info('=' * 80)
        
        output_path = self.paths['data_output'] / 'simulacion_final_escenarios.csv'
        self.results_df.to_csv(output_path, index=False)
        logger.info(f'✅ simulacion_final_escenarios.csv exportado: {len(self.results_df):,} registros')
        logger.info(f'   📁 {output_path}')
        
        return output_path
    
    def generate_scenario_summary(self):
        """Generar resumen por escenario"""
        logger.info('Generando resumen por escenario...')
        
        resumen_data = []
        
        for escenario in self.results_df['escenario'].unique():
            scenario_data = self.results_df[self.results_df['escenario'] == escenario]
            
            resumen_data.append({
                'escenario': escenario,
                'descripcion': scenario_data['descripcion_escenario'].iloc[0],
                'total_productos': len(scenario_data),
                'productos_en_riesgo': int(scenario_data['riesgo_rotura'].sum()),
                'pct_productos_riesgo': round(scenario_data['riesgo_rotura'].mean() * 100, 2),
                'riesgo_promedio': round(scenario_data['probabilidad_rotura'].mean() * 100, 2),
                'compra_total_recomendada': round(scenario_data['compra_recomendada'].sum(), 0),
                'stock_seguridad_total': round(scenario_data['stock_seguridad'].sum(), 0),
                'unidades_faltantes_total': round(scenario_data['unidades_faltantes'].sum(), 0),
                'fill_rate_promedio': round(scenario_data['fill_rate'].mean() * 100, 2),
                'dias_cobertura_promedio': round(scenario_data['dias_cobertura'].mean(), 2),
                'demanda_base_promedio': round(scenario_data['demanda_base'].mean(), 4),
                'demanda_ajustada_promedio': round(scenario_data['demanda_ajustada'].mean(), 4),
                'impacto_demanda_pct': round(
                    (scenario_data['demanda_ajustada'].mean() / scenario_data['demanda_base'].mean() - 1) * 100 
                    if scenario_data['demanda_base'].mean() > 0 else 0, 2
                )
            })
        
        resumen_df = pd.DataFrame(resumen_data)
        
        output_path = self.paths['reports'] / 'resumen_escenarios.csv'
        resumen_df.to_csv(output_path, index=False)
        logger.info(f'✅ resumen_escenarios.csv exportado')
        logger.info(f'   📁 {output_path}')
        
        return resumen_df
    
    def generate_family_summary(self):
        """Generar resumen por familia"""
        logger.info('Generando resumen por familia...')
        
        family_summary = self.results_df.groupby('family').agg({
            'total_productos': ('item_nbr', 'nunique'),
            'productos_en_riesgo': ('riesgo_rotura', 'sum'),
            'compra_total_recomendada': ('compra_recomendada', 'sum'),
            'stock_seguridad_total': ('stock_seguridad', 'sum'),
            'unidades_faltantes_total': ('unidades_faltantes', 'sum'),
            'fill_rate_promedio': ('fill_rate', 'mean'),
            'demanda_base_promedio': ('demanda_base', 'mean'),
            'demanda_ajustada_promedio': ('demanda_ajustada', 'mean')
        }).round(2)
        
        family_summary = family_summary.reset_index()
        family_summary.columns = ['family', 'total_productos', 'productos_en_riesgo', 
                                   'compra_total_recomendada', 'stock_seguridad_total',
                                   'unidades_faltantes_total', 'fill_rate_promedio',
                                   'demanda_base_promedio', 'demanda_ajustada_promedio']
        
        output_path = self.paths['reports'] / 'resumen_por_familia.csv'
        family_summary.to_csv(output_path, index=False)
        logger.info(f'✅ resumen_por_familia.csv exportado')
        
        return family_summary
    
    def generate_leadtime_summary(self):
        """Generar resumen por lead time"""
        logger.info('Generando resumen por lead time...')
        
        leadtime_summary = self.results_df.groupby('lead_time').agg({
            'total_productos': ('item_nbr', 'nunique'),
            'productos_en_riesgo': ('riesgo_rotura', 'sum'),
            'compra_total_recomendada': ('compra_recomendada', 'sum'),
            'stock_seguridad_promedio': ('stock_seguridad', 'mean'),
            'punto_pedido_promedio': ('punto_pedido', 'mean'),
            'fill_rate_promedio': ('fill_rate', 'mean'),
            'dias_cobertura_promedio': ('dias_cobertura', 'mean')
        }).round(2)
        
        leadtime_summary = leadtime_summary.reset_index()
        leadtime_summary.columns = ['lead_time', 'total_productos', 'productos_en_riesgo',
                                     'compra_total_recomendada', 'stock_seguridad_promedio',
                                     'punto_pedido_promedio', 'fill_rate_promedio', 
                                     'dias_cobertura_promedio']
        
        output_path = self.paths['reports'] / 'resumen_por_leadtime.csv'
        leadtime_summary.to_csv(output_path, index=False)
        logger.info(f'✅ resumen_por_leadtime.csv exportado')
        
        return leadtime_summary

# =============================================================================
# GENERADOR DE VISUALIZACIONES
# =============================================================================

class VisualizationGenerator:
    """Clase para generar visualizaciones"""
    
    def __init__(self, results_df, resumen_df, paths):
        self.results_df = results_df
        self.resumen_df = resumen_df
        self.paths = paths
        
    def generate_riesgo_por_escenario(self):
        """01_riesgo_por_escenario.png"""
        logger.info('Generando visualización: 01_riesgo_por_escenario.png...')
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análisis de Riesgo por Escenario', fontsize=16, fontweight='bold')
        
        # Gráfico 1: Productos en riesgo por escenario
        ax = axes[0, 0]
        riesgo_data = self.resumen_df.sort_values('pct_productos_riesgo', ascending=True)
        colores = [COLORS['danger'] if x > 30 else COLORS['warning'] if x > 15 else COLORS['success'] 
                   for x in riesgo_data['pct_productos_riesgo']]
        barras = ax.barh(riesgo_data['escenario'], riesgo_data['pct_productos_riesgo'], color=colores)
        ax.set_xlabel('Productos con Riesgo (%)', fontsize=12)
        ax.set_ylabel('Escenario', fontsize=12)
        ax.set_title('Porcentaje de Productos con Riesgo por Escenario', fontsize=12, fontweight='bold')
        for barra, valor in zip(barras, riesgo_data['pct_productos_riesgo']):
            ax.text(barra.get_width() + 0.5, barra.get_y() + barra.get_height()/2,
                    f'{valor:.1f}%', va='center', fontsize=9)
        
        # Gráfico 2: Probabilidad de rotura promedio
        ax = axes[0, 1]
        prob_data = self.resumen_df.sort_values('riesgo_promedio', ascending=True)
        barras = ax.barh(prob_data['escenario'], prob_data['riesgo_promedio'], color=COLORS['primary'])
        ax.set_xlabel('Probabilidad de Rotura Promedio (%)', fontsize=12)
        ax.set_ylabel('Escenario', fontsize=12)
        ax.set_title('Probabilidad de Rotura Promedio por Escenario', fontsize=12, fontweight='bold')
        
        # Gráfico 3: Unidades faltantes totales
        ax = axes[1, 0]
        faltantes_data = self.resumen_df.sort_values('unidades_faltantes_total', ascending=True)
        barras = ax.barh(faltantes_data['escenario'], faltantes_data['unidades_faltantes_total'] / 1000, 
                         color=COLORS['tertiary'])
        ax.set_xlabel('Unidades Faltantes (miles)', fontsize=12)
        ax.set_ylabel('Escenario', fontsize=12)
        ax.set_title('Unidades Faltantes Totales por Escenario', fontsize=12, fontweight='bold')
        
        # Gráfico 4: Fill rate promedio
        ax = axes[1, 1]
        fill_data = self.resumen_df.sort_values('fill_rate_promedio', ascending=False)
        barras = ax.barh(fill_data['escenario'], fill_data['fill_rate_promedio'], color=COLORS['dark_green'])
        ax.set_xlabel('Fill Rate Promedio (%)', fontsize=12)
        ax.set_ylabel('Escenario', fontsize=12)
        ax.set_title('Fill Rate Promedio por Escenario', fontsize=12, fontweight='bold')
        ax.axvline(x=95, color=COLORS['danger'], linestyle='--', label='Objetivo 95%')
        ax.legend()
        
        plt.tight_layout()
        output_path = self.paths['visuals'] / '01_riesgo_por_escenario.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f'  ✅ {output_path}')
        
    def generate_compra_por_escenario(self):
        """02_compra_recomendada_por_escenario.png"""
        logger.info('Generando visualización: 02_compra_recomendada_por_escenario.png...')
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análisis de Compras Recomendadas por Escenario', fontsize=16, fontweight='bold')
        
        # Gráfico 1: Compra total recomendada por escenario
        ax = axes[0, 0]
        compra_data = self.resumen_df.sort_values('compra_total_recomendada', ascending=True)
        barras = ax.barh(compra_data['escenario'], compra_data['compra_total_recomendada'] / 1000, 
                         color=COLORS['primary'])
        ax.set_xlabel('Compra Total Recomendada (miles)', fontsize=12)
        ax.set_ylabel('Escenario', fontsize=12)
        ax.set_title('Compra Total Recomendada por Escenario', fontsize=12, fontweight='bold')
        
        # Gráfico 2: Compra recomendada por familia (top 10)
        ax = axes[0, 1]
        family_compras = self.results_df.groupby('family')['compra_recomendada'].sum().sort_values(ascending=False).head(10)
        barras = ax.barh(family_compras.index, family_compras.values / 1000, color=COLORS['secondary'])
        ax.set_xlabel('Compra Recomendada (miles)', fontsize=12)
        ax.set_ylabel('Familia', fontsize=12)
        ax.set_title('Top 10 Familias - Compra Recomendada', fontsize=12, fontweight='bold')
        
        # Gráfico 3: Distribución de compras recomendadas
        ax = axes[1, 0]
        compras_positivas = self.results_df[self.results_df['compra_recomendada'] > 0]['compra_recomendada']
        ax.hist(compras_positivas / 100, bins=30, color=COLORS['tertiary'], edgecolor='white', alpha=0.7)
        ax.set_xlabel('Compra Recomendada (cientos)', fontsize=12)
        ax.set_ylabel('Frecuencia', fontsize=12)
        ax.set_title('Distribución de Compras Recomendadas', fontsize=12, fontweight='bold')
        if len(compras_positivas) > 0:
            ax.axvline(x=compras_positivas.mean() / 100, color=COLORS['danger'], linestyle='--',
                       label=f'Media: {compras_positivas.mean():.0f} unidades')
            ax.legend()
        
        # Gráfico 4: Compra por lead time
        ax = axes[1, 1]
        lt_compras = self.results_df.groupby('lead_time')['compra_recomendada'].sum()
        barras = ax.bar(lt_compras.index, lt_compras.values / 1000, color=COLORS['info'])
        ax.set_xlabel('Lead Time (días)', fontsize=12)
        ax.set_ylabel('Compra Recomendada (miles)', fontsize=12)
        ax.set_title('Compra Recomendada por Lead Time', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        output_path = self.paths['visuals'] / '02_compra_recomendada_por_escenario.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f'  ✅ {output_path}')
        
    def generate_impacto_festivos(self):
        """03_impacto_festivos.png"""
        logger.info('Generando visualización: 03_impacto_festivos.png...')
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Impacto de Festivos en la Demanda', fontsize=16, fontweight='bold')
        
        # Filtrar escenarios con efecto festivo
        festivo_scenarios = ['FESTIVO', 'FESTIVO_MAS_PROMOCION']
        festivo_data = self.results_df[self.results_df['escenario'].isin(festivo_scenarios)]
        baseline_data = self.results_df[self.results_df['escenario'] == 'BASELINE']
        
        # Gráfico 1: Comparativa demanda base vs ajustada
        ax = axes[0, 0]
        for scenario in festivo_scenarios:
            scenario_data = festivo_data[festivo_data['escenario'] == scenario]
            avg_base = scenario_data['demanda_base'].mean()
            avg_ajustada = scenario_data['demanda_ajustada'].mean()
            ax.bar([f'Base\n{scenario}'], [avg_base], color=COLORS['info'], alpha=0.7)
            ax.bar([f'Ajustada\n{scenario}'], [avg_ajustada], color=COLORS['danger'], alpha=0.7)
        
        ax.set_ylabel('Demanda Promedio Diaria', fontsize=12)
        ax.set_title('Demanda Base vs Ajustada con Festivos', fontsize=12, fontweight='bold')
        
        # Gráfico 2: Impacto festivo por familia (top 10)
        ax = axes[0, 1]
        festivo_familias = festivo_data.groupby('family').agg({
            'demanda_base': 'mean',
            'demanda_ajustada': 'mean'
        }).reset_index()
        festivo_familias['impacto_pct'] = ((festivo_familias['demanda_ajustada'] / festivo_familias['demanda_base']) - 1) * 100
        festivo_familias = festivo_familias.sort_values('impacto_pct', ascending=False).head(10)
        
        colores_familia = [COLORS['danger'] if x > 10 else COLORS['warning'] if x > 5 else COLORS['info'] 
                          for x in festivo_familias['impacto_pct']]
        barras = ax.barh(festivo_familias['family'], festivo_familias['impacto_pct'], color=colores_familia)
        ax.set_xlabel('Impacto Festivo (%)', fontsize=12)
        ax.set_ylabel('Familia', fontsize=12)
        ax.set_title('Impacto Festivo por Familia (Top 10)', fontsize=12, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # Gráfico 3: Distribución del impacto festivo
        ax = axes[1, 0]
        festivo_data_copy = festivo_data.copy()
        festivo_data_copy['impacto_absoluto'] = festivo_data_copy['demanda_ajustada'] - festivo_data_copy['demanda_base']
        ax.hist(festivo_data_copy['impacto_absoluto'], bins=30, color=COLORS['purple'], edgecolor='white', alpha=0.7)
        ax.set_xlabel('Incremento de Demanda (unidades)', fontsize=12)
        ax.set_ylabel('Frecuencia', fontsize=12)
        ax.set_title('Distribución del Incremento de Demanda por Festivos', fontsize=12, fontweight='bold')
        ax.axvline(x=festivo_data_copy['impacto_absoluto'].mean(), color=COLORS['danger'], linestyle='--',
                   label=f'Media: {festivo_data_copy["impacto_absoluto"].mean():.1f}')
        ax.legend()
        
        # Gráfico 4: Riesgo de rotura en escenarios festivos
        ax = axes[1, 1]
        riesgo_festivo = festivo_data.groupby('escenario')['riesgo_rotura'].mean() * 100
        riesgo_baseline = baseline_data.groupby('escenario')['riesgo_rotura'].mean() * 100
        
        x = range(len(riesgo_festivo) + len(riesgo_baseline))
        barras1 = ax.bar([0, 2], [riesgo_baseline.values[0]] * 2, 0.35, label='Baseline', color=COLORS['info'], alpha=0.7)
        barras2 = ax.bar([1, 3], riesgo_festivo.values, 0.35, label='Festivo', color=COLORS['danger'], alpha=0.7)
        ax.set_xticks([0.5, 2.5])
        ax.set_xticklabels(list(riesgo_festivo.index))
        ax.set_ylabel('Productos con Riesgo (%)', fontsize=12)
        ax.set_title('Riesgo de Rotura: Baseline vs Festivo', fontsize=12, fontweight='bold')
        ax.legend()
        
        plt.tight_layout()
        output_path = self.paths['visuals'] / '03_impacto_festivos.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f'  ✅ {output_path}')
        
    def generate_impacto_promociones(self):
        """04_impacto_promociones.png"""
        logger.info('Generando visualización: 04_impacto_promociones.png...')
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Impacto de Promociones en la Demanda', fontsize=16, fontweight='bold')
        
        # Filtrar escenarios con promoción
        promo_scenarios = ['PROMOCION_SUAVE', 'PROMOCION_MEDIA', 'PROMOCION_AGRESIVA', 'FESTIVO_MAS_PROMOCION']
        promo_data = self.results_df[self.results_df['escenario'].isin(promo_scenarios)]
        baseline_data = self.results_df[self.results_df['escenario'] == 'BASELINE']
        
        # Gráfico 1: Demanda ajustada por nivel de promoción
        ax = axes[0, 0]
        promo_labels = ['Suave (10%)', 'Media (20%)', 'Agresiva (30%)', 'Festivo+Promo']
        promo_means = []
        for scenario in promo_scenarios:
            scenario_data = promo_data[promo_data['escenario'] == scenario]
            promo_means.append(scenario_data['demanda_ajustada'].mean())
        
        colores_promo = [COLORS['success'], COLORS['warning'], COLORS['danger'], COLORS['purple']]
        barras = ax.bar(promo_labels, promo_means, color=colores_promo, alpha=0.8)
        ax.set_ylabel('Demanda Ajustada Promedio', fontsize=12)
        ax.set_title('Demanda Ajustada por Nivel de Promoción', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=15)
        
        # Gráfico 2: Impacto promoción por familia (top 10)
        ax = axes[0, 1]
        promo_familias = promo_data.groupby('family').agg({
            'demanda_base': 'mean',
            'demanda_ajustada': 'mean'
        }).reset_index()
        promo_familias['impacto_pct'] = ((promo_familias['demanda_ajustada'] / promo_familias['demanda_base']) - 1) * 100
        promo_familias = promo_familias.sort_values('impacto_pct', ascending=False).head(10)
        
        barras = ax.barh(promo_familias['family'], promo_familias['impacto_pct'], color=COLORS['tertiary'])
        ax.set_xlabel('Impacto Promoción (%)', fontsize=12)
        ax.set_ylabel('Familia', fontsize=12)
        ax.set_title('Impacto Promoción por Familia (Top 10)', fontsize=12, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # Gráfico 3: Elasticidad promoción-demanda
        ax = axes[1, 0]
        promo_effects = []
        discount_levels = [10, 20, 30]
        for disc in discount_levels:
            scenario = f'PROMOCION_{"SUAVE" if disc == 10 else "MEDIA" if disc == 20 else "AGRESIVA"}'
            scenario_data = promo_data[promo_data['escenario'] == scenario]
            avg_demand = scenario_data['demanda_ajustada'].mean()
            baseline_avg = baseline_data['demanda_base'].mean()
            elasticity = ((avg_demand - baseline_avg) / baseline_avg) / (disc / 100)
            promo_effects.append(elasticity)
        
        ax.plot(discount_levels, promo_effects, 'o-', color=COLORS['primary'], linewidth=2, markersize=8)
        ax.set_xlabel('Descuento (%)', fontsize=12)
        ax.set_ylabel('Elasticidad Precio-Demanda', fontsize=12)
        ax.set_title('Elasticidad Promoción-Demanda', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Gráfico 4: Compra recomendada por escenario de promoción
        ax = axes[1, 1]
        promo_compras = promo_data.groupby('escenario')['compra_recomendada'].sum() / 1000
        barras = ax.bar(range(len(promo_compras)), promo_compras.values, color=COLORS['secondary'], alpha=0.8)
        ax.set_xticks(range(len(promo_compras)))
        ax.set_xticklabels(['Suave', 'Media', 'Agresiva', 'Festivo+Promo'], rotation=15)
        ax.set_ylabel('Compra Recomendada (miles)', fontsize=12)
        ax.set_title('Compra Recomendada por Escenario Promocional', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        output_path = self.paths['visuals'] / '04_impacto_promociones.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f'  ✅ {output_path}')
        
    def generate_dashboard_escenarios(self):
        """05_dashboard_escenarios.png"""
        logger.info('Generando visualización: 05_dashboard_escenarios.png...')
        
        fig = plt.figure(figsize=(20, 14))
        fig.suptitle('Dashboard de Simulación de Escenarios - FLOWMAP ANALYTICS', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        # Crear grid personalizado
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        # 1. Resumen ejecutivo - KPIs principales
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        
        # Calcular KPIs globales
        total_productos = len(self.results_df['item_nbr'].unique())
        avg_riesgo = self.resumen_df['riesgo_promedio'].mean()
        total_compras = self.resumen_df['compra_total_recomendada'].sum()
        avg_fill_rate = self.resumen_df['fill_rate_promedio'].mean()
        
        kpi_text = f"""
        ╔══════════════════════════════════════════════════════════════════════════════╗
        ║                           RESUMEN EJECUTIVO                                  ║
        ╠══════════════════════════════════════════════════════════════════════════════╣
        ║  📊 Productos Analizados: {total_productos:>8}          📈 Riesgo Promedio: {avg_riesgo:>6.1f}%           ║
        ║  💰 Compra Total Recomendada: {total_compras:>12,.0f}     ✅ Fill Rate Promedio: {avg_fill_rate:>6.1f}%     ║
        ║  📦 Escenarios Simulados: {len(self.resumen_df):>8}          🔗 Lead Times: {len(params.lead_times):>3} niveles        ║
        ╚══════════════════════════════════════════════════════════════════════════════╝
        """
        ax1.text(0.05, 0.5, kpi_text, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='center', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
        
        # 2. Riesgo por escenario
        ax2 = fig.add_subplot(gs[1, 0])
        riesgo_data = self.resumen_df.sort_values('pct_productos_riesgo', ascending=True)
        colores_riesgo = [COLORS['danger'] if x > 30 else COLORS['warning'] if x > 15 else COLORS['success'] 
                         for x in riesgo_data['pct_productos_riesgo']]
        ax2.barh(riesgo_data['escenario'], riesgo_data['pct_productos_riesgo'], color=colores_riesgo)
        ax2.set_xlabel('% Productos con Riesgo', fontsize=10)
        ax2.set_title('Riesgo por Escenario', fontsize=11, fontweight='bold')
        ax2.tick_params(labelsize=8)
        
        # 3. Compra recomendada por escenario
        ax3 = fig.add_subplot(gs[1, 1])
        compra_data = self.resumen_df.sort_values('compra_total_recomendada', ascending=True)
        ax3.barh(compra_data['escenario'], compra_data['compra_total_recomendada'] / 1000, color=COLORS['primary'])
        ax3.set_xlabel('Compra (miles)', fontsize=10)
        ax3.set_title('Compra Recomendada', fontsize=11, fontweight='bold')
        ax3.tick_params(labelsize=8)
        
        # 4. Fill rate por escenario
        ax4 = fig.add_subplot(gs[1, 2])
        fill_data = self.resumen_df.sort_values('fill_rate_promedio', ascending=False)
        ax4.barh(fill_data['escenario'], fill_data['fill_rate_promedio'], color=COLORS['dark_green'])
        ax4.set_xlabel('Fill Rate (%)', fontsize=10)
        ax4.set_title('Fill Rate Promedio', fontsize=11, fontweight='bold')
        ax4.axvline(x=95, color=COLORS['danger'], linestyle='--', linewidth=1)
        ax4.tick_params(labelsize=8)
        
        # 5. Impacto festivo por familia
        ax5 = fig.add_subplot(gs[2, 0])
        festivo_scenarios = ['FESTIVO', 'FESTIVO_MAS_PROMOCION']
        festivo_data = self.results_df[self.results_df['escenario'].isin(festivo_scenarios)]
        festivo_familias = festivo_data.groupby('family').agg({
            'demanda_base': 'mean',
            'demanda_ajustada': 'mean'
        }).reset_index()
        festivo_familias['impacto_pct'] = ((festivo_familias['demanda_ajustada'] / festivo_familias['demanda_base']) - 1) * 100
        festivo_familias = festivo_familias.sort_values('impacto_pct', ascending=False).head(8)
        
        colores_festivo = [COLORS['danger'] if x > 10 else COLORS['warning'] if x > 5 else COLORS['info'] 
                          for x in festivo_familias['impacto_pct']]
        ax5.barh(festivo_familias['family'], festivo_familias['impacto_pct'], color=colores_festivo)
        ax5.set_xlabel('Impacto Festivo (%)', fontsize=10)
        ax5.set_title('Top Familias - Impacto Festivo', fontsize=11, fontweight='bold')
        ax5.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax5.tick_params(labelsize=8)
        
        # 6. Impacto promoción por familia
        ax6 = fig.add_subplot(gs[2, 1])
        promo_scenarios = ['PROMOCION_SUAVE', 'PROMOCION_MEDIA', 'PROMOCION_AGRESIVA']
        promo_data = self.results_df[self.results_df['escenario'].isin(promo_scenarios)]
        promo_familias = promo_data.groupby('family').agg({
            'demanda_base': 'mean',
            'demanda_ajustada': 'mean'
        }).reset_index()
        promo_familias['impacto_pct'] = ((promo_familias['demanda_ajustada'] / promo_familias['demanda_base']) - 1) * 100
        promo_familias = promo_familias.sort_values('impacto_pct', ascending=False).head(8)
        
        ax6.barh(promo_familias['family'], promo_familias['impacto_pct'], color=COLORS['tertiary'])
        ax6.set_xlabel('Impacto Promoción (%)', fontsize=10)
        ax6.set_title('Top Familias - Impacto Promoción', fontsize=11, fontweight='bold')
        ax6.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax6.tick_params(labelsize=8)
        
        # 7. Matriz de riesgo - Lead Time vs Nivel de Servicio
        ax7 = fig.add_subplot(gs[2, 2])
        matriz_riesgo = self.results_df.groupby(['lead_time', 'nivel_servicio'])['riesgo_rotura'].mean().unstack()
        sns.heatmap(matriz_riesgo * 100, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                    ax=ax7, vmin=0, vmax=100, cbar_kws={'label': 'Riesgo (%)'})
        ax7.set_title('Matriz de Riesgo: LT × Nivel Servicio', fontsize=11, fontweight='bold')
        ax7.set_xlabel('Nivel de Servicio', fontsize=10)
        ax7.set_ylabel('Lead Time (días)', fontsize=10)
        
        output_path = self.paths['visuals'] / '05_dashboard_escenarios.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f'  ✅ {output_path}')
    
    def generate_all_visualizations(self):
        """Generar todas las visualizaciones"""
        logger.info('')
        logger.info('=' * 80)
        logger.info('GENERANDO VISUALIZACIONES')
        logger.info('=' * 80)
        
        self.generate_riesgo_por_escenario()
        self.generate_compra_por_escenario()
        self.generate_impacto_festivos()
        self.generate_impacto_promociones()
        self.generate_dashboard_escenarios()
        
        logger.info(f'\n📁 Todas las visualizaciones guardadas en: {self.paths["visuals"]}')

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """Función principal del script"""
    
    start_time = datetime.now()
    logger.info(f'🚀 Inicio del proceso: {start_time.strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info('')
    
    try:
        # 1. Cargar datos
        logger.info('=' * 80)
        logger.info('FASE 1: CARGA DE DATOS')
        logger.info('=' * 80)
        
        data_loader = DataLoader(paths)
        items_df = data_loader.load_items()
        holidays_df = data_loader.load_holidays()
        stores_df = data_loader.load_stores()
        transactions_df = data_loader.load_transactions()
        train_df = data_loader.load_train_chunked()
        
        logger.info('')
        
        # 2. Análisis histórico (usando métodos optimizados en chunks)
        logger.info('=' * 80)
        logger.info('FASE 2: ANÁLISIS HISTÓRICO')
        logger.info('=' * 80)
        
        # No necesitamos train_df completo, usamos procesamiento en chunks
        del train_df  # Liberar memoria
        
        historical_analyzer = HistoricalAnalyzer(paths, items_df, holidays_df, stores_df, transactions_df)
        
        holiday_analysis = historical_analyzer.analyze_holiday_effect_chunked()
        promotion_analysis = historical_analyzer.analyze_promotion_effect_chunked()
        transaction_analysis = historical_analyzer.analyze_transaction_effect()
        variability_stats = historical_analyzer.analyze_demand_variability_chunked()
        abc_df = historical_analyzer.classify_abc_chunked(items_df)
        
        historical_analysis = {
            'holiday': holiday_analysis,
            'promotion': promotion_analysis,
            'transaction': transaction_analysis
        }
        
        # Actualizar parámetros con valores históricos
        if holiday_analysis['general'] is not None:
            params.scenarios['FESTIVO']['holiday_effect'] = holiday_analysis['general']
            params.scenarios['FESTIVO_MAS_PROMOCION']['holiday_effect'] = holiday_analysis['general']
        
        logger.info('')
        
        # 3. Simulación de escenarios
        logger.info('=' * 80)
        logger.info('FASE 3: SIMULACIÓN DE ESCENARIOS')
        logger.info('=' * 80)
        
        simulator = ScenarioSimulator(abc_df, variability_stats, historical_analysis, items_df, stores_df)
        results_df = simulator.run_all_scenarios()
        
        logger.info('')
        
        # 4. Generar reportes
        logger.info('=' * 80)
        logger.info('FASE 4: GENERACIÓN DE REPORTES')
        logger.info('=' * 80)
        
        report_generator = ReportGenerator(results_df, paths)
        report_generator.export_simulation_results()
        resumen_df = report_generator.generate_scenario_summary()
        report_generator.generate_family_summary()
        report_generator.generate_leadtime_summary()
        
        logger.info('')
        
        # 5. Generar visualizaciones
        logger.info('=' * 80)
        logger.info('FASE 5: GENERACIÓN DE VISUALIZACIONES')
        logger.info('=' * 80)
        
        viz_generator = VisualizationGenerator(results_df, resumen_df, paths)
        viz_generator.generate_all_visualizations()
        
        # 6. Resumen ejecutivo final
        logger.info('')
        logger.info('=' * 80)
        logger.info('RESUMEN EJECUTIVO FINAL')
        logger.info('=' * 80)
        
        total_productos = len(results_df['item_nbr'].unique())
        total_combinaciones = len(results_df)
        avg_riesgo = resumen_df['riesgo_promedio'].mean()
        total_compras = resumen_df['compra_total_recomendada'].sum()
        avg_fill_rate = resumen_df['fill_rate_promedio'].mean()
        
        logger.info(f'''
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SIMULACIÓN DE ESCENARIOS COMPLETADA                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  📊 Productos Analizados: {total_productos:>8}          🔗 Combinaciones: {total_combinaciones:>8}           ║
║  📈 Riesgo Promedio: {avg_riesgo:>8.1f}%          ✅ Fill Rate: {avg_fill_rate:>6.1f}%                     ║
║  💰 Compra Total Recomendada: {total_compras:>12,.0f} unidades                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

📁 ARCHIVOS GENERADOS:
   • simulacion_final_escenarios.csv (datos completos)
   • resumen_escenarios.csv (resumen por escenario)
   • resumen_por_familia.csv (resumen por familia)
   • resumen_por_leadtime.csv (resumen por lead time)

📈 VISUALIZACIONES GENERADAS:
   • 01_riesgo_por_escenario.png
   • 02_compra_recomendada_por_escenario.png
   • 03_impacto_festivos.png
   • 04_impacto_promociones.png
   • 05_dashboard_escenarios.png

📊 ESCENARIOS SIMULADOS:
''')
        
        for scenario_name, scenario_config in params.scenarios.items():
            logger.info(f'   • {scenario_name}: {scenario_config["description"]}')
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f'''
══════════════════════════════════════════════════════════════════════════════
⏱️  Tiempo de ejecución: {duration:.2f} segundos ({duration/60:.2f} minutos)
🚀 Proceso completado exitosamente
══════════════════════════════════════════════════════════════════════════════
''')
        
    except Exception as e:
        logger.error(f'❌ Error durante la ejecución: {str(e)}')
        raise

if __name__ == '__main__':
    main()