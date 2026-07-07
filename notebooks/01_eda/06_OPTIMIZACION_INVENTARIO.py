# =============================================================================
# OPTIMIZACION_INVENTARIO: Sistema de Optimización de Inventario para Retail
# =============================================================================
# Sistema de optimización de inventario que convierte la demanda prevista en
# decisiones operativas: stock de seguridad, punto de pedido, cobertura,
# simulaciones de escenarios y recomendaciones de compra.
#
# Autor: Lead Data Scientist - Inventory Optimization & Supply Chain
# Versión: 1.0 - Optimizado para datasets grandes
# =============================================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
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
    'light_blue': '#5DADE2',
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
    forecasting_reports = project_root / 'reports' / 'forecasting'
    
    # Rutas de salida
    inventory_reports = project_root / 'reports' / 'inventory'
    inventory_reports.mkdir(parents=True, exist_ok=True)
    
    visuals_dir = project_root / 'visuals' / 'inventory'
    visuals_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        'project_root': project_root,
        'forecasting_reports': forecasting_reports,
        'inventory_reports': inventory_reports,
        'visuals': visuals_dir,
        'forecast_30d': forecasting_reports / 'forecast_30d.csv'
    }

paths = setup_paths()

print('=' * 80)
print('SISTEMA DE OPTIMIZACIÓN DE INVENTARIO')
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
        self.z_score = stats.norm.ppf(0.95)  # 1.645
        
        # Lead time en días
        self.lead_time = 7
        
        # Impactos detectados en EDA
        self.promocion_impact = 0.698  # +69.8%
        self.festivo_impact = 0.118    # +11.8%
        
        # Umbrales de riesgo
        self.risk_thresholds = {
            'bajo': 0.10,    # < 10% probabilidad de rotura
            'medio': 0.25,   # 10-25% probabilidad de rotura
            'alto': 0.25     # > 25% probabilidad de rotura
        }
        
        # Días de cobertura objetivo
        self.target_coverage_days = 30
        
    def get_config_summary(self):
        """Retornar resumen de configuración"""
        return {
            'Nivel de servicio': f'{self.service_level * 100:.1f}%',
            'Z-score': f'{self.z_score:.3f}',
            'Lead time': f'{self.lead_time} días',
            'Impacto promoción': f'+{self.promocion_impact * 100:.1f}%',
            'Impacto festivo': f'+{self.festivo_impact * 100:.1f}%',
            'Cobertura objetivo': f'{self.target_coverage_days} días'
        }

params = InventoryParameters()

print('CONFIGURACIÓN DEL SISTEMA:')
for key, value in params.get_config_summary().items():
    print(f'  • {key}: {value}')
print()

# =============================================================================
# CARGA DE DATOS DE FORECASTING
# =============================================================================

print('=' * 80)
print('CARGANDO DATOS DE FORECASTING')
print('=' * 80)

# Cargar forecast a 30 días
forecast_df = pd.read_csv(paths['forecast_30d'], parse_dates=['date'])
print(f'✅ Forecast 30d cargado: {len(forecast_df)} registros')

# Obtener familias únicas
familias = forecast_df['family'].unique().tolist()
print(f'📊 Familias analizadas: {len(familias)}')
for familia in familias:
    total_forecast = forecast_df[forecast_df['family'] == familia]['forecast'].sum()
    print(f'  • {familia}: ${total_forecast:,.0f} (30 días)')
print()

# =============================================================================
# CÁLCULO DE ESTADÍSTICAS DE DEMANDA
# =============================================================================

print('=' * 80)
print('CÁLCULO DE ESTADÍSTICAS DE DEMANDA')
print('=' * 80)

# Diccionario para almacenar estadísticas por familia
demanda_stats = {}

for familia in familias:
    familia_forecast = forecast_df[forecast_df['family'] == familia]['forecast'].values
    
    # Estadísticas básicas
    media_diaria = np.mean(familia_forecast)
    std_diaria = np.std(familia_forecast, ddof=1)  # ddof=1 para muestra
    cv = (std_diaria / media_diaria) * 100 if media_diaria > 0 else 0  # Coeficiente de variación
    max_diario = np.max(familia_forecast)
    min_diario = np.min(familia_forecast)
    
    demanda_stats[familia] = {
        'media_diaria': media_diaria,
        'std_diaria': std_diaria,
        'cv': cv,
        'max_diario': max_diario,
        'min_diario': min_diario,
        'total_30d': np.sum(familia_forecast),
        'forecast_completo': familia_forecast
    }
    
    print(f'\n  📊 {familia}:')
    print(f'    • Demanda diaria promedio: ${media_diaria:,.0f}')
    print(f'    • Desviación estándar: ${std_diaria:,.0f}')
    print(f'    • Coeficiente de variación: {cv:.2f}%')
    print(f'    • Rango: ${min_diario:,.0f} - ${max_diario:,.0f}')
    print(f'    • Total 30 días: ${np.sum(familia_forecast):,.0f}')

print()

# =============================================================================
# CÁLCULO DE STOCK DE SEGURIDAD
# =============================================================================

print('=' * 80)
print('CÁLCULO DE STOCK DE SEGURIDAD')
print('=' * 80)

stock_seguridad_data = []

for familia in familias:
    stats_fam = demanda_stats[familia]
    
    # Fórmula: SS = Z * σ_d * √(Lead Time)
    # Donde σ_d es la desviación estándar de la demanda diaria
    stock_seguridad = params.z_score * stats_fam['std_diaria'] * np.sqrt(params.lead_time)
    
    # Stock de seguridad ajustado por coeficiente de variación
    # Si CV es alto, aumentamos el stock de seguridad
    cv_factor = 1 + (stats_fam['cv'] / 100) * 0.5  # Factor de ajuste por volatilidad
    stock_seguridad_ajustado = stock_seguridad * cv_factor
    
    stock_seguridad_data.append({
        'familia': familia,
        'demanda_promedio_diaria': stats_fam['media_diaria'],
        'std_demanda_diaria': stats_fam['std_diaria'],
        'cv_porcentaje': stats_fam['cv'],
        'z_score': params.z_score,
        'lead_time_dias': params.lead_time,
        'stock_seguridad_base': stock_seguridad,
        'factor_ajuste_cv': cv_factor,
        'stock_seguridad_ajustado': stock_seguridad_ajustado,
        'dias_cobertura_ss': stock_seguridad_ajustado / stats_fam['media_diaria'] if stats_fam['media_diaria'] > 0 else 0
    })
    
    print(f'\n  🛡️  {familia}:')
    print(f'    • Stock seguridad base: ${stock_seguridad:,.0f}')
    print(f'    • Factor ajuste CV: {cv_factor:.3f}')
    print(f'    • Stock seguridad ajustado: ${stock_seguridad_ajustado:,.0f}')
    print(f'    • Días de cobertura del SS: {stock_seguridad_ajustado / stats_fam["media_diaria"]:.1f} días')

stock_seguridad_df = pd.DataFrame(stock_seguridad_data)
print()

# =============================================================================
# CÁLCULO DE PUNTO DE PEDIDO (REORDER POINT)
# =============================================================================

print('=' * 80)
print('CÁLCULO DE PUNTO DE PEDIDO (REORDER POINT)')
print('=' * 80)

punto_pedido_data = []

for _, row in stock_seguridad_df.iterrows():
    familia = row['familia']
    demanda_promedio = row['demanda_promedio_diaria']
    stock_seguridad = row['stock_seguridad_ajustado']
    
    # Fórmula: ROP = Demanda promedio durante lead time + Stock de seguridad
    # ROP = d * L + SS
    demanda_during_leadtime = demanda_promedio * params.lead_time
    punto_pedido = demanda_during_leadtime + stock_seguridad
    
    # Punto de pedido con margen de seguridad adicional (10%)
    punto_pedido_con_margen = punto_pedido * 1.10
    
    punto_pedido_data.append({
        'familia': familia,
        'demanda_promedio_diaria': demanda_promedio,
        'demanda_during_leadtime': demanda_during_leadtime,
        'stock_seguridad': stock_seguridad,
        'punto_pedido': punto_pedido,
        'punto_pedido_con_margen': punto_pedido_con_margen,
        'margen_seguridad_pct': 10.0
    })
    
    print(f'\n  📦 {familia}:')
    print(f'    • Demanda durante lead time ({params.lead_time} días): ${demanda_during_leadtime:,.0f}')
    print(f'    • Stock de seguridad: ${stock_seguridad:,.0f}')
    print(f'    • Punto de pedido (ROP): ${punto_pedido:,.0f}')
    print(f'    • Punto de pedido con margen 10%: ${punto_pedido_con_margen:,.0f}')

punto_pedido_df = pd.DataFrame(punto_pedido_data)
print()

# =============================================================================
# CÁLCULO DE COBERTURA DE INVENTARIO Y DÍAS DE INVENTARIO
# =============================================================================

print('=' * 80)
print('CÁLCULO DE COBERTURA Y DÍAS DE INVENTARIO')
print('=' * 80)

cobertura_data = []

for _, row in stock_seguridad_df.iterrows():
    familia = row['familia']
    demanda_diaria = row['demanda_promedio_diaria']
    stock_seguridad = row['stock_seguridad_ajustado']
    
    # Calcular inventario promedio (asumiendo política de revisar punto de pedido)
    # Inventario promedio = Stock seguridad + (Demanda durante lead time / 2)
    inventario_promedio = stock_seguridad + (demanda_diaria * params.lead_time / 2)
    
    # Cobertura de inventario (días que dura el inventario actual)
    cobertura_dias = inventario_promedio / demanda_diaria if demanda_diaria > 0 else 0
    
    # Rotación de inventario (veces por mes)
    rotacion_mensual = 30 / cobertura_dias if cobertura_dias > 0 else 0
    
    # Capital inmovilizado estimado (valor del inventario promedio)
    capital_inmovilizado = inventario_promedio
    
    cobertura_data.append({
        'familia': familia,
        'demanda_diaria_promedio': demanda_diaria,
        'stock_seguridad': stock_seguridad,
        'inventario_promedio': inventario_promedio,
        'cobertura_dias': cobertura_dias,
        'rotacion_mensual': rotacion_mensual,
        'capital_inmovilizado': capital_inmovilizado,
        'cobertura_objetivo': params.target_coverage_days,
        'desviacion_cobertura': cobertura_dias - params.target_coverage_days
    })
    
    print(f'\n  📊 {familia}:')
    print(f'    • Inventario promedio: ${inventario_promedio:,.0f}')
    print(f'    • Cobertura: {cobertura_dias:.1f} días')
    print(f'    • Rotación mensual: {rotacion_mensual:.2f} veces')
    print(f'    • Capital inmovilizado: ${capital_inmovilizado:,.0f}')

cobertura_df = pd.DataFrame(cobertura_data)
print()

# =============================================================================
# CÁLCULO DE ÍNDICE DE RIESGO DE ROTURA
# =============================================================================

print('=' * 80)
print('CÁLCULO DE ÍNDICE DE RIESGO DE ROTURA')
print('=' * 80)

riesgo_data = []

for _, row in stock_seguridad_df.iterrows():
    familia = row['familia']
    demanda_diaria = row['demanda_promedio_diaria']
    std_diaria = row['std_demanda_diaria']
    stock_seguridad = row['stock_seguridad_ajustado']
    cv = row['cv_porcentaje']
    
    # Probabilidad de rotura basada en el nivel de servicio
    # Si el stock de seguridad es menor al necesario para el nivel de servicio
    # calculamos la probabilidad de que la demanda exceda el inventario
    
    # Z calculado basado en el stock de seguridad actual
    z_calculado = stock_seguridad / (std_diaria * np.sqrt(params.lead_time)) if std_diaria > 0 else params.z_score
    
    # Probabilidad de no rotura (nivel de servicio efectivo)
    servicio_efectivo = stats.norm.cdf(z_calculado)
    
    # Probabilidad de rotura
    probabilidad_rotura = 1 - servicio_efectivo
    
    # Índice de riesgo (0-1, donde 1 es máximo riesgo)
    indice_riesgo = probabilidad_rotura
    
    # Clasificación de riesgo
    if indice_riesgo <= params.risk_thresholds['bajo']:
        riesgo_nivel = 'Bajo'
        riesgo_color = COLORS['success']
    elif indice_riesgo <= params.risk_thresholds['medio']:
        riesgo_nivel = 'Medio'
        riesgo_color = COLORS['warning']
    else:
        riesgo_nivel = 'Alto'
        riesgo_color = COLORS['danger']
    
    # Factor de criticidad basado en volumen de ventas
    total_ventas = demanda_stats[familia]['total_30d']
    total_general = sum(demanda_stats[f]['total_30d'] for f in familias)
    peso_ventas = total_ventas / total_general if total_general > 0 else 0
    
    # Riesgo ponderado (combina probabilidad de rotura con impacto en ventas)
    riesgo_ponderado = indice_riesgo * peso_ventas * 10  # Escala 0-10
    
    riesgo_data.append({
        'familia': familia,
        'stock_seguridad': stock_seguridad,
        'demanda_diaria_promedio': demanda_diaria,
        'std_demanda_diaria': std_diaria,
        'z_calculado': z_calculado,
        'servicio_efectivo': servicio_efectivo,
        'probabilidad_rotura': probabilidad_rotura,
        'indice_riesgo': indice_riesgo,
        'riesgo_nivel': riesgo_nivel,
        'peso_ventas_pct': peso_ventas * 100,
        'riesgo_ponderado': riesgo_ponderado,
        'cv_porcentaje': cv
    })
    
    print(f'\n  ⚠️  {familia}:')
    print(f'    • Servicio efectivo: {servicio_efectivo * 100:.2f}%')
    print(f'    • Probabilidad rotura: {probabilidad_rotura * 100:.2f}%')
    print(f'    • Índice de riesgo: {indice_riesgo:.4f}')
    print(f'    • Nivel de riesgo: {riesgo_nivel}')
    print(f'    • Peso en ventas: {peso_ventas * 100:.1f}%')
    print(f'    • Riesgo ponderado: {riesgo_ponderado:.2f}/10')

riesgo_df = pd.DataFrame(riesgo_data)
print()

# =============================================================================
# SIMULACIONES DE ESCENARIOS
# =============================================================================

print('=' * 80)
print('SIMULACIONES DE ESCENARIOS')
print('=' * 80)

escenarios = {
    'Base': 1.0,
    'Demanda +20%': 1.20,
    'Demanda -20%': 0.80,
    'Promoción (+69.8%)': 1 + params.promocion_impact,
    'Festivo (+11.8%)': 1 + params.festivo_impact
}

simulacion_data = []

for familia in familias:
    stats_fam = demanda_stats[familia]
    demanda_base = stats_fam['media_diaria']
    std_base = stats_fam['std_diaria']
    
    for escenario_nombre, factor_demanda in escenarios.items():
        # Nueva demanda bajo el escenario
        demanda_escenario = demanda_base * factor_demanda
        
        # La desviación estándar también escala con la demanda
        std_escenario = std_base * factor_demanda
        
        # Stock de seguridad bajo el escenario
        stock_seguridad_escenario = params.z_score * std_escenario * np.sqrt(params.lead_time)
        cv_factor = 1 + ((std_escenario / demanda_escenario * 100) / 100) * 0.5 if demanda_escenario > 0 else 1
        stock_seguridad_ajustado = stock_seguridad_escenario * cv_factor
        
        # Punto de pedido bajo el escenario
        demanda_during_lt = demanda_escenario * params.lead_time
        punto_pedido_escenario = demanda_during_lt + stock_seguridad_ajustado
        
        # Probabilidad de rotura bajo el escenario
        z_escenario = stock_seguridad_ajustado / (std_escenario * np.sqrt(params.lead_time)) if std_escenario > 0 else params.z_score
        servicio_efectivo = stats.norm.cdf(z_escenario)
        probabilidad_rotura = 1 - servicio_efectivo
        
        # Clasificación de riesgo
        if probabilidad_rotura <= params.risk_thresholds['bajo']:
            riesgo_nivel = 'Bajo'
        elif probabilidad_rotura <= params.risk_thresholds['medio']:
            riesgo_nivel = 'Medio'
        else:
            riesgo_nivel = 'Alto'
        
        # Variación respecto al escenario base
        if factor_demanda == 1.0:
            variacion_ss = 0
            variacion_rop = 0
            variacion_riesgo = 0
        else:
            # Comparar con el escenario base
            base_row = next(r for r in simulacion_data if r['familia'] == familia and r['escenario'] == 'Base')
            variacion_ss = ((stock_seguridad_ajustado - base_row['stock_seguridad']) / base_row['stock_seguridad'] * 100) if base_row['stock_seguridad'] > 0 else 0
            variacion_rop = ((punto_pedido_escenario - base_row['punto_pedido']) / base_row['punto_pedido'] * 100) if base_row['punto_pedido'] > 0 else 0
            variacion_riesgo = probabilidad_rotura - base_row['probabilidad_rotura']
        
        simulacion_data.append({
            'familia': familia,
            'escenario': escenario_nombre,
            'factor_demanda': factor_demanda,
            'demanda_diaria': demanda_escenario,
            'std_demanda': std_escenario,
            'stock_seguridad': stock_seguridad_ajustado,
            'punto_pedido': punto_pedido_escenario,
            'probabilidad_rotura': probabilidad_rotura,
            'riesgo_nivel': riesgo_nivel,
            'variacion_ss_pct': variacion_ss,
            'variacion_rop_pct': variacion_rop,
            'variacion_riesgo': variacion_riesgo
        })

simulacion_df = pd.DataFrame(simulacion_data)

# Mostrar resumen de simulaciones
for familia in familias:
    print(f'\n  📊 {familia}:')
    familia_sim = simulacion_df[simulacion_df['familia'] == familia]
    for _, row in familia_sim.iterrows():
        print(f'    • {row["escenario"]:>20}: SS=${row["stock_seguridad"]:>10,.0f} | ROP=${row["punto_pedido"]:>12,.0f} | Riesgo={row["riesgo_nivel"]:>5} ({row["probabilidad_rotura"]*100:.1f}%)')

print()

# =============================================================================
# RANKING DE FAMILIAS SEGÚN RIESGO
# =============================================================================

print('=' * 80)
print('RANKING DE FAMILIAS SEGÚN RIESGO')
print('=' * 80)

# Crear ranking combinando índice de riesgo y peso en ventas
ranking_data = []

for _, row in riesgo_df.iterrows():
    familia = row['familia']
    
    # Score de riesgo (combina probabilidad de rotura con impacto comercial)
    score_riesgo = row['riesgo_ponderado']
    
    # Score de volatilidad (CV normalizado)
    max_cv = riesgo_df['cv_porcentaje'].max()
    score_volatilidad = (row['cv_porcentaje'] / max_cv * 10) if max_cv > 0 else 0
    
    # Score de impacto comercial (peso en ventas normalizado)
    score_impacto = row['peso_ventas_pct'] / 10  # Escala 0-10
    
    # Score total ponderado
    score_total = (score_riesgo * 0.4) + (score_volatilidad * 0.3) + (score_impacto * 0.3)
    
    ranking_data.append({
        'familia': familia,
        'indice_riesgo': row['indice_riesgo'],
        'probabilidad_rotura_pct': row['probabilidad_rotura'] * 100,
        'cv_porcentaje': row['cv_porcentaje'],
        'peso_ventas_pct': row['peso_ventas_pct'],
        'score_riesgo': score_riesgo,
        'score_volatilidad': score_volatilidad,
        'score_impacto': score_impacto,
        'score_total': score_total,
        'nivel_riesgo': row['riesgo_nivel']
    })

ranking_df = pd.DataFrame(ranking_data).sort_values('score_total', ascending=False).reset_index(drop=True)
ranking_df['ranking'] = ranking_df.index + 1

print('\n🏆 RANKING DE FAMILIAS POR RIESGO:')
for _, row in ranking_df.iterrows():
    icon = '🔴' if row['nivel_riesgo'] == 'Alto' else '🟡' if row['nivel_riesgo'] == 'Medio' else '🟢'
    print(f'  {row["ranking"]}. {icon} {row["familia"]:15} | Score: {row["score_total"]:5.2f} | Riesgo: {row["probabilidad_rotura_pct"]:5.2f}% | CV: {row["cv_porcentaje"]:5.2f}% | Impacto: {row["peso_ventas_pct"]:5.1f}%')

print()

# =============================================================================
# GENERAR RECOMENDACIONES AUTOMÁTICAS DE COMPRA
# =============================================================================

print('=' * 80)
print('GENERANDO RECOMENDACIONES AUTOMÁTICAS DE COMPRA')
print('=' * 80)

recomendaciones_data = []

for familia in familias:
    # Obtener datos relevantes
    pp_row = punto_pedido_df[punto_pedido_df['familia'] == familia].iloc[0]
    ss_row = stock_seguridad_df[stock_seguridad_df['familia'] == familia].iloc[0]
    riesgo_row = riesgo_df[riesgo_df['familia'] == familia].iloc[0]
    ranking_row = ranking_df[ranking_df['familia'] == familia].iloc[0]
    stats_fam = demanda_stats[familia]
    
    # Calcular cantidad de pedido sugerida (EOQ simplificado)
    # Q = √(2 * D * S / H) donde:
    # D = demanda anual, S = costo de pedido, H = costo de mantenimiento
    # Usamos una aproximación: pedido = demanda durante lead time * 2 + stock seguridad
    demanda_anual = stats_fam['media_diaria'] * 365
    cantidad_pedido = (stats_fam['media_diaria'] * params.lead_time * 2) + ss_row['stock_seguridad_ajustado']
    
    # Prioridad de compra basada en riesgo y ranking
    prioridad = 'ALTA' if ranking_row['nivel_riesgo'] == 'Alto' or ranking_row['ranking'] <= 2 else 'MEDIA' if ranking_row['nivel_riesgo'] == 'Medio' else 'BAJA'
    
    # Frecuencia de pedido sugerida
    if prioridad == 'ALTA':
        frecuencia = 'Semanal'
        dias_frecuencia = 7
    elif prioridad == 'MEDIA':
        frecuencia = 'Quincenal'
        dias_frecuencia = 15
    else:
        frecuencia = 'Mensual'
        dias_frecuencia = 30
    
    # Recomendación específica
    if riesgo_row['riesgo_nivel'] == 'Alto':
        recomendacion = f'URGENTE: Aumentar stock de seguridad en {int(ranking_row["score_volatilidad"] * 10)}%. Considerar pedidos de emergencia.'
    elif riesgo_row['riesgo_nivel'] == 'Medio':
        recomendacion = f'MONITOREAR: Revisar niveles de inventario cada {dias_frecuencia} días. Considerar aumento preventivo del {int(ranking_row["score_volatilidad"] * 5)}%.'
    else:
        recomendacion = f'MANTENER: Niveles actuales son adecuados. Revisión estándar cada {frecuencia.lower()}.'
    
    # Acción específica
    if prioridad == 'ALTA':
        accion = f'Pedir ${cantidad_pedido:,.0f} inmediatamente. Revisar diariamente.'
    elif prioridad == 'MEDIA':
        accion = f'Programar pedido de ${cantidad_pedido:,.0f} para la próxima semana.'
    else:
        accion = f'Incluir en próximo pedido mensual de ${cantidad_pedido:,.0f}.'
    
    recomendaciones_data.append({
        'familia': familia,
        'prioridad': prioridad,
        'cantidad_pedido_sugerida': cantidad_pedido,
        'frecuencia_pedido': frecuencia,
        'dias_frecuencia': dias_frecuencia,
        'punto_pedido': pp_row['punto_pedido_con_margen'],
        'stock_seguridad': ss_row['stock_seguridad_ajustado'],
        'nivel_riesgo': riesgo_row['riesgo_nivel'],
        'ranking_riesgo': int(ranking_row['ranking']),
        'recomendacion': recomendacion,
        'accion': accion,
        'demanda_diaria_promedio': stats_fam['media_diaria'],
        'total_forecast_30d': stats_fam['total_30d']
    })
    
    print(f'\n  📋 {familia} (Prioridad: {prioridad}):')
    print(f'    • Cantidad pedido sugerida: ${cantidad_pedido:,.0f}')
    print(f'    • Frecuencia: {frecuencia}')
    print(f'    • Punto de pedido: ${pp_row["punto_pedido_con_margen"]:,.0f}')
    print(f'    • Recomendación: {recomendacion}')
    print(f'    • Acción: {accion}')

recomendaciones_df = pd.DataFrame(recomendaciones_data)
print()

# =============================================================================
# EXPORTAR RESULTADOS
# =============================================================================

print('=' * 80)
print('EXPORTANDO RESULTADOS')
print('=' * 80)

# 1. Stock de Seguridad
stock_seguridad_export = stock_seguridad_df[[
    'familia', 'demanda_promedio_diaria', 'std_demanda_diaria', 'cv_porcentaje',
    'z_score', 'lead_time_dias', 'stock_seguridad_base', 'stock_seguridad_ajustado',
    'dias_cobertura_ss'
]].copy()
stock_seguridad_export.to_csv(paths['inventory_reports'] / 'stock_seguridad.csv', index=False)
print('✅ stock_seguridad.csv exportado')

# 2. Punto de Pedido
punto_pedido_export = punto_pedido_df[[
    'familia', 'demanda_promedio_diaria', 'demanda_during_leadtime',
    'stock_seguridad', 'punto_pedido', 'punto_pedido_con_margen', 'margen_seguridad_pct'
]].copy()
punto_pedido_export.to_csv(paths['inventory_reports'] / 'punto_pedido.csv', index=False)
print('✅ punto_pedido.csv exportado')

# 3. Simulación de Escenarios
simulacion_export = simulacion_df[[
    'familia', 'escenario', 'factor_demanda', 'demanda_diaria', 'std_demanda',
    'stock_seguridad', 'punto_pedido', 'probabilidad_rotura', 'riesgo_nivel',
    'variacion_ss_pct', 'variacion_rop_pct', 'variacion_riesgo'
]].copy()
simulacion_export.to_csv(paths['inventory_reports'] / 'simulacion_escenarios.csv', index=False)
print('✅ simulacion_escenarios.csv exportado')

# 4. Ranking de Riesgo
ranking_export = ranking_df[[
    'ranking', 'familia', 'indice_riesgo', 'probabilidad_rotura_pct',
    'cv_porcentaje', 'peso_ventas_pct', 'score_total', 'nivel_riesgo'
]].copy()
ranking_export.to_csv(paths['inventory_reports'] / 'ranking_riesgo.csv', index=False)
print('✅ ranking_riesgo.csv exportado')

# 5. Recomendaciones de Compra
recomendaciones_export = recomendaciones_df[[
    'familia', 'prioridad', 'cantidad_pedido_sugerida', 'frecuencia_pedido',
    'punto_pedido', 'stock_seguridad', 'nivel_riesgo', 'ranking_riesgo',
    'recomendacion', 'accion'
]].copy()
recomendaciones_export.to_csv(paths['inventory_reports'] / 'recomendaciones_compra.csv', index=False)
print('✅ recomendaciones_compra.csv exportado')

# 6. Resumen Ejecutivo
resumen_ejecutivo_data = []

for familia in familias:
    pp_row = punto_pedido_df[punto_pedido_df['familia'] == familia].iloc[0]
    ss_row = stock_seguridad_df[stock_seguridad_df['familia'] == familia].iloc[0]
    riesgo_row = riesgo_df[riesgo_df['familia'] == familia].iloc[0]
    cobertura_row = cobertura_df[cobertura_df['familia'] == familia].iloc[0]
    ranking_row = ranking_df[ranking_df['familia'] == familia].iloc[0]
    stats_fam = demanda_stats[familia]
    
    resumen_ejecutivo_data.append({
        'familia': familia,
        'demanda_diaria_promedio': round(stats_fam['media_diaria'], 2),
        'demanda_total_30d': round(stats_fam['total_30d'], 2),
        'stock_seguridad': round(ss_row['stock_seguridad_ajustado'], 2),
        'punto_pedido': round(pp_row['punto_pedido_con_margen'], 2),
        'cobertura_dias': round(cobertura_row['cobertura_dias'], 1),
        'rotacion_mensual': round(cobertura_row['rotacion_mensual'], 2),
        'capital_inmovilizado': round(cobertura_row['capital_inmovilizado'], 2),
        'nivel_riesgo_rotura': riesgo_row['riesgo_nivel'],
        'probabilidad_rotura_pct': round(riesgo_row['probabilidad_rotura'] * 100, 2),
        'ranking_riesgo': int(ranking_row['ranking']),
        'cv_porcentaje': round(riesgo_row['cv_porcentaje'], 2)
    })

resumen_df = pd.DataFrame(resumen_ejecutivo_data)
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

# 1. Stock de Seguridad por Familia
fig, ax = plt.subplots(figsize=(12, 6))
barras = ax.barh(
    stock_seguridad_df['familia'],
    stock_seguridad_df['stock_seguridad_ajustado'],
    color=[COLORS['primary'], COLORS['secondary'], COLORS['tertiary'], COLORS['info'], COLORS['purple']]
)
ax.set_xlabel('Stock de Seguridad ($)', fontsize=12)
ax.set_ylabel('Familia', fontsize=12)
ax.set_title('Stock de Seguridad por Familia', fontsize=14, fontweight='bold')
for barra, valor in zip(barras, stock_seguridad_df['stock_seguridad_ajustado']):
    ax.text(barra.get_width() + 1000, barra.get_y() + barra.get_height()/2,
            f'${valor:,.0f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig(paths['visuals'] / '01_stock_seguridad_por_familia.png', dpi=150, bbox_inches='tight')
plt.close()
print('  ✅ 01_stock_seguridad_por_familia.png')

# 2. Punto de Pedido por Familia
fig, ax = plt.subplots(figsize=(12, 6))
barras = ax.bar(
    punto_pedido_df['familia'],
    punto_pedido_df['punto_pedido_con_margen'],
    color=[COLORS['primary'], COLORS['secondary'], COLORS['tertiary'], COLORS['info'], COLORS['purple']]
)
ax.set_xlabel('Familia', fontsize=12)
ax.set_ylabel('Punto de Pedido ($)', fontsize=12)
ax.set_title('Punto de Pedido por Familia (con margen 10%)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
for barra, valor in zip(barras, punto_pedido_df['punto_pedido_con_margen']):
    ax.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 5000,
            f'${valor:,.0f}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(paths['visuals'] / '02_punto_pedido_por_familia.png', dpi=150, bbox_inches='tight')
plt.close()
print('  ✅ 02_punto_pedido_por_familia.png')

# 3. Riesgo de Rotura por Familia
fig, ax = plt.subplots(figsize=(12, 6))
colores_riesgo = []
for _, row in riesgo_df.iterrows():
    if row['riesgo_nivel'] == 'Alto':
        colores_riesgo.append(COLORS['danger'])
    elif row['riesgo_nivel'] == 'Medio':
        colores_riesgo.append(COLORS['warning'])
    else:
        colores_riesgo.append(COLORS['success'])

barras = ax.bar(riesgo_df['familia'], riesgo_df['probabilidad_rotura'] * 100, color=colores_riesgo)
ax.axhline(y=params.risk_thresholds['bajo'] * 100, color=COLORS['success'], linestyle='--', alpha=0.7, label='Umbral Bajo')
ax.axhline(y=params.risk_thresholds['medio'] * 100, color=COLORS['warning'], linestyle='--', alpha=0.7, label='Umbral Medio')
ax.set_xlabel('Familia', fontsize=12)
ax.set_ylabel('Probabilidad de Rotura (%)', fontsize=12)
ax.set_title('Índice de Riesgo de Rotura por Familia', fontsize=14, fontweight='bold')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(paths['visuals'] / '03_riesgo_rotura_por_familia.png', dpi=150, bbox_inches='tight')
plt.close()
print('  ✅ 03_riesgo_rotura_por_familia.png')

# 4. Comparativa de Escenarios
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Simulación de Escenarios - Impacto en Parámetros de Inventario', fontsize=16, fontweight='bold')

# Stock de seguridad por escenario
ax = axes[0, 0]
for familia in familias:
    fam_sim = simulacion_df[simulacion_df['familia'] == familia]
    ax.plot(fam_sim['escenario'], fam_sim['stock_seguridad'], marker='o', label=familia, linewidth=2)
ax.set_title('Stock de Seguridad por Escenario', fontsize=12, fontweight='bold')
ax.set_ylabel('Stock de Seguridad ($)')
ax.legend(fontsize=8)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

# Punto de pedido por escenario
ax = axes[0, 1]
for familia in familias:
    fam_sim = simulacion_df[simulacion_df['familia'] == familia]
    ax.plot(fam_sim['escenario'], fam_sim['punto_pedido'], marker='s', label=familia, linewidth=2)
ax.set_title('Punto de Pedido por Escenario', fontsize=12, fontweight='bold')
ax.set_ylabel('Punto de Pedido ($)')
ax.legend(fontsize=8)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

# Probabilidad de rotura por escenario
ax = axes[1, 0]
for familia in familias:
    fam_sim = simulacion_df[simulacion_df['familia'] == familia]
    ax.plot(fam_sim['escenario'], fam_sim['probabilidad_rotura'] * 100, marker='^', label=familia, linewidth=2)
ax.axhline(y=params.risk_thresholds['medio'] * 100, color='red', linestyle='--', alpha=0.5)
ax.set_title('Probabilidad de Rotura por Escenario', fontsize=12, fontweight='bold')
ax.set_ylabel('Probabilidad de Rotura (%)')
ax.legend(fontsize=8)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

# Variación porcentual del stock de seguridad
ax = axes[1, 1]
for familia in familias:
    fam_sim = simulacion_df[simulacion_df['familia'] == familia]
    fam_sim_base = fam_sim[fam_sim['escenario'] == 'Base']
    if len(fam_sim_base) > 0:
        base_ss = fam_sim_base['stock_seguridad'].values[0]
        variaciones = ((fam_sim['stock_seguridad'] - base_ss) / base_ss * 100).tolist()
        ax.plot(fam_sim['escenario'], variaciones, marker='d', label=familia, linewidth=2)
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
ax.set_title('Variación % Stock de Seguridad vs Base', fontsize=12, fontweight='bold')
ax.set_ylabel('Variación (%)')
ax.legend(fontsize=8)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig(paths['visuals'] / '04_comparativa_escenarios.png', dpi=150, bbox_inches='tight')
plt.close()
print('  ✅ 04_comparativa_escenarios.png')

# 5. Ranking de Familias Críticas
fig, ax = plt.subplots(figsize=(12, 6))
ranking_ordenado = ranking_df.sort_values('score_total', ascending=True)
colores_ranking = []
for _, row in ranking_ordenado.iterrows():
    if row['nivel_riesgo'] == 'Alto':
        colores_ranking.append(COLORS['danger'])
    elif row['nivel_riesgo'] == 'Medio':
        colores_ranking.append(COLORS['warning'])
    else:
        colores_ranking.append(COLORS['success'])

barras = ax.barh(ranking_ordenado['familia'], ranking_ordenado['score_total'], color=colores_ranking)
ax.set_xlabel('Score de Riesgo Total', fontsize=12)
ax.set_ylabel('Familia', fontsize=12)
ax.set_title('Ranking de Familias por Riesgo (Mayor a Menor Criticidad)', fontsize=14, fontweight='bold')
for barra, score, nivel in zip(barras, ranking_ordenado['score_total'], ranking_ordenado['nivel_riesgo']):
    ax.text(barra.get_width() + 0.2, barra.get_y() + barra.get_height()/2,
            f'{score:.2f} ({nivel})', va='center', fontsize=10)
plt.tight_layout()
plt.savefig(paths['visuals'] / '05_ranking_familias_criticas.png', dpi=150, bbox_inches='tight')
plt.close()
print('  ✅ 05_ranking_familias_criticas.png')

# 6. Dashboard integrado - Cobertura y Rotación
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Métricas de Cobertura y Rotación de Inventario', fontsize=16, fontweight='bold')

ax = axes[0]
barras = ax.bar(cobertura_df['familia'], cobertura_df['cobertura_dias'],
               color=[COLORS['primary'], COLORS['secondary'], COLORS['tertiary'], COLORS['info'], COLORS['purple']])
ax.axhline(y=params.target_coverage_days, color='red', linestyle='--', linewidth=2, label=f'Objetivo ({params.target_coverage_days} días)')
ax.set_title('Cobertura de Inventario (Días)', fontsize=12, fontweight='bold')
ax.set_ylabel('Días de cobertura')
ax.legend()
for barra, valor in zip(barras, cobertura_df['cobertura_dias']):
    ax.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 0.5,
            f'{valor:.1f}', ha='center', fontsize=10)

ax = axes[1]
barras = ax.bar(cobertura_df['familia'], cobertura_df['rotacion_mensual'],
               color=[COLORS['primary'], COLORS['secondary'], COLORS['tertiary'], COLORS['info'], COLORS['purple']])
ax.set_title('Rotación de Inventario (veces/mes)', fontsize=12, fontweight='bold')
ax.set_ylabel('Rotación mensual')
for barra, valor in zip(barras, cobertura_df['rotacion_mensual']):
    ax.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 0.05,
            f'{valor:.2f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(paths['visuals'] / '06_cobertura_rotacion.png', dpi=150, bbox_inches='tight')
plt.close()
print('  ✅ 06_cobertura_rotacion.png')

print(f'\n📁 Visualizaciones guardadas en: {paths["visuals"]}')
print()

# =============================================================================
# CONCLUSIONES EJECUTIVAS Y RECOMENDACIONES
# =============================================================================

print('=' * 80)
print('CONCLUSIONES EJECUTIVAS Y RECOMENDACIONES')
print('=' * 80)

# Calcular métricas globales
total_capital_inmovilizado = cobertura_df['capital_inmovilizado'].sum()
total_demanda_diaria = sum(demanda_stats[f]['media_diaria'] for f in familias)
familias_alto_riesgo = riesgo_df[riesgo_df['riesgo_nivel'] == 'Alto']['familia'].tolist()
familias_medio_riesgo = riesgo_df[riesgo_df['riesgo_nivel'] == 'Medio']['familia'].tolist()

mejor_cobertura = cobertura_df.loc[cobertura_df['cobertura_dias'].idxmax()]
peor_cobertura = cobertura_df.loc[cobertura_df['cobertura_dias'].idxmin()]

print(f'''
╔══════════════════════════════════════════════════════════════════════════════╗
║                           RESUMEN EJECUTIVO                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 MÉTRICAS GLOBALES:
   • Capital total inmovilizado: ${total_capital_inmovilizado:,.0f}
   • Demanda diaria total: ${total_demanda_diaria:,.0f}
   • Familias con riesgo ALTO: {len(familias_alto_riesgo)} ({', '.join(familias_alto_riesgo) if familias_alto_riesgo else 'Ninguna'})
   • Familias con riesgo MEDIO: {len(familias_medio_riesgo)} ({', '.join(familias_medio_riesgo) if familias_medio_riesgo else 'Ninguna'})

🏆 MEJORES Y PEORES MÉTRICAS:
   • Mejor cobertura: {mejor_cobertura['familia']} ({mejor_cobertura['cobertura_dias']:.1f} días)
   • Peor cobertura: {peor_cobertura['familia']} ({peor_cobertura['cobertura_dias']:.1f} días)

═══════════════════════════════════════════════════════════════════════════════

💡 RECOMENDACIONES ESTRATÉGICAS:

1️⃣  REDUCCIÓN DE ROTURAS:
''')

if familias_alto_riesgo:
    print(f'   ⚠️  ACCIÓN INMEDIATA requerida para: {", ".join(familias_alto_riesgo)}')
    print('   • Aumentar stock de seguridad en un 15-25%')
    print('   • Implementar monitoreo diario de niveles')
    print('   • Considerar proveedores alternativos para reducir lead time')
else:
    print('   ✅ No se identifican familias con riesgo alto de rotura')
    print('   • Mantener políticas actuales de inventario')

print(f'''
2️⃣  OPTIMIZACIÓN DE INVENTARIO:
   • Revisar políticas de {peor_cobertura['familia']} (cobertura: {peor_cobertura['cobertura_dias']:.1f} días)
   • Reducir exceso de inventario en {mejor_cobertura['familia']} (cobertura: {mejor_cobertura['cobertura_dias']:.1f} días)
   • Implementar sistema de revisión continua (no periódica)
   • Considerar modelo de reposición automática basado en punto de pedido

3️⃣  REDUCCIÓN DE CAPITAL INMOVILIZADO:
   • Capital total inmovilizado: ${total_capital_inmovilizado:,.0f}
   • Oportunidad de reducción: 10-15% mediante optimización de stock de seguridad
   • Ahorro potencial estimado: ${total_capital_inmovilizado * 0.125:,.0f}
   • Implementar clasificación ABC para priorizar gestión

4️⃣  MEJORA DEL NIVEL DE SERVICIO:
   • Objetivo: Mantener nivel de servicio ≥ 95%
   • Familias que requieren atención: {len(familias_alto_riesgo) + len(familias_medio_riesgo)}
   • Implementar alertas tempranas cuando inventario < punto de pedido
   • Revisar nivel de servicio semanalmente

═══════════════════════════════════════════════════════════════════════════════

📈 PLAN DE ACCIÓN INMEDIATO (Próximos 7 días):

''')

# Generar plan de acción basado en recomendaciones
for _, rec in recomendaciones_df.iterrows():
    if rec['prioridad'] == 'ALTA':
        print(f'   🔴 {rec["familia"]}:')
        print(f'      {rec["accion"]}')
        print()

print('''   📋 ACCIONES GENERALES:
      1. Revisar y ajustar puntos de pedido en sistema ERP
      2. Configurar alertas automáticas de stock bajo
      3. Establecer reunión semanal de revisión de inventario
      4. Capacitar equipo de compras en nuevas políticas

═══════════════════════════════════════════════════════════════════════════════

🔄 PRÓXIMOS PASOS:

   1. Implementar dashboard de seguimiento en tiempo real
   2. Integrar con sistema de gestión de almacén (WMS)
   3. Automatizar generación de órdenes de compra
   4. Expandir análisis a nivel SKU (producto individual)
   5. Incorporar machine learning para predicción de demanda más precisa
   6. Desarrollar modelo de optimización multi-echelon

═══════════════════════════════════════════════════════════════════════════════
''')

# Tabla resumen final
print('''
╔══════════════════════════════════════════════════════════════════════════════╗
║                         TABLA RESUMEN POR FAMILIA                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
''')

print(f'{"Familia":<15} {"Demanda/ Día":>12} {"Stock Seg.":>12} {"Pto. Pedido":>13} {"Cobertura":>10} {"Riesgo":>8}')
print('─' * 75)

for _, row in resumen_df.iterrows():
    riesgo_icon = '🔴' if row['nivel_riesgo_rotura'] == 'Alto' else '🟡' if row['nivel_riesgo_rotura'] == 'Medio' else '🟢'
    print(f'{row["familia"]:<15} ${row["demanda_diaria_promedio"]:>10,.0f} ${row["stock_seguridad"]:>10,.0f} ${row["punto_pedido"]:>11,.0f} {row["cobertura_dias"]:>8.1f}d {riesgo_icon} {row["nivel_riesgo_rotura"]}')

print()
print('=' * 80)
print('✅ ANÁLISIS DE OPTIMIZACIÓN DE INVENTARIO COMPLETADO')
print('=' * 80)
print(f'\n📁 Reports exportados: {paths["inventory_reports"]}')
print(f'📁 Visualizaciones: {paths["visuals"]}')
print()
print('📊 Archivos generados:')
print('   • stock_seguridad.csv')
print('   • punto_pedido.csv')
print('   • simulacion_escenarios.csv')
print('   • ranking_riesgo.csv')
print('   • recomendaciones_compra.csv')
print('   • resumen_ejecutivo.csv')
print()
print('📈 Gráficos generados:')
print('   • 01_stock_seguridad_por_familia.png')
print('   • 02_punto_pedido_por_familia.png')
print('   • 03_riesgo_rotura_por_familia.png')
print('   • 04_comparativa_escenarios.png')
print('   • 05_ranking_familias_criticas.png')
print('   • 06_cobertura_rotacion.png')
print('=' * 80)