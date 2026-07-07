"""
================================================================================
08_SIMULADOR_INVENTARIO.py
================================================================================

Aplicación profesional en Streamlit para simulación de inventario basada en
resultados de forecasting y optimización.

Objetivo:
---------
Construir una herramienta interactiva para simulación de inventario basada en
los resultados generados en los archivos de forecast y optimización.

Estructura:
-----------
- Página 1: Resumen Ejecutivo (KPIs y gráficos principales)
- Página 2: Simulador Interactivo (cálculos en tiempo real)
- Página 3: Productos Críticos (tabla interactiva)
- Página 4: Forecast + Inventario (análisis combinado)

Autor: FLOWMAP ANALYTICS
Fecha: 2026
Versión: 1.0.0
================================================================================
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add app directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from utils.calculations import (
    calculate_safety_stock,
    calculate_reorder_point,
    calculate_demand_adjustment,
    calculate_fill_rate,
    calculate_stock_coverage_days,
    calculate_stockout_probability
)
from utils.data_loader import load_all_data, load_csv_safe
from utils.exporters import export_to_csv, export_to_excel

# =============================================================================
# CONFIGURACIÓN DE PÁGINA
# =============================================================================

st.set_page_config(
    page_title="Simulador de Inventario | FLOWMAP ANALYTICS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# ESTILOS PERSONALIZADOS
# =============================================================================

st.markdown("""
<style>
    /* Colores corporativos */
    :root {
        --primary-color: #1E3A5F;
        --secondary-color: #2ECC71;
        --warning-color: #F39C12;
        --danger-color: #E74C3C;
        --light-bg: #F8F9FA;
    }
   
    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
   
    .kpi-value {
        font-size: 2.5em;
        font-weight: bold;
        margin: 10px 0;
    }
   
    .kpi-label {
        font-size: 0.9em;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
   
    /* Métricas en columnas */
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #1E3A5F;
    }
   
    /* Alertas */
    .alert-critical {
        background-color: #ffebee;
        border-left: 4px solid #E74C3C;
        padding: 15px;
        margin: 10px 0;
        border-radius: 4px;
    }
   
    .alert-warning {
        background-color: #fff3e0;
        border-left: 4px solid #F39C12;
        padding: 15px;
        margin: 10px 0;
        border-radius: 4px;
    }
   
    .alert-success {
        background-color: #e8f5e9;
        border-left: 4px solid #2ECC71;
        padding: 15px;
        margin: 10px 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CARGA DE DATOS
# =============================================================================

@st.cache_data
def cargar_datos():
    """Carga todos los datasets necesarios para la aplicación."""
    # Ruta relativa al directorio de datos
    data_dir = os.path.join(os.path.dirname(__file__), 'segundo proyecto', 'data')
   
    # Si no existe, intentar ruta alternativa
    if not os.path.exists(data_dir):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
   
    datasets = load_all_data(data_dir)
    return datasets


# Cargar datos al inicio
with st.spinner('🔄 Cargando datos...'):
    datasets = cargar_datos()

# =============================================================================
# BARRA LATERAL - NAVEGACIÓN
# =============================================================================

st.sidebar.image(
    "https://via.placeholder.com/300x100/1E3A5F/FFFFFF?text=FLOWMAP+ANALYTICS",
    use_container_width=True
)

st.sidebar.markdown("---")
st.sidebar.title("🧭 Navegación")

menu_options = {
    "📊 Resumen Ejecutivo": "page1",
    "🎛️ Simulador Interactivo": "page2",
    "⚠️ Productos Críticos": "page3",
    "📈 Forecast + Inventario": "page4"
}

selected_option = st.sidebar.radio(
    "Selecciona una página:",
    options=list(menu_options.keys()),
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("📁 Archivos de Datos")

# Mostrar estado de carga de archivos
for name, df in datasets.items():
    if df is not None:
        st.sidebar.success(f"✓ {name}.csv ({len(df)} filas)")
    else:
        st.sidebar.error(f"✗ {name}.csv (no encontrado)")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### ℹ️ Acerca de
**Simulador de Inventario v1.0**

Herramienta profesional para:
- Análisis de riesgo de rotura
- Optimización de stock
- Simulación de escenarios
- Planificación de compras

*FLOWMAP ANALYTICS © 2026*
""")

# =============================================================================
# FUNCIÓN PARA MOSTRAR KPI CARDS
# =============================================================================

def show_kpi_card(label, value, unit="", color="blue", icon="📊"):
    """Muestra una tarjeta KPI estilizada."""
    color_map = {
        "blue": "#667eea",
        "green": "#2ECC71",
        "orange": "#F39C12",
        "red": "#E74C3C",
        "purple": "#9B59B6"
    }
   
    color_hex = color_map.get(color, "#667eea")
   
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color_hex} 0%, {color_hex}cc 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    ">
        <div style="font-size: 0.9em; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px;">
            {icon} {label}
        </div>
        <div style="font-size: 2.5em; font-weight: bold; margin: 10px 0;">
            {value}{unit}
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PÁGINA 1: RESUMEN EJECUTIVO
# =============================================================================

if selected_option == "📊 Resumen Ejecutivo":
    st.title("📊 Resumen Ejecutivo de Inventario")
    st.markdown("""
    <div style="text-align: justify;">
        Vista general de los indicadores clave de inventario. Esta página proporciona
        una visión panorámica del estado actual del inventario, identificando productos
        críticos, riesgos de rotura y recomendaciones de compra.
    </div>
    """, unsafe_allow_html=True)
   
    st.markdown("---")
   
    # Obtener datos principales
    inventario_df = datasets.get('inventario')
    riesgo_df = datasets.get('riesgo_rotura')
    compras_df = datasets.get('compras_recomendadas')
   
    if inventario_df is not None:
        # Calcular KPIs principales
        total_productos = len(inventario_df)
       
        # Identificar columnas relevantes (intentar múltiples nombres)
        posibles_stock = ['stock_actual', 'STOCK_ACTUAL', 'stock', 'current_stock', 'inventario']
        stock_col = next((col for col in posibles_stock if col in inventario_df.columns), None)
       
        posibles_familia = ['familia', 'FAMILIA', 'family', 'categoria']
        familia_col = next((col for col in posibles_familia if col in inventario_df.columns), None)
       
        posibles_abc = ['abc_class', 'ABC_CLASS', 'clase_abc', 'claseABC']
        abc_col = next((col for col in posibles_abc if col in inventario_df.columns), None)
       
        # Calcular productos con riesgo
        productos_riesgo = 0
        productos_criticos = 0
       
        if riesgo_df is not None:
            posibles_riesgo = ['riesgo_rotura', 'RIESGO_ROTURA', 'stockout_risk', 'risk_level']
            riesgo_col = next((col for col in posibles_riesgo if col in riesgo_df.columns), None)
           
            if riesgo_col:
                productos_riesgo = len(riesgo_df[riesgo_df[riesgo_col] > 0.5])
                productos_criticos = len(riesgo_df[riesgo_df[riesgo_col] > 0.8])
       
        # Calcular unidades a comprar
        unidades_comprar = 0
        if compras_df is not None:
            posibles_cantidad = ['cantidad_recomendada', 'CANTIDAD_RECOMENDADA', 'qty_recommended', 'quantity']
            qty_col = next((col for col in posibles_cantidad if col in compras_df.columns), None)
           
            if qty_col:
                unidades_comprar = compras_df[qty_col].sum()
       
        # Stock de seguridad total
        stock_seguridad_total = 0
        if stock_col and inventario_df[stock_col].notna().any():
            # Asumir que el 20% del stock es seguridad (estimación)
            stock_seguridad_total = inventario_df[stock_col].sum() * 0.2
       
        # Punto de pedido promedio
        punto_pedido_promedio = 0
        if stock_col:
            punto_pedido_promedio = inventario_df[stock_col].mean() * 1.5  # Estimación
       
        # Mostrar KPIs en columnas
        col1, col2, col3, col4, col5, col6 = st.columns(6)
       
        with col1:
            show_kpi_card("Productos Analizados", total_productos, icon="📦", color="blue")
       
        with col2:
            show_kpi_card("Con Riesgo Rotura", productos_riesgo, icon="⚠️", color="orange")
       
        with col3:
            show_kpi_card("Productos Críticos", productos_criticos, icon="🔴", color="red")
       
        with col4:
            show_kpi_card("Unidades a Comprar", int(unidades_comprar), icon="🛒", color="green")
       
        with col5:
            show_kpi_card("Stock Seguridad", int(stock_seguridad_total), icon="🛡️", color="purple")
       
        with col6:
            show_kpi_card("Punto Pedido Prom.", f"{punto_pedido_promedio:.0f}", icon="📍", color="blue")
       
        st.markdown("---")
       
        # FILA 2: GRÁFICOS
        st.subheader("📊 Visualizaciones Principales")
       
        col_graf1, col_graf2 = st.columns(2)
       
        with col_graf1:
            # Gráfico: Riesgo por Familia
            if familia_col and riesgo_df is not None:
                st.markdown("### Riesgo de Rotura por Familia")
               
                # Agrupar riesgo por familia
                if familia_col in riesgo_df.columns:
                    riesgo_por_familia = riesgo_df.groupby(familia_col)['riesgo_rotura'].mean().sort_values(ascending=False).head(10)
                   
                    fig_riesgo = px.bar(
                        x=riesgo_por_familia.values,
                        y=riesgo_por_familia.index,
                        orientation='h',
                        title='Top 10 Familias con Mayor Riesgo',
                        labels={'x': 'Nivel de Riesgo', 'y': 'Familia'},
                        color=riesgo_por_familia.values,
                        color_continuous_scale='RdYlGn_r'
                    )
                    fig_riesgo.update_layout(height=400)
                    st.plotly_chart(fig_riesgo, use_container_width=True)
           
            # Gráfico: Compras Recomendadas por Familia
            if compras_df is not None and familia_col and familia_col in compras_df.columns:
                st.markdown("### Compras Recomendadas por Familia")
               
                posibles_qty = ['cantidad_recomendada', 'CANTIDAD_RECOMENDADA', 'qty_recommended']
                qty_col = next((col for col in posibles_qty if col in compras_df.columns), None)
               
                if qty_col:
                    compras_por_familia = compras_df.groupby(familia_col)[qty_col].sum().sort_values(ascending=False).head(10)
                   
                    fig_compras = px.bar(
                        x=compras_por_familia.values,
                        y=compras_por_familia.index,
                        orientation='h',
                        title='Top 10 Familias - Compras Recomendadas',
                        labels={'x': 'Unidades a Comprar', 'y': 'Familia'},
                        color=compras_por_familia.values,
                        color_continuous_scale='Blues'
                    )
                    fig_compras.update_layout(height=400)
                    st.plotly_chart(fig_compras, use_container_width=True)
       
        with col_graf2:
            # Distribución ABC
            if abc_col and inventario_df is not None:
                st.markdown("### Distribución ABC")
               
                abc_counts = inventario_df[abc_col].value_counts().sort_index()
               
                fig_abc = px.pie(
                    values=abc_counts.values,
                    names=abc_counts.index,
                    title='Distribución de Productos por Clase ABC',
                    color=abc_counts.index,
                    color_discrete_map={'A': '#E74C3C', 'B': '#F39C12', 'C': '#2ECC71'}
                )
                fig_abc.update_traces(textposition='inside', textinfo='percent+label')
                fig_abc.update_layout(height=400)
                st.plotly_chart(fig_abc, use_container_width=True)
           
            # Top Productos Críticos
            if riesgo_df is not None:
                st.markdown("### Top 10 Productos Críticos")
               
                posibles_riesgo = ['riesgo_rotura', 'RIESGO_ROTURA', 'stockout_risk']
                riesgo_col = next((col for col in posibles_riesgo if col in riesgo_df.columns), None)
                posibles_prod = ['producto', 'PRODUCTO', 'product_id', 'ProductId']
                prod_col = next((col for col in posibles_prod if col in riesgo_df.columns), None)
               
                if riesgo_col and prod_col:
                    top_criticos = riesgo_df.nlargest(10, riesgo_col)[[prod_col, riesgo_col]]
                   
                    fig_top = px.bar(
                        x=top_criticos[riesgo_col],
                        y=top_criticos[prod_col],
                        orientation='h',
                        title='Productos con Mayor Riesgo de Rotura',
                        labels={'x': 'Nivel de Riesgo', 'y': 'Producto'},
                        color=top_criticos[riesgo_col],
                        color_continuous_scale='Reds'
                    )
                    fig_top.update_layout(height=400)
                    st.plotly_chart(fig_top, use_container_width=True)
       
        # Exportar datos
        st.markdown("---")
        st.subheader("📥 Exportar Datos")
       
        col_exp1, col_exp2 = st.columns(2)
       
        with col_exp1:
            if inventario_df is not None:
                csv_data = export_to_csv(inventario_df, "inventario_completo.csv")
                st.download_button(
                    label="📄 Descargar Inventario (CSV)",
                    data=csv_data,
                    file_name="inventario_completo.csv",
                    mime="text/csv",
                    use_container_width=True
                )
       
        with col_exp2:
            if inventario_df is not None:
                excel_data = export_to_excel(inventario_df, "inventario_completo.xlsx")
                st.download_button(
                    label="📊 Descargar Inventario (Excel)",
                    data=excel_data,
                    file_name="inventario_completo.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
   
    else:
        st.warning("⚠️ No se pudo cargar el archivo de inventario. Por favor verifica que el archivo 'inventario_productos.csv' exista en el directorio de datos.")

# =============================================================================
# PÁGINA 2: SIMULADOR INTERACTIVO
# =============================================================================

elif selected_option == "🎛️ Simulador Interactivo":
    st.title("🎛️ Simulador Interactivo de Inventario")
    st.markdown("""
    <div style="text-align: justify;">
        Herramienta de simulación para evaluar diferentes escenarios de inventario.
        Ajusta los parámetros para ver cómo afectan los niveles de stock, puntos de
        pedido y recomendaciones de compra.
    </div>
    """, unsafe_allow_html=True)
   
    st.markdown("---")
   
    # Obtener datos para filtros
    inventario_df = datasets.get('inventario')
   
    if inventario_df is not None:
        # Identificar columnas
        posibles_prod = ['producto', 'PRODUCTO', 'product_id', 'ProductId']
        prod_col = next((col for col in posibles_prod if col in inventario_df.columns), None)
       
        posibles_familia = ['familia', 'FAMILIA', 'family']
        familia_col = next((col for col in posibles_familia if col in inventario_df.columns), None)
       
        posibles_abc = ['abc_class', 'ABC_CLASS', 'clase_abc']
        abc_col = next((col for col in posibles_abc if col in inventario_df.columns), None)
       
        posibles_stock = ['stock_actual', 'STOCK_ACTUAL', 'stock', 'current_stock']
        stock_col = next((col for col in posibles_stock if col in inventario_df.columns), None)
       
        posibles_demanda = ['demanda_diaria', 'DEMANDA_DIARIA', 'daily_demand', 'avg_daily_demand']
        demanda_col = next((col for col in posibles_demanda if col in inventario_df.columns), None)
       
        # FILTROS
        st.subheader("🔍 Filtros de Producto")
       
        col_filtro1, col_filtro2, col_filtro3 = st.columns(3)
       
        with col_filtro1:
            if familia_col:
                familias_disponibles = ['Todas'] + sorted(inventario_df[familia_col].unique().tolist())
                familia_seleccionada = st.selectbox("Familia:", familias_disponibles)
            else:
                familia_seleccionada = 'Todas'
       
        with col_filtro2:
            if abc_col:
                abc_disponibles = ['Todas'] + sorted(inventario_df[abc_col].unique().tolist())
                abc_seleccionado = st.selectbox("Clase ABC:", abc_disponibles)
            else:
                abc_seleccionado = 'Todas'
       
        with col_filtro3:
            if prod_col:
                if familia_seleccionada != 'Todas':
                    productos_filtrados = inventario_df[inventario_df[familia_col] == familia_seleccionada][prod_col].unique().tolist()
                else:
                    productos_filtrados = inventario_df[prod_col].unique().tolist()
               
                producto_seleccionado = st.selectbox("Producto:", ['Todos'] + productos_filtrados)
            else:
                producto_seleccionado = 'Todos'
       
        # Aplicar filtros
        df_filtrado = inventario_df.copy()
       
        if familia_seleccionada != 'Todas' and familia_col:
            df_filtrado = df_filtrado[df_filtrado[familia_col] == familia_seleccionada]
       
        if abc_seleccionado != 'Todas' and abc_col:
            df_filtrado = df_filtrado[df_filtrado[abc_col] == abc_seleccionado]
       
        if producto_seleccionado != 'Todos' and prod_col:
            df_filtrado = df_filtrado[df_filtrado[prod_col] == producto_seleccionado]
       
        st.markdown("---")
       
        # SLIDERS DE PARÁMETROS
        st.subheader("⚙️ Parámetros de Simulación")
       
        col_slider1, col_slider2, col_slider3 = st.columns(3)
       
        with col_slider1:
            lead_time = st.slider(
                "Lead Time (días)",
                min_value=1,
                max_value=30,
                value=7,
                step=1,
                help="Tiempo de entrega del proveedor en días"
            )
           
            nivel_servicio = st.slider(
                "Nivel de Servicio (%)",
                min_value=80.0,
                max_value=99.9,
                value=95.0,
                step=0.1,
                help="Probabilidad de no tener rotura de stock"
            )
       
        with col_slider2:
            variacion_demanda = st.slider(
                "Variación de Demanda (%)",
                min_value=-30,
                max_value=100,
                value=0,
                step=5,
                help="Variación esperada en la demanda (-30% a +100%)"
            )
           
            impacto_promocional = st.slider(
                "Impacto Promocional (%)",
                min_value=0,
                max_value=80,
                value=0,
                step=5,
                help="Impacto adicional por promociones"
            )
       
        with col_slider3:
            variabilidad = st.selectbox(
                "Variabilidad de Demanda",
                options=['Baja', 'Media', 'Alta'],
                index=1,
                help="Nivel de variabilidad histórica de la demanda"
            )
       
        # Convertir porcentajes a decimales
        nivel_servicio_decimal = nivel_servicio / 100
        variacion_demanda_decimal = variacion_demanda / 100
        impacto_promocional_decimal = impacto_promocional / 100
       
        st.markdown("---")
       
        # CÁLCULOS EN TIEMPO REAL
        st.subheader("📊 Resultados de la Simulación")
       
        if demanda_col and stock_col and prod_col:
            # Calcular para cada producto filtrado
            resultados = []
           
            for _, row in df_filtrado.iterrows():
                demanda_base = row.get(demanda_col, 0)
                stock_actual = row.get(stock_col, 0)
                producto = row.get(prod_col, 'Unknown')
                familia = row.get(familia_col, 'N/A') if familia_col else 'N/A'
                abc = row.get(abc_col, 'N/A') if abc_col else 'N/A'
               
                # Demanda ajustada
                demanda_ajustada = calculate_demand_adjustment(
                    demanda_base,
                    variacion_demanda_decimal,
                    impacto_promocional_decimal
                )
               
                # Stock de seguridad
                stock_seguridad = calculate_safety_stock(
                    demanda_ajustada,
                    lead_time,
                    nivel_servicio_decimal,
                    variabilidad
                )
               
                # Punto de pedido
                punto_pedido = calculate_reorder_point(
                    demanda_ajustada,
                    lead_time,
                    stock_seguridad
                )
               
                # Cobertura en días
                cobertura_dias = calculate_stock_coverage_days(
                    stock_actual,
                    demanda_ajustada
                )
               
                # Riesgo de rotura
                riesgo_rotura = calculate_stockout_probability(
                    stock_actual,
                    demanda_ajustada,
                    lead_time,
                    variabilidad
                )
               
                # Fill Rate estimado
                fill_rate = calculate_fill_rate(
                    stock_seguridad,
                    demanda_ajustada,
                    lead_time,
                    variabilidad
                )
               
                # Compra recomendada
                compra_recomendada = max(0, punto_pedido - stock_actual + demanda_ajustada * lead_time)
               
                resultados.append({
                    'Producto': producto,
                    'Familia': familia,
                    'Clase ABC': abc,
                    'Demanda Diaria': round(demanda_ajustada, 2),
                    'Stock Actual': stock_actual,
                    'Stock Seguridad': stock_seguridad,
                    'Punto de Pedido': round(punto_pedido, 2),
                    'Cobertura (días)': cobertura_dias,
                    'Compra Recomendada': round(compra_recomendada, 2),
                    'Riesgo Rotura': round(riesgo_rotura, 4),
                    'Fill Rate': round(fill_rate, 4)
                })
           
            df_resultados = pd.DataFrame(resultados)
           
            # Mostrar resultados agregados
            col_agg1, col_agg2, col_agg3, col_agg4 = st.columns(4)
           
            with col_agg1:
                st.metric(
                    label="Demanda Total Diaria",
                    value=f"{df_resultados['Demanda Diaria'].sum():,.0f} un."
                )
           
            with col_agg2:
                st.metric(
                    label="Stock Seguridad Total",
                    value=f"{df_resultados['Stock Seguridad'].sum():,.0f} un."
                )
           
            with col_agg3:
                st.metric(
                    label="Compra Total Recomendada",
                    value=f"{df_resultados['Compra Recomendada'].sum():,.0f} un."
                )
           
            with col_agg4:
                riesgo_promedio = df_resultados['Riesgo Rotura'].mean()
                st.metric(
                    label="Riesgo Promedio",
                    value=f"{riesgo_promedio*100:.1f}%",
                    delta=f"{'Alto' if riesgo_promedio > 0.3 else 'Bajo'}"
                )
           
            # Mostrar tabla de resultados
            st.markdown("### 📋 Detalle por Producto")
           
            # Formatear tabla para visualización
            df_display = df_resultados.copy()

            # Asegurar que las columnas numéricas sigan siendo numéricas
            df_display['Riesgo Rotura'] = pd.to_numeric(df_display['Riesgo Rotura'], errors='coerce')
            df_display['Fill Rate'] = pd.to_numeric(df_display['Fill Rate'], errors='coerce')
            df_display['Cobertura (días)'] = pd.to_numeric(df_display['Cobertura (días)'], errors='coerce')
            df_display['Demanda Diaria'] = pd.to_numeric(df_display['Demanda Diaria'], errors='coerce')
            df_display['Stock Actual'] = pd.to_numeric(df_display['Stock Actual'], errors='coerce')
            df_display['Stock Seguridad'] = pd.to_numeric(df_display['Stock Seguridad'], errors='coerce')
            df_display['Punto de Pedido'] = pd.to_numeric(df_display['Punto de Pedido'], errors='coerce')
            df_display['Compra Recomendada'] = pd.to_numeric(df_display['Compra Recomendada'], errors='coerce')

            st.dataframe(
                df_display.style.format({
                    'Demanda Diaria': '{:,.2f}',
                    'Stock Actual': '{:,.0f}',
                    'Stock Seguridad': '{:,.2f}',
                    'Punto de Pedido': '{:,.2f}',
                    'Cobertura (días)': '{:,.1f}',
                    'Compra Recomendada': '{:,.2f}',
                    'Riesgo Rotura': '{:.1%}',
                    'Fill Rate': '{:.1%}'
                }).background_gradient(
                    subset=['Riesgo Rotura'],
                    cmap='RdYlGn_r',
                    vmin=0,
                    vmax=1
                ),
                use_container_width=True,
                height=400
            )
           
            # Fórmulas utilizadas
            with st.expander("📐 Ver Fórmulas Utilizadas"):
                st.markdown("""
                #### Fórmulas de Cálculo:
               
                1. **Demanda Ajustada** = Demanda Base × (1 + Variación) × (1 + Impacto Promocional)
               
                2. **Stock de Seguridad** = Z × σ_d × √L
                   - Z = Z-score para nivel de servicio
                   - σ_d = Desviación estándar de demanda diaria
                   - L = Lead time en días
               
                3. **Punto de Pedido** = Demanda Diaria × Lead Time + Stock de Seguridad
               
                4. **Cobertura (días)** = Stock Actual / Demanda Diaria Ajustada
               
                5. **Riesgo de Rotura** = 1 - Φ((Stock Actual - Demanda × L) / (σ_d × √L))
               
                6. **Fill Rate** ≈ 1 - (Función de Pérdida / Cantidad de Pedido)
               
                7. **Compra Recomendada** = max(0, Punto de Pedido - Stock Actual + Demanda × Lead Time)
                """)
           
            # Exportar resultados
            col_exp1, col_exp2 = st.columns(2)
           
            with col_exp1:
                csv_simulacion = export_to_csv(df_resultados, "simulacion_resultados.csv")
                st.download_button(
                    label="📄 Exportar Resultados (CSV)",
                    data=csv_simulacion,
                    file_name="simulacion_resultados.csv",
                    mime="text/csv",
                    use_container_width=True
                )
           
            with col_exp2:
                excel_simulacion = export_to_excel(df_resultados, "simulacion_resultados.xlsx")
                st.download_button(
                    label="📊 Exportar Resultados (Excel)",
                    data=excel_simulacion,
                    file_name="simulacion_resultados.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
       
        else:
            st.warning("⚠️ No se encontraron las columnas necesarias para realizar los cálculos.")
   
    else:
        st.warning("⚠️ No se pudo cargar el archivo de inventario.")

# =============================================================================
# PÁGINA 3: PRODUCTOS CRÍTICOS
# =============================================================================

elif selected_option == "⚠️ Productos Críticos":
    st.title("⚠️ Productos Críticos")
    st.markdown("""
    <div style="text-align: justify;">
        Listado detallado de productos con mayor riesgo de rotura de stock.
        Esta vista permite identificar rápidamente los productos que requieren
        atención inmediata y acciones correctivas.
    </div>
    """, unsafe_allow_html=True)
   
    st.markdown("---")
   
    # Cargar datos
    riesgo_df = datasets.get('riesgo_rotura')
    inventario_df = datasets.get('inventario')
   
    if riesgo_df is not None:
        # Identificar columnas
        posibles_riesgo = ['riesgo_rotura', 'RIESGO_ROTURA', 'stockout_risk']
        riesgo_col = next((col for col in posibles_riesgo if col in riesgo_df.columns), None)
       
        posibles_prod = ['producto', 'PRODUCTO', 'product_id', 'ProductId']
        prod_col = next((col for col in posibles_prod if col in riesgo_df.columns), None)
       
        posibles_familia = ['familia', 'FAMILIA', 'family']
        familia_col = next((col for col in posibles_familia if col in riesgo_df.columns), None)
       
        posibles_abc = ['abc_class', 'ABC_CLASS', 'clase_abc']
        abc_col = next((col for col in posibles_abc if col in riesgo_df.columns), None)
       
        # Filtros
        st.subheader("🔍 Filtros")
       
        col_f1, col_f2, col_f3 = st.columns(3)
       
        with col_f1:
            umbral_riesgo = st.slider(
                "Umbral de Riesgo Mínimo",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Mostrar solo productos con riesgo superior a este valor"
            )
       
        with col_f2:
            if familia_col:
                familias_riesgo = ['Todas'] + sorted(riesgo_df[familia_col].unique().tolist())
                familia_filtro = st.selectbox("Familia:", familias_riesgo)
            else:
                familia_filtro = 'Todas'
       
        with col_f3:
            if abc_col:
                abc_riesgo = ['Todas'] + sorted(riesgo_df[abc_col].unique().tolist())
                abc_filtro = st.selectbox("Clase ABC:", abc_riesgo)
            else:
                abc_filtro = 'Todas'
       
        # Aplicar filtros
        df_riesgo_filtrado = riesgo_df.copy()
       
        if riesgo_col:
            df_riesgo_filtrado = df_riesgo_filtrado[df_riesgo_filtrado[riesgo_col] >= umbral_riesgo]
       
        if familia_filtro != 'Todas' and familia_col:
            df_riesgo_filtrado = df_riesgo_filtrado[df_riesgo_filtrado[familia_col] == familia_filtro]
       
        if abc_filtro != 'Todas' and abc_col:
            df_riesgo_filtrado = df_riesgo_filtrado[df_riesgo_filtrado[abc_col] == abc_filtro]
       
        # Ordenar por riesgo descendente
        if riesgo_col:
            df_riesgo_filtrado = df_riesgo_filtrado.sort_values(riesgo_col, ascending=False)
       
        st.markdown(f"**{len(df_riesgo_filtrado)} productos** cumplen con los criterios de riesgo.")
       
        # Preparar datos para visualización
        columnas_mostrar = []
        if prod_col:
            columnas_mostrar.append(prod_col)
        if familia_col:
            columnas_mostrar.append(familia_col)
        if abc_col:
            columnas_mostrar.append(abc_col)
        if riesgo_col:
            columnas_mostrar.append(riesgo_col)
       
        # Agregar columnas de inventario si están disponibles
        if inventario_df is not None:
            posibles_stock = ['stock_actual', 'STOCK_ACTUAL', 'stock']
            stock_col = next((col for col in posibles_stock if col in inventario_df.columns), None)
           
            posibles_demanda = ['demanda_diaria', 'DEMANDA_DIARIA', 'daily_demand']
            demanda_col = next((col for col in posibles_demanda if col in inventario_df.columns), None)
           
            posibles_seguridad = ['stock_seguridad', 'STOCK_SEGURIDAD', 'safety_stock']
            seguridad_col = next((col for col in posibles_seguridad if col in inventario_df.columns), None)
           
            posibles_punto = ['punto_pedido', 'PUNTO_PEDIDO', 'reorder_point']
            punto_col = next((col for col in posibles_punto if col in inventario_df.columns), None)
           
            posibles_compra = ['compra_recomendada', 'COMPRA_RECOMENDADA', 'recommended_purchase']
            compra_col = next((col for col in posibles_compra if col in inventario_df.columns), None)
       
        # Mostrar tabla interactiva
        st.subheader("📋 Listado de Productos Críticos")
       
        if prod_col and riesgo_col:
            # Seleccionar columnas para mostrar
            cols_tabla = [prod_col, riesgo_col]
           
            if familia_col:
                cols_tabla.append(familia_col)
            if abc_col:
                cols_tabla.append(abc_col)
           
            # Renombrar columnas para mejor legibilidad
            nombres_columnas = {
                prod_col: 'Producto',
                riesgo_col: 'Riesgo Rotura',
                familia_col: 'Familia',
                abc_col: 'Clase ABC'
            }
           
            df_tabla = df_riesgo_filtrado[cols_tabla].copy()
            df_tabla = df_tabla.rename(columns=nombres_columnas)

            # Mantener Riesgo Rotura como numérico
            if 'Riesgo Rotura' in df_tabla.columns:
                df_tabla['Riesgo Rotura'] = pd.to_numeric(df_tabla['Riesgo Rotura'], errors='coerce')

            # Mostrar tabla con formato condicional
            st.dataframe(
                df_tabla.style.format({
                    'Riesgo Rotura': '{:.1%}'
                }).background_gradient(
                    subset=['Riesgo Rotura'],
                    cmap='RdYlGn_r',
                    vmin=0,
                    vmax=1
                ),
                use_container_width=True,
                height=500
            )
           
            # Gráfico de distribución de riesgo
            st.subheader("📊 Distribución de Riesgo")
           
            col_graf1, col_graf2 = st.columns(2)
           
            with col_graf1:
                # Histograma de riesgo
                fig_hist = px.histogram(
                    df_riesgo_filtrado,
                    x=riesgo_col,
                    nbins=20,
                    title='Distribución de Nivel de Riesgo',
                    labels={riesgo_col: 'Nivel de Riesgo'},
                    color_discrete_sequence=['#E74C3C']
                )
                fig_hist.update_layout(height=400)
                st.plotly_chart(fig_hist, use_container_width=True)
           
            with col_graf2:
                # Gráfico de dispersión Riesgo vs Familia
                if familia_col:
                    riesgo_por_familia = df_riesgo_filtrado.groupby(familia_col)[riesgo_col].agg(['mean', 'count'])
                   
                    fig_scatter = px.scatter(
                        riesgo_por_familia,
                        x='mean',
                        y='count',
                        size='count',
                        title='Riesgo Promedio por Familia',
                        labels={'mean': 'Riesgo Promedio', 'count': 'Cantidad de Productos'},
                        color='mean',
                        color_continuous_scale='RdYlGn_r'
                    )
                    fig_scatter.update_layout(height=400)
                    st.plotly_chart(fig_scatter, use_container_width=True)
           
            # Exportar
            st.markdown("---")
            st.subheader("📥 Exportar Listado")
           
            col_exp1, col_exp2 = st.columns(2)
           
            with col_exp1:
                csv_criticos = export_to_csv(df_tabla, "productos_criticos.csv")
                st.download_button(
                    label="📄 Descargar Productos Críticos (CSV)",
                    data=csv_criticos,
                    file_name="productos_criticos.csv",
                    mime="text/csv",
                    use_container_width=True
                )
           
            with col_exp2:
                excel_criticos = export_to_excel(df_tabla, "productos_criticos.xlsx")
                st.download_button(
                    label="📊 Descargar Productos Críticos (Excel)",
                    data=excel_criticos,
                    file_name="productos_criticos.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
       
        else:
            st.warning("⚠️ No se encontraron las columnas necesarias para mostrar los productos críticos.")
   
    else:
        st.warning("⚠️ No se pudo cargar el archivo de riesgo de rotura.")

# =============================================================================
# PÁGINA 4: FORECAST + INVENTARIO
# =============================================================================

elif selected_option == "📈 Forecast + Inventario":
    st.title("📈 Forecast + Inventario")
    st.markdown("""
    <div style="text-align: justify;">
        Análisis combinado de predicciones de demanda (forecast) y niveles de
        inventario actual. Permite identificar desabastecimientos futuros y
        generar alertas automáticas basadas en la proyección de stock.
    </div>
    """, unsafe_allow_html=True)
   
    st.markdown("---")
   
    # Cargar datos
    forecast_7d = datasets.get('forecast_7d')
    forecast_30d = datasets.get('forecast_30d')
    forecast_90d = datasets.get('forecast_90d')
    inventario_df = datasets.get('inventario')
   
    if forecast_7d is not None or forecast_30d is not None or forecast_90d is not None:
        # Selector de horizonte de forecast
        st.subheader("📅 Horizonte de Forecast")
       
        horizonte = st.radio(
            "Selecciona el horizonte de forecast:",
            options=["7 días", "30 días", "90 días"],
            index=0,
            horizontal=True
        )
       
        # Seleccionar dataframe según horizonte
        if horizonte == "7 días":
            forecast_df = forecast_7d
            dias = 7
        elif horizonte == "30 días":
            forecast_df = forecast_30d
            dias = 30
        else:
            forecast_df = forecast_90d
            dias = 90
       
        if forecast_df is not None:
            # Identificar columnas
            posibles_prod = ['producto', 'PRODUCTO', 'product_id', 'ProductId']
            prod_col = next((col for col in posibles_prod if col in forecast_df.columns), None)
           
            posibles_demanda = ['demanda_forecast', 'DEMANDA_FORECAST', 'forecast_demand', 'predicted_demand']
            forecast_col = next((col for col in posibles_demanda if col in forecast_df.columns), None)
           
            # Filtro de producto
            if prod_col:
                productos_forecast = st.multiselect(
                    "Selecciona productos para visualizar:",
                    options=sorted(forecast_df[prod_col].unique().tolist()),
                    default=sorted(forecast_df[prod_col].unique().tolist())[:10],
                    max_selections=20
                )
               
                if productos_forecast:
                    forecast_filtrado = forecast_df[forecast_df[prod_col].isin(productos_forecast)]
                   
                    # Agrupar por producto y mostrar evolución temporal
                    st.subheader(f"📊 Evolución del Forecast ({horizonte})")
                   
                    # Si hay columna de fecha
                    posibles_fecha = ['fecha', 'FECHA', 'date', 'ds']
                    fecha_col = next((col for col in posibles_fecha if col in forecast_filtrado.columns), None)
                   
                    if fecha_col and forecast_col:
                        # Convertir fecha
                        forecast_filtrado[fecha_col] = pd.to_datetime(forecast_filtrado[fecha_col])
                       
                        # Agrupar por fecha
                        forecast_por_fecha = forecast_filtrado.groupby(fecha_col)[forecast_col].sum().reset_index()
                       
                        fig_forecast = px.line(
                            forecast_por_fecha,
                            x=fecha_col,
                            y=forecast_col,
                            title=f'Demanda Forecast Total - {horizonte}',
                            labels={forecast_col: 'Unidades', fecha_col: 'Fecha'},
                            markers=True
                        )
                        fig_forecast.update_layout(height=400)
                        st.plotly_chart(fig_forecast, use_container_width=True)
                       
                        # Mostrar tabla resumen
                        st.subheader("📋 Resumen del Forecast")
                       
                        col_res1, col_res2, col_res3 = st.columns(3)
                       
                        with col_res1:
                            st.metric(
                                label="Demanda Total Forecast",
                                value=f"{forecast_por_fecha[forecast_col].sum():,.0f} un."
                            )
                       
                        with col_res2:
                            st.metric(
                                label="Demanda Diaria Promedio",
                                value=f"{forecast_por_fecha[forecast_col].mean():,.0f} un."
                            )
                       
                        with col_res3:
                            st.metric(
                                label="Día de Máxima Demanda",
                                value=forecast_por_fecha.loc[forecast_por_fecha[forecast_col].idxmax(), fecha_col].strftime('%d/%m/%Y')
                            )
                   
                    # Tabla detallada por producto
                    st.subheader("📋 Detalle por Producto")
                   
                    if prod_col and forecast_col:
                        resumen_productos = forecast_filtrado.groupby(prod_col)[forecast_col].agg(['sum', 'mean', 'max']).reset_index()
                        resumen_productos.columns = ['Producto', 'Demanda Total', 'Demanda Promedio', 'Demanda Máxima']
                       
                        st.dataframe(
                            resumen_productos.style.format({
                                'Demanda Total': '{:,.0f}',
                                'Demanda Promedio': '{:,.1f}',
                                'Demanda Máxima': '{:,.0f}'
                            }).background_gradient(
                                subset=['Demanda Total'],
                                cmap='Blues'
                            ),
                            use_container_width=True,
                            height=400
                        )
                   
                    # Alertas automáticas
                    st.subheader("🚨 Alertas Automáticas")
                   
                    if inventario_df is not None and prod_col in inventario_df.columns:
                        posibles_stock = ['stock_actual', 'STOCK_ACTUAL', 'stock']
                        stock_col = next((col for col in posibles_stock if col in inventario_df.columns), None)
                       
                        if stock_col:
                            # Calcular alertas
                            alertas = []
                           
                            for producto in productos_forecast:
                                # Demanda del producto
                                demanda_producto = forecast_filtrado[forecast_filtrado[prod_col] == producto][forecast_col].sum()
                               
                                # Stock actual
                                stock_producto_df = inventario_df[inventario_df[prod_col] == producto]
                               
                                if not stock_producto_df.empty and stock_col in stock_producto_df.columns:
                                    stock_producto = stock_producto_df[stock_col].values[0]
                                   
                                    # Calcular cobertura
                                    cobertura = stock_producto / (demanda_producto / dias) if demanda_producto > 0 else float('inf')
                                   
                                    if cobertura < 7:
                                        alertas.append({
                                            'Producto': producto,
                                            'Tipo': '🔴 CRÍTICA',
                                            'Mensaje': f'Stock actual ({stock_producto:.0f} un.) cubre solo {cobertura:.1f} días de demanda forecast ({demanda_producto:.0f} un. en {dias} días)',
                                            'Stock': stock_producto,
                                            'Demanda Forecast': demanda_producto,
                                            'Cobertura': cobertura
                                        })
                                    elif cobertura < 14:
                                        alertas.append({
                                            'Producto': producto,
                                            'Tipo': '🟡 PRECAUCIÓN',
                                            'Mensaje': f'Stock actual ({stock_producto:.0f} un.) cubre {cobertura:.1f} días de demanda forecast ({demanda_producto:.0f} un. en {dias} días)',
                                            'Stock': stock_producto,
                                            'Demanda Forecast': demanda_producto,
                                            'Cobertura': cobertura
                                        })
                           
                            if alertas:
                                df_alertas = pd.DataFrame(alertas)
                               
                                # Mostrar alertas críticas primero
                                alertas_criticas = df_alertas[df_alertas['Tipo'].str.contains('CRÍTICA')]
                                alertas_precaucion = df_alertas[df_alertas['Tipo'].str.contains('PRECAUCIÓN')]
                               
                                if not alertas_criticas.empty:
                                    st.error(f"⚠️ **{len(alertas_criticas)} alertas críticas detectadas**")
                                    for _, alerta in alertas_criticas.iterrows():
                                        st.markdown(f"""
                                        <div class="alert-critical">
                                            <strong>{alerta['Producto']}</strong>: {alerta['Mensaje']}
                                        </div>
                                        """, unsafe_allow_html=True)
                               
                                if not alertas_precaucion.empty:
                                    st.warning(f"⚠️ **{len(alertas_precaucion)} alertas de precaución**")
                                    for _, alerta in alertas_precaucion.iterrows():
                                        st.markdown(f"""
                                        <div class="alert-warning">
                                            <strong>{alerta['Producto']}</strong>: {alerta['Mensaje']}
                                        </div>
                                        """, unsafe_allow_html=True)
                            else:
                                st.success("✅ No se detectaron alertas de desabastecimiento para los productos seleccionados.")
                   
                    # Exportar forecast
                    st.markdown("---")
                    st.subheader("📥 Exportar Datos de Forecast")
                   
                    col_exp1, col_exp2 = st.columns(2)
                   
                    with col_exp1:
                        csv_forecast = export_to_csv(forecast_filtrado, f"forecast_{dias}d.csv")
                        st.download_button(
                            label="📄 Descargar Forecast (CSV)",
                            data=csv_forecast,
                            file_name=f"forecast_{dias}d.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                   
                    with col_exp2:
                        excel_forecast = export_to_excel(forecast_filtrado, f"forecast_{dias}d.xlsx")
                        st.download_button(
                            label="📊 Descargar Forecast (Excel)",
                            data=excel_forecast,
                            file_name=f"forecast_{dias}d.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                else:
                    st.info("ℹ️ Selecciona al menos un producto para visualizar.")
           
            else:
                st.warning("⚠️ No se encontraron las columnas necesarias en el forecast.")
       
        else:
            st.warning(f"⚠️ No se encontró archivo de forecast a {dias} días.")
   
    else:
        st.warning("⚠️ No se encontraron archivos de forecast.")

# =============================================================================
# PIE DE PÁGINA
# =============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p><strong>FLOWMAP ANALYTICS</strong> | Simulador de Inventario v1.0</p>
    <p>Herramienta desarrollada para análisis de inventario y planificación de compras</p>
    <p>© 2026 - Todos los derechos reservados</p>
</div>
""", unsafe_allow_html=True)