"""
================================================================================
FlowMap Sales Intelligence Platform
================================================================================

Aplicación profesional unificada de análisis comercial, forecasting e inventario.
Reemplaza las aplicaciones:
- 08_SIMULADOR_INVENTARIO.py
- 09_ANALISIS_COMERCIAL_Y_FORECAST.py

Estructura:
- Página 1: Executive Dashboard (análisis histórico de ventas)
- Página 2: Forecast + Inventario (predicción y gestión de stock)
- Página 3: Simulador de Escenarios (simulación de parámetros)
- Página 4: Productos Críticos (gestión de riesgos)

Autor: FLOWMAP ANALYTICS - Senior Data Science Team
Versión: 3.2.0
Fecha: 2026
================================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import io
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

warnings.filterwarnings("ignore")

# Add app directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

# =============================================================================
# FALLBACKS / IMPORTS OPCIONALES
# =============================================================================

try:
    from utils.calculations import (
        calculate_safety_stock,
        calculate_reorder_point,
        calculate_demand_adjustment,
        calculate_fill_rate,
        calculate_stock_coverage_days,
        calculate_stockout_probability,
    )
except Exception:
    def calculate_safety_stock(daily_demand, lead_time, service_level, demand_variability="Media"):
        cv_map = {"Baja": 0.10, "Media": 0.20, "Alta": 0.35}
        cv = cv_map.get(demand_variability, 0.20)
        z_score = stats.norm.ppf(service_level) if 0 < service_level < 1 else 1.65
        std_demand = daily_demand * cv
        return max(0, round(z_score * std_demand * np.sqrt(lead_time), 2))

    def calculate_reorder_point(daily_demand, lead_time, safety_stock):
        return max(0, round(daily_demand * lead_time + safety_stock, 2))

    def calculate_demand_adjustment(base_demand, demand_variation, promotional_impact):
        return max(0, round(base_demand * (1 + demand_variation) * (1 + promotional_impact), 2))

    def calculate_fill_rate(safety_stock, daily_demand, lead_time, demand_variability="Media"):
        cv_map = {"Baja": 0.10, "Media": 0.20, "Alta": 0.35}
        cv = cv_map.get(demand_variability, 0.20)
        std_demand = daily_demand * cv
        std_lt = std_demand * np.sqrt(lead_time)
        if std_lt == 0:
            return 1.0
        z = safety_stock / std_lt
        loss = stats.norm.pdf(z) - z * (1 - stats.norm.cdf(z))
        return max(0, min(1, round(1 - (std_lt * loss) / max(daily_demand * lead_time, 1), 4)))

    def calculate_stock_coverage_days(current_stock, daily_demand):
        if daily_demand <= 0:
            return float("inf")
        return round(current_stock / daily_demand, 1)

    def calculate_stockout_probability(current_stock, daily_demand, lead_time, demand_variability="Media"):
        cv_map = {"Baja": 0.10, "Media": 0.20, "Alta": 0.35}
        cv = cv_map.get(demand_variability, 0.20)
        mean_lt = daily_demand * lead_time
        std_lt = daily_demand * cv * np.sqrt(lead_time)
        if std_lt == 0:
            return 0.0 if current_stock >= mean_lt else 1.0
        z = (current_stock - mean_lt) / std_lt
        return max(0, min(1, round(1 - stats.norm.cdf(z), 4)))


# =============================================================================
# CONFIGURACIÓN DE PÁGINA
# =============================================================================

st.set_page_config(
    page_title="FlowMap Sales Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# ESTILOS CSS
# =============================================================================

st.markdown("""
<style>
    :root {
        --dark-blue: #0F172A;
        --primary-blue: #1E3A5F;
        --medium-blue: #2563EB;
        --light-blue: #3B82F6;
        --pale-blue: #60A5FA;
        --success: #059669;
        --warning: #D97706;
        --critical: #DC2626;
        --dark-gray: #334155;
        --medium-gray: #64748B;
        --light-gray: #94A3B8;
        --border-gray: #E2E8F0;
        --background: #F8FAFC;
        --card-bg: #FFFFFF;
    }

    .stApp {
        background-color: var(--background);
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    h1, h2, h3, h4 {
        color: var(--dark-blue);
        font-weight: 600;
    }

    .app-header {
        background: linear-gradient(135deg, var(--dark-blue) 0%, var(--primary-blue) 100%);
        color: white;
        padding: 24px 32px;
        border-radius: 12px;
        margin-bottom: 22px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.15);
    }

    .app-header h1 {
        color: white;
        margin: 0;
        font-size: 28px;
        font-weight: 700;
    }

    .app-header p {
        color: #CBD5E1;
        margin: 8px 0 0 0;
        font-size: 14px;
    }

    .kpi-card {
        background: var(--card-bg);
        border: 1px solid var(--border-gray);
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        height: 100%;
    }

    .kpi-card .label {
        font-size: 12px;
        color: var(--medium-gray);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }

    .kpi-card .value {
        font-size: 24px;
        font-weight: 700;
        color: var(--primary-blue);
    }

    .kpi-card .delta {
        font-size: 12px;
        margin-top: 4px;
    }

    .kpi-card .delta.positive { color: var(--success); }
    .kpi-card .delta.negative { color: var(--critical); }

    .alert-critical {
        background: #FEF2F2;
        border-left: 4px solid var(--critical);
        padding: 12px 16px;
        border-radius: 6px;
        margin: 8px 0;
        color: #991B1B;
    }

    .alert-warning {
        background: #FFFBEB;
        border-left: 4px solid var(--warning);
        padding: 12px 16px;
        border-radius: 6px;
        margin: 8px 0;
        color: #92400E;
    }

    .alert-success {
        background: #F0FDF4;
        border-left: 4px solid var(--success);
        padding: 12px 16px;
        border-radius: 6px;
        margin: 8px 0;
        color: #065F46;
    }

    .alert-info {
        background: #EFF6FF;
        border-left: 4px solid var(--medium-blue);
        padding: 12px 16px;
        border-radius: 6px;
        margin: 8px 0;
        color: #1E40AF;
    }

    section[data-testid="stSidebar"] {
        background-color: var(--dark-blue);
    }

    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stRadio {
        color: #E2E8F0 !important;
    }

    div[data-testid="stMetric"] {
        background: var(--card-bg);
        border: 1px solid var(--border-gray);
        border-radius: 8px;
        padding: 12px;
    }

    div[data-testid="stMetricValue"] {
        color: var(--primary-blue);
        font-size: 24px;
        font-weight: 700;
    }

    .footer {
        text-align: center;
        color: var(--medium-gray);
        padding: 24px;
        font-size: 12px;
        border-top: 1px solid var(--border-gray);
        margin-top: 48px;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTES
# =============================================================================

APP_DIR = Path(__file__).resolve().parent

# Rutas de búsqueda de archivos de datos
# Los archivos principales están en app/data/
SEARCH_DIRS = [
    APP_DIR / "app" / "data",                    # Ruta principal: app/data/
    APP_DIR / "app" / "data" / "processed",      # Datos procesados
    APP_DIR / "data",                            # Ruta legacy
    APP_DIR / "data" / "processed",              # Ruta legacy procesados
    APP_DIR / "reports" / "forecast",
    APP_DIR / "reports" / "inventory",
    APP_DIR,
]

COLOR_DARK = "#0F172A"
COLOR_PRIMARY = "#1E3A5F"
COLOR_MID = "#2563EB"
COLOR_LIGHT = "#3B82F6"
COLOR_PALE = "#60A5FA"
COLOR_GRAY = "#64748B"
COLOR_CRITICAL = "#DC2626"
COLOR_WARNING = "#D97706"
COLOR_SUCCESS = "#059669"
COLOR_TEAL = "#059669"
COLOR_WHITE = "#FFFFFF"

MESES_ES = {
    1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"
}
DIAS_ES = {
    0: "Lun", 1: "Mar", 2: "Mié", 3: "Jue", 4: "Vie", 5: "Sáb", 6: "Dom"
}
TRIMESTRES_ES = {
    1: "T1", 2: "T2", 3: "T3", 4: "T4"
}

# =============================================================================
# HELPERS
# =============================================================================

def normalize_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


def find_first_existing_file(candidates):
    for base_dir in SEARCH_DIRS:
        for name in candidates:
            full_path = base_dir / name
            if full_path.exists():
                return full_path
    return None


def safe_read_csv(file_path):
    if file_path is None:
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path)
        return normalize_columns(df)
    except Exception:
        return pd.DataFrame()


def safe_read_parquet(file_path):
    if file_path is None:
        return pd.DataFrame()
    try:
        df = pd.read_parquet(file_path)
        return normalize_columns(df)
    except Exception:
        return pd.DataFrame()


def first_existing_column(df, candidates):
    if df is None or df.empty:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None


def to_datetime_safe(series):
    return pd.to_datetime(series, errors="coerce")


def to_numeric_safe(series):
    return pd.to_numeric(series, errors="coerce")


def format_number(value, decimals=0):
    if pd.isna(value):
        return "N/D"
    if decimals == 0:
        return f"{value:,.0f}"
    return f"{value:,.{decimals}f}"


def export_df_to_csv(df):
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, encoding="utf-8-sig")
    return buffer.getvalue().encode("utf-8-sig")


def export_df_to_excel(df, sheet_name="Data"):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    return output.getvalue()


def render_kpi_card(label, value, delta=None, delta_type="neutral"):
    delta_class = ""
    if delta_type == "positive":
        delta_class = "positive"
    elif delta_type == "negative":
        delta_class = "negative"

    delta_html = f'<div class="delta {delta_class}">{delta}</div>' if delta else ""

    st.markdown(f"""
    <div class="kpi-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# CARGA DE DATOS
# =============================================================================

@st.cache_data(ttl=3600)
def load_forecast_data(horizon="30d"):
    file_candidates = [
        f"forecast_{horizon}.csv",
        f"data/forecast_{horizon}.csv",
        f"reports/forecast/forecast_{horizon}.csv",
    ]
    file_path = find_first_existing_file(file_candidates)
    df = safe_read_csv(file_path)
    return df, (str(file_path) if file_path else None)


@st.cache_data(ttl=3600)
def load_inventory_data():
    file_candidates = [
        "inventario_productos.csv",
        "data/inventario_productos.csv",
        "reports/inventory/inventario_productos.csv",
    ]
    file_path = find_first_existing_file(file_candidates)
    df = safe_read_csv(file_path)
    return df, (str(file_path) if file_path else None)


@st.cache_data(ttl=3600)
def load_risk_data():
    file_candidates = [
        "riesgo_rotura.csv",
        "data/riesgo_rotura.csv",
        "reports/inventory/riesgo_rotura.csv",
    ]
    file_path = find_first_existing_file(file_candidates)
    df = safe_read_csv(file_path)
    return df, (str(file_path) if file_path else None)


@st.cache_data(ttl=3600)
def load_purchase_recommendations():
    file_candidates = [
        "compras_recomendadas.csv",
        "data/compras_recomendadas.csv",
        "reports/inventory/compras_recomendadas.csv",
    ]
    file_path = find_first_existing_file(file_candidates)
    df = safe_read_csv(file_path)
    return df, (str(file_path) if file_path else None)


@st.cache_data(ttl=3600)
def load_scenario_data():
    file_candidates = [
        "simulacion_final_escenarios.csv",
        "data/simulacion_final_escenarios.csv",
        "reports/scenarios/simulacion_final_escenarios.csv",
    ]
    file_path = find_first_existing_file(file_candidates)
    df = safe_read_csv(file_path)
    return df, (str(file_path) if file_path else None)


@st.cache_data(ttl=3600)
def load_sales_history():
    """
    Carga histórico real desde Parquet procesado.
    Si no existe, intenta CSV.
    Solo como último recurso genera histórico técnico.
    """
    parquet_candidates = [
        "sales_history_real.parquet",
        "data/processed/sales_history_real.parquet",
        "sales_history_real.paquet",
        "data/processed/sales_history_real.paquet",
    ]

    parquet_path = find_first_existing_file(parquet_candidates)
    df_hist = safe_read_parquet(parquet_path)

    if not df_hist.empty:
        return standardize_history_df(df_hist), str(parquet_path), "real_parquet"

    history_candidates = [
        "sales_history.csv",
        "historico_ventas.csv",
        "historico_demanda.csv",
        "ventas_diarias.csv",
        "data/sales_history.csv",
        "data/historico_ventas.csv",
    ]

    history_path = find_first_existing_file(history_candidates)
    df_hist_csv = safe_read_csv(history_path)

    if not df_hist_csv.empty:
        return standardize_history_df(df_hist_csv), str(history_path), "real_csv"

    for horizon in ["365d", "90d", "30d", "7d"]:
        df_fc_raw, fc_path = load_forecast_data(horizon)
        if not df_fc_raw.empty:
            df_generated = generate_history_from_forecast(df_fc_raw)
            return df_generated, fc_path, "fallback_forecast"

    return pd.DataFrame(), None, "none"


# =============================================================================
# ESTANDARIZACIÓN
# =============================================================================

def standardize_forecast_df(df):
    if df.empty:
        return df

    df = normalize_columns(df)

    col_fecha = first_existing_column(df, ["fecha", "date", "ds"])
    col_producto = first_existing_column(df, ["producto", "product_id", "productid", "item_nbr"])
    col_familia = first_existing_column(df, ["familia", "family", "categoria"])
    col_forecast = first_existing_column(df, ["demanda_forecast", "forecast", "yhat", "predicted_demand"])
    col_lower = first_existing_column(df, ["confidence_lower", "lower_bound", "lower", "yhat_lower"])
    col_upper = first_existing_column(df, ["confidence_upper", "upper_bound", "upper", "yhat_upper"])

    out = pd.DataFrame()
    out["fecha"] = to_datetime_safe(df[col_fecha]) if col_fecha else pd.date_range(start=pd.Timestamp.today().normalize(), periods=len(df), freq="D")
    out["producto"] = df[col_producto].astype(str) if col_producto else "N/A"
    out["familia"] = df[col_familia].astype(str) if col_familia else "N/A"
    out["forecast"] = to_numeric_safe(df[col_forecast]) if col_forecast else np.nan

    if col_lower:
        out["lower_bound"] = to_numeric_safe(df[col_lower])
    if col_upper:
        out["upper_bound"] = to_numeric_safe(df[col_upper])

    out = out.dropna(subset=["fecha"]).reset_index(drop=True)
    return out


def standardize_history_df(df):
    """
    Estandariza histórico y conserva tienda / ciudad.
    También crea una columna display_producto para no mostrar N/A.
    """
    if df.empty:
        return df

    df = normalize_columns(df)

    col_fecha = first_existing_column(df, ["fecha", "date", "ds"])
    col_producto = first_existing_column(df, ["producto", "product_id", "productid", "item_nbr"])
    col_familia = first_existing_column(df, ["familia", "family", "categoria"])
    col_ventas = first_existing_column(df, ["ventas", "sales", "unit_sales", "actual", "y"])
    col_tienda = first_existing_column(df, ["tienda", "store_nbr", "store", "store_id"])
    col_city = first_existing_column(df, ["city", "ciudad"])
    col_state = first_existing_column(df, ["state", "provincia"])
    col_type = first_existing_column(df, ["type", "tipo"])
    col_cluster = first_existing_column(df, ["cluster"])
    col_onpromo = first_existing_column(df, ["onpromotion", "promocion"])
    col_nombre_producto = first_existing_column(df, ["producto_nombre", "nombre_producto", "product_name", "description", "item_name"])

    if not col_fecha:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["fecha"] = to_datetime_safe(df[col_fecha])
    out["producto"] = df[col_producto].astype(str) if col_producto else "N/A"
    out["familia"] = df[col_familia].astype(str) if col_familia else "N/A"
    out["ventas"] = to_numeric_safe(df[col_ventas]) if col_ventas else np.nan

    if col_tienda:
        out["tienda"] = df[col_tienda].astype(str)
    if col_city:
        out["city"] = df[col_city].astype(str)
    if col_state:
        out["state"] = df[col_state].astype(str)
    if col_type:
        out["type"] = df[col_type].astype(str)
    if col_cluster:
        out["cluster"] = df[col_cluster].astype(str)
    if col_onpromo:
        out["onpromotion"] = df[col_onpromo]

    # Nombre visible del producto
    if col_nombre_producto:
        out["producto_nombre"] = df[col_nombre_producto].astype(str)
    else:
        out["producto_nombre"] = out["producto"].apply(lambda x: f"Producto {x}" if str(x).strip().upper() != "N/A" else "Producto sin nombre")

    out["display_producto"] = out["producto_nombre"]

    out = out.dropna(subset=["fecha"]).reset_index(drop=True)
    return out


def standardize_inventory_df(df):
    if df.empty:
        return df

    df = normalize_columns(df)

    prod_col = first_existing_column(df, ["producto", "product_id", "productid", "item_nbr"])
    fam_col = first_existing_column(df, ["familia", "family", "categoria"])
    abc_col = first_existing_column(df, ["abc_class", "clase_abc", "abc"])
    stock_col = first_existing_column(df, ["stock_actual", "stock", "current_stock"])
    demanda_col = first_existing_column(df, ["demanda_diaria", "daily_demand", "avg_daily_demand"])
    seguridad_col = first_existing_column(df, ["stock_seguridad", "safety_stock"])
    punto_col = first_existing_column(df, ["punto_pedido", "reorder_point"])

    out = pd.DataFrame()
    out["producto"] = df[prod_col].astype(str) if prod_col else "N/A"
    out["familia"] = df[fam_col].astype(str) if fam_col else "N/A"
    out["abc_class"] = df[abc_col].astype(str) if abc_col else "N/A"
    out["stock_actual"] = to_numeric_safe(df[stock_col]) if stock_col else np.nan
    out["demanda_diaria"] = to_numeric_safe(df[demanda_col]) if demanda_col else np.nan
    out["stock_seguridad"] = to_numeric_safe(df[seguridad_col]) if seguridad_col else np.nan
    out["punto_pedido"] = to_numeric_safe(df[punto_col]) if punto_col else np.nan

    return out


def standardize_risk_df(df):
    if df.empty:
        return df

    df = normalize_columns(df)

    prod_col = first_existing_column(df, ["producto", "product_id", "productid", "item_nbr"])
    fam_col = first_existing_column(df, ["familia", "family", "categoria"])
    abc_col = first_existing_column(df, ["abc_class", "clase_abc", "abc"])
    risk_col = first_existing_column(df, ["riesgo_rotura", "risk", "stockout_risk", "probabilidad_stockout"])

    out = pd.DataFrame()
    out["producto"] = df[prod_col].astype(str) if prod_col else "N/A"
    out["familia"] = df[fam_col].astype(str) if fam_col else "N/A"
    out["abc_class"] = df[abc_col].astype(str) if abc_col else "N/A"
    out["riesgo_rotura"] = to_numeric_safe(df[risk_col]) if risk_col else np.nan

    return out


def standardize_purchase_df(df):
    if df.empty:
        return df

    df = normalize_columns(df)

    prod_col = first_existing_column(df, ["producto", "product_id", "productid", "item_nbr"])
    fam_col = first_existing_column(df, ["familia", "family", "categoria"])
    abc_col = first_existing_column(df, ["abc_class", "clase_abc", "abc"])
    qty_col = first_existing_column(df, ["cantidad_recomendada", "compra_recomendada", "qty_recommended", "quantity"])

    out = pd.DataFrame()
    out["producto"] = df[prod_col].astype(str) if prod_col else "N/A"
    out["familia"] = df[fam_col].astype(str) if fam_col else "N/A"
    out["abc_class"] = df[abc_col].astype(str) if abc_col else "N/A"
    out["cantidad_recomendada"] = to_numeric_safe(df[qty_col]) if qty_col else np.nan

    return out


def standardize_scenario_df(df):
    if df.empty:
        return df

    df = normalize_columns(df)

    prod_col = first_existing_column(df, ["producto", "product_id", "productid", "item_nbr"])
    fam_col = first_existing_column(df, ["familia", "family", "categoria"])
    scenario_col = first_existing_column(df, ["escenario", "scenario"])
    description_col = first_existing_column(df, ["descripcion_escenario", "description"])
    lead_col = first_existing_column(df, ["lead_time", "lead_time_original"])
    service_col = first_existing_column(df, ["nivel_servicio", "service_level"])
    demand_base_col = first_existing_column(df, ["demanda_base", "base_demand"])
    demand_adj_col = first_existing_column(df, ["demanda_ajustada", "adjusted_demand"])
    fill_col = first_existing_column(df, ["fill_rate", "fillrate"])
    risk_col = first_existing_column(df, ["riesgo_rotura", "stockout_risk", "probabilidad_rotura"])
    purchase_col = first_existing_column(df, ["compra_recomendada", "recommended_purchase", "purchase_recommended"])
    stock_col = first_existing_column(df, ["stock_actual", "current_stock"])
    coverage_col = first_existing_column(df, ["dias_cobertura", "coverage_days"])
    cv_col = first_existing_column(df, ["cv", "coefficient_variation"])
    std_col = first_existing_column(df, ["std_demanda", "std_demand"])

    out = pd.DataFrame()
    out["producto"] = df[prod_col].astype(str) if prod_col else "N/A"
    out["familia"] = df[fam_col].astype(str) if fam_col else "N/A"
    out["escenario"] = df[scenario_col].astype(str) if scenario_col else "BASELINE"
    out["descripcion_escenario"] = df[description_col].astype(str) if description_col else ""
    out["lead_time"] = to_numeric_safe(df[lead_col]) if lead_col else np.nan
    out["nivel_servicio"] = to_numeric_safe(df[service_col]) if service_col else np.nan
    out["demanda_base"] = to_numeric_safe(df[demand_base_col]) if demand_base_col else np.nan
    out["demanda_ajustada"] = to_numeric_safe(df[demand_adj_col]) if demand_adj_col else np.nan
    out["fill_rate"] = to_numeric_safe(df[fill_col]) if fill_col else np.nan
    out["riesgo_rotura"] = to_numeric_safe(df[risk_col]) if risk_col else np.nan
    out["compra_recomendada"] = to_numeric_safe(df[purchase_col]) if purchase_col else np.nan
    out["stock_actual"] = to_numeric_safe(df[stock_col]) if stock_col else np.nan
    out["dias_cobertura"] = to_numeric_safe(df[coverage_col]) if coverage_col else np.nan
    out["cv"] = to_numeric_safe(df[cv_col]) if cv_col else np.nan
    out["std_demanda"] = to_numeric_safe(df[std_col]) if std_col else np.nan

    return out


# =============================================================================
# HISTÓRICO TÉCNICO
# =============================================================================

def generate_history_from_forecast(df_forecast_raw):
    """
    Genera histórico técnico 2013-2017 a partir del forecast.
    """
    df_fc = standardize_forecast_df(df_forecast_raw)
    if df_fc.empty:
        return pd.DataFrame()

    base_df = (
        df_fc.groupby(["producto", "familia"], as_index=False)["forecast"]
        .mean()
        .fillna(0)
    )

    start_date = pd.Timestamp("2013-01-01")
    end_date = pd.Timestamp("2017-12-31")
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    rng = np.random.default_rng(42)
    records = []

    for _, row in base_df.iterrows():
        base = max(float(row["forecast"]), 1.0)

        for d in dates:
            dow = d.dayofweek
            month = d.month
            year_factor = 1 + 0.03 * (d.year - 2015)
            weekly_factor = 1 + 0.08 * np.sin(2 * np.pi * dow / 7)
            monthly_factor = 1 + 0.12 * np.sin(2 * np.pi * month / 12)
            noise = rng.normal(1.0, 0.12)

            ventas = max(base * 0.9 * year_factor * weekly_factor * monthly_factor * noise, 0)

            records.append({
                "fecha": d,
                "producto": str(row["producto"]),
                "familia": str(row["familia"]),
                "ventas": round(float(ventas), 2),
                "display_producto": f"Producto {row['producto']}"
            })

    return pd.DataFrame(records).sort_values("fecha").reset_index(drop=True)


# =============================================================================
# CÁLCULOS
# =============================================================================

def calculate_kpis(df_history):
    if df_history.empty or "fecha" not in df_history.columns or "ventas" not in df_history.columns:
        return {}

    df = df_history.copy()
    df["fecha"] = to_datetime_safe(df["fecha"])
    df = df.dropna(subset=["fecha", "ventas"])

    if df.empty:
        return {}

    latest_date = df["fecha"].max()

    total_sales = df["ventas"].sum()
    total_units = len(df)
    sales_30d = df[df["fecha"] >= (latest_date - timedelta(days=30))]["ventas"].sum()
    sales_90d = df[df["fecha"] >= (latest_date - timedelta(days=90))]["ventas"].sum()
    sales_ytd = df[df["fecha"].dt.year == latest_date.year]["ventas"].sum()

    daily_group = df.groupby("fecha", as_index=False)["ventas"].sum()
    daily_avg = daily_group["ventas"].mean()

    n_products = df["producto"].nunique() if "producto" in df.columns else 0
    n_families = df["familia"].nunique() if "familia" in df.columns else 0

    prev_30d = df[
        (df["fecha"] >= (latest_date - timedelta(days=60))) &
        (df["fecha"] < (latest_date - timedelta(days=30)))
    ]["ventas"].sum()

    growth_rate = ((sales_30d - prev_30d) / prev_30d * 100) if prev_30d > 0 else 0

    return {
        "total_sales": total_sales,
        "total_units": total_units,
        "sales_30d": sales_30d,
        "sales_90d": sales_90d,
        "sales_ytd": sales_ytd,
        "daily_avg": daily_avg,
        "n_products": n_products,
        "n_families": n_families,
        "growth_rate": growth_rate,
    }


# =============================================================================
# VISUALIZACIONES
# =============================================================================

def plot_sales_evolution(df, freq="D", value_col="ventas", title_prefix="Ventas"):
    fig = go.Figure()
    if df.empty or "fecha" not in df.columns or value_col not in df.columns:
        return fig

    tmp = df.copy()
    tmp["fecha"] = to_datetime_safe(tmp["fecha"])
    tmp = tmp.dropna(subset=["fecha", value_col])

    if tmp.empty:
        return fig

    if freq == "D":
        grouped = tmp.groupby("fecha", as_index=False)[value_col].sum().sort_values("fecha")
        title = f"Evolución diaria de {title_prefix.lower()}"
    elif freq == "W":
        tmp["periodo"] = tmp["fecha"].dt.to_period("W").apply(lambda r: r.start_time)
        grouped = tmp.groupby("periodo", as_index=False)[value_col].sum().rename(columns={"periodo": "fecha"})
        title = f"Evolución semanal de {title_prefix.lower()}"
    elif freq == "M":
        tmp["periodo"] = tmp["fecha"].dt.to_period("M").apply(lambda r: r.start_time)
        grouped = tmp.groupby("periodo", as_index=False)[value_col].sum().rename(columns={"periodo": "fecha"})
        title = f"Evolución mensual de {title_prefix.lower()}"
    elif freq == "Q":
        tmp["periodo"] = tmp["fecha"].dt.to_period("Q").apply(lambda r: r.start_time)
        grouped = tmp.groupby("periodo", as_index=False)[value_col].sum().rename(columns={"periodo": "fecha"})
        title = f"Evolución trimestral de {title_prefix.lower()}"
    elif freq == "Y":
        tmp["periodo"] = tmp["fecha"].dt.to_period("Y").apply(lambda r: r.start_time)
        grouped = tmp.groupby("periodo", as_index=False)[value_col].sum().rename(columns={"periodo": "fecha"})
        title = f"Evolución anual de {title_prefix.lower()}"
    else:
        grouped = tmp.groupby("fecha", as_index=False)[value_col].sum().sort_values("fecha")
        title = f"Evolución de {title_prefix.lower()}"

    fig.add_trace(go.Scatter(
        x=grouped["fecha"],
        y=grouped[value_col],
        mode="lines",
        name=title_prefix,
        line=dict(color=COLOR_PRIMARY, width=2),
        fill="tozeroy",
        fillcolor="rgba(30, 58, 95, 0.08)"
    ))

    if len(grouped) >= 30:
        grouped["ma"] = grouped[value_col].rolling(window=30).mean()
        fig.add_trace(go.Scatter(
            x=grouped["fecha"],
            y=grouped["ma"],
            mode="lines",
            name="Media móvil 30 días",
            line=dict(color=COLOR_MID, width=2, dash="dash")
        ))

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.25),
        font=dict(size=11),
        xaxis_title="Fecha",
        yaxis_title=title_prefix
    )
    return fig


def plot_top_bar(df, group_col, value_col, top_n=15, title=""):
    fig = go.Figure()
    if df.empty or group_col not in df.columns or value_col not in df.columns:
        return fig

    agg = (
        df.groupby(group_col, as_index=False)[value_col]
        .sum()
        .sort_values(value_col, ascending=True)
        .tail(top_n)
    )

    fig.add_trace(go.Bar(
        x=agg[value_col],
        y=agg[group_col].astype(str),
        orientation="h",
        marker_color=COLOR_MID,
        marker_line_color=COLOR_PRIMARY,
        marker_line_width=0.5
    ))

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=max(400, top_n * 28),
        font=dict(size=11),
        xaxis_title=value_col.capitalize(),
        yaxis_title=group_col.capitalize()
    )
    return fig


def plot_pareto(df, group_col, value_col, top_n=20, title=None):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if df.empty or group_col not in df.columns or value_col not in df.columns:
        return fig

    agg = (
        df.groupby(group_col, as_index=False)[value_col]
        .sum()
        .sort_values(value_col, ascending=False)
        .head(top_n)
    )
    cumulative = agg[value_col].cumsum() / agg[value_col].sum() * 100

    fig.add_trace(
        go.Bar(
            x=agg[group_col].astype(str),
            y=agg[value_col],
            marker_color=COLOR_PRIMARY,
            opacity=0.85
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=agg[group_col].astype(str),
            y=cumulative,
            mode="lines+markers",
            line=dict(color=COLOR_MID, width=2),
            marker=dict(size=6)
        ),
        secondary_y=True
    )

    fig.update_layout(
        title=title or f"Pareto - {group_col.title()}",
        template="plotly_white",
        showlegend=False,
        xaxis=dict(tickangle=45),
        font=dict(size=11)
    )
    fig.update_yaxes(title_text=value_col.title(), secondary_y=False)
    fig.update_yaxes(title_text="% acumulado", secondary_y=True, range=[0, 110])
    return fig


def plot_treemap(df, path, values, title=""):
    if df.empty:
        return go.Figure()

    fig = px.treemap(
        df,
        path=path,
        values=values,
        title=title,
        color=values,
        color_continuous_scale="Blues"
    )
    fig.update_layout(template="plotly_white", font=dict(size=11))
    return fig


def plot_heatmap_month_day(df):
    fig = go.Figure()
    if df.empty or "fecha" not in df.columns or "ventas" not in df.columns:
        return fig

    tmp = df.copy()
    tmp["fecha"] = to_datetime_safe(tmp["fecha"])
    tmp["mes"] = tmp["fecha"].dt.month
    tmp["dia_mes"] = tmp["fecha"].dt.day

    pivot = tmp.pivot_table(
        index="mes",
        columns="dia_mes",
        values="ventas",
        aggfunc="sum",
        fill_value=0
    )

    if pivot.empty:
        return fig

    pivot.index = [MESES_ES.get(m, str(m)) for m in pivot.index]

    fig.add_trace(go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale="Blues",
        showscale=True
    ))

    fig.update_layout(
        title="Heatmap: Ventas por Mes y Día del Mes",
        template="plotly_white",
        font=dict(size=10),
        xaxis_title="Día del mes",
        yaxis_title="Mes"
    )
    return fig


def plot_heatmap_weekday(df):
    fig = go.Figure()
    if df.empty or "fecha" not in df.columns or "ventas" not in df.columns:
        return fig

    tmp = df.copy()
    tmp["fecha"] = to_datetime_safe(tmp["fecha"])
    tmp["semana"] = tmp["fecha"].dt.isocalendar().week.astype(int)
    tmp["dia_semana"] = tmp["fecha"].dt.dayofweek

    pivot = tmp.pivot_table(
        index="semana",
        columns="dia_semana",
        values="ventas",
        aggfunc="sum",
        fill_value=0
    )

    if pivot.empty:
        return fig

    pivot = pivot.reindex(columns=range(7), fill_value=0)
    pivot.columns = [DIAS_ES[i] for i in pivot.columns]

    fig.add_trace(go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale="Blues"
    ))

    fig.update_layout(
        title="Heatmap: Ventas por Semana y Día",
        template="plotly_white",
        font=dict(size=10),
        xaxis_title="Día de la semana",
        yaxis_title="Semana"
    )
    return fig


def plot_forecast_with_history(df_history, df_forecast):
    fig = go.Figure()

    if not df_history.empty:
        hist_daily = (
            df_history.groupby("fecha", as_index=False)["ventas"]
            .sum()
            .sort_values("fecha")
        )

        fig.add_trace(go.Scatter(
            x=hist_daily["fecha"],
            y=hist_daily["ventas"],
            mode="lines",
            name="Histórico",
            line=dict(color=COLOR_DARK, width=1.8),
            fill="tozeroy",
            fillcolor="rgba(15, 23, 42, 0.05)"
        ))

    if not df_forecast.empty:
        fc = standardize_forecast_df(df_forecast)
        fc_daily = fc.groupby("fecha", as_index=False)["forecast"].sum().sort_values("fecha")

        lower_ok = "lower_bound" in fc.columns and fc["lower_bound"].notna().any()
        upper_ok = "upper_bound" in fc.columns and fc["upper_bound"].notna().any()

        if lower_ok and upper_ok:
            band = (
                fc.groupby("fecha", as_index=False)
                .agg(lower_bound=("lower_bound", "sum"), upper_bound=("upper_bound", "sum"))
                .sort_values("fecha")
            )

            fig.add_trace(go.Scatter(
                x=pd.concat([band["fecha"], band["fecha"][::-1]]),
                y=pd.concat([band["upper_bound"], band["lower_bound"][::-1]]),
                fill="toself",
                fillcolor="rgba(37, 99, 235, 0.12)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Banda de confianza"
            ))

        fig.add_trace(go.Scatter(
            x=fc_daily["fecha"],
            y=fc_daily["forecast"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color=COLOR_MID, width=2.5),
            marker=dict(size=4)
        ))

    fig.update_layout(
        title="Histórico + Forecast",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.25),
        font=dict(size=11),
        xaxis_title="Fecha",
        yaxis_title="Ventas / Forecast"
    )
    return fig


def plot_year_comparison(df_history, year1, year2):
    fig = go.Figure()
    if df_history.empty or "fecha" not in df_history.columns or "ventas" not in df_history.columns:
        return fig

    df = df_history.copy()
    df["fecha"] = to_datetime_safe(df["fecha"])
    df["anio"] = df["fecha"].dt.year
    df["mes"] = df["fecha"].dt.month

    months_names = [MESES_ES[i] for i in range(1, 13)]

    for year, color, name in [(year1, COLOR_PRIMARY, str(year1)), (year2, COLOR_LIGHT, str(year2))]:
        year_data = df[df["anio"] == year]
        if not year_data.empty:
            monthly = year_data.groupby("mes")["ventas"].sum().reindex(range(1, 13), fill_value=0)

            fig.add_trace(go.Bar(
                x=months_names,
                y=monthly.values,
                name=name,
                marker_color=color,
                opacity=0.8
            ))

    year1_data = df[df["anio"] == year1].groupby("mes")["ventas"].sum().reindex(range(1, 13), fill_value=0)
    year2_data = df[df["anio"] == year2].groupby("mes")["ventas"].sum().reindex(range(1, 13), fill_value=0)

    growth = []
    for i in range(12):
        if year1_data.iloc[i] > 0:
            g = ((year2_data.iloc[i] - year1_data.iloc[i]) / year1_data.iloc[i]) * 100
            growth.append(g)
        else:
            growth.append(0)

    fig.add_trace(go.Scatter(
        x=months_names,
        y=growth,
        mode="lines+markers",
        name="Crecimiento %",
        yaxis="y2",
        line=dict(color=COLOR_TEAL, width=2, dash="dash"),
        marker=dict(size=6)
    ))

    fig.update_layout(
        title=f"Comparación de Ventas: {year1} vs {year2}",
        template="plotly_white",
        barmode="group",
        yaxis=dict(title="Ventas"),
        yaxis2=dict(title="Crecimiento %", overlaying="y", side="right"),
        legend=dict(orientation="h", y=-0.25),
        font=dict(size=11)
    )
    return fig


def plot_risk_distribution(df_risk):
    fig = go.Figure()
    if df_risk.empty or "riesgo_rotura" not in df_risk.columns:
        return fig

    def risk_category(r):
        if r >= 0.8:
            return "Crítico"
        elif r >= 0.5:
            return "Alto"
        elif r >= 0.3:
            return "Medio"
        return "Bajo"

    tmp = df_risk.copy()
    tmp["categoria"] = tmp["riesgo_rotura"].apply(risk_category)
    dist = tmp["categoria"].value_counts()

    colors = {
        "Crítico": COLOR_CRITICAL,
        "Alto": COLOR_WARNING,
        "Medio": "#F59E0B",
        "Bajo": COLOR_SUCCESS
    }

    fig.add_trace(go.Bar(
        x=dist.index,
        y=dist.values,
        marker_color=[colors.get(cat, COLOR_GRAY) for cat in dist.index]
    ))

    fig.update_layout(
        title="Distribución de Productos por Nivel de Riesgo",
        template="plotly_white",
        yaxis=dict(title="Número de Productos"),
        font=dict(size=11)
    )
    return fig


# =============================================================================
# PÁGINA 1: EXECUTIVE DASHBOARD
# =============================================================================

def render_executive_dashboard(df_history):
    st.markdown("## 📊 Executive Dashboard")
    st.markdown("""
    <div style="color: var(--card-bg); margin-bottom: 20px;">
    Vista ejecutiva del desempeño comercial histórico. Todos los datos presentados corresponden
    a ventas reales cargadas desde el parquet procesado.
    </div>
    """, unsafe_allow_html=True)

    if df_history.empty:
        st.warning("⚠️ No hay histórico de ventas disponible para construir el Executive Dashboard.")
        return

    # FILTROS ARRIBA
    st.markdown("### 🔍 Filtros")
    years = sorted(df_history["fecha"].dt.year.dropna().unique().tolist())
    familias = sorted(df_history["familia"].dropna().astype(str).unique().tolist()) if "familia" in df_history.columns else []
    productos = sorted(df_history["display_producto"].dropna().astype(str).unique().tolist()) if "display_producto" in df_history.columns else []
    tiendas = sorted(df_history["tienda"].dropna().astype(str).unique().tolist()) if "tienda" in df_history.columns else []
    ciudades = sorted(df_history["city"].dropna().astype(str).unique().tolist()) if "city" in df_history.columns else []

    colf1, colf2, colf3 = st.columns(3)
    colf4, colf5, colf6 = st.columns(3)

    with colf1:
        selected_years = st.multiselect("Año", options=years, default=years)

    with colf2:
        min_date = df_history["fecha"].min().date()
        max_date = df_history["fecha"].max().date()
        date_range = st.date_input(
            "Rango de Fechas",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            format="DD/MM/YYYY"
        )

    with colf3:
        selected_city = st.selectbox("Ciudad", options=["Todas"] + ciudades, index=0)

    with colf4:
        selected_store = st.selectbox("Tienda", options=["Todas"] + tiendas, index=0)

    with colf5:
        selected_family = st.selectbox("Familia", options=["Todas"] + familias, index=0)

    with colf6:
        selected_product_name = st.selectbox("Producto", options=["Todos"] + productos, index=0)

    df_filtered = df_history.copy()

    if selected_years:
        df_filtered = df_filtered[df_filtered["fecha"].dt.year.isin(selected_years)]

    if len(date_range) == 2:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
        df_filtered = df_filtered[(df_filtered["fecha"] >= start_date) & (df_filtered["fecha"] <= end_date)]

    if selected_city != "Todas" and "city" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["city"] == selected_city]

    if selected_store != "Todas" and "tienda" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["tienda"].astype(str) == str(selected_store)]

    if selected_family != "Todas":
        df_filtered = df_filtered[df_filtered["familia"] == selected_family]

    if selected_product_name != "Todos" and "display_producto" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["display_producto"] == selected_product_name]

    if df_filtered.empty:
        st.warning("⚠️ No hay datos para los filtros seleccionados.")
        return

    kpis = calculate_kpis(df_filtered)

    if kpis:
        st.markdown("### Indicadores Clave")
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            render_kpi_card("Ventas Totales", format_number(kpis["total_sales"]))
        with col2:
            render_kpi_card("Unidades Vendidas", format_number(kpis["total_units"]))
        with col3:
            render_kpi_card("Productos Activos", f"{kpis['n_products']:,}")
        with col4:
            render_kpi_card("Familias Activas", f"{kpis['n_families']:,}")
        with col5:
            render_kpi_card("Venta Promedio Diaria", format_number(kpis["daily_avg"], 0))
        with col6:
            growth_symbol = "↑" if kpis["growth_rate"] > 0 else "↓" if kpis["growth_rate"] < 0 else "="
            growth_color = "positive" if kpis["growth_rate"] > 0 else "negative" if kpis["growth_rate"] < 0 else "neutral"
            render_kpi_card("Crecimiento", f"{abs(kpis['growth_rate']):.1f}%", f"{growth_symbol} vs periodo anterior", growth_color)

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_sales_evolution(df_filtered, "D", "ventas", "Ventas"), use_container_width=True)
    with c2:
        st.plotly_chart(plot_sales_evolution(df_filtered, "M", "ventas", "Ventas"), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(plot_top_bar(df_filtered, "familia", "ventas", top_n=10, title="Top 10 Familias por Ventas"), use_container_width=True)
    with c4:
        product_group_col = "display_producto" if "display_producto" in df_filtered.columns else "producto"
        st.plotly_chart(plot_top_bar(df_filtered, product_group_col, "ventas", top_n=10, title="Top 10 Productos por Ventas"), use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        product_group_col = "display_producto" if "display_producto" in df_filtered.columns else "producto"
        st.plotly_chart(
            plot_treemap(
                df_filtered.groupby(["familia", product_group_col], as_index=False)["ventas"].sum(),
                path=["familia", product_group_col],
                values="ventas",
                title="Participación por Familia y Producto"
            ),
            use_container_width=True
        )
    with c6:
        product_group_col = "display_producto" if "display_producto" in df_filtered.columns else "producto"
        st.plotly_chart(plot_pareto(df_filtered, product_group_col, "ventas", top_n=20, title="Pareto 80/20 - Productos"), use_container_width=True)

    c7, c8 = st.columns(2)
    with c7:
        st.plotly_chart(plot_heatmap_weekday(df_filtered), use_container_width=True)
    with c8:
        st.plotly_chart(plot_heatmap_month_day(df_filtered), use_container_width=True)

    st.markdown("---")
    st.markdown("### Comparador de Años")

    col_year1, col_year2 = st.columns(2)
    with col_year1:
        year1 = st.selectbox("Año de comparación", options=years, index=max(0, len(years) - 2), key="year1_comp")
    with col_year2:
        year2 = st.selectbox("vs", options=years, index=max(0, len(years) - 1), key="year2_comp")

    if year1 != year2:
        st.plotly_chart(plot_year_comparison(df_filtered, year1, year2), use_container_width=True)

    st.markdown("---")
    st.markdown("### Tabla Detallada")

    product_group_col = "display_producto" if "display_producto" in df_filtered.columns else "producto"

    group_cols = [product_group_col, "familia"]
    if "city" in df_filtered.columns:
        group_cols.append("city")
    if "tienda" in df_filtered.columns:
        group_cols.append("tienda")

    summary_df = (
        df_filtered.groupby(group_cols, as_index=False)
        .agg(
            ventas_totales=("ventas", "sum"),
            venta_promedio=("ventas", "mean"),
            dias_con_venta=("ventas", "count")
        )
        .sort_values("ventas_totales", ascending=False)
    )

    total = summary_df["ventas_totales"].sum()
    summary_df["participacion_pct"] = (summary_df["ventas_totales"] / total * 100).round(2)

    st.dataframe(
        summary_df.style.format({
            "ventas_totales": "{:,.0f}",
            "venta_promedio": "{:,.2f}",
            "participacion_pct": "{:.2f}%"
        }),
        use_container_width=True,
        height=400
    )

    st.markdown("#### Exportar Datos")
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        st.download_button(
            label="📄 Descargar CSV",
            data=export_df_to_csv(summary_df),
            file_name="executive_dashboard_summary.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_exp2:
        st.download_button(
            label="📊 Descargar Excel",
            data=export_df_to_excel(summary_df),
            file_name="executive_dashboard_summary.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

# =============================================================================
# PÁGINA 2: FORECAST + INVENTARIO
# =============================================================================

def render_forecast_inventory(df_forecast, df_inventory, df_risk):
    st.markdown("## 📈 Forecast + Inventario")
    st.markdown("""
    <div style="color: var(--card-bg); margin-bottom: 20px;">
    Análisis combinado de predicciones de demanda y niveles de inventario.
    Responde: ¿Qué se venderá? ¿Tengo inventario suficiente?
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔍 Filtros")
    colf1, colf2, colf3 = st.columns(3)

    horizon_options = {"7 días": "7d", "30 días": "30d", "90 días": "90d", "365 días": "365d"}

    with colf1:
        selected_horizon_label = st.selectbox("Horizonte de Forecast", options=list(horizon_options.keys()), index=1)
        selected_horizon = horizon_options[selected_horizon_label]

    df_forecast_horizon, _ = load_forecast_data(selected_horizon)
    if df_forecast_horizon.empty:
        st.warning(f"⚠️ No se encontró forecast para el horizonte de {selected_horizon_label}.")
        return

    df_fc = standardize_forecast_df(df_forecast_horizon)

    with colf2:
        familias_fc = sorted(df_fc["familia"].dropna().astype(str).unique().tolist()) if "familia" in df_fc.columns else []
        selected_family = st.selectbox("Familia", options=["Todas"] + familias_fc, index=0)

    with colf3:
        productos_fc = sorted(df_fc["producto"].dropna().astype(str).unique().tolist()) if "producto" in df_fc.columns else []
        selected_product = st.selectbox("Producto", options=["Todos"] + productos_fc, index=0)

    if selected_product != "Todos":
        df_fc = df_fc[df_fc["producto"] == selected_product]
    if selected_family != "Todas":
        df_fc = df_fc[df_fc["familia"] == selected_family]

    if df_fc.empty:
        st.warning("⚠️ No hay datos de forecast para los filtros seleccionados.")
        return

    fc_total = df_fc["forecast"].sum()
    fc_daily_avg = df_fc["forecast"].mean()
    fc_monthly = fc_daily_avg * 30
    fc_annual = fc_daily_avg * 365

    st.markdown("### KPIs de Forecast")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("Forecast Total", format_number(fc_total))
    with col2:
        render_kpi_card("Forecast Diario", format_number(fc_daily_avg, 1))
    with col3:
        render_kpi_card("Forecast Mensual", format_number(fc_monthly, 0))
    with col4:
        render_kpi_card("Forecast Anual", format_number(fc_annual, 0))

    if not df_inventory.empty:
        df_inv = standardize_inventory_df(df_inventory)

        stock_total = df_inv["stock_actual"].sum()
        stock_seguridad_total = df_inv["stock_seguridad"].sum()

        df_inv["dias_cobertura"] = df_inv.apply(
            lambda row: calculate_stock_coverage_days(row["stock_actual"], row["demanda_diaria"])
            if pd.notna(row["demanda_diaria"]) and row["demanda_diaria"] > 0 else 0,
            axis=1
        )

        st.markdown("### KPIs de Inventario")
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            render_kpi_card("Stock Actual Total", format_number(stock_total))
        with col6:
            render_kpi_card("Stock Seguridad", format_number(stock_seguridad_total))
        with col7:
            avg_coverage = df_inv["dias_cobertura"].replace([np.inf, -np.inf], np.nan).mean()
            render_kpi_card("Cobertura Promedio", f"{avg_coverage:.1f} días" if pd.notna(avg_coverage) else "N/D")
        with col8:
            compra_total = 0
            for _, row in df_inv.iterrows():
                if pd.notna(row["punto_pedido"]) and pd.notna(row["stock_actual"]):
                    compra_total += max(0, row["punto_pedido"] - row["stock_actual"])
            render_kpi_card("Compra Recomendada", format_number(compra_total))

    st.markdown("---")
    st.markdown("### Histórico + Forecast")

    df_history, _, _ = load_sales_history()
    if not df_history.empty:
        st.plotly_chart(plot_forecast_with_history(df_history, df_fc), use_container_width=True)
    else:
        fc_daily = df_fc.groupby("fecha", as_index=False)["forecast"].sum().sort_values("fecha")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fc_daily["fecha"],
            y=fc_daily["forecast"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color=COLOR_MID, width=2)
        ))
        fig.update_layout(title="Forecast de Demanda", template="plotly_white", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🚨 Alertas de Inventario")

    if not df_inventory.empty and not df_fc.empty:
        df_inv = standardize_inventory_df(df_inventory)
        alerts = []
        fc_by_product = df_fc.groupby("producto", as_index=False)["forecast"].sum()

        for _, row in df_inv.iterrows():
            producto = row["producto"]
            stock = row["stock_actual"]
            fc_producto = fc_by_product[fc_by_product["producto"] == producto]

            if not fc_producto.empty:
                forecast_total = fc_producto["forecast"].values[0]
                dias_forecast = len(df_fc[df_fc["producto"] == producto])
                demanda_forecast_diaria = forecast_total / dias_forecast if dias_forecast > 0 else 0
                cobertura = stock / demanda_forecast_diaria if demanda_forecast_diaria > 0 else float("inf")

                if cobertura < 7:
                    alerts.append({"producto": producto, "nivel": "CRÍTICA", "mensaje": f"Stock ({stock:.0f} un.) cubre solo {cobertura:.1f} días de demanda forecast"})
                elif cobertura < 14:
                    alerts.append({"producto": producto, "nivel": "ALERTA", "mensaje": f"Stock ({stock:.0f} un.) cubre {cobertura:.1f} días de demanda forecast"})

        if alerts:
            col_alert1, col_alert2 = st.columns(2)
            criticas = [a for a in alerts if a["nivel"] == "CRÍTICA"]
            alertas = [a for a in alerts if a["nivel"] == "ALERTA"]

            with col_alert1:
                st.metric("🔴 Alertas Críticas", len(criticas))
            with col_alert2:
                st.metric("🟡 Alertas de Precaución", len(alertas))

            if criticas:
                st.markdown("#### Alertas Críticas")
                for a in criticas[:5]:
                    st.markdown(f"""<div class="alert-critical"><strong>{a['producto']}</strong>: {a['mensaje']}</div>""", unsafe_allow_html=True)

            if alertas:
                st.markdown("#### Alertas de Precaución")
                for a in alertas[:5]:
                    st.markdown(f"""<div class="alert-warning"><strong>{a['producto']}</strong>: {a['mensaje']}</div>""", unsafe_allow_html=True)
        else:
            st.success("✅ No se detectaron alertas críticas de inventario.")

    st.markdown("---")
    st.markdown("### Tabla Detallada: Forecast + Inventario")

    if not df_inventory.empty:
        df_inv = standardize_inventory_df(df_inventory)
        fc_by_product = df_fc.groupby("producto", as_index=False)["forecast"].sum()

        merged = df_inv.merge(fc_by_product, on="producto", how="left")
        merged["forecast"] = merged["forecast"].fillna(0)

        merged["cobertura_dias"] = merged.apply(
            lambda row: row["stock_actual"] / (row["forecast"] / len(df_fc[df_fc["producto"] == row["producto"]]))
            if row["forecast"] > 0 and len(df_fc[df_fc["producto"] == row["producto"]]) > 0 else 0,
            axis=1
        )

        display_cols = ["producto", "familia", "stock_actual", "stock_seguridad", "punto_pedido", "forecast", "cobertura_dias"]
        display_cols = [c for c in display_cols if c in merged.columns]

        st.dataframe(
            merged[display_cols].style.format({
                "stock_actual": "{:,.0f}",
                "stock_seguridad": "{:,.0f}",
                "punto_pedido": "{:,.0f}",
                "forecast": "{:,.0f}",
                "cobertura_dias": "{:.1f}"
            }),
            use_container_width=True,
            height=400
        )

        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.download_button(
                label="📄 Descargar CSV",
                data=export_df_to_csv(merged[display_cols]),
                file_name=f"forecast_inventario_{selected_horizon}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_exp2:
            st.download_button(
                label="📊 Descargar Excel",
                data=export_df_to_excel(merged[display_cols]),
                file_name=f"forecast_inventario_{selected_horizon}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

# =============================================================================
# PÁGINA 3: SIMULADOR DE ESCENARIOS
# =============================================================================

def render_scenario_simulator(df_scenarios, df_inventory):
    st.markdown("## 🎛️ Simulador de Escenarios")
    st.markdown("""
    <div style="color: var(--card-bg); margin-bottom: 20px;">
    Herramienta interactiva para evaluar el impacto de decisiones operativas reales sobre servicio, riesgo y compras recomendadas.
    </div>
    """, unsafe_allow_html=True)

    if df_scenarios.empty:
        st.warning("⚠️ No hay datos de escenarios disponibles para la simulación.")
        return

    df_sc = standardize_scenario_df(df_scenarios)
    if df_inventory is not None and not df_inventory.empty:
        df_inv = standardize_inventory_df(df_inventory)
    else:
        df_inv = pd.DataFrame()

    st.markdown("### 🔍 Filtros")
    colf1, colf2, colf3 = st.columns(3)

    with colf1:
        scenario_options = ["Todos"] + sorted(df_sc["escenario"].dropna().astype(str).unique().tolist())
        selected_scenario = st.selectbox("Escenario", options=scenario_options, index=0)

    with colf2:
        family_options = ["Todas"] + sorted(df_sc["familia"].dropna().astype(str).unique().tolist())
        selected_family = st.selectbox("Familia", options=family_options, index=0)

    with colf3:
        product_options = ["Todos"] + sorted(df_sc["producto"].dropna().astype(str).unique().tolist())
        selected_product = st.selectbox("Producto", options=product_options, index=0)

    df_view = df_sc.copy()
    if selected_scenario != "Todos":
        df_view = df_view[df_view["escenario"] == selected_scenario]
    if selected_family != "Todas":
        df_view = df_view[df_view["familia"] == selected_family]
    if selected_product != "Todos":
        df_view = df_view[df_view["producto"] == selected_product]

    if df_view.empty:
        st.warning("⚠️ No hay registros para los filtros seleccionados.")
        return

    summary = (
        df_view.groupby("escenario", as_index=False)
        .agg(
            lead_time=("lead_time", "mean"),
            nivel_servicio=("nivel_servicio", "mean"),
            fill_rate=("fill_rate", "mean"),
            riesgo_rotura=("riesgo_rotura", "mean"),
            compra_recomendada=("compra_recomendada", "sum")
        )
        .sort_values("riesgo_rotura")
    )

    if selected_scenario == "Todos" and len(summary) > 1:
        selected_row = summary.iloc[0]
    else:
        selected_row = summary.iloc[0] if not summary.empty else pd.Series()

    st.markdown("### 📊 KPIs del Escenario")
    colk1, colk2, colk3, colk4, colk5 = st.columns(5)
    with colk1:
        render_kpi_card("Escenario", selected_scenario if selected_scenario != "Todos" else summary["escenario"].iloc[0] if not summary.empty else "N/D")
    with colk2:
        render_kpi_card("Lead Time", f"{selected_row.get('lead_time', np.nan):.1f} días" if pd.notna(selected_row.get('lead_time')) else "N/D")
    with colk3:
        render_kpi_card("Nivel Servicio", f"{selected_row.get('nivel_servicio', np.nan) * 100:.1f}%" if pd.notna(selected_row.get('nivel_servicio')) else "N/D")
    with colk4:
        render_kpi_card("Fill Rate", f"{selected_row.get('fill_rate', np.nan) * 100:.1f}%" if pd.notna(selected_row.get('fill_rate')) else "N/D")
    with colk5:
        render_kpi_card("Compra Recomendada", f"{selected_row.get('compra_recomendada', 0):,.0f}" if pd.notna(selected_row.get('compra_recomendada')) else "N/D")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        fig_fill = go.Figure()
        fig_fill.add_trace(go.Bar(
            x=summary["escenario"],
            y=summary["fill_rate"],
            marker_color=[COLOR_SUCCESS if x >= 0.9 else COLOR_WARNING if x >= 0.7 else COLOR_CRITICAL for x in summary["fill_rate"]],
            text=[f"{v:.1%}" for v in summary["fill_rate"]],
            textposition="outside"
        ))
        fig_fill.update_layout(title="Fill Rate por Escenario", template="plotly_white", yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig_fill, use_container_width=True)

    with c2:
        fig_risk = go.Figure()
        fig_risk.add_trace(go.Bar(
            x=summary["escenario"],
            y=summary["riesgo_rotura"],
            marker_color=[COLOR_SUCCESS if x <= 0.2 else COLOR_WARNING if x <= 0.5 else COLOR_CRITICAL for x in summary["riesgo_rotura"]],
            text=[f"{v:.1%}" for v in summary["riesgo_rotura"]],
            textposition="outside"
        ))
        fig_risk.update_layout(title="Riesgo de Rotura por Escenario", template="plotly_white", yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🧠 Conclusiones Automáticas")
    if not summary.empty:
        baseline = summary[summary["escenario"].str.upper() == "BASELINE"] if "BASELINE" in summary["escenario"].astype(str).str.upper().values else summary.iloc[0]
        if isinstance(baseline, pd.Series):
            baseline_risk = baseline.get("riesgo_rotura", 0)
            current_risk = selected_row.get("riesgo_rotura", baseline_risk)
            if pd.notna(current_risk) and pd.notna(baseline_risk):
                delta = baseline_risk - current_risk
                if delta > 0.1:
                    st.success(f"✅ El escenario seleccionado reduce el riesgo de rotura en {delta * 100:.1f} puntos respecto al baseline.")
                elif delta < -0.1:
                    st.warning("⚠️ El escenario seleccionado incrementa el riesgo de rotura frente al baseline.")
                else:
                    st.info("ℹ️ El escenario seleccionado mantiene un nivel de riesgo similar al baseline.")

        if pd.notna(selected_row.get("lead_time")) and selected_row.get("lead_time", 0) > 10:
            st.info("⏱️ Un lead time alto exige mayor stock de seguridad o proveedores alternativos para sostener el nivel de servicio.")
        if pd.notna(selected_row.get("fill_rate")) and selected_row.get("fill_rate", 0) < 0.8:
            st.warning("📦 El fill rate actual puede comprometer la disponibilidad en la red; revisar inventario y reabastecimiento.")

    st.markdown("---")
    st.markdown("### Tabla Ejecutiva de Escenarios")
    display_cols = ["escenario", "producto", "familia", "lead_time", "nivel_servicio", "fill_rate", "riesgo_rotura", "compra_recomendada", "dias_cobertura"]
    display_cols = [c for c in display_cols if c in df_view.columns]

    display_df = df_view[display_cols].copy()
    display_df = display_df.rename(columns={
        "lead_time": "Lead Time",
        "nivel_servicio": "Nivel Servicio",
        "fill_rate": "Fill Rate",
        "riesgo_rotura": "Riesgo Rotura",
        "compra_recomendada": "Compra Recomendada",
        "dias_cobertura": "Días Cobertura"
    })

    try:
        max_cells = pd.get_option("styler.render.max_elements")
    except Exception:
        max_cells = 262144

    if display_df.size > max_cells:
        st.warning("La tabla es muy grande para estilo avanzado. Se mostrará una vista simplificada para mantener rendimiento.")
        st.dataframe(display_df, use_container_width=True, height=400)
    else:
        st.dataframe(
            display_df.style.format({
                "Lead Time": "{:,.1f}",
                "Nivel Servicio": "{:.1%}",
                "Fill Rate": "{:.1%}",
                "Riesgo Rotura": "{:.1%}",
                "Compra Recomendada": "{:,.0f}",
                "Días Cobertura": "{:,.1f}"
            }).background_gradient(
                subset=[col for col in ["Riesgo Rotura"] if col in display_df.columns],
                cmap="RdYlGn_r",
                vmin=0,
                vmax=1
            ),
            use_container_width=True,
            height=400
        )

    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        st.download_button(
            label="📄 Descargar CSV",
            data=export_df_to_csv(df_view[display_cols]),
            file_name="simulacion_escenarios.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_exp2:
        st.download_button(
            label="📊 Descargar Excel",
            data=export_df_to_excel(df_view[display_cols]),
            file_name="simulacion_escenarios.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

# =============================================================================
# PÁGINA 4: PRODUCTOS CRÍTICOS
# =============================================================================

def render_critical_products(df_risk, df_inventory, df_purchases):
    st.markdown("## ⚠️ Productos Críticos")
    st.markdown("""
    <div style="color: var(--card-bg); margin-bottom: 20px;">
    Identificación y gestión de productos con riesgo de rotura de stock.
    Permite priorizar acciones correctivas y compras de emergencia.
    </div>
    """, unsafe_allow_html=True)

    if df_risk.empty:
        st.warning("⚠️ No hay datos de riesgo de rotura disponibles.")
        return

    df_risk_std = standardize_risk_df(df_risk)

    st.markdown("### 🔍 Filtros")
    colf1, colf2, colf3 = st.columns(3)

    with colf1:
        risk_threshold = st.slider("Umbral de Riesgo Mínimo", 0.0, 1.0, 0.3, 0.05)

    with colf2:
        familias = sorted(df_risk_std["familia"].dropna().astype(str).unique().tolist()) if "familia" in df_risk_std.columns else []
        selected_family = st.selectbox("Familia", options=["Todas"] + familias, index=0)

    with colf3:
        abc_classes = sorted(df_risk_std["abc_class"].dropna().astype(str).unique().tolist()) if "abc_class" in df_risk_std.columns else []
        selected_abc = st.selectbox("Clase ABC", options=["Todas"] + abc_classes, index=0)

    df_filtered = df_risk_std[df_risk_std["riesgo_rotura"] >= risk_threshold].copy()

    if selected_family != "Todas":
        df_filtered = df_filtered[df_filtered["familia"] == selected_family]

    if selected_abc != "Todas":
        df_filtered = df_filtered[df_filtered["abc_class"] == selected_abc]

    df_filtered = df_filtered.sort_values("riesgo_rotura", ascending=False)

    if df_filtered.empty:
        st.warning("⚠️ No hay productos que cumplan con los filtros seleccionados.")
        return

    n_critical = len(df_filtered[df_filtered["riesgo_rotura"] >= 0.8])
    n_high_risk = len(df_filtered[(df_filtered["riesgo_rotura"] >= 0.5) & (df_filtered["riesgo_rotura"] < 0.8)])
    n_medium_risk = len(df_filtered[(df_filtered["riesgo_rotura"] >= 0.3) & (df_filtered["riesgo_rotura"] < 0.5)])

    total_purchase = 0
    if not df_purchases.empty:
        df_purchases_std = standardize_purchase_df(df_purchases)
        purchases_filtered = df_purchases_std[df_purchases_std["producto"].isin(df_filtered["producto"])]
        if not purchases_filtered.empty and "cantidad_recomendada" in purchases_filtered.columns:
            total_purchase = purchases_filtered["cantidad_recomendada"].sum()

    st.markdown("### Indicadores de Riesgo")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("Productos Críticos", str(n_critical), delta_type="negative")
    with col2:
        render_kpi_card("Alto Riesgo", str(n_high_risk), delta_type="negative")
    with col3:
        render_kpi_card("Riesgo Medio", str(n_medium_risk))
    with col4:
        render_kpi_card("Compra Recomendada", format_number(total_purchase))

    st.markdown("---")
    st.markdown("### Visualizaciones de Riesgo")

    c1, c2 = st.columns(2)
    with c1:
        fig_top_risk = plot_top_bar(
            df_filtered.head(15),
            "producto",
            "riesgo_rotura",
            top_n=15,
            title="Top 15 Productos con Mayor Riesgo"
        )
        fig_top_risk.update_traces(marker_color=COLOR_CRITICAL)
        st.plotly_chart(fig_top_risk, use_container_width=True)

    with c2:
        if "familia" in df_filtered.columns:
            risk_by_family = (
                df_filtered.groupby("familia", as_index=False)["riesgo_rotura"]
                .mean()
                .sort_values("riesgo_rotura", ascending=False)
            )

            fig_family_risk = go.Figure()
            fig_family_risk.add_trace(go.Bar(
                x=risk_by_family["familia"],
                y=risk_by_family["riesgo_rotura"],
                marker_color=[COLOR_CRITICAL if r > 0.5 else COLOR_WARNING if r > 0.3 else COLOR_TEAL for r in risk_by_family["riesgo_rotura"]]
            ))
            fig_family_risk.update_layout(
                title="Riesgo Promedio por Familia",
                template="plotly_white",
                yaxis=dict(title="Riesgo Promedio", range=[0, 1])
            )
            st.plotly_chart(fig_family_risk, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        if "abc_class" in df_filtered.columns:
            risk_by_abc = (
                df_filtered.groupby("abc_class", as_index=False)
                .agg(
                    riesgo_promedio=("riesgo_rotura", "mean"),
                    cantidad=("riesgo_rotura", "count")
                )
                .sort_values("riesgo_promedio", ascending=False)
            )

            color_map_abc = {"A": COLOR_CRITICAL, "B": COLOR_WARNING, "C": COLOR_TEAL}

            fig_abc_risk = go.Figure()
            fig_abc_risk.add_trace(go.Bar(
                x=risk_by_abc["abc_class"],
                y=risk_by_abc["riesgo_promedio"],
                marker_color=[color_map_abc.get(str(x), COLOR_GRAY) for x in risk_by_abc["abc_class"]],
                text=risk_by_abc["cantidad"].apply(lambda x: f"n={x}"),
                textposition="outside"
            ))
            fig_abc_risk.update_layout(
                title="Riesgo por Clase ABC",
                template="plotly_white",
                yaxis=dict(title="Riesgo Promedio", range=[0, 1])
            )
            st.plotly_chart(fig_abc_risk, use_container_width=True)

    with c4:
        st.plotly_chart(plot_risk_distribution(df_filtered), use_container_width=True)

    st.markdown("---")
    st.markdown("### Tabla de Productos Críticos")

    if not df_inventory.empty:
        df_inv_std = standardize_inventory_df(df_inventory)
        df_table = df_filtered.merge(df_inv_std, on=["producto", "familia"], how="left")
    else:
        df_table = df_filtered.copy()

    if not df_purchases.empty:
        df_purchases_std = standardize_purchase_df(df_purchases)
        df_table = df_table.merge(
            df_purchases_std[["producto", "cantidad_recomendada"]],
            on="producto",
            how="left"
        )

    display_cols = ["producto", "familia", "abc_class", "stock_actual", "stock_seguridad", "punto_pedido", "cantidad_recomendada", "riesgo_rotura"]
    display_cols = [c for c in display_cols if c in df_table.columns]

    products_df = df_table[display_cols].copy()
    products_df = products_df.rename(columns={
        "stock_actual": "Stock Actual",
        "stock_seguridad": "Stock Seguridad",
        "punto_pedido": "Punto de Pedido",
        "cantidad_recomendada": "Compra Recomendada",
        "riesgo_rotura": "Riesgo Rotura"
    })

    try:
        max_cells = pd.get_option("styler.render.max_elements")
    except Exception:
        max_cells = 262144

    if products_df.size > max_cells:
        st.warning("La tabla es muy grande para formato enriquecido. Se mostrará una versión simplificada para mejorar rendimiento.")
        st.dataframe(products_df, use_container_width=True, height=500)
    else:
        st.dataframe(
            products_df.style.format({
                "Stock Actual": "{:,.0f}",
                "Stock Seguridad": "{:,.0f}",
                "Punto de Pedido": "{:,.0f}",
                "Compra Recomendada": "{:,.0f}",
                "Riesgo Rotura": "{:.1%}"
            }).background_gradient(
                subset=[col for col in ["Riesgo Rotura"] if col in products_df.columns],
                cmap="RdYlGn_r",
                vmin=0,
                vmax=1
            ),
            use_container_width=True,
            height=500
        )

    st.markdown("#### Exportar Datos")
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        st.download_button(
            label="📄 Descargar CSV",
            data=export_df_to_csv(df_table[display_cols]),
            file_name="productos_criticos.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_exp2:
        st.download_button(
            label="📊 Descargar Excel",
            data=export_df_to_excel(df_table[display_cols]),
            file_name="productos_criticos.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

# =============================================================================
# APP PRINCIPAL
# =============================================================================

def main():
    st.markdown("""
    <div class="app-header">
        <h1>📊 FlowMap Sales Intelligence Platform</h1>
        <p>Análisis Comercial · Forecast · Optimización de Inventario · Simulación de Escenarios</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("🔄 Cargando datos..."):
        df_forecast_raw, _ = load_forecast_data("30d")
        df_inventory_raw, _ = load_inventory_data()
        df_history_raw, history_path, history_mode = load_sales_history()
        df_risk_raw, _ = load_risk_data()
        df_purchases_raw, _ = load_purchase_recommendations()
        df_scenarios_raw, scenario_path = load_scenario_data()

    df_forecast = standardize_forecast_df(df_forecast_raw) if not df_forecast_raw.empty else pd.DataFrame()
    df_inventory = standardize_inventory_df(df_inventory_raw) if not df_inventory_raw.empty else pd.DataFrame()
    df_history = standardize_history_df(df_history_raw) if not df_history_raw.empty else pd.DataFrame()
    df_risk = standardize_risk_df(df_risk_raw) if not df_risk_raw.empty else pd.DataFrame()
    df_purchases = standardize_purchase_df(df_purchases_raw) if not df_purchases_raw.empty else pd.DataFrame()
    df_scenarios = standardize_scenario_df(df_scenarios_raw) if not df_scenarios_raw.empty else pd.DataFrame()

    with st.sidebar:
    # Código CSS para forzar letras blancas en la barra lateral
        st.markdown(
        """
        <style>
        [data-testid="stSidebar"] *, [data-testid="stSidebar"] p {
            color: #FFFFFF !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
        logo_path = APP_DIR / "app" / "data" / "LOGO.png"
        if logo_path.exists():
            st.image(str(logo_path), width=180)
        else:
            st.markdown("")
        st.markdown("##  Panel de Navegación")
        page = st.radio(
            "Selecciona una sección",
            [
                "📊 Executive Dashboard",
                "📈 Forecast + Inventario",
                "🎛️ Simulador de Escenarios",
                "⚠️ Productos Críticos"
            ],
            index=0
        )

        st.markdown("---")
        st.markdown("### ℹ️ Estado de carga")

        status_data = {
            "Dataset": ["Histórico", "Forecast", "Inventario", "Riesgo", "Compras", "Escenarios"],
            "Estado": [
                f"✓ {len(df_history)} registros" if not df_history.empty else "✗ No disponible",
                f"✓ {len(df_forecast)} registros" if not df_forecast.empty else "✗ No disponible",
                f"✓ {len(df_inventory)} productos" if not df_inventory.empty else "✗ No disponible",
                f"✓ {len(df_risk)} productos" if not df_risk.empty else "✗ No disponible",
                f"✓ {len(df_purchases)} productos" if not df_purchases.empty else "✗ No disponible",
                f"✓ {len(df_scenarios)} registros" if not df_scenarios.empty else "✗ No disponible",
            ]
        }
        st.dataframe(pd.DataFrame(status_data), use_container_width=True, hide_index=True)

        if history_path:
            st.markdown("#### Histórico detectado")
            st.code(str(history_path), language="text")
        if scenario_path:
            st.markdown("#### Escenarios detectados")
            st.code(str(scenario_path), language="text")

        st.markdown("#### Modo histórico")
        st.write(history_mode)

    if history_mode == "fallback_forecast" and not df_history.empty:
        st.markdown("""
        <div class="alert-info">
        ℹ️ Se ha generado un histórico técnico a partir del forecast disponible.
        Para validación de negocio, se recomienda usar un histórico real.
        </div>
        """, unsafe_allow_html=True)

    if history_mode == "real_parquet" and not df_history.empty:
        st.markdown("""
        <div class="alert-success">
        ✅ La app está usando el histórico real cargado desde sales_history_real.parquet
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    if page == "📊 Executive Dashboard":
        render_executive_dashboard(df_history)
    elif page == "📈 Forecast + Inventario":
        render_forecast_inventory(df_forecast, df_inventory, df_risk)
    elif page == "🎛️ Simulador de Escenarios":
        render_scenario_simulator(df_scenarios, df_inventory)
    elif page == "⚠️ Productos Críticos":
        render_critical_products(df_risk, df_inventory, df_purchases)

    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p><strong>FlowMap Sales Intelligence Platform</strong> | v3.2.0</p>
        <p>Plataforma unificada de análisis comercial, forecasting y optimización de inventario</p>
        <p>FLOWMAP ANALYTICS © 2026 - Todos los derechos reservados</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()