"""
================================================================================
09_ANALISIS_COMERCIAL_Y_FORECAST.py
Sales & Demand Intelligence Center
================================================================================

Aplicación profesional de análisis comercial y forecast de demanda.
Forma parte del proyecto empresarial completo de Forecasting, Optimización
de Inventario y Simulación de Compras.

Enfoque exclusivo:
- ANÁLISIS COMERCIAL
- INTELIGENCIA DE VENTAS
- FORECAST DE DEMANDA
- ESTACIONALIDAD
- EVOLUCIÓN TEMPORAL

Autor: Senior Data Science Team
Versión: 2.1.0
Fecha: 2026
================================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import io
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURACIÓN DE PÁGINA
# =============================================================================

st.set_page_config(
    page_title="Sales & Demand Intelligence Center",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# ESTILOS CSS PERSONALIZADOS
# =============================================================================

st.markdown("""
<style>
    :root {
        --dark-blue: #0F172A;
        --primary-blue: #1E3A5F;
        --medium-blue: #2563EB;
        --light-blue: #60A5FA;
        --dark-gray: #334155;
        --medium-gray: #64748B;
        --light-gray: #CBD5E1;
        --background: #F8FAFC;
        --card-bg: #FFFFFF;
    }

    .stApp {
        background-color: var(--background);
    }

    h1, h2, h3 {
        color: var(--dark-blue);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid var(--light-gray);
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }

    div[data-testid="stMetricValue"] {
        color: var(--primary-blue);
        font-size: 26px;
        font-weight: 700;
    }

    div[data-testid="stMetricDelta"] {
        color: var(--medium-blue);
    }

    section[data-testid="stSidebar"] {
        background-color: var(--dark-blue);
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    .app-header {
        background: linear-gradient(135deg, var(--dark-blue) 0%, var(--primary-blue) 100%);
        color: white;
        padding: 22px 28px;
        border-radius: 12px;
        margin-bottom: 22px;
    }

    .app-header h1 {
        color: white;
        margin: 0;
        font-size: 28px;
    }

    .app-header p {
        color: #E2E8F0;
        margin: 6px 0 0 0;
        font-size: 14px;
    }

    .note-box {
        background: #EFF6FF;
        border-left: 4px solid #2563EB;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 10px 0;
        color: #1E3A5F;
    }

    .soft-caption {
        font-size: 12px;
        color: #64748B;
        margin-top: -6px;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTES Y RUTAS
# =============================================================================

APP_DIR = Path(__file__).resolve().parent

SEARCH_DIRS = [
    APP_DIR,
    APP_DIR / "data",
    APP_DIR / "reports",
    APP_DIR / "reports" / "forecast",
    APP_DIR / "reports" / "inventory",
    APP_DIR / "segundo proyecto",
    APP_DIR / "segundo proyecto" / "data",
    APP_DIR / "segundo proyecto" / "reports",
    APP_DIR / "segundo proyecto" / "reports" / "forecast",
    APP_DIR / "segundo proyecto" / "reports" / "inventory",
]

COLOR_DARK = "#0F172A"
COLOR_PRIMARY = "#1E3A5F"
COLOR_MID = "#2563EB"
COLOR_LIGHT = "#60A5FA"
COLOR_GRAY = "#64748B"
COLOR_CRITICAL="#DC2626"   # rojo suave
COLOR_WARNING = "#D97706"    # ámbar sobrio
COLOR_TEAL = "#059669"       # verde azulado
# =============================================================================
# HELPERS
# =============================================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


def find_first_existing_file(candidates: List[str]) -> Optional[Path]:
    """Busca el primer archivo existente entre múltiples rutas posibles."""
    for base_dir in SEARCH_DIRS:
        for name in candidates:
            full_path = base_dir / name
            if full_path.exists():
                return full_path
    return None


def safe_read_csv(file_path: Optional[Path]) -> pd.DataFrame:
    """Lee un CSV sin romper la app."""
    if file_path is None:
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path)
        df = normalize_columns(df)
        return df
    except Exception:
        return pd.DataFrame()


def first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Devuelve la primera columna encontrada dentro de una lista de candidatos."""
    if df is None or df.empty:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None


def to_datetime_safe(series: pd.Series) -> pd.Series:
    """Convierte una serie a datetime sin romper."""
    return pd.to_datetime(series, errors="coerce")


def to_numeric_safe(series: pd.Series) -> pd.Series:
    """Convierte una serie a numérica sin romper."""
    return pd.to_numeric(series, errors="coerce")


def format_currency(value: float, currency: str = "USD") -> str:
    """Formatea un valor monetario."""
    symbol_map = {
        "USD": "$",
        "EUR": "€",
        "MXN": "MX$",
        "COP": "COP "
    }
    symbol = symbol_map.get(currency, "$")
    if pd.isna(value):
        value = 0
    return f"{symbol}{value:,.0f}"


def export_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Exporta DataFrame a bytes CSV."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, encoding="utf-8-sig")
    return buffer.getvalue().encode("utf-8-sig")


def export_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Data") -> bytes:
    """Exporta DataFrame a bytes Excel."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    return output.getvalue()


def add_download_buttons(df: pd.DataFrame, base_filename: str, col1, col2):
    """Añade botones de descarga CSV y Excel."""
    with col1:
        st.download_button(
            label="📄 Descargar CSV",
            data=export_to_csv_bytes(df),
            file_name=f"{base_filename}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        st.download_button(
            label="📊 Descargar Excel",
            data=export_to_excel_bytes(df),
            file_name=f"{base_filename}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )


def render_empty_message(message: str):
    """Muestra mensaje estándar cuando no hay datos."""
    st.info(message)


# =============================================================================
# CARGA DE DATOS
# =============================================================================

@st.cache_data(ttl=3600)
def load_forecast_data(horizon: str = "30d") -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Carga los datos de forecast para el horizonte solicitado.
    """
    file_candidates = [
        f"forecast_{horizon}.csv",
        f"reports/forecast/forecast_{horizon}.csv",
        f"data/forecast_{horizon}.csv",
    ]
    file_path = find_first_existing_file(file_candidates)
    df = safe_read_csv(file_path)
    return df, (str(file_path) if file_path else None)


@st.cache_data(ttl=3600)
def load_inventory_data() -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Carga datos de inventario si existen.
    """
    file_candidates = [
        "inventario_productos.csv",
        "reports/inventory/inventario_productos.csv",
        "data/inventario_productos.csv",
    ]
    file_path = find_first_existing_file(file_candidates)
    df = safe_read_csv(file_path)
    return df, (str(file_path) if file_path else None)


@st.cache_data(ttl=3600)
def load_risk_data() -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Carga datos de riesgo de rotura si existen.
    """
    file_candidates = [
        "riesgo_rotura.csv",
        "reports/inventory/riesgo_rotura.csv",
        "data/riesgo_rotura.csv",
    ]
    file_path = find_first_existing_file(file_candidates)
    df = safe_read_csv(file_path)
    return df, (str(file_path) if file_path else None)


@st.cache_data(ttl=3600)
def load_purchase_recommendations() -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Carga recomendaciones de compra si existen.
    """
    file_candidates = [
        "compras_recomendadas.csv",
        "reports/inventory/compras_recomendadas.csv",
        "data/compras_recomendadas.csv",
    ]
    file_path = find_first_existing_file(file_candidates)
    df = safe_read_csv(file_path)
    return df, (str(file_path) if file_path else None)


@st.cache_data(ttl=3600)
def load_sales_history() -> Tuple[pd.DataFrame, Optional[str], str]:
    """
    Intenta cargar un histórico real.
    Si no existe, lo genera a partir del forecast disponible.
    """
    history_candidates = [
        "sales_history.csv",
        "historico_ventas.csv",
        "historico_demanda.csv",
        "ventas_diarias.csv",
        "history_sales.csv",
        "reports/forecast/sales_history.csv",
        "reports/eda/historico_ventas.csv",
        "data/sales_history.csv",
    ]

    history_path = find_first_existing_file(history_candidates)
    df_hist = safe_read_csv(history_path)

    if not df_hist.empty:
        return standardize_history_df(df_hist), str(history_path), "real"

    # fallback usando 365 -> 90 -> 30 -> 7
    for horizon in ["365d", "90d", "30d", "7d"]:
        df_fc_raw, fc_path = load_forecast_data(horizon)
        if not df_fc_raw.empty:
            df_generated = generate_history_from_forecast(df_fc_raw)
            return df_generated, fc_path, "fallback_forecast"

    return pd.DataFrame(), None, "none"


# =============================================================================
# ESTANDARIZACIÓN
# =============================================================================

def standardize_forecast_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza forecast a columnas:
    fecha, producto, familia, forecast, actual, lower_bound, upper_bound
    """
    if df.empty:
        return df

    df = normalize_columns(df)

    col_fecha = first_existing_column(df, ["fecha", "date", "ds"])
    col_producto = first_existing_column(df, ["producto", "product_id", "productid", "item_nbr"])
    col_familia = first_existing_column(df, ["familia", "family", "categoria"])
    col_forecast = first_existing_column(df, ["forecast", "demanda_forecast", "forecast_demand", "predicted_demand", "yhat"])
    col_actual = first_existing_column(df, ["actual", "ventas", "real", "sales", "y"])
    col_lower = first_existing_column(df, ["lower_bound", "lower", "forecast_lower", "yhat_lower"])
    col_upper = first_existing_column(df, ["upper_bound", "upper", "forecast_upper", "yhat_upper"])

    out = pd.DataFrame()

    if col_fecha:
        out["fecha"] = to_datetime_safe(df[col_fecha])
    else:
        start_date = pd.Timestamp.today().normalize()
        out["fecha"] = pd.date_range(start=start_date, periods=len(df), freq="D")

    if col_producto:
        out["producto"] = df[col_producto].astype(str)
    else:
        out["producto"] = "N/A"

    if col_familia:
        out["familia"] = df[col_familia].astype(str)
    else:
        out["familia"] = "N/A"

    if col_forecast:
        out["forecast"] = to_numeric_safe(df[col_forecast])
    elif col_actual:
        out["forecast"] = to_numeric_safe(df[col_actual])
    else:
        out["forecast"] = np.nan

    if col_actual:
        out["actual"] = to_numeric_safe(df[col_actual])

    if col_lower:
        out["lower_bound"] = to_numeric_safe(df[col_lower])

    if col_upper:
        out["upper_bound"] = to_numeric_safe(df[col_upper])

    out = out.dropna(subset=["fecha"]).reset_index(drop=True)
    return out


def standardize_history_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza histórico a columnas:
    fecha, producto, familia, ventas
    """
    if df.empty:
        return df

    df = normalize_columns(df)

    col_fecha = first_existing_column(df, ["fecha", "date", "ds"])
    col_producto = first_existing_column(df, ["producto", "product_id", "productid", "item_nbr"])
    col_familia = first_existing_column(df, ["familia", "family", "categoria"])
    col_ventas = first_existing_column(df, ["ventas", "sales", "unit_sales", "actual", "y"])

    out = pd.DataFrame()

    if col_fecha:
        out["fecha"] = to_datetime_safe(df[col_fecha])
    else:
        return pd.DataFrame()

    if col_producto:
        out["producto"] = df[col_producto].astype(str)
    else:
        out["producto"] = "N/A"

    if col_familia:
        out["familia"] = df[col_familia].astype(str)
    else:
        out["familia"] = "N/A"

    if col_ventas:
        out["ventas"] = to_numeric_safe(df[col_ventas])
    else:
        out["ventas"] = np.nan

    out = out.dropna(subset=["fecha"]).reset_index(drop=True)
    return out


def standardize_inventory_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza inventario.
    """
    if df.empty:
        return df

    df = normalize_columns(df)

    out = pd.DataFrame()

    prod_col = first_existing_column(df, ["producto", "product_id", "productid", "item_nbr"])
    fam_col = first_existing_column(df, ["familia", "family", "categoria"])
    abc_col = first_existing_column(df, ["abc_class", "clase_abc", "abc"])
    stock_col = first_existing_column(df, ["stock_actual", "stock", "current_stock"])
    demanda_col = first_existing_column(df, ["demanda_diaria", "daily_demand", "avg_daily_demand"])
    cobertura_col = first_existing_column(df, ["cobertura_dias", "dias_suministro", "days_supply"])
    min_col = first_existing_column(df, ["stock_minimo", "min_stock"])
    max_col = first_existing_column(df, ["stock_maximo", "max_stock"])

    out["producto"] = df[prod_col].astype(str) if prod_col else "N/A"
    out["familia"] = df[fam_col].astype(str) if fam_col else "N/A"
    out["abc_class"] = df[abc_col].astype(str) if abc_col else "N/A"
    out["stock_actual"] = to_numeric_safe(df[stock_col]) if stock_col else np.nan
    out["demanda_diaria"] = to_numeric_safe(df[demanda_col]) if demanda_col else np.nan
    out["cobertura_dias"] = to_numeric_safe(df[cobertura_col]) if cobertura_col else np.nan
    out["stock_minimo"] = to_numeric_safe(df[min_col]) if min_col else np.nan
    out["stock_maximo"] = to_numeric_safe(df[max_col]) if max_col else np.nan

    return out


def standardize_risk_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza riesgo de rotura.
    """
    if df.empty:
        return df

    df = normalize_columns(df)

    prod_col = first_existing_column(df, ["producto", "product_id", "productid", "item_nbr"])
    fam_col = first_existing_column(df, ["familia", "family", "categoria"])
    risk_col = first_existing_column(df, ["riesgo_rotura", "risk", "stockout_risk"])

    out = pd.DataFrame()
    out["producto"] = df[prod_col].astype(str) if prod_col else "N/A"
    out["familia"] = df[fam_col].astype(str) if fam_col else "N/A"
    out["riesgo_rotura"] = to_numeric_safe(df[risk_col]) if risk_col else np.nan

    return out


def standardize_purchase_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza compras recomendadas.
    """
    if df.empty:
        return df

    df = normalize_columns(df)

    prod_col = first_existing_column(df, ["producto", "product_id", "productid", "item_nbr"])
    fam_col = first_existing_column(df, ["familia", "family", "categoria"])
    qty_col = first_existing_column(df, ["compra_recomendada", "cantidad_recomendada", "qty_recommended", "quantity"])

    out = pd.DataFrame()
    out["producto"] = df[prod_col].astype(str) if prod_col else "N/A"
    out["familia"] = df[fam_col].astype(str) if fam_col else "N/A"
    out["compra_recomendada"] = to_numeric_safe(df[qty_col]) if qty_col else np.nan

    return out


# =============================================================================
# GENERACIÓN DE HISTÓRICO TÉCNICO
# =============================================================================

def generate_history_from_forecast(df_forecast_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Genera histórico técnico a partir del forecast disponible.
    """
    df_fc = standardize_forecast_df(df_forecast_raw)
    if df_fc.empty:
        return pd.DataFrame()

    # Si existe actual, lo usamos directamente
    if "actual" in df_fc.columns and df_fc["actual"].notna().any():
        hist = df_fc[["fecha", "producto", "familia", "actual"]].copy()
        hist = hist.rename(columns={"actual": "ventas"})
        return hist.dropna(subset=["fecha", "ventas"]).sort_values("fecha").reset_index(drop=True)

    # Si no existe actual, generamos 365 días sintéticos
    base_df = (
        df_fc.groupby(["producto", "familia"], as_index=False)["forecast"]
        .mean()
        .fillna(0)
    )

    end_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    start_date = end_date - pd.Timedelta(days=364)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    rng = np.random.default_rng(42)
    records = []

    for _, row in base_df.iterrows():
        base = max(float(row["forecast"]), 1.0)

        for d in dates:
            dow = d.dayofweek
            month = d.month
            weekly_factor = 1 + 0.08 * np.sin(2 * np.pi * dow / 7)
            monthly_factor = 1 + 0.12 * np.sin(2 * np.pi * month / 12)
            noise = rng.normal(1.0, 0.12)

            ventas = max(base * 0.9 * weekly_factor * monthly_factor * noise, 0)

            records.append({
                "fecha": d,
                "producto": str(row["producto"]),
                "familia": str(row["familia"]),
                "ventas": round(float(ventas), 2)
            })

    return pd.DataFrame(records).sort_values("fecha").reset_index(drop=True)


# =============================================================================
# CÁLCULOS DE NEGOCIO
# =============================================================================

def calculate_kpis(df_history: pd.DataFrame) -> Dict:
    """
    Calcula KPIs principales.
    """
    if df_history.empty or "fecha" not in df_history.columns or "ventas" not in df_history.columns:
        return {}

    df = df_history.copy()
    df["fecha"] = to_datetime_safe(df["fecha"])
    df = df.dropna(subset=["fecha", "ventas"])

    if df.empty:
        return {}

    latest_date = df["fecha"].max()

    total_sales = df["ventas"].sum()
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
        "sales_30d": sales_30d,
        "sales_90d": sales_90d,
        "sales_ytd": sales_ytd,
        "daily_avg": daily_avg,
        "n_products": n_products,
        "n_families": n_families,
        "growth_rate": growth_rate
    }


def calculate_abc_classification(df_history: pd.DataFrame, threshold_a: float = 0.80, threshold_b: float = 0.95) -> pd.DataFrame:
    """
    Clasificación ABC.
    """
    if df_history.empty or "producto" not in df_history.columns or "ventas" not in df_history.columns:
        return pd.DataFrame()

    abc = (
        df_history.groupby("producto", as_index=False)["ventas"]
        .sum()
        .sort_values("ventas", ascending=False)
    )

    abc["participacion_acumulada"] = abc["ventas"].cumsum() / abc["ventas"].sum()
    abc["clase_abc"] = "C"
    abc.loc[abc["participacion_acumulada"] <= threshold_a, "clase_abc"] = "A"
    abc.loc[
        (abc["participacion_acumulada"] > threshold_a) &
        (abc["participacion_acumulada"] <= threshold_b),
        "clase_abc"
    ] = "B"

    return abc


def calculate_seasonality(df_history: pd.DataFrame) -> Dict:
    """
    Estacionalidad mensual, semanal y trimestral.
    """
    if df_history.empty or "fecha" not in df_history.columns or "ventas" not in df_history.columns:
        return {}

    df = df_history.copy()
    df["fecha"] = to_datetime_safe(df["fecha"])
    df = df.dropna(subset=["fecha"])

    overall_avg = df["ventas"].mean()
    if overall_avg == 0:
        return {}

    df["mes"] = df["fecha"].dt.month
    df["dia_semana"] = df["fecha"].dt.dayofweek
    df["trimestre"] = df["fecha"].dt.quarter

    monthly = ((df.groupby("mes")["ventas"].mean() / overall_avg) - 1) * 100
    dow = ((df.groupby("dia_semana")["ventas"].mean() / overall_avg) - 1) * 100
    quarter = ((df.groupby("trimestre")["ventas"].mean() / overall_avg) - 1) * 100

    return {
        "monthly": monthly,
        "dow": dow,
        "quarter": quarter
    }


def generate_forecast_insights(df_forecast: pd.DataFrame, df_history: pd.DataFrame) -> List[str]:
    """
    Genera insights automáticos.
    """
    insights: List[str] = []

    if df_forecast.empty:
        return insights

    df_fc = standardize_forecast_df(df_forecast)
    df_hist = standardize_history_df(df_history) if not df_history.empty else pd.DataFrame()

    if df_fc.empty:
        return insights

    forecast_avg = df_fc["forecast"].mean()

    if not df_hist.empty:
        recent_daily = df_hist.groupby("fecha", as_index=False)["ventas"].sum()
        recent_avg = recent_daily.tail(30)["ventas"].mean() if len(recent_daily) >= 30 else recent_daily["ventas"].mean()

        if pd.notna(recent_avg) and recent_avg > 0:
            diff_pct = (forecast_avg / recent_avg - 1) * 100
            if diff_pct > 10:
                insights.append(f"📈 El forecast proyecta un crecimiento del {diff_pct:.1f}% frente al promedio reciente.")
            elif diff_pct < -10:
                insights.append(f"📉 El forecast proyecta una caída del {abs(diff_pct):.1f}% frente al promedio reciente.")

        monthly_sales = df_hist.groupby(df_hist["fecha"].dt.month)["ventas"].mean()
        if not monthly_sales.empty:
            current_month = datetime.now().month
            if current_month in monthly_sales.index:
                month_avg = monthly_sales[current_month]
                overall_avg = monthly_sales.mean()
                if overall_avg > 0:
                    month_diff = (month_avg / overall_avg - 1) * 100
                    if month_diff > 10:
                        insights.append(f"📅 El mes actual presenta un {month_diff:.0f}% más ventas que la media anual.")
                    elif month_diff < -10:
                        insights.append(f"📅 El mes actual presenta un {abs(month_diff):.0f}% menos ventas que la media anual.")

        dow_sales = df_hist.groupby(df_hist["fecha"].dt.dayofweek)["ventas"].mean()
        if not dow_sales.empty:
            days = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
            max_day = int(dow_sales.idxmax())
            share = dow_sales[max_day] / dow_sales.mean() * 100
            insights.append(f"📆 {days[max_day]} concentra el {share:.0f}% de la demanda media diaria relativa.")

    if "familia" in df_fc.columns and df_fc["familia"].notna().any():
        fam_sum = (
            df_fc.groupby("familia", as_index=False)["forecast"]
            .sum()
            .sort_values("forecast", ascending=False)
        )
        if not fam_sum.empty:
            top_fam = fam_sum.iloc[0]
            insights.append(
                f"🏷️ La familia con mayor forecast es **{top_fam['familia']}** con {top_fam['forecast']:,.0f}."
            )

    return insights


# =============================================================================
# VISUALIZACIONES
# =============================================================================

def plot_sales_evolution(df: pd.DataFrame, freq: str = "D", value_col: str = "ventas", title_prefix: str = "Ventas") -> go.Figure:
    """
    Serie temporal agregada por frecuencia.
    """
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
        mode="lines+markers",
        name=title_prefix,
        line=dict(color=COLOR_PRIMARY, width=2),
        marker=dict(size=4, color=COLOR_MID)
    ))

    if len(grouped) >= 7:
        window = min(7, len(grouped))
        grouped["ma"] = grouped[value_col].rolling(window=window).mean()

        fig.add_trace(go.Scatter(
            x=grouped["fecha"],
            y=grouped["ma"],
            mode="lines",
            name=f"Media móvil {window}",
            line=dict(color=COLOR_LIGHT, width=2, dash="dash")
        ))

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.25)
    )

    return fig


def plot_pareto(df: pd.DataFrame, group_col: str, value_col: str, top_n: int = 20, title: Optional[str] = None) -> go.Figure:
    """
    Gráfico de Pareto.
    """
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
            opacity=0.85,
            name="Valor"
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=agg[group_col].astype(str),
            y=cumulative,
            mode="lines+markers",
            name="% acumulado",
            line=dict(color=COLOR_LIGHT, width=2),
            marker=dict(size=6)
        ),
        secondary_y=True
    )

    fig.update_layout(
        title=title or f"Pareto - {group_col.title()}",
        template="plotly_white",
        showlegend=False,
        xaxis=dict(tickangle=45)
    )
    fig.update_yaxes(title_text=value_col.title(), secondary_y=False)
    fig.update_yaxes(title_text="% acumulado", secondary_y=True, range=[0, 110])

    return fig


def plot_top_bar(df: pd.DataFrame, group_col: str, value_col: str, top_n: int = 20, title: str = "") -> go.Figure:
    """
    Barra horizontal top N.
    """
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
        height=520
    )

    return fig


def plot_heatmap_weekday(df: pd.DataFrame) -> go.Figure:
    """
    Heatmap semana x día de semana.
    """
    fig = go.Figure()

    if df.empty or "fecha" not in df.columns or "ventas" not in df.columns:
        return fig

    tmp = df.copy()
    tmp["fecha"] = to_datetime_safe(tmp["fecha"])
    tmp["semana"] = tmp["fecha"].dt.isocalendar().week.astype(int)
    tmp["dia_semana"] = tmp["fecha"].dt.dayofweek

    days = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]

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
    pivot.columns = [days[i] for i in pivot.columns]

    fig.add_trace(go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale="Blues"
    ))

    fig.update_layout(
        title="Heatmap de ventas por semana y día",
        template="plotly_white"
    )

    return fig


def plot_heatmap_month_day(df: pd.DataFrame) -> go.Figure:
    """
    Heatmap mes x día del mes.
    """
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

    months = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    pivot.index = [months[m - 1] for m in pivot.index]

    fig.add_trace(go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale="Blues"
    ))

    fig.update_layout(
        title="Heatmap de ventas por mes y día del mes",
        template="plotly_white"
    )

    return fig


def plot_seasonality_year_month(df: pd.DataFrame) -> go.Figure:
    """
    Heatmap año x mes.
    """
    fig = go.Figure()

    if df.empty or "fecha" not in df.columns or "ventas" not in df.columns:
        return fig

    tmp = df.copy()
    tmp["fecha"] = to_datetime_safe(tmp["fecha"])
    tmp["anio"] = tmp["fecha"].dt.year
    tmp["mes"] = tmp["fecha"].dt.month

    pivot = tmp.pivot_table(
        index="anio",
        columns="mes",
        values="ventas",
        aggfunc="sum",
        fill_value=0
    )

    if pivot.empty:
        return fig

    months = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    pivot.columns = [months[m - 1] for m in pivot.columns]

    fig.add_trace(go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale="Blues"
    ))

    fig.update_layout(
        title="Heatmap de estacionalidad: año vs mes",
        template="plotly_white"
    )

    return fig


def plot_forecast_with_confidence(df_history: pd.DataFrame, df_forecast: pd.DataFrame) -> go.Figure:
    """
    Histórico + forecast + banda de confianza.
    """
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
            line=dict(color=COLOR_DARK, width=1.8)
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
                fillcolor="rgba(37, 99, 235, 0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Banda de confianza"
            ))

        fig.add_trace(go.Scatter(
            x=fc_daily["fecha"],
            y=fc_daily["forecast"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color=COLOR_MID, width=2.4),
            marker=dict(size=4)
        ))

    fig.update_layout(
        title="Histórico + Forecast",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.25)
    )

    return fig


def plot_forecast_accumulated(df_forecast: pd.DataFrame) -> go.Figure:
    """
    Forecast acumulado.
    """
    fig = go.Figure()

    if df_forecast.empty:
        return fig

    fc = standardize_forecast_df(df_forecast)
    fc_daily = fc.groupby("fecha", as_index=False)["forecast"].sum().sort_values("fecha")
    fc_daily["forecast_acumulado"] = fc_daily["forecast"].cumsum()

    fig.add_trace(go.Scatter(
        x=fc_daily["fecha"],
        y=fc_daily["forecast_acumulado"],
        mode="lines+markers",
        line=dict(color=COLOR_PRIMARY, width=2),
        marker=dict(size=4)
    ))

    fig.update_layout(
        title="Forecast acumulado",
        template="plotly_white",
        hovermode="x unified"
    )

    return fig


def plot_real_vs_forecast(df_history: pd.DataFrame, df_forecast: pd.DataFrame) -> go.Figure:
    """
    Comparación real vs forecast.
    """
    fig = go.Figure()

    if df_history.empty or df_forecast.empty:
        return fig

    hist_daily = (
        df_history.groupby("fecha", as_index=False)["ventas"]
        .sum()
        .sort_values("fecha")
    )

    fc = standardize_forecast_df(df_forecast)
    fc_daily = (
        fc.groupby("fecha", as_index=False)["forecast"]
        .sum()
        .sort_values("fecha")
    )

    merged = hist_daily.merge(fc_daily, on="fecha", how="inner")
    if merged.empty:
        return fig

    fig.add_trace(go.Scatter(
        x=merged["fecha"],
        y=merged["ventas"],
        mode="lines",
        name="Real",
        line=dict(color=COLOR_DARK, width=2)
    ))

    fig.add_trace(go.Scatter(
        x=merged["fecha"],
        y=merged["forecast"],
        mode="lines",
        name="Forecast",
        line=dict(color=COLOR_LIGHT, width=2, dash="dash")
    ))

    fig.update_layout(
        title="Comparación real vs forecast",
        template="plotly_white",
        hovermode="x unified"
    )

    return fig


# =============================================================================
# APP PRINCIPAL
# =============================================================================

def main():
    # -------------------------------------------------------------------------
    # Header
    # -------------------------------------------------------------------------
    st.markdown("""
    <div class="app-header">
        <h1>📊 Sales & Demand Intelligence Center</h1>
        <p>Análisis Comercial · Forecast de Demanda · Inteligencia de Ventas</p>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Sidebar
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 🎯 Navegación")
        page = st.radio(
            "Selecciona una sección:",
            [
                "📊 Executive Dashboard",
                "📈 Forecast Center",
                "🏷️ Análisis por Familia",
                "📦 Análisis por Producto",
                "🔥 Top Movers",
                "📅 Estacionalidad",
                "🎯 Escenarios de Negocio"
            ],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("### ⚙️ Configuración")

        forecast_horizon_label = st.selectbox(
            "Horizonte de Forecast",
            ["7 días", "30 días", "90 días", "365 días"],
            index=1
        )

        currency = st.selectbox(
            "Moneda",
            ["USD", "EUR", "MXN", "COP"],
            index=0
        )

        export_format = st.selectbox(
            "Formato de exportación",
            ["CSV", "Excel"],
            index=0
        )

        st.markdown("---")
        st.markdown("### 📁 Rutas de búsqueda")

        with st.expander("Ver carpetas donde busca archivos"):
            for d in SEARCH_DIRS:
                st.write(str(d))

    horizon_map = {
        "7 días": "7d",
        "30 días": "30d",
        "90 días": "90d",
        "365 días": "365d"
    }
    selected_horizon = horizon_map[forecast_horizon_label]

    # -------------------------------------------------------------------------
    # Carga de datos
    # -------------------------------------------------------------------------
    with st.spinner("Cargando datos..."):
        df_forecast_raw, forecast_path = load_forecast_data(selected_horizon)
        df_inventory_raw, inventory_path = load_inventory_data()
        df_history_raw, history_path, history_mode = load_sales_history()
        df_risk_raw, risk_path = load_risk_data()
        df_purchase_raw, purchase_path = load_purchase_recommendations()

    # Estandarización
    df_forecast = standardize_forecast_df(df_forecast_raw) if not df_forecast_raw.empty else pd.DataFrame()
    df_inventory = standardize_inventory_df(df_inventory_raw) if not df_inventory_raw.empty else pd.DataFrame()
    df_history = standardize_history_df(df_history_raw) if not df_history_raw.empty else pd.DataFrame()
    df_risk = standardize_risk_df(df_risk_raw) if not df_risk_raw.empty else pd.DataFrame()
    df_purchases = standardize_purchase_df(df_purchase_raw) if not df_purchase_raw.empty else pd.DataFrame()

    # -------------------------------------------------------------------------
    # Estado de carga
    # -------------------------------------------------------------------------
    with st.expander("ℹ️ Estado de carga de archivos"):
        status_rows = [
            ["Forecast seleccionado", forecast_path if forecast_path else "No encontrado"],
            ["Histórico de ventas", history_path if history_path else "No encontrado"],
            ["Modo histórico", history_mode],
            ["Inventario", inventory_path if inventory_path else "No encontrado"],
            ["Riesgo de rotura", risk_path if risk_path else "No encontrado"],
            ["Compras recomendadas", purchase_path if purchase_path else "No encontrado"],
        ]
        status_df = pd.DataFrame(status_rows, columns=["Dataset", "Ruta / Estado"])
        st.dataframe(status_df, use_container_width=True)

    if history_mode == "fallback_forecast" and not df_history.empty:
        st.markdown("""
        <div class="note-box">
        Se ha generado un histórico técnico a partir del forecast disponible porque no se encontró un archivo histórico real.
        La app sigue funcionando, pero para validación de negocio es preferible usar un histórico real exportado.
        </div>
        """, unsafe_allow_html=True)

    if df_forecast.empty and df_history.empty:
        st.warning("No se pudo cargar forecast ni histórico. La navegación sigue disponible, pero varias secciones no mostrarán datos.")

    # =============================================================================
    # PÁGINA 1 - EXECUTIVE DASHBOARD
    # =============================================================================
    if page == "📊 Executive Dashboard":
        st.markdown("## 📊 Executive Dashboard")
        st.markdown("Visión general del desempeño comercial, evolución temporal y concentración de ventas.")

        if df_history.empty:
            render_empty_message("No hay histórico disponible para construir el Executive Dashboard.")
        else:
            kpis = calculate_kpis(df_history)

            if kpis:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Ventas Totales", format_currency(kpis["total_sales"], currency))
                with col2:
                    st.metric("Ventas Últimos 30 días", format_currency(kpis["sales_30d"], currency), f"{kpis['growth_rate']:+.1f}%")
                with col3:
                    st.metric("Ventas Últimos 90 días", format_currency(kpis["sales_90d"], currency))
                with col4:
                    st.metric("Ventas Acumuladas Año", format_currency(kpis["sales_ytd"], currency))

                col5, col6, col7, col8 = st.columns(4)
                with col5:
                    st.metric("Productos Activos", f"{kpis['n_products']:,}")
                with col6:
                    st.metric("Familias Activas", f"{kpis['n_families']:,}")
                with col7:
                    st.metric("Venta Promedio Diaria", format_currency(kpis["daily_avg"], currency))
                with col8:
                    st.metric("Crecimiento %", f"{kpis['growth_rate']:+.1f}%")

            st.markdown("---")

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(plot_sales_evolution(df_history, "D", "ventas", "Ventas"), use_container_width=True)
            with c2:
                st.plotly_chart(plot_sales_evolution(df_history, "W", "ventas", "Ventas"), use_container_width=True)

            c3, c4 = st.columns(2)
            with c3:
                st.plotly_chart(plot_sales_evolution(df_history, "M", "ventas", "Ventas"), use_container_width=True)
            with c4:
                st.plotly_chart(plot_sales_evolution(df_history, "Y", "ventas", "Ventas"), use_container_width=True)

            c5, c6 = st.columns(2)
            with c5:
                st.plotly_chart(plot_pareto(df_history, "producto", "ventas", top_n=20, title="Pareto de ventas por producto"), use_container_width=True)
            with c6:
                st.plotly_chart(plot_top_bar(df_history, "producto", "ventas", top_n=20, title="Top 20 productos"), use_container_width=True)

            c7, c8 = st.columns(2)
            with c7:
                st.plotly_chart(plot_top_bar(df_history, "familia", "ventas", top_n=20, title="Top 20 familias"), use_container_width=True)
            with c8:
                st.plotly_chart(plot_heatmap_weekday(df_history), use_container_width=True)

            st.plotly_chart(plot_heatmap_month_day(df_history), use_container_width=True)

            with st.expander("📋 Tabla resumen exportable"):
                summary_df = (
                    df_history.groupby(["producto", "familia"], as_index=False)
                    .agg(
                        ventas_totales=("ventas", "sum"),
                        venta_promedio=("ventas", "mean"),
                        desv_estandar=("ventas", "std"),
                        dias_con_venta=("ventas", "count")
                    )
                    .sort_values("ventas_totales", ascending=False)
                )
                st.dataframe(summary_df, use_container_width=True)

                colx1, colx2 = st.columns(2)
                add_download_buttons(summary_df, "executive_dashboard_summary", colx1, colx2)

    # =============================================================================
    # PÁGINA 2 - FORECAST CENTER
    # =============================================================================
    elif page == "📈 Forecast Center":
        st.markdown("## 📈 Forecast Center")
        st.markdown("Centro de forecast de demanda, comparativas y desagregación comercial.")

        if df_forecast.empty:
            render_empty_message(f"No se encontró el archivo de forecast para {selected_horizon}.")
        else:
            fc_daily = df_forecast.groupby("fecha", as_index=False)["forecast"].sum().sort_values("fecha")

            total_forecast = fc_daily["forecast"].sum()
            daily_avg = fc_daily["forecast"].mean()
            weekly_forecast = daily_avg * 7
            monthly_forecast = daily_avg * 30
            annual_forecast = daily_avg * 365

            next_day = fc_daily["forecast"].iloc[0] if len(fc_daily) > 0 else 0
            next_week = fc_daily.head(7)["forecast"].sum()
            next_month = fc_daily.head(30)["forecast"].sum()
            next_quarter = fc_daily.head(90)["forecast"].sum()
            next_year = fc_daily.head(365)["forecast"].sum()

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Forecast Total Esperado", format_currency(total_forecast, currency))
            with col2:
                st.metric("Forecast Diario", format_currency(daily_avg, currency))
            with col3:
                st.metric("Forecast Semanal", format_currency(weekly_forecast, currency))
            with col4:
                st.metric("Forecast Mensual", format_currency(monthly_forecast, currency))
            with col5:
                st.metric("Forecast Anual", format_currency(annual_forecast, currency))

            col6, col7, col8, col9, col10 = st.columns(5)
            with col6:
                st.metric("Próximo Día", format_currency(next_day, currency))
            with col7:
                st.metric("Próxima Semana", format_currency(next_week, currency))
            with col8:
                st.metric("Próximo Mes", format_currency(next_month, currency))
            with col9:
                st.metric("Próximo Trimestre", format_currency(next_quarter, currency))
            with col10:
                st.metric("Próximo Año", format_currency(next_year, currency))

            st.markdown("---")

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(plot_forecast_with_confidence(df_history, df_forecast), use_container_width=True)
            with c2:
                st.plotly_chart(plot_forecast_accumulated(df_forecast), use_container_width=True)

            c3, c4 = st.columns(2)
            with c3:
                st.plotly_chart(plot_real_vs_forecast(df_history, df_forecast), use_container_width=True)
            with c4:
                if "familia" in df_forecast.columns:
                    st.plotly_chart(plot_top_bar(df_forecast, "familia", "forecast", top_n=20, title="Forecast por familia"), use_container_width=True)
                else:
                    render_empty_message("El forecast cargado no contiene columna familia.")

            c5, c6 = st.columns(2)
            with c5:
                if "producto" in df_forecast.columns:
                    st.plotly_chart(plot_top_bar(df_forecast, "producto", "forecast", top_n=20, title="Top 20 productos forecast"), use_container_width=True)
                else:
                    render_empty_message("El forecast cargado no contiene columna producto.")
            with c6:
                fig_dist = px.histogram(
                    df_forecast,
                    x="forecast",
                    nbins=40,
                    title="Distribución del forecast",
                    color_discrete_sequence=[COLOR_PRIMARY]
                )
                fig_dist.update_layout(template="plotly_white")
                st.plotly_chart(fig_dist, use_container_width=True)

            st.markdown("---")
            st.markdown("### 🧠 Insights automáticos")
            insights = generate_forecast_insights(df_forecast, df_history)
            if insights:
                for insight in insights:
                    st.info(insight)
            else:
                st.info("No se pudieron generar insights automáticos con los datos disponibles.")

            with st.expander("📋 Exportar datos de forecast"):
                st.dataframe(df_forecast, use_container_width=True)
                colx1, colx2 = st.columns(2)
                add_download_buttons(df_forecast, f"forecast_center_{selected_horizon}", colx1, colx2)

    # =============================================================================
    # PÁGINA 3 - ANÁLISIS POR FAMILIA
    # =============================================================================
    elif page == "🏷️ Análisis por Familia":
        st.markdown("## 🏷️ Análisis por Familia")
        st.markdown("Desempeño comercial, forecast y estacionalidad a nivel familia.")

        if df_history.empty:
            render_empty_message("No hay histórico disponible para análisis por familia.")
        else:
            familias = sorted(df_history["familia"].dropna().astype(str).unique().tolist()) if "familia" in df_history.columns else []
            selected_family = st.selectbox("Selecciona una familia", ["Todas"] + familias)

            min_date = df_history["fecha"].min().date()
            max_date = df_history["fecha"].max().date()

            date_range = st.date_input(
                "Rango de fechas",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )

            df_filtered = df_history.copy()

            if selected_family != "Todas":
                df_filtered = df_filtered[df_filtered["familia"] == selected_family]

            if len(date_range) == 2:
                start_date = pd.to_datetime(date_range[0])
                end_date = pd.to_datetime(date_range[1])
                df_filtered = df_filtered[(df_filtered["fecha"] >= start_date) & (df_filtered["fecha"] <= end_date)]

            if df_filtered.empty:
                render_empty_message("No hay datos para la familia y rango seleccionados.")
            else:
                ventas_familia = df_filtered["ventas"].sum()

                forecast_familia = np.nan
                if not df_forecast.empty and "familia" in df_forecast.columns:
                    df_fc_fam = df_forecast[df_forecast["familia"] == selected_family] if selected_family != "Todas" else df_forecast
                    if not df_fc_fam.empty:
                        forecast_familia = df_fc_fam["forecast"].sum()

                total_sales = df_history["ventas"].sum()
                participation = (ventas_familia / total_sales * 100) if total_sales > 0 else 0

                if len(df_filtered) >= 60:
                    daily = df_filtered.groupby("fecha", as_index=False)["ventas"].sum()
                    recent = daily.tail(30)["ventas"].sum()
                    prev = daily.iloc[-60:-30]["ventas"].sum()
                    growth = ((recent - prev) / prev * 100) if prev > 0 else 0
                else:
                    growth = 0

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Ventas", format_currency(ventas_familia, currency))
                with col2:
                    st.metric("Forecast", format_currency(forecast_familia, currency) if pd.notna(forecast_familia) else "N/D")
                with col3:
                    st.metric("Participación %", f"{participation:.1f}%")
                with col4:
                    st.metric("Crecimiento %", f"{growth:+.1f}%")

                st.markdown("---")

                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(plot_sales_evolution(df_filtered, "D", "ventas", "Ventas familia"), use_container_width=True)
                with c2:
                    st.plotly_chart(plot_sales_evolution(df_filtered, "M", "ventas", "Ventas familia"), use_container_width=True)

                c3, c4 = st.columns(2)
                with c3:
                    if not df_forecast.empty and "familia" in df_forecast.columns:
                        df_fc_fam = df_forecast[df_forecast["familia"] == selected_family] if selected_family != "Todas" else df_forecast
                        if not df_fc_fam.empty:
                            st.plotly_chart(plot_sales_evolution(df_fc_fam.rename(columns={"forecast": "ventas"}), "D", "ventas", "Forecast familia"), use_container_width=True)
                        else:
                            render_empty_message("No hay forecast disponible para la familia seleccionada.")
                    else:
                        render_empty_message("No hay forecast con columna familia.")
                with c4:
                    part_df = (
                        df_history.groupby("familia", as_index=False)["ventas"]
                        .sum()
                        .sort_values("ventas", ascending=False)
                    )
                    fig_part = px.pie(
                        part_df.head(15),
                        names="familia",
                        values="ventas",
                        title="Participación por familia",
                        color_discrete_sequence=px.colors.sequential.Blues_r
                    )
                    fig_part.update_layout(template="plotly_white")
                    st.plotly_chart(fig_part, use_container_width=True)

                c5, c6 = st.columns(2)
                with c5:
                    st.plotly_chart(plot_top_bar(df_history, "familia", "ventas", top_n=20, title="Ranking familias por ventas"), use_container_width=True)
                with c6:
                    st.plotly_chart(plot_pareto(df_history, "familia", "ventas", top_n=20, title="Pareto familias"), use_container_width=True)

                st.plotly_chart(plot_heatmap_month_day(df_filtered), use_container_width=True)

                with st.expander("📋 Tabla exportable"):
                    family_table = (
                        df_history.groupby("familia", as_index=False)
                        .agg(
                            ventas=("ventas", "sum"),
                            venta_promedio=("ventas", "mean"),
                            dias=("ventas", "count"),
                            productos=("producto", "nunique")
                        )
                        .sort_values("ventas", ascending=False)
                    )
                    st.dataframe(family_table, use_container_width=True)
                    colx1, colx2 = st.columns(2)
                    add_download_buttons(family_table, "analisis_por_familia", colx1, colx2)

    # =============================================================================
    # PÁGINA 4 - ANÁLISIS POR PRODUCTO
    # =============================================================================
    elif page == "📦 Análisis por Producto":
        st.markdown("## 📦 Análisis por Producto")
        st.markdown("Ficha ejecutiva y análisis individual del producto.")

        if df_history.empty or "producto" not in df_history.columns:
            render_empty_message("No hay datos suficientes para análisis por producto.")
        else:
            productos = sorted(df_history["producto"].dropna().astype(str).unique().tolist())
            selected_product = st.selectbox("Selecciona un producto", productos)

            df_product = df_history[df_history["producto"] == selected_product].copy()

            if df_product.empty:
                render_empty_message("No se encontraron datos para el producto seleccionado.")
            else:
                total_sales = df_product["ventas"].sum()
                daily_avg = df_product["ventas"].mean()

                forecast_value = np.nan
                if not df_forecast.empty and "producto" in df_forecast.columns:
                    fc_prod = df_forecast[df_forecast["producto"] == selected_product]
                    if not fc_prod.empty:
                        forecast_value = fc_prod["forecast"].sum()

                if len(df_product) >= 60:
                    daily = df_product.groupby("fecha", as_index=False)["ventas"].sum()
                    recent_avg = daily.tail(30)["ventas"].mean()
                    old_avg = daily.iloc[-60:-30]["ventas"].mean() if len(daily) >= 60 else daily.head(30)["ventas"].mean()
                    growth = ((recent_avg - old_avg) / old_avg * 100) if old_avg > 0 else 0
                else:
                    growth = 0

                cv = (df_product["ventas"].std() / df_product["ventas"].mean() * 100) if df_product["ventas"].mean() > 0 else 0

                abc_data = calculate_abc_classification(df_history)
                abc_class = "N/D"
                if not abc_data.empty:
                    abc_match = abc_data[abc_data["producto"] == selected_product]
                    if not abc_match.empty:
                        abc_class = abc_match["clase_abc"].iloc[0]

                col1, col2, col3, col4, col5, col6 = st.columns(6)
                with col1:
                    st.metric("Venta total", format_currency(total_sales, currency))
                with col2:
                    st.metric("Venta diaria promedio", format_currency(daily_avg, currency))
                with col3:
                    st.metric("Forecast", format_currency(forecast_value, currency) if pd.notna(forecast_value) else "N/D")
                with col4:
                    st.metric("Crecimiento", f"{growth:+.1f}%")
                with col5:
                    st.metric("CV demanda", f"{cv:.1f}%")
                with col6:
                    st.metric("Clase ABC", abc_class)

                st.markdown("---")

                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(plot_sales_evolution(df_product, "D", "ventas", "Ventas producto"), use_container_width=True)
                with c2:
                    if not df_forecast.empty and "producto" in df_forecast.columns:
                        fc_prod = df_forecast[df_forecast["producto"] == selected_product]
                        if not fc_prod.empty:
                            st.plotly_chart(plot_sales_evolution(fc_prod.rename(columns={"forecast": "ventas"}), "D", "ventas", "Forecast producto"), use_container_width=True)
                        else:
                            render_empty_message("No hay forecast disponible para el producto.")
                    else:
                        render_empty_message("No hay forecast por producto disponible.")

                c3, c4 = st.columns(2)
                with c3:
                    st.plotly_chart(plot_sales_evolution(df_product, "W", "ventas", "Demanda semanal"), use_container_width=True)
                with c4:
                    st.plotly_chart(plot_sales_evolution(df_product, "M", "ventas", "Demanda mensual"), use_container_width=True)

                c5, c6 = st.columns(2)
                with c5:
                    fig_hist = px.histogram(
                        df_product,
                        x="ventas",
                        nbins=30,
                        title="Histograma de demanda",
                        color_discrete_sequence=[COLOR_PRIMARY]
                    )
                    fig_hist.update_layout(template="plotly_white")
                    st.plotly_chart(fig_hist, use_container_width=True)

                with c6:
                    fig_box = px.box(
                        df_product,
                        y="ventas",
                        title="Boxplot de demanda",
                        color_discrete_sequence=[COLOR_MID]
                    )
                    fig_box.update_layout(template="plotly_white")
                    st.plotly_chart(fig_box, use_container_width=True)

                if not df_inventory.empty and "producto" in df_inventory.columns:
                    prod_inv = df_inventory[df_inventory["producto"] == selected_product]
                    if not prod_inv.empty:
                        st.markdown("---")
                        st.markdown("### 📦 Cobertura de inventario (si disponible)")
                        inv = prod_inv.iloc[0]

                        c7, c8, c9, c10 = st.columns(4)
                        with c7:
                            st.metric("Stock actual", f"{inv.get('stock_actual', np.nan):,.0f} un." if pd.notna(inv.get("stock_actual", np.nan)) else "N/D")
                        with c8:
                            st.metric("Cobertura", f"{inv.get('cobertura_dias', np.nan):,.1f} días" if pd.notna(inv.get("cobertura_dias", np.nan)) else "N/D")
                        with c9:
                            st.metric("Stock mínimo", f"{inv.get('stock_minimo', np.nan):,.0f} un." if pd.notna(inv.get("stock_minimo", np.nan)) else "N/D")
                        with c10:
                            st.metric("Stock máximo", f"{inv.get('stock_maximo', np.nan):,.0f} un." if pd.notna(inv.get("stock_maximo", np.nan)) else "N/D")

                with st.expander("📋 Datos del producto"):
                    st.dataframe(df_product, use_container_width=True)
                    colx1, colx2 = st.columns(2)
                    add_download_buttons(df_product, f"analisis_producto_{selected_product}", colx1, colx2)

    # =============================================================================
    # PÁGINA 5 - TOP MOVERS
    # =============================================================================
    elif page == "🔥 Top Movers":
        st.markdown("## 🔥 Top Movers")
        st.markdown("Rankings dinámicos de productos y señales de negocio.")

        if df_history.empty:
            render_empty_message("No hay histórico disponible para calcular Top Movers.")
        else:
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
                "🏆 Top ventas",
                "📈 Top crecimiento",
                "📉 Top caída",
                "🔮 Top forecast",
                "⚡ Top volatilidad",
                "⚠️ Top riesgo rotura",
                "🛒 Top compra recomendada",
                "🎯 Top estratégicos"
            ])

            growth_df = pd.DataFrame()

            with tab1:
                top_sales = (
                    df_history.groupby(["producto", "familia"], as_index=False)["ventas"]
                    .sum()
                    .sort_values("ventas", ascending=False)
                    .head(20)
                )
                st.dataframe(top_sales, use_container_width=True)
                st.plotly_chart(plot_top_bar(top_sales, "producto", "ventas", 20, "Top ventas"), use_container_width=True)

            with tab2:
                daily = (
                    df_history.groupby(["producto", "fecha"], as_index=False)["ventas"]
                    .sum()
                    .sort_values("fecha")
                )
                max_date = daily["fecha"].max()

                recent = (
                    daily[daily["fecha"] >= (max_date - timedelta(days=30))]
                    .groupby("producto", as_index=False)["ventas"]
                    .sum()
                    .rename(columns={"ventas": "ventas_recientes"})
                )

                previous = (
                    daily[
                        (daily["fecha"] >= (max_date - timedelta(days=60))) &
                        (daily["fecha"] < (max_date - timedelta(days=30)))
                    ]
                    .groupby("producto", as_index=False)["ventas"]
                    .sum()
                    .rename(columns={"ventas": "ventas_anteriores"})
                )

                growth_df = recent.merge(previous, on="producto", how="left").fillna(0)
                growth_df["crecimiento_pct"] = np.where(
                    growth_df["ventas_anteriores"] > 0,
                    (growth_df["ventas_recientes"] - growth_df["ventas_anteriores"]) / growth_df["ventas_anteriores"] * 100,
                    100
                )
                growth_df = growth_df.sort_values("crecimiento_pct", ascending=False).head(20)
                st.dataframe(growth_df, use_container_width=True)

            with tab3:
                if not growth_df.empty:
                    decline_df = growth_df.sort_values("crecimiento_pct", ascending=True).head(20)
                    st.dataframe(decline_df, use_container_width=True)
                else:
                    render_empty_message("No fue posible calcular ranking de caída.")

            with tab4:
                if not df_forecast.empty and "producto" in df_forecast.columns:
                    top_fc = (
                        df_forecast.groupby("producto", as_index=False)["forecast"]
                        .sum()
                        .sort_values("forecast", ascending=False)
                        .head(20)
                    )
                    st.dataframe(top_fc, use_container_width=True)
                    st.plotly_chart(plot_top_bar(top_fc, "producto", "forecast", 20, "Top forecast"), use_container_width=True)
                else:
                    render_empty_message("No existe forecast por producto para esta vista.")

            with tab5:
                vol = (
                    df_history.groupby("producto", as_index=False)
                    .agg(media=("ventas", "mean"), std=("ventas", "std"), dias=("ventas", "count"))
                )
                vol["cv_pct"] = np.where(vol["media"] > 0, vol["std"] / vol["media"] * 100, np.nan)
                vol = vol.sort_values("cv_pct", ascending=False).head(20)
                st.dataframe(vol, use_container_width=True)

            with tab6:
                if not df_risk.empty:
                    risk_top = df_risk.sort_values("riesgo_rotura", ascending=False).head(20)
                    st.dataframe(risk_top, use_container_width=True)
                else:
                    render_empty_message("No hay archivo de riesgo de rotura disponible.")

            with tab7:
                if not df_purchases.empty:
                    top_buy = df_purchases.sort_values("compra_recomendada", ascending=False).head(20)
                    st.dataframe(top_buy, use_container_width=True)
                else:
                    render_empty_message("No hay archivo de compras recomendadas disponible.")

            with tab8:
                abc = calculate_abc_classification(df_history)
                if not abc.empty:
                    strategic = abc[abc["clase_abc"] == "A"].sort_values("ventas", ascending=False).head(20)
                    st.dataframe(strategic, use_container_width=True)
                else:
                    render_empty_message("No se pudo calcular clasificación ABC.")

    # =============================================================================
    # PÁGINA 6 - ESTACIONALIDAD
    # =============================================================================
    elif page == "📅 Estacionalidad":
        st.markdown("## 📅 Estacionalidad")
        st.markdown("Patrones temporales, estacionalidad e insights automáticos.")

        if df_history.empty:
            render_empty_message("No hay histórico disponible para análisis de estacionalidad.")
        else:
            seasonality = calculate_seasonality(df_history)

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(plot_seasonality_year_month(df_history), use_container_width=True)
            with c2:
                st.plotly_chart(plot_heatmap_month_day(df_history), use_container_width=True)

            c3, c4 = st.columns(2)
            with c3:
                st.plotly_chart(plot_heatmap_weekday(df_history), use_container_width=True)
            with c4:
                st.plotly_chart(plot_sales_evolution(df_history, "M", "ventas", "Tendencia mensual"), use_container_width=True)

            if seasonality:
                months = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
                days = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

                c5, c6, c7 = st.columns(3)
                with c5:
                    monthly_df = pd.DataFrame({
                        "Mes": months,
                        "Índice %": [seasonality["monthly"].get(i + 1, np.nan) for i in range(12)]
                    })
                    fig_month = px.bar(
                        monthly_df,
                        x="Mes",
                        y="Índice %",
                        title="Estacionalidad mensual",
                        color="Índice %",
                        color_continuous_scale="Blues"
                    )
                    fig_month.update_layout(template="plotly_white")
                    st.plotly_chart(fig_month, use_container_width=True)

                with c6:
                    dow_df = pd.DataFrame({
                        "Día": days,
                        "Índice %": [seasonality["dow"].get(i, np.nan) for i in range(7)]
                    })
                    fig_dow = px.bar(
                        dow_df,
                        x="Día",
                        y="Índice %",
                        title="Estacionalidad por día de semana",
                        color="Índice %",
                        color_continuous_scale="Blues"
                    )
                    fig_dow.update_layout(template="plotly_white")
                    st.plotly_chart(fig_dow, use_container_width=True)

                with c7:
                    quarter_df = pd.DataFrame({
                        "Trimestre": ["Q1", "Q2", "Q3", "Q4"],
                        "Índice %": [seasonality["quarter"].get(i + 1, np.nan) for i in range(4)]
                    })
                    fig_q = px.bar(
                        quarter_df,
                        x="Trimestre",
                        y="Índice %",
                        title="Estacionalidad trimestral",
                        color="Índice %",
                        color_continuous_scale="Blues"
                    )
                    fig_q.update_layout(template="plotly_white")
                    st.plotly_chart(fig_q, use_container_width=True)

                st.markdown("---")
                st.markdown("### 🧠 Insights automáticos")

                if not seasonality["monthly"].empty:
                    best_month_idx = seasonality["monthly"].idxmax()
                    worst_month_idx = seasonality["monthly"].idxmin()

                    best_month_name = months[best_month_idx - 1]
                    worst_month_name = months[worst_month_idx - 1]

                    st.info(f"{best_month_name} presenta un {seasonality['monthly'].max():.1f}% más ventas que la media anual.")
                    st.info(f"{worst_month_name} presenta un {abs(seasonality['monthly'].min()):.1f}% menos ventas que la media anual.")

                if not seasonality["dow"].empty:
                    best_day_idx = int(seasonality["dow"].idxmax())
                    best_day_name = days[best_day_idx]
                    st.info(f"{best_day_name} concentra la mayor intensidad relativa de demanda semanal (+{seasonality['dow'].max():.1f}%).")

            st.markdown("---")
            st.markdown("### 📅 Calendar View simplificado")

            calendar_df = (
                df_history.groupby("fecha", as_index=False)["ventas"]
                .sum()
                .sort_values("fecha")
            )
            calendar_df["año"] = calendar_df["fecha"].dt.year
            calendar_df["mes"] = calendar_df["fecha"].dt.month
            calendar_df["día"] = calendar_df["fecha"].dt.day

            latest_year = calendar_df["año"].max()
            cal_year = calendar_df[calendar_df["año"] == latest_year]

            fig_calendar = px.scatter(
                cal_year,
                x="día",
                y="mes",
                size="ventas",
                color="ventas",
                color_continuous_scale="Blues",
                title=f"Calendar View {latest_year}"
            )
            fig_calendar.update_layout(template="plotly_white")
            st.plotly_chart(fig_calendar, use_container_width=True)

    # =============================================================================
    # PÁGINA 7 - ESCENARIOS DE NEGOCIO
    # =============================================================================
    elif page == "🎯 Escenarios de Negocio":
        st.markdown("## 🎯 Escenarios de Negocio")
        st.markdown("Simulador estratégico de demanda, forecast e impacto operativo.")

        if df_forecast.empty:
            render_empty_message("No hay forecast disponible para simular escenarios.")
        else:
            st.markdown("### 🔧 Parámetros de simulación")

            col1, col2, col3 = st.columns(3)

            with col1:
                demand_change = st.slider("Cambio demanda (%)", -50, 100, 0, 5)
                annual_growth = st.slider("Crecimiento anual (%)", 0, 50, 0, 2)
                promotions = st.slider("Promociones (%)", 0, 80, 0, 5)

            with col2:
                new_stores = st.slider("Apertura nuevas tiendas", 0, 50, 0, 1)
                inflation = st.slider("Inflación (%)", 0, 20, 0, 1)
                lead_time = st.slider("Lead Time (días)", 1, 30, 7, 1)

            with col3:
                service_level = st.slider("Nivel de servicio (%)", 80.0, 99.9, 95.0, 0.1)
                safety_stock_days = st.slider("Días stock seguridad", 0, 30, 7, 1)

            fc_daily = df_forecast.groupby("fecha", as_index=False)["forecast"].sum().sort_values("fecha")

            base_forecast_total = fc_daily["forecast"].sum()
            base_daily = fc_daily["forecast"].mean() if len(fc_daily) > 0 else 0

            demand_factor = 1 + demand_change / 100
            growth_factor = 1 + annual_growth / 100
            promo_factor = 1 + promotions / 100
            store_factor = 1 + new_stores * 0.02
            inflation_factor = 1 + inflation / 100

            total_factor = demand_factor * growth_factor * promo_factor * store_factor
            adjusted_forecast = base_forecast_total * total_factor * inflation_factor
            adjusted_daily = base_daily * total_factor * inflation_factor

            hist_std = df_history["ventas"].std() if not df_history.empty else max(base_daily * 0.3, 1)

            z_map = {
                80: 0.84,
                85: 1.04,
                90: 1.28,
                95: 1.65,
                97: 1.88,
                99: 2.33
            }
            closest_key = min(z_map.keys(), key=lambda x: abs(x - int(round(service_level))))
            z_score = z_map[closest_key]

            safety_stock = z_score * hist_std * np.sqrt(lead_time) + adjusted_daily * safety_stock_days
            reorder_point = adjusted_daily * lead_time + safety_stock
            avg_inventory = adjusted_daily * lead_time / 2 + safety_stock

            risk_factor = 0
            if base_daily > 0:
                risk_factor = min(max((hist_std / max(base_daily, 1)) * lead_time / 30 * 100, 0), 100)

            compra_recomendada = 0
            coverage = np.nan

            if not df_inventory.empty:
                stock_total = df_inventory["stock_actual"].fillna(0).sum()
                coverage = stock_total / adjusted_daily if adjusted_daily > 0 else np.nan
                compra_recomendada = max(reorder_point - stock_total, 0)

            st.markdown("---")
            st.markdown("### 📊 Impacto del escenario")

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            with c1:
                st.metric("Forecast", format_currency(adjusted_forecast, currency), f"{(total_factor - 1) * 100:+.1f}%")
            with c2:
                st.metric("Ventas esperadas / día", format_currency(adjusted_daily, currency))
            with c3:
                st.metric("Stock seguridad", f"{safety_stock:,.0f} un.")
            with c4:
                st.metric("Punto de reorden", f"{reorder_point:,.0f} un.")
            with c5:
                st.metric("Compra recomendada", f"{compra_recomendada:,.0f} un.")
            with c6:
                st.metric("Cobertura", f"{coverage:.1f} días" if pd.notna(coverage) else "N/D")

            c7, c8, c9 = st.columns(3)
            with c7:
                st.metric("Inventario promedio", f"{avg_inventory:,.0f} un.")
            with c8:
                st.metric("Riesgo rotura", f"{risk_factor:.1f}%")
            with c9:
                st.metric("Nivel de servicio", f"{service_level:.1f}%")

            st.markdown("---")

            cc1, cc2 = st.columns(2)

            with cc1:
                comparison_df = pd.DataFrame({
                    "Métrica": ["Forecast total", "Venta diaria", "Stock seguridad", "Punto reorden"],
                    "Base": [base_forecast_total, base_daily, base_daily * 7, base_daily * 7 + base_daily * 7],
                    "Escenario": [adjusted_forecast, adjusted_daily, safety_stock, reorder_point]
                })

                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(
                    x=comparison_df["Métrica"],
                    y=comparison_df["Base"],
                    name="Base",
                    marker_color=COLOR_GRAY
                ))
                fig_comp.add_trace(go.Bar(
                    x=comparison_df["Métrica"],
                    y=comparison_df["Escenario"],
                    name="Escenario",
                    marker_color=COLOR_MID
                ))
                fig_comp.update_layout(
                    title="Comparación base vs escenario",
                    template="plotly_white",
                    barmode="group"
                )
                st.plotly_chart(fig_comp, use_container_width=True)

            with cc2:
                impact_df = pd.DataFrame({
                    "Variable": [
                        f"Demanda {demand_change:+d}%",
                        f"Crecimiento {annual_growth:+d}%",
                        f"Promociones {promotions:+d}%",
                        f"Nuevas tiendas +{new_stores}",
                        f"Inflación {inflation:+d}%"
                    ],
                    "Impacto %": [
                        demand_change,
                        annual_growth,
                        promotions,
                        new_stores * 2,
                        inflation
                    ]
                })

                fig_impact = px.bar(
                    impact_df,
                    x="Variable",
                    y="Impacto %",
                    title="Impacto por variable",
                    color="Impacto %",
                    color_continuous_scale="Blues"
                )
                fig_impact.update_layout(template="plotly_white")
                st.plotly_chart(fig_impact, use_container_width=True)

            st.markdown("---")
            st.markdown("### 💡 Recomendaciones automáticas")

            recs = []

            if demand_change > 20:
                recs.append("Aumento relevante de demanda: revisar capacidad de suministro y disponibilidad comercial.")
            if lead_time > 14:
                recs.append("Lead Time elevado: revisar proveedores o anticipar pedidos.")
            if service_level >= 99:
                recs.append("Nivel de servicio muy alto: puede tensionar inventario y capital inmovilizado.")
            if promotions > 30:
                recs.append("Promociones agresivas: validar impacto sobre forecast y cobertura.")
            if new_stores > 10:
                recs.append("Expansión comercial relevante: revisar planificación de distribución y forecast por cluster.")
            if not recs:
                recs.append("Escenario equilibrado: no se detectan alertas críticas en los parámetros simulados.")

            for r in recs:
                st.info(r)

            with st.expander("📋 Exportar escenario"):
                scenario_df = pd.DataFrame({
                    "Métrica": [
                        "Forecast base",
                        "Forecast ajustado",
                        "Venta diaria ajustada",
                        "Stock seguridad",
                        "Punto de reorden",
                        "Compra recomendada",
                        "Inventario promedio",
                        "Riesgo rotura",
                        "Cobertura"
                    ],
                    "Valor": [
                        f"{base_forecast_total:,.2f}",
                        f"{adjusted_forecast:,.2f}",
                        f"{adjusted_daily:,.2f}",
                        f"{safety_stock:,.2f}",
                        f"{reorder_point:,.2f}",
                        f"{compra_recomendada:,.2f}",
                        f"{avg_inventory:,.2f}",
                        f"{risk_factor:.2f}%",
                        f"{coverage:.2f} días" if pd.notna(coverage) else "N/D"
                    ]
                })
                st.dataframe(scenario_df, use_container_width=True)
                colx1, colx2 = st.columns(2)
                add_download_buttons(scenario_df, "escenario_de_negocio", colx1, colx2)

    # -------------------------------------------------------------------------
    # Footer
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #64748B; padding: 20px;'>
        <p><strong>Sales & Demand Intelligence Center</strong> | v2.1.0</p>
        <p>Aplicación corporativa de análisis comercial, forecast y estacionalidad</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()