# 📊 FlowMap Sales Intelligence Platform

## Descripción

Aplicación profesional en Streamlit para análisis comercial, forecasting, optimización de inventario y simulación de escenarios. Esta plataforma está pensada para líderes retail: CEO, gerentes comerciales, planners de demanda, supply chain managers y equipos de operaciones.

## Objetivo

Ofrecer una experiencia ejecutiva con:
- análisis histórico de ventas,
- forecast por horizontes 7/30/90/365 días,
- monitoreo de inventario y riesgo de rotura,
- simulador de escenarios operativos,
- recomendaciones de compra basadas en datos.

## Archivo principal

- `FlowMap_Sales_Intelligence.py`

## Estructura del proyecto

```
segundo proyecto/
├── FlowMap_Sales_Intelligence.py      # App Streamlit final
├── app/
│   ├── utils/
│   │   ├── calculations.py           # Fórmulas de inventario y riesgo
│   │   ├── data_loader.py            # Carga segura de archivos
│   │   └── exporters.py              # Exportar CSV/Excel
│   └── data/                         # Datos usados por la app
│       ├── compras_recomendadas.csv
│       ├── forecast_7d.csv
│       ├── forecast_30d.csv
│       ├── forecast_90d.csv
│       ├── forecast_365d.csv
│       ├── inventario_productos.csv
│       ├── riesgo_rotura.csv
│       ├── sales_history_real.parquet
│       ├── simulacion_final_escenarios.csv
│       └── LOGO.png
├── requirements.txt                  # Dependencias Python
└── README.md                         # Este archivo
```

## Requisitos

- Python 3.10+ recomendado
- Dependencias instaladas desde `requirements.txt`

## Instalación y ejecución

1. Abrir terminal en la carpeta `segundo proyecto`
2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

3. Ejecutar la app:

```bash
streamlit run FlowMap_Sales_Intelligence.py
```

4. Abrir en el navegador:

```text
http://localhost:8501
```

> Si el puerto 8501 ya está en uso, Streamlit elegirá otro automáticamente. También puedes especificar puerto manual:

```bash
streamlit run FlowMap_Sales_Intelligence.py --server.port 8502
```

## Contenido de la app

### Página 1 — Executive Dashboard
- Ventas totales y promedio diario
- Familias y productos activos
- Crecimiento YoY y variación mensual
- Gráficos de tendencia, comparativo año vs año, top familias/productos, participación por familia, heatmap mensual y ranking dinámico.
- Insight ejecutivo automático.

### Página 2 — Forecast + Inventario
- Selección de horizonte: 7, 30, 90, 365 días
- Forecast por familia y familia/producto
- Banda de confianza
- Comparación forecast vs histórico
- KPIs de forecast e inventario
- Alertas de stock peligroso
- Tabla detallada y exportación

### Página 3 — Simulador de Escenarios
- Filtros por escenario, familia y producto
- Evaluación de lead time, nivel de servicio, fill rate, riesgo de rotura y compra recomendada
- Comparación de escenarios, radar/columnas y conclusiones automáticas
- Tabla ejecutiva de resultados y descarga

### Página 4 — Productos Críticos
- Identificación de productos con riesgo de rotura
- KPIs de riesgo alto/medio/bajo
- Matriz y ranking de riesgo
- Riesgo por familia y cobertura vs riesgo
- Tabla ejecutiva con semáforos y exportación

## Datos utilizados

Los datos cargados están en `app/data/` y contienen:
- `sales_history_real.parquet` — histórico real de ventas
- `forecast_7d.csv`, `forecast_30d.csv`, `forecast_90d.csv`, `forecast_365d.csv`
- `inventario_productos.csv`
- `riesgo_rotura.csv`
- `compras_recomendadas.csv`
- `simulacion_final_escenarios.csv`

## Compatibilidad de columnas

La app es flexible con nombres de columnas y busca alternativas para encontrar los campos clave.

### Columnas principales
- Producto: `item_nbr`, `producto`, `product_id`, `productid`
- Familia: `familia`, `family`, `categoria`
- Clase ABC: `clase_abc`, `abc_class`, `abc`
- Forecast: `forecast`, `yhat`, `predicted_demand`
- Fecha: `date`, `fecha`, `ds`
- Stock actual: `stock_actual`, `stock`, `current_stock`
- Demanda diaria: `demanda_promedio_diaria`, `demanda_diaria`, `daily_demand`
- Riesgo: `riesgo_rotura`, `nivel_riesgo`, `probabilidad_rotura`

## Cómo preparar para producción

- Subir todo el repositorio a GitHub
- Incluir `requirements.txt` y `README.md`
- Usar `FlowMap_Sales_Intelligence.py` como entrypoint
- Asegurar que los datos estén en `app/data/`
- Para Streamlit Cloud, agregar el repo y seleccionar `FlowMap_Sales_Intelligence.py`

## Despliegue recomendado

### 1) Streamlit Cloud
- Crear repositorio GitHub
- Subir proyecto completo
- En Streamlit Cloud, conectar el repo
- Configurar `main.py` o `FlowMap_Sales_Intelligence.py` como entrypoint

### 2) GitHub Pages (solo front-end docs)
- Añadir README e instrucciones
- No sirve para ejecutar Streamlit directamente

### 3) Servidor propio
- Crear entorno virtual
- Instalar dependencias
- Ejecutar con `streamlit run FlowMap_Sales_Intelligence.py`
- Usar `nginx` o `caddy` como proxy si se necesita acceso público

## Notas finales

- Este README está preparado para que tengas la app final lista en `segundo proyecto`.
- Si deseas, puedo crear también un `requirements_prod.txt` con versiones exactas y un archivo `Procfile` para Streamlit Cloud.

**Riesgo:**
- `riesgo_rotura`, `RIESGO_ROTURA`, `stockout_risk`, `risk_level`

## 🤝 Soporte

Para consultas o problemas técnicos, contactar a:
- **FLOWMAP ANALYTICS**
- Email: soporte@flowmapanalytics.com

## 📄 Licencia

© 2026 FLOWMAP ANALYTICS - Todos los derechos reservados

---

*Documento generado el 12/06/2026*