# 🚀 FlowMap Sales Intelligence Platform

### Retail Analytics · Demand Forecasting · Inventory Optimization · Scenario Simulation

A Streamlit application designed to support data-driven decision-making across retail, supply chain, and commercial operations.

FlowMap Sales Intelligence Platform combines historical sales analytics, demand forecasting, inventory optimization, stockout risk monitoring, and scenario simulation into a single executive-grade solution.

Designed for:

* CEOs and Business Executives
* Commercial Managers
* Demand Planners
* Supply Chain Managers
* Operations Teams
* Data & Analytics Professionals

---

## Business Objective

Provide a comprehensive decision-support platform capable of:

* Analyzing historical sales performance
* Forecasting demand across multiple planning horizons
* Monitoring inventory health and stockout risk
* Simulating operational scenarios
* Generating purchasing recommendations
* Supporting strategic and operational planning

---

## Key Capabilities

✔ Executive Sales Intelligence

✔ Multi-Horizon Forecasting (7 / 30 / 90 / 365 Days)

✔ Inventory Optimization

✔ Stockout Risk Monitoring

✔ Purchasing Recommendations

✔ Scenario Simulation

✔ Interactive Business Dashboards

✔ Executive Insights & KPI Tracking

---

## Main Application

* `FlowMap_Sales_Intelligence.py`

---

## Project Structure

```text
second_project/
├── FlowMap_Sales_Intelligence.py      # Main Streamlit Application
├── app/
│   ├── utils/
│   │   ├── calculations.py           # Inventory & Risk Calculations
│   │   ├── data_loader.py            # Data Loading Utilities
│   │   └── exporters.py              # CSV / Excel Export Functions
│   └── data/
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
├── requirements.txt
└── README.md
```

---

## Technology Stack

### Data Processing

* Python
* Pandas
* NumPy
* PyArrow

### Business Intelligence & Analytics

* Streamlit
* Plotly
* Power BI

### Supply Chain & Forecasting

* Demand Forecasting Models
* Inventory Optimization Logic
* Stockout Risk Analysis
* Scenario Simulation Framework

---

## Requirements

* Python 3.10+
* Dependencies installed from `requirements.txt`

---

## Installation

Open a terminal in the project directory and install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run FlowMap_Sales_Intelligence.py
```

Open:

```text
http://localhost:8501
```

If port 8501 is already in use:

```bash
streamlit run FlowMap_Sales_Intelligence.py --server.port 8502
```

---

# Application Modules

## 1. Executive Dashboard

Executive-level overview of business performance.

### Features

* Total Sales
* Average Daily Sales
* Active Product Families
* Active Products
* Year-over-Year Growth
* Monthly Performance Trends
* Interactive Filters
* Executive Business Insights

<img width="1848" height="1792" alt="image" src="https://github.com/user-attachments/assets/b4add7e5-84c8-44ca-894f-0ab8ae4d79c6" />

### Visualizations

* Sales Trend Analysis
* Year vs Previous Year Comparison
* Top Product Families
* Top Products
* Family Contribution Analysis
* Monthly Heatmap
* Dynamic Rankings

---

## 2. Forecast & Inventory

Integrated demand planning and inventory monitoring environment.

### Features

* Forecast Horizon Selection:

  * 7 Days
  * 30 Days
  * 90 Days
  * 365 Days

* Confidence Intervals

* Historical vs Forecast Comparison

* Forecast KPIs

* Inventory KPIs

* Low Stock Alerts

* Detailed Analysis Tables

* Data Export

---

## 3. Scenario Simulator

Evaluate operational decisions before implementation.

### Features

* Scenario Selection
* Family & Product Filters
* Lead Time Analysis
* Service Level Analysis
* Fill Rate Evaluation
* Stockout Risk Assessment
* Recommended Purchase Quantities

### Visualizations

* Scenario Comparison
* Risk Analysis
* Performance Radar Charts
* Operational Impact Assessment

---

## 4. Critical Products Monitor

Identify products with the highest operational risk.

### Features

* High / Medium / Low Risk Classification
* Stockout Risk Ranking
* Risk Matrix
* Inventory Coverage Analysis
* Family-Level Risk Assessment
* Executive Action Tables

---

## Data Sources

Data files are stored in:

```text
app/data/
```

### Available Datasets

* `sales_history_real.parquet`
* `forecast_7d.csv`
* `forecast_30d.csv`
* `forecast_90d.csv`
* `forecast_365d.csv`
* `inventario_productos.csv`
* `riesgo_rotura.csv`
* `compras_recomendadas.csv`
* `simulacion_final_escenarios.csv`

---

## Deployment

### Streamlit Cloud

1. Push repository to GitHub
2. Connect repository to Streamlit Cloud
3. Select `FlowMap_Sales_Intelligence.py` as entry point
4. Deploy

### Self-Hosted Server

```bash
streamlit run FlowMap_Sales_Intelligence.py
```

Optionally use:

* Nginx
* Caddy
* Reverse Proxy Configuration

for public access and production environments.

---

## Business Value

FlowMap enables retail organizations to:

* Improve demand planning accuracy
* Reduce stockout risk
* Optimize inventory levels
* Support purchasing decisions
* Increase service levels
* Evaluate operational trade-offs
* Improve executive visibility through analytics

---

## Solution Architecture

```text
Raw Data
    ↓
Feature Engineering
    ↓
Demand Forecast Models
    ↓
Inventory Optimization
    ↓
Scenario Simulation
    ↓
Streamlit Analytics Platform
    ↓
Power BI Executive Dashboard
```

---

## Future Enhancements

* Automated Model Retraining
* Multi-Store Forecasting
* Promotion Impact Simulation
* Advanced Inventory Policies
* Real-Time Data Integration
* API-Based Deployment

---

### FLOWMAP ANALYTICS

Specialized in:

* Retail Analytics
* Business Intelligence
* Demand Forecasting
* Supply Chain Analytics
* Inventory Optimization
* Data Science Solutions
