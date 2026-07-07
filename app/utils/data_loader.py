"""
Data loading utilities for the inventory simulator.
Handles loading and validation of CSV files.
"""

import pandas as pd
import os
from typing import Optional, Dict


# Default data directory path
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data"
)


def load_csv_safe(
    filepath: str,
    required_columns: Optional[list] = None,
    **kwargs
) -> Optional[pd.DataFrame]:
    """
    Safely load a CSV file with error handling and validation.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    required_columns : list, optional
        List of required column names to validate
    **kwargs
        Additional arguments passed to pd.read_csv
    
    Returns:
    --------
    pd.DataFrame or None
        Loaded DataFrame or None if loading failed
    """
    try:
        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}")
            return None
        
        df = pd.read_csv(filepath, **kwargs)
        
        if required_columns:
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                print(f"Warning: Missing columns in {filepath}: {missing}")
                return None
        
        return df
        
    except Exception as e:
        print(f"Error loading {filepath}: {str(e)}")
        return None


def load_forecast_7d(data_dir: str = DEFAULT_DATA_DIR) -> Optional[pd.DataFrame]:
    """Load 7-day forecast data."""
    filepath = os.path.join(data_dir, "forecast_7d.csv")
    return load_csv_safe(filepath)


def load_forecast_30d(data_dir: str = DEFAULT_DATA_DIR) -> Optional[pd.DataFrame]:
    """Load 30-day forecast data."""
    filepath = os.path.join(data_dir, "forecast_30d.csv")
    return load_csv_safe(filepath)


def load_forecast_90d(data_dir: str = DEFAULT_DATA_DIR) -> Optional[pd.DataFrame]:
    """Load 90-day forecast data."""
    filepath = os.path.join(data_dir, "forecast_90d.csv")
    return load_csv_safe(filepath)


def load_inventario_productos(data_dir: str = DEFAULT_DATA_DIR) -> Optional[pd.DataFrame]:
    """Load product inventory data."""
    filepath = os.path.join(data_dir, "inventario_productos.csv")
    return load_csv_safe(filepath)


def load_riesgo_rotura(data_dir: str = DEFAULT_DATA_DIR) -> Optional[pd.DataFrame]:
    """Load stockout risk data."""
    filepath = os.path.join(data_dir, "riesgo_rotura.csv")
    return load_csv_safe(filepath)


def load_compras_recomendadas(data_dir: str = DEFAULT_DATA_DIR) -> Optional[pd.DataFrame]:
    """Load recommended purchases data."""
    filepath = os.path.join(data_dir, "compras_recomendadas.csv")
    return load_csv_safe(filepath)


def load_all_data(data_dir: str = DEFAULT_DATA_DIR) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Load all data files and return as a dictionary.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the CSV files
    
    Returns:
    --------
    dict
        Dictionary with keys for each dataset
    """
    datasets = {
        'forecast_7d': load_forecast_7d(data_dir),
        'forecast_30d': load_forecast_30d(data_dir),
        'forecast_90d': load_forecast_90d(data_dir),
        'inventario': load_inventario_productos(data_dir),
        'riesgo_rotura': load_riesgo_rotura(data_dir),
        'compras_recomendadas': load_compras_recomendadas(data_dir)
    }
    
    # Print summary of loaded data
    for name, df in datasets.items():
        if df is not None:
            print(f"✓ Loaded {name}: {len(df)} rows, {len(df.columns)} columns")
        else:
            print(f"✗ Failed to load {name}")
    
    return datasets


def get_available_families(datasets: Dict[str, pd.DataFrame]) -> list:
    """
    Get list of unique product families from loaded data.
    
    Parameters:
    -----------
    datasets : dict
        Dictionary of loaded DataFrames
    
    Returns:
    --------
    list
        Sorted list of unique families
    """
    # Try multiple possible column names
    family_columns = ['familia', 'FAMILIA', 'family', 'categoria', 'categoria']
    
    for df in datasets.values():
        if df is None:
            continue
        for col in family_columns:
            if col in df.columns:
                return sorted(df[col].unique().tolist())
    
    return []


def get_available_abc_classes(datasets: Dict[str, pd.DataFrame]) -> list:
    """
    Get list of unique ABC classes from loaded data.
    
    Parameters:
    -----------
    datasets : dict
        Dictionary of loaded DataFrames
    
    Returns:
    --------
    list
        Sorted list of unique ABC classes
    """
    abc_columns = ['abc_class', 'ABC_CLASS', 'clase_abc', 'claseABC', 'categoria_abc']
    
    for df in datasets.values():
        if df is None:
            continue
        for col in abc_columns:
            if col in df.columns:
                return sorted(df[col].unique().tolist())
    
    return ['A', 'B', 'C']  # Default if not found


def prepare_inventory_dataset(
    inventario_df: pd.DataFrame,
    riesgo_df: pd.DataFrame,
    compras_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge inventory, risk, and purchase recommendation data into a single dataset.
    
    Parameters:
    -----------
    inventario_df : pd.DataFrame
        Inventory data
    riesgo_df : pd.DataFrame
        Stockout risk data
    compras_df : pd.DataFrame
        Recommended purchases data
    
    Returns:
    --------
    pd.DataFrame
        Merged dataset
    """
    # Identify common key columns
    key_columns = ['producto', 'PRODUCTO', 'product_id', 'ProductId', 'id_producto']
    family_columns = ['familia', 'FAMILIA', 'family']
    abc_columns = ['abc_class', 'ABC_CLASS', 'clase_abc', 'claseABC']
    
    # Find the actual column names used
    inv_keys = [col for col in key_columns if col in inventario_df.columns]
    risk_keys = [col for col in key_columns if col in riesgo_df.columns]
    comp_keys = [col for col in key_columns if col in compras_df.columns]
    
    if not inv_keys or not risk_keys or not comp_keys:
        return inventario_df
    
    key_col = inv_keys[0]
    
    # Start with inventory data
    merged = inventario_df.copy()
    
    # Merge risk data if keys match
    if risk_keys:
        risk_key = risk_keys[0]
        risk_suffix = '_risk'
        merged = merged.merge(
            riesgo_df.add_suffix(risk_suffix),
            left_on=key_col,
            right_on=f"{risk_key}{risk_suffix}",
            how='left'
        )
    
    # Merge purchase recommendations if keys match
    if comp_keys:
        comp_key = comp_keys[0]
        comp_suffix = '_comp'
        merged = merged.merge(
            compras_df.add_suffix(comp_suffix),
            left_on=key_col,
            right_on=f"{comp_key}{comp_suffix}",
            how='left'
        )
    
    return merged