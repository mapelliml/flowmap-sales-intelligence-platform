"""
Inventory calculation functions for the simulator.
All formulas based on standard inventory management theory.
"""

import numpy as np
from scipy import stats


def calculate_safety_stock(
    daily_demand: float,
    lead_time: int,
    service_level: float,
    demand_variability: str = "Media"
) -> float:
    """
    Calculate safety stock based on service level and demand variability.
    
    Formula: SS = Z * σ_d * √L
    where:
        Z = Z-score for service level
        σ_d = standard deviation of daily demand
        L = lead time in days
    
    Parameters:
    -----------
    daily_demand : float
        Average daily demand units
    lead_time : int
        Lead time in days
    service_level : float
        Service level as decimal (0.80 to 0.999)
    demand_variability : str
        'Baja', 'Media', or 'Alta' - affects coefficient of variation
    
    Returns:
    --------
    float : Safety stock units
    """
    # Z-score for service level
    z_score = stats.norm.ppf(service_level)
    
    # Coefficient of variation based on variability level
    cv_map = {'Baja': 0.10, 'Media': 0.20, 'Alta': 0.35}
    cv = cv_map.get(demand_variability, 0.20)
    
    # Standard deviation of daily demand
    std_demand = daily_demand * cv
    
    # Safety stock calculation
    safety_stock = z_score * std_demand * np.sqrt(lead_time)
    
    return max(0, round(safety_stock, 2))


def calculate_reorder_point(
    daily_demand: float,
    lead_time: int,
    safety_stock: float
) -> float:
    """
    Calculate reorder point (ROP).
    
    Formula: ROP = d * L + SS
    where:
        d = daily demand
        L = lead time
        SS = safety stock
    
    Parameters:
    -----------
    daily_demand : float
        Average daily demand units
    lead_time : int
        Lead time in days
    safety_stock : float
        Safety stock units
    
    Returns:
    --------
    float : Reorder point units
    """
    demand_during_lead_time = daily_demand * lead_time
    rop = demand_during_lead_time + safety_stock
    
    return max(0, round(rop, 2))


def calculate_demand_adjustment(
    base_demand: float,
    demand_variation: float,
    promotional_impact: float
) -> float:
    """
    Calculate adjusted demand considering variations and promotions.
    
    Formula: Adjusted Demand = Base * (1 + variation) * (1 + promo_impact)
    
    Parameters:
    -----------
    base_demand : float
        Base daily demand
    demand_variation : float
        Expected demand variation as decimal (-0.30 to +1.00)
    promotional_impact : float
        Promotional impact as decimal (0.00 to 0.80)
    
    Returns:
    --------
    float : Adjusted daily demand
    """
    adjusted = base_demand * (1 + demand_variation) * (1 + promotional_impact)
    return max(0, round(adjusted, 2))


def calculate_fill_rate(
    safety_stock: float,
    daily_demand: float,
    lead_time: int,
    demand_variability: str = "Media"
) -> float:
    """
    Estimate fill rate based on safety stock and demand parameters.
    
    Fill Rate ≈ 1 - (Expected Shortage / Order Quantity)
    Simplified: Based on service level probability
    
    Parameters:
    -----------
    safety_stock : float
        Safety stock units
    daily_demand : float
        Average daily demand
    lead_time : int
        Lead time in days
    demand_variability : str
        'Baja', 'Media', or 'Alta'
    
    Returns:
    --------
    float : Estimated fill rate (0 to 1)
    """
    cv_map = {'Baja': 0.10, 'Media': 0.20, 'Alta': 0.35}
    cv = cv_map.get(demand_variability, 0.20)
    std_demand = daily_demand * cv
    std_lead_time_demand = std_demand * np.sqrt(lead_time)
    
    if std_lead_time_demand == 0:
        return 1.0
    
    # Calculate the normalized safety stock (Z equivalent)
    z = safety_stock / std_lead_time_demand if std_lead_time_demand > 0 else 0
    
    # Loss function approximation
    loss = stats.norm.pdf(z) - z * (1 - stats.norm.cdf(z))
    
    # Fill rate approximation
    fill_rate = 1 - (std_lead_time_demand * loss) / max(daily_demand * lead_time, 1)
    
    return max(0, min(1, round(fill_rate, 4)))


def calculate_stock_coverage_days(
    current_stock: float,
    daily_demand: float
) -> float:
    """
    Calculate how many days current stock will last.
    
    Formula: Coverage = Current Stock / Daily Demand
    
    Parameters:
    -----------
    current_stock : float
        Current inventory units
    daily_demand : float
        Average daily demand
    
    Returns:
    --------
    float : Coverage in days
    """
    if daily_demand <= 0:
        return float('inf')
    
    return round(current_stock / daily_demand, 1)


def calculate_break_even_quantity(
    ordering_cost: float,
    annual_demand: float,
    holding_cost_rate: float,
    unit_cost: float
) -> float:
    """
    Calculate Economic Order Quantity (EOQ).
    
    Formula: EOQ = √((2 * D * S) / (H * C))
    where:
        D = annual demand
        S = ordering cost per order
        H = holding cost rate
        C = unit cost
    
    Parameters:
    -----------
    ordering_cost : float
        Cost per order
    annual_demand : float
        Annual demand units
    holding_cost_rate : float
        Annual holding cost as decimal
    unit_cost : float
        Cost per unit
    
    Returns:
    --------
    float : EOQ units
    """
    if holding_cost_rate <= 0 or unit_cost <= 0:
        return 0
    
    holding_cost = holding_cost_rate * unit_cost
    eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
    
    return round(eoq, 0)


def calculate_stockout_probability(
    current_stock: float,
    daily_demand: float,
    lead_time: int,
    demand_variability: str = "Media"
) -> float:
    """
    Calculate probability of stockout during lead time.
    
    Parameters:
    -----------
    current_stock : float
        Current inventory units
    daily_demand : float
        Average daily demand
    lead_time : int
        Lead time in days
    demand_variability : str
        'Baja', 'Media', or 'Alta'
    
    Returns:
    --------
    float : Probability of stockout (0 to 1)
    """
    cv_map = {'Baja': 0.10, 'Media': 0.20, 'Alta': 0.35}
    cv = cv_map.get(demand_variability, 0.20)
    
    mean_demand_lt = daily_demand * lead_time
    std_demand_lt = daily_demand * cv * np.sqrt(lead_time)
    
    if std_demand_lt == 0:
        return 0.0 if current_stock >= mean_demand_lt else 1.0
    
    z = (current_stock - mean_demand_lt) / std_demand_lt
    stockout_prob = 1 - stats.norm.cdf(z)
    
    return max(0, min(1, round(stockout_prob, 4)))