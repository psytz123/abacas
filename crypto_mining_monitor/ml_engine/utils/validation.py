"""
Validation utilities for the ML recommendation engine.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

from .logging_config import logger

class ValidationError(Exception):
    """Exception raised for validation errors."""
    pass

def validate_dataframe(df: pd.DataFrame, required_columns: List[str], name: str = "DataFrame") -> bool:
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        name: Name of the DataFrame for error messages
        
    Returns:
        True if validation passes
        
    Raises:
        ValidationError: If validation fails
    """
    # Check if DataFrame is empty
    if df.empty:
        raise ValidationError(f"{name} is empty")
    
    # Check for required columns
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValidationError(f"{name} missing required columns: {missing_columns}")
    
    # Check for all NaN columns
    nan_columns = [col for col in required_columns if df[col].isna().all()]
    if nan_columns:
        raise ValidationError(f"{name} has all NaN values in columns: {nan_columns}")
    
    return True

def handle_missing_data(df: pd.DataFrame, strategy: str = 'conservative') -> pd.DataFrame:
    """
    Handle missing data with domain-specific strategies.
    
    Args:
        df: DataFrame with missing data
        strategy: Strategy for handling missing data ('conservative', 'mean', 'median', 'zero')
        
    Returns:
        DataFrame with missing data handled
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Log missing data statistics
    missing_stats = df.isna().sum()
    missing_cols = missing_stats[missing_stats > 0]
    if not missing_cols.empty:
        logger.info(f"Handling missing data in columns: {missing_cols.to_dict()}")
    
    # Handle missing data based on column type and domain knowledge
    if strategy == 'conservative':
        # For hashrate, use median as it's less sensitive to outliers
        if 'hashrate_th_s' in df.columns and df['hashrate_th_s'].isna().any():
            median_hashrate = df['hashrate_th_s'].median()
            df['hashrate_th_s'] = df['hashrate_th_s'].fillna(median_hashrate)
            logger.debug(f"Filled missing hashrate_th_s with median: {median_hashrate}")
        
        # For power, use median or estimate from hashrate
        if 'power_consumption_w' in df.columns and df['power_consumption_w'].isna().any():
            if df['power_consumption_w'].notna().any():
                median_power = df['power_consumption_w'].median()
                df['power_consumption_w'] = df['power_consumption_w'].fillna(median_power)
                logger.debug(f"Filled missing power_consumption_w with median: {median_power}")
            elif 'hashrate_th_s' in df.columns:
                # Estimate power from hashrate using typical efficiency
                typical_efficiency = 35.0  # J/TH
                df['power_consumption_w'] = df['power_consumption_w'].fillna(
                    df['hashrate_th_s'] * typical_efficiency
                )
                logger.debug("Estimated missing power_consumption_w from hashrate")
        
        # For efficiency, calculate from power and hashrate if possible
        if 'efficiency_j_th' in df.columns and df['efficiency_j_th'].isna().any():
            if 'power_consumption_w' in df.columns and 'hashrate_th_s' in df.columns:
                mask = df['efficiency_j_th'].isna() & df['power_consumption_w'].notna() & df['hashrate_th_s'].notna() & (df['hashrate_th_s'] > 0)
                df.loc[mask, 'efficiency_j_th'] = df.loc[mask, 'power_consumption_w'] / df.loc[mask, 'hashrate_th_s']
                logger.debug("Calculated missing efficiency_j_th from power and hashrate")
            
            # For remaining NaNs, use median
            if df['efficiency_j_th'].isna().any():
                median_efficiency = df['efficiency_j_th'].median()
                df['efficiency_j_th'] = df['efficiency_j_th'].fillna(median_efficiency)
                logger.debug(f"Filled remaining missing efficiency_j_th with median: {median_efficiency}")
        
        # For temperature, use median
        if 'avg_chip_temp_c' in df.columns and df['avg_chip_temp_c'].isna().any():
            median_temp = df['avg_chip_temp_c'].median()
            df['avg_chip_temp_c'] = df['avg_chip_temp_c'].fillna(median_temp)
            logger.debug(f"Filled missing avg_chip_temp_c with median: {median_temp}")
        
        # For categorical data like primary_coin, use mode
        if 'primary_coin' in df.columns and df['primary_coin'].isna().any():
            mode_coin = df['primary_coin'].mode()[0]
            df['primary_coin'] = df['primary_coin'].fillna(mode_coin)
            logger.debug(f"Filled missing primary_coin with mode: {mode_coin}")
    
    elif strategy == 'mean':
        # Fill all numeric columns with mean
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mean())
                logger.debug(f"Filled missing {col} with mean: {df[col].mean()}")
    
    elif strategy == 'median':
        # Fill all numeric columns with median
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
                logger.debug(f"Filled missing {col} with median: {df[col].median()}")
    
    elif strategy == 'zero':
        # Fill all numeric columns with zero
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(0)
                logger.debug(f"Filled missing {col} with zero")
    
    # Fill any remaining NaNs with appropriate defaults
    df = df.fillna({
        'fan_speed_percent': 70.0,
        'accepted_shares': 0,
        'rejected_shares': 0,
        'overclock_profile': 'normal',
        'merge_mining_enabled': False,
        'price_change_24h_percent': 0.0,
        'transaction_fees_percent': 5.0
    })
    
    return df

def handle_outliers(df: pd.DataFrame, method: str = 'winsorize') -> pd.DataFrame:
    """
    Handle outliers in numerical columns.
    
    Args:
        df: DataFrame with potential outliers
        method: Method for handling outliers ('winsorize', 'zscore')
        
    Returns:
        DataFrame with outliers handled
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Identify numerical columns
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Handle outliers based on method
    if method == 'winsorize':
        for col in num_cols:
            # Skip columns with all NaN
            if df[col].isna().all():
                continue
            
            # Calculate percentiles
            q1 = df[col].quantile(0.01)
            q3 = df[col].quantile(0.99)
            
            # Winsorize (clip) values outside the 1st and 99th percentiles
            df[col] = df[col].clip(lower=q1, upper=q3)
            
            # Log clipped values
            clipped_count = ((df[col] == q1) | (df[col] == q3)).sum()
            if clipped_count > 0:
                logger.debug(f"Clipped {clipped_count} outliers in column {col}")
    
    elif method == 'zscore':
        for col in num_cols:
            # Skip columns with all NaN
            if df[col].isna().all():
                continue
            
            # Calculate z-scores
            mean = df[col].mean()
            std = df[col].std()
            z_scores = (df[col] - mean) / std
            
            # Identify outliers (|z| > 3)
            outliers = (z_scores.abs() > 3)
            
            # Replace outliers with mean
            if outliers.any():
                df.loc[outliers, col] = mean
                logger.debug(f"Replaced {outliers.sum()} outliers in column {col}")
    
    return df
