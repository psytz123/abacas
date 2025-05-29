"""
Feature engineering pipeline for processing mining telemetry and market data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
import os
from pathlib import Path

from ml_engine.utils.logging_config import logger
from ml_engine.utils.validation import handle_missing_data, handle_outliers, ValidationError
from ml_engine.config import FEATURE_CONFIG


class FeatureEngineeringPipeline:
    """
    Pipeline for transforming raw mining telemetry and market data into features
    suitable for machine learning models.
    """
    
    def __init__(self):
        """Initialize the feature engineering pipeline."""
        self.time_windows = FEATURE_CONFIG["time_windows"]
        self.agg_functions = FEATURE_CONFIG["aggregation_functions"]
        self.electricity_cost_per_kwh = FEATURE_CONFIG["electricity_cost_per_kwh"]
        self.baseline_percentage = FEATURE_CONFIG["baseline_percentage"]
        self.trend_analysis = FEATURE_CONFIG["trend_analysis"]
    
    def process_miner_telemetry(self, telemetry_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw miner telemetry data to extract relevant features.
        
        Args:
            telemetry_data: DataFrame containing raw miner telemetry data
                Expected columns: miner_id, timestamp, hashrate, power, temperature, etc.
                
        Returns:
            DataFrame with processed telemetry features
        """
        try:
            # Validate input
            required_columns = ['miner_id', 'timestamp', 'hashrate_th_s', 'power_consumption_w']
            self._validate_input(telemetry_data, required_columns, "Telemetry data")
            
            # Make a copy to avoid modifying the original
            telemetry_data = telemetry_data.copy()
            
            # Ensure timestamp is datetime
            telemetry_data['timestamp'] = pd.to_datetime(telemetry_data['timestamp'])
            
            # Sort by miner_id and timestamp
            telemetry_data = telemetry_data.sort_values(['miner_id', 'timestamp'])
            
            # Handle missing values
            telemetry_data = handle_missing_data(telemetry_data, strategy='conservative')
            
            # Handle outliers
            telemetry_data = handle_outliers(telemetry_data, method='winsorize')
            
            # Calculate basic efficiency metrics
            telemetry_data['efficiency_j_th'] = telemetry_data['power_consumption_w'] / telemetry_data['hashrate_th_s']
            logger.debug(f"Added efficiency_j_th column")
            
            # Calculate stability indicators using vectorized operations
            stability_features = self._calculate_stability_indicators_vectorized(telemetry_data)
            
            # Calculate health indicators using vectorized operations
            health_features = self._calculate_health_indicators_vectorized(telemetry_data)
            
            # Combine all features efficiently
            result = telemetry_data.copy()
            
            # Add stability features
            for col in stability_features.columns:
                if col not in ['miner_id', 'timestamp']:
                    result = result.merge(
                        stability_features[['miner_id', 'timestamp', col]],
                        on=['miner_id', 'timestamp'],
                        how='left'
                    )
            
            # Add health features
            for col in health_features.columns:
                if col not in ['miner_id', 'timestamp']:
                    result = result.merge(
                        health_features[['miner_id', 'timestamp', col]],
                        on=['miner_id', 'timestamp'],
                        how='left'
                    )
            
            logger.info(f"Processed {len(telemetry_data)} telemetry records into {len(result)} feature records")
            return result
            
        except Exception as e:
            logger.error(f"Error processing miner telemetry: {str(e)}")
            raise
    
    def process_pool_performance(self, pool_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process pool performance data to extract relevant features.
        
        Args:
            pool_data: DataFrame containing pool performance data
                Expected columns: worker_id, timestamp, hashrate, earnings, etc.
                
        Returns:
            DataFrame with processed pool performance features
        """
        try:
            # Validate input
            required_columns = ['worker_id', 'timestamp', 'effective_hashrate_th_s', 'earnings_usd_24h']
            self._validate_input(pool_data, required_columns, "Pool data")
            
            # Make a copy to avoid modifying the original
            pool_data = pool_data.copy()
            
            # Ensure timestamp is datetime
            pool_data['timestamp'] = pd.to_datetime(pool_data['timestamp'])
            
            # Sort by worker_id and timestamp
            pool_data = pool_data.sort_values(['worker_id', 'timestamp'])
            
            # Handle missing values
            pool_data = handle_missing_data(pool_data, strategy='conservative')
            
            # Handle outliers
            pool_data = handle_outliers(pool_data, method='winsorize')
            
            # Calculate reward efficiency metrics
            pool_data['earnings_per_th_usd'] = pool_data['earnings_usd_24h'] / pool_data['effective_hashrate_th_s']
            
            # Calculate reported vs effective hashrate ratio if reported hashrate is available
            if 'hashrate_th_s' in pool_data.columns:
                pool_data['reported_vs_effective_ratio'] = pool_data['hashrate_th_s'] / pool_data['effective_hashrate_th_s']
            
            # Calculate temporal patterns using vectorized operations
            temporal_features = self._calculate_temporal_patterns_vectorized(pool_data)
            
            # Combine all features efficiently
            result = pool_data.copy()
            
            # Add temporal features
            for col in temporal_features.columns:
                if col not in ['worker_id', 'timestamp']:
                    result = result.merge(
                        temporal_features[['worker_id', 'timestamp', col]],
                        on=['worker_id', 'timestamp'],
                        how='left'
                    )
            
            logger.info(f"Processed {len(pool_data)} pool records into {len(result)} feature records")
            return result
            
        except Exception as e:
            logger.error(f"Error processing pool performance: {str(e)}")
            raise
    
    def process_market_data(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process cryptocurrency market data to extract relevant features.
        
        Args:
            market_data: DataFrame containing market data
                Expected columns: coin_id, timestamp, price_usd, volume, etc.
                
        Returns:
            DataFrame with processed market features
        """
        try:
            # Validate input
            required_columns = ['coin_id', 'timestamp', 'price_usd', 'network_difficulty']
            self._validate_input(market_data, required_columns, "Market data")
            
            # Make a copy to avoid modifying the original
            market_data = market_data.copy()
            
            # Ensure timestamp is datetime
            market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])
            
            # Sort by coin_id and timestamp
            market_data = market_data.sort_values(['coin_id', 'timestamp'])
            
            # Handle missing values
            market_data = handle_missing_data(market_data, strategy='conservative')
            
            # Handle outliers
            market_data = handle_outliers(market_data, method='winsorize')
            
            # Calculate price dynamics using vectorized operations
            price_features = self._calculate_price_dynamics_vectorized(market_data)
            
            # Calculate mining economics using vectorized operations
            mining_features = self._calculate_mining_economics_vectorized(market_data)
            
            # Combine all features efficiently
            result = market_data.copy()
            
            # Add price features
            for col in price_features.columns:
                if col not in ['coin_id', 'timestamp']:
                    result = result.merge(
                        price_features[['coin_id', 'timestamp', col]],
                        on=['coin_id', 'timestamp'],
                        how='left'
                    )
            
            # Add mining features
            for col in mining_features.columns:
                if col not in ['coin_id', 'timestamp']:
                    result = result.merge(
                        mining_features[['coin_id', 'timestamp', col]],
                        on=['coin_id', 'timestamp'],
                        how='left'
                    )
            
            logger.info(f"Processed {len(market_data)} market records into {len(result)} feature records")
            return result
            
        except Exception as e:
            logger.error(f"Error processing market data: {str(e)}")
            raise
    
    def combine_features(self, 
                         miner_features: pd.DataFrame, 
                         pool_features: pd.DataFrame, 
                         market_features: pd.DataFrame) -> pd.DataFrame:
        """
        Combine features from different sources into a unified feature set.
        
        Args:
            miner_features: Processed miner telemetry features
            pool_features: Processed pool performance features
            market_features: Processed market data features
            
        Returns:
            DataFrame with combined features
        """
        try:
            # Validate inputs
            if miner_features.empty or pool_features.empty or market_features.empty:
                logger.warning("One or more feature sets are empty")
                return pd.DataFrame()
            
            # Log column names for debugging
            logger.debug(f"Miner features columns: {miner_features.columns.tolist()}")
            logger.debug(f"Pool features columns: {pool_features.columns.tolist()}")
            logger.debug(f"Market features columns: {market_features.columns.tolist()}")
            
            # Make copies to avoid modifying the originals
            miner_features = miner_features.copy()
            pool_features = pool_features.copy()
            market_features = market_features.copy()
            
            # Map miner_id to worker_id (assuming 1:1 mapping for simplicity)
            # In a real implementation, this would use a proper mapping table
            miner_features = miner_features.rename(columns={'miner_id': 'worker_id'})
            
            # Merge miner and pool features
            combined = pd.merge(
                miner_features,
                pool_features,
                on=['worker_id', 'timestamp'],
                how='inner',
                suffixes=('_miner', '_pool')
            )
            
            # Get the latest market data for each coin
            market_features_latest = market_features.sort_values('timestamp').groupby('coin_id').last().reset_index()
            
            # Join with market data based on primary_coin
            if 'primary_coin' in combined.columns:
                # Merge on primary_coin
                combined = pd.merge(
                    combined,
                    market_features_latest,
                    left_on='primary_coin',
                    right_on='coin_id',
                    how='left',
                    suffixes=('', '_market')
                )
            else:
                # If primary_coin is not available, use a more complex approach
                # For each row in combined, find the closest market data timestamp
                combined_with_market = []
                
                for _, row in combined.iterrows():
                    # Find market data closest to this timestamp
                    # In reality, you'd join based on the coin being mined at this timestamp
                    market_row = market_features_latest.iloc[0].to_dict()
                    combined_row = {**row.to_dict(), **market_row}
                    combined_with_market.append(combined_row)
                
                combined = pd.DataFrame(combined_with_market)
            
            # Calculate cross-source derived features
            combined['actual_profit_margin_percent'] = self._calculate_profit_margin(combined)
            
            # Calculate power cost per dollar earned if columns exist
            if 'power_consumption_w' in combined.columns and 'earnings_usd_24h' in combined.columns:
                combined['power_cost_per_dollar_earned'] = (
                    combined['power_consumption_w'] * 24 / 1000 * self.electricity_cost_per_kwh
                ) / combined['earnings_usd_24h']
            else:
                # Default value if columns are missing
                combined['power_cost_per_dollar_earned'] = pd.Series([0.2] * len(combined))
            
            # Ensure efficiency_j_th is preserved
            if 'efficiency_j_th' not in combined.columns:
                # If efficiency_j_th is not available, calculate it
                if 'power_consumption_w' in combined.columns and 'hashrate_th_s' in combined.columns:
                    combined['efficiency_j_th'] = combined['power_consumption_w'] / combined['hashrate_th_s']
                else:
                    # Default value if columns are missing
                    combined['efficiency_j_th'] = pd.Series([33.5] * len(combined))  # Typical J/TH value
            
            # Add hashrate_th_s if missing
            if 'hashrate_th_s' not in combined.columns and 'hashrate_th_s_miner' in combined.columns:
                combined['hashrate_th_s'] = combined['hashrate_th_s_miner']
            elif 'hashrate_th_s' not in combined.columns and 'effective_hashrate_th_s' in combined.columns:
                combined['hashrate_th_s'] = combined['effective_hashrate_th_s']
            elif 'hashrate_th_s' not in combined.columns:
                combined['hashrate_th_s'] = pd.Series([100.0] * len(combined))  # Default value
            
            # Calculate market-adjusted efficiency
            if 'efficiency_j_th' in combined.columns and 'price_usd' in combined.columns:
                combined['market_adjusted_efficiency'] = combined['efficiency_j_th'] / combined['price_usd']
            
            # Calculate risk metrics
            if 'hashrate_variance_24h' in combined.columns and 'price_volatility_24h' in combined.columns:
                combined['operational_risk_score'] = (
                    combined['hashrate_variance_24h'] * 0.7 + 
                    combined['price_volatility_24h'] * 0.3
                )
            
            # Handle missing values in the combined dataset
            combined = handle_missing_data(combined, strategy='conservative')
            
            # Log final columns for debugging
            logger.debug(f"Combined features columns: {combined.columns.tolist()}")
            logger.info(f"Created {len(combined)} combined feature records")
            
            return combined
            
        except Exception as e:
            logger.error(f"Error combining features: {str(e)}")
            raise
    
    def _validate_input(self, df: pd.DataFrame, required_columns: List[str], name: str) -> bool:
        """Validate input DataFrame."""
        if df.empty:
            raise ValidationError(f"{name} is empty")
        
        # Check for required columns
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValidationError(f"{name} missing required columns: {missing_columns}")
        
        return True
    
    def _calculate_stability_indicators_vectorized(self, telemetry_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate stability indicators from telemetry data using vectorized operations."""
        # Group by miner_id
        grouped = telemetry_data.groupby('miner_id')
        
        # Initialize result DataFrame
        result_rows = []
        
        # Process each miner group
        for miner_id, group in grouped:
            # Sort by timestamp
            group = group.sort_values('timestamp')
            
            # For each timestamp, calculate window-based metrics
            for i, row in group.iterrows():
                timestamp = row['timestamp']
                
                # Calculate metrics for different time windows
                stability_metrics = {'miner_id': miner_id, 'timestamp': timestamp}
                
                for window in self.time_windows:
                    window_minutes = window['minutes']
                    window_name = window['name']
                    
                    # Get data within the time window
                    start_time = timestamp - timedelta(minutes=window_minutes)
                    window_data = group[(group['timestamp'] >= start_time) & (group['timestamp'] <= timestamp)]
                    
                    if len(window_data) > 1:
                        # Calculate hashrate variance
                        stability_metrics[f'hashrate_variance_{window_name}'] = window_data['hashrate_th_s'].var()
                        
                        # Calculate temperature stability
                        if 'avg_chip_temp_c' in window_data.columns:
                            stability_metrics[f'temp_stability_{window_name}'] = window_data['avg_chip_temp_c'].std()
                        
                        # Calculate share acceptance ratio
                        if 'accepted_shares' in window_data.columns and 'rejected_shares' in window_data.columns:
                            total_shares = window_data['accepted_shares'].sum() + window_data['rejected_shares'].sum()
                            if total_shares > 0:
                                stability_metrics[f'share_acceptance_ratio_{window_name}'] = window_data['accepted_shares'].sum() / total_shares
                
                result_rows.append(stability_metrics)
        
        # Convert to DataFrame
        return pd.DataFrame(result_rows)
    
    def _calculate_health_indicators_vectorized(self, telemetry_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate health indicators from telemetry data using vectorized operations."""
        # Group by miner_id
        grouped = telemetry_data.groupby('miner_id')
        
        # Initialize result DataFrame
        result_rows = []
        
        # Process each miner group
        for miner_id, group in grouped:
            # Sort by timestamp
            group = group.sort_values('timestamp')
            
            # Calculate baseline metrics (using first 10% of data as baseline)
            baseline_count = max(1, int(len(group) * self.baseline_percentage))
            baseline = group.head(baseline_count)
            
            baseline_temp = baseline['avg_chip_temp_c'].mean() if 'avg_chip_temp_c' in baseline.columns else 65.0
            baseline_hashrate = baseline['hashrate_th_s'].mean() if baseline['hashrate_th_s'].mean() > 0 else 100.0
            
            # Calculate health metrics for each timestamp
            for i, row in group.iterrows():
                timestamp = row['timestamp']
                
                # Calculate health metrics
                health_metrics = {
                    'miner_id': miner_id,
                    'timestamp': timestamp
                }
                
                if 'avg_chip_temp_c' in row:
                    health_metrics['temp_deviation_from_baseline'] = row['avg_chip_temp_c'] - baseline_temp
                
                health_metrics['hashrate_percent_of_baseline'] = (row['hashrate_th_s'] / baseline_hashrate) * 100
                
                # Calculate fan performance index if fan data is available
                if 'fan_speed_percent' in row and 'avg_chip_temp_c' in row and row['avg_chip_temp_c'] > 0:
                    health_metrics['fan_performance_index'] = row['fan_speed_percent'] / row['avg_chip_temp_c']
                
                result_rows.append(health_metrics)
        
        # Convert to DataFrame
        return pd.DataFrame(result_rows)
    
    def _calculate_temporal_patterns_vectorized(self, pool_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate temporal patterns from pool performance data using vectorized operations."""
        # Group by worker_id
        grouped = pool_data.groupby('worker_id')
        
        # Initialize result DataFrame
        result_rows = []
        
        # Process each worker group
        for worker_id, group in grouped:
            # Sort by timestamp
            group = group.sort_values('timestamp')
            
            # Calculate temporal metrics for each timestamp
            for i, row in group.iterrows():
                timestamp = row['timestamp']
                
                # Extract time components
                hour_of_day = timestamp.hour
                day_of_week = timestamp.dayofweek
                
                # Calculate temporal metrics
                temporal_metrics = {
                    'worker_id': worker_id,
                    'timestamp': timestamp,
                    'hour_of_day': hour_of_day,
                    'day_of_week': day_of_week
                }
                
                # Calculate time-based profitability patterns
                for window in self.time_windows:
                    window_minutes = window['minutes']
                    window_name = window['name']
                    
                    # Get data within the time window
                    start_time = timestamp - timedelta(minutes=window_minutes)
                    window_data = group[(group['timestamp'] >= start_time) & (group['timestamp'] <= timestamp)]
                    
                    if len(window_data) > 1:
                        # Calculate profitability trends
                        if 'earnings_per_th_usd' in window_data.columns:
                            trend_features = self._calculate_trend_features(
                                window_data['timestamp'], 
                                window_data['earnings_per_th_usd']
                            )
                            
                            for trend_name, trend_value in trend_features.items():
                                temporal_metrics[f'{trend_name}_{window_name}'] = trend_value
                
                result_rows.append(temporal_metrics)
        
        # Convert to DataFrame
        return pd.DataFrame(result_rows)
    
    def _calculate_price_dynamics_vectorized(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate price dynamics from market data using vectorized operations."""
        # Group by coin_id
        grouped = market_data.groupby('coin_id')
        
        # Initialize result DataFrame
        result_rows = []
        
        # Process each coin group
        for coin_id, group in grouped:
            # Sort by timestamp
            group = group.sort_values('timestamp')
            
            # Calculate price dynamics for each timestamp
            for i, row in group.iterrows():
                timestamp = row['timestamp']
                
                # Calculate price dynamics metrics
                dynamics_metrics = {
                    'coin_id': coin_id,
                    'timestamp': timestamp
                }
                
                for window in self.time_windows:
                    window_minutes = window['minutes']
                    window_name = window['name']
                    
                    # Get data within the time window
                    start_time = timestamp - timedelta(minutes=window_minutes)
                    window_data = group[(group['timestamp'] >= start_time) & (group['timestamp'] <= timestamp)]
                    
                    if len(window_data) > 1:
                        # Calculate price volatility
                        if 'price_usd' in window_data.columns:
                            price_mean = window_data['price_usd'].mean()
                            if price_mean > 0:
                                dynamics_metrics[f'price_volatility_{window_name}'] = window_data['price_usd'].std() / price_mean
                            
                            # Calculate price momentum and other trend features
                            trend_features = self._calculate_trend_features(
                                window_data['timestamp'], 
                                window_data['price_usd']
                            )
                            
                            for trend_name, trend_value in trend_features.items():
                                dynamics_metrics[f'price_{trend_name}_{window_name}'] = trend_value
                
                result_rows.append(dynamics_metrics)
        
        # Convert to DataFrame
        return pd.DataFrame(result_rows)
    
    def _calculate_mining_economics_vectorized(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate mining economics from market data using vectorized operations."""
        # Group by coin_id
        grouped = market_data.groupby('coin_id')
        
        # Initialize result DataFrame
        result_rows = []
        
        # Process each coin group
        for coin_id, group in grouped:
            # Sort by timestamp
            group = group.sort_values('timestamp')
            
            # Calculate mining economics for each timestamp
            for i, row in group.iterrows():
                timestamp = row['timestamp']
                
                # Calculate mining economics metrics
                economics_metrics = {
                    'coin_id': coin_id,
                    'timestamp': timestamp
                }
                
                # Calculate mining profitability indicators
                if all(col in row for col in ['price_usd', 'block_reward_usd', 'network_difficulty']) and row['network_difficulty'] > 0:
                    # Simple profitability indicator (higher is better)
                    economics_metrics['mining_profitability_indicator'] = (row['block_reward_usd'] * row['price_usd']) / row['network_difficulty']
                
                # Calculate difficulty adjusted return
                for window in self.time_windows:
                    window_minutes = window['minutes']
                    window_name = window['name']
                    
                    # Get data within the time window
                    start_time = timestamp - timedelta(minutes=window_minutes)
                    window_data = group[(group['timestamp'] >= start_time) & (group['timestamp'] <= timestamp)]
                    
                    if len(window_data) > 1:
                        # Calculate difficulty trend
                        if 'network_difficulty' in window_data.columns:
                            trend_features = self._calculate_trend_features(
                                window_data['timestamp'], 
                                window_data['network_difficulty']
                            )
                            
                            for trend_name, trend_value in trend_features.items():
                                economics_metrics[f'difficulty_{trend_name}_{window_name}'] = trend_value
                
                result_rows.append(economics_metrics)
        
        # Convert to DataFrame
        return pd.DataFrame(result_rows)
    
    def _calculate_trend_features(self, time_series: pd.Series, values: pd.Series) -> Dict[str, float]:
        """
        Calculate multiple trend-related features from time series data.
        
        Args:
            time_series: Series of timestamps
            values: Series of values
            
        Returns:
            Dictionary of trend features
        """
        if len(time_series) < 2 or len(values) < 2:
            return {'trend': 0, 'momentum': 0, 'volatility': 0}
        
        # Convert timestamps to numeric (seconds since epoch)
        x_numeric = np.array([(t - time_series.min()).total_seconds() for t in time_series])
        
        # Simple linear trend
        slope, intercept = np.polyfit(x_numeric, values, 1)
        
        # Non-linear trend (polynomial)
        poly_degree = self.trend_analysis.get('polynomial_degree', 3)
        if len(values) > poly_degree:
            try:
                poly_coefs = np.polyfit(x_numeric, values, poly_degree)
                poly_trend = poly_coefs[0]  # Highest order coefficient
            except:
                poly_trend = 0
        else:
            poly_trend = 0
        
        # Volatility (standard deviation)
        volatility = values.std()
        
        # Momentum (recent trend vs overall trend)
        recent_percentage = self.trend_analysis.get('recent_percentage', 0.3)
        recent_idx = int(len(values) * recent_percentage)  # Last 30%
        
        if recent_idx > 1:
            try:
                recent_slope, _ = np.polyfit(x_numeric[-recent_idx:], values[-recent_idx:], 1)
                momentum = recent_slope / slope if slope != 0 else 0
            except:
                momentum = 0
        else:
            momentum = 0
        
        return {
            'trend': slope,
            'trend_intercept': intercept,
            'poly_trend': poly_trend,
            'volatility': volatility,
            'momentum': momentum
        }
    
    def _calculate_profit_margin(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate profit margin percentage.
        
        Args:
            data: DataFrame with earnings and power consumption data
            
        Returns:
            Series with profit margin percentages
        """
        # Check if the required columns exist
        if 'power_consumption_w' not in data.columns or 'earnings_usd_24h' not in data.columns:
            # Return a default value if columns are missing
            return pd.Series([50.0] * len(data))
        
        # Calculate daily power cost (power_consumption_w is in watts)
        # 24 hours * power in kW * cost per kWh
        daily_power_cost = data['power_consumption_w'] * 24 / 1000 * self.electricity_cost_per_kwh
        
        # Calculate profit margin
        revenue = data['earnings_usd_24h']
        profit = revenue - daily_power_cost
        
        # Avoid division by zero
        profit_margin = np.where(revenue > 0, (profit / revenue) * 100, 0)
        
        return pd.Series(profit_margin, index=data.index)


class MockDataGenerator:
    """
    Generate mock data for development and testing of the feature engineering pipeline.
    """
    
    def __init__(self, num_miners: int = 5, num_coins: int = 3, days: int = 7, interval_minutes: int = 60):
        """
        Initialize the mock data generator.
        
        Args:
            num_miners: Number of miners to simulate
            num_coins: Number of cryptocurrencies to simulate
            days: Number of days of historical data to generate
            interval_minutes: Interval between data points in minutes
        """
        self.num_miners = num_miners
        self.num_coins = num_coins
        self.days = days
        self.interval_minutes = interval_minutes
        
        # Generate miner IDs
        self.miner_ids = [f"miner_{i:03d}" for i in range(1, num_miners + 1)]
        
        # Generate worker IDs (1:1 mapping with miners for simplicity)
        self.worker_ids = [f"worker_{i:03d}" for i in range(1, num_miners + 1)]
        
        # Generate coin IDs
        self.coin_ids = ["BTC", "ETH", "LTC"][:num_coins]
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        logger.info(f"Initialized mock data generator with {num_miners} miners, {num_coins} coins, {days} days of data")
    
    def generate_miner_telemetry(self) -> pd.DataFrame:
        """
        Generate mock miner telemetry data.
        
        Returns:
            DataFrame with mock telemetry data
        """
        # Calculate number of data points
        num_intervals = (self.days * 24 * 60) // self.interval_minutes
        
        data = []
        
        for miner_id in self.miner_ids:
            # Base parameters for this miner
            base_hashrate = np.random.uniform(80, 120)  # TH/s
            base_power = np.random.uniform(3000, 3500)  # Watts
            base_temp = np.random.uniform(60, 70)  # Celsius
            
            # Generate time series with some realistic patterns
            end_time = datetime.now()
            start_time = end_time - timedelta(days=self.days)
            
            for i in range(num_intervals):
                timestamp = start_time + timedelta(minutes=i * self.interval_minutes)
                
                # Add time-based patterns
                hour_factor = 1.0 + 0.05 * np.sin(timestamp.hour * np.pi / 12)  # Daily cycle
                day_factor = 1.0 + 0.02 * np.sin(timestamp.weekday() * np.pi / 3.5)  # Weekly cycle
                
                # Add some random noise
                noise_factor = np.random.normal(1.0, 0.03)
                
                # Calculate values with patterns and noise
                hashrate = base_hashrate * hour_factor * day_factor * noise_factor
                power = base_power * (0.9 + 0.1 * (hashrate / base_hashrate))  # Power correlates with hashrate
                temp = base_temp * (0.95 + 0.05 * (hashrate / base_hashrate))  # Temperature correlates with hashrate
                
                # Add some random errors and anomalies
                if np.random.random() < 0.01:  # 1% chance of an anomaly
                    hashrate *= np.random.uniform(0.7, 0.9)  # Sudden drop in hashrate
                    temp *= np.random.uniform(1.1, 1.2)  # Temperature spike
                
                # Generate accepted and rejected shares
                shares_per_interval = int(hashrate * 10)  # Approximate shares based on hashrate
                rejected_rate = np.random.uniform(0.001, 0.01)  # 0.1% to 1% rejection rate
                accepted_shares = int(shares_per_interval * (1 - rejected_rate))
                rejected_shares = int(shares_per_interval * rejected_rate)
                
                # Fan speed correlates with temperature
                fan_speed = min(100, 50 + (temp - base_temp) * 5)
                
                data.append({
                    'miner_id': miner_id,
                    'timestamp': timestamp,
                    'hashrate_th_s': hashrate,
                    'power_consumption_w': power,
                    'avg_chip_temp_c': temp,
                    'max_chip_temp_c': temp + np.random.uniform(2, 5),
                    'fan_speed_percent': fan_speed,
                    'accepted_shares': accepted_shares,
                    'rejected_shares': rejected_shares,
                    'overclock_profile': 'normal'
                })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} miner telemetry records")
        return df
    
    def generate_pool_performance(self) -> pd.DataFrame:
        """
        Generate mock pool performance data.
        
        Returns:
            DataFrame with mock pool performance data
        """
        # Calculate number of data points (pool data might be less frequent)
        pool_interval_minutes = self.interval_minutes * 2  # Pool data every 2 hours
        num_intervals = (self.days * 24 * 60) // pool_interval_minutes
        
        data = []
        
        for i, worker_id in enumerate(self.worker_ids):
            # Map to corresponding miner for consistency
            miner_id = self.miner_ids[i]
            
            # Base parameters
            base_effective_hashrate = np.random.uniform(75, 115)  # TH/s
            base_earnings = np.random.uniform(10, 20)  # USD per day per 100 TH/s
            
            # Assign a primary coin to this worker
            primary_coin = np.random.choice(self.coin_ids)
            
            end_time = datetime.now()
            start_time = end_time - timedelta(days=self.days)
            
            for i in range(num_intervals):
                timestamp = start_time + timedelta(minutes=i * pool_interval_minutes)
                
                # Add time-based patterns
                hour_factor = 1.0 + 0.03 * np.sin(timestamp.hour * np.pi / 12)  # Daily cycle
                day_factor = 1.0 + 0.01 * np.sin(timestamp.weekday() * np.pi / 3.5)  # Weekly cycle
                
                # Add some random noise
                noise_factor = np.random.normal(1.0, 0.02)
                
                # Calculate values with patterns and noise
                effective_hashrate = base_effective_hashrate * hour_factor * day_factor * noise_factor
                
                # Earnings fluctuate based on market conditions
                market_factor = 1.0 + 0.1 * np.sin(i * np.pi / (num_intervals / 4))  # Market cycles
                earnings_per_th = base_earnings * market_factor * noise_factor
                earnings_usd_24h = effective_hashrate * earnings_per_th / 100
                
                # Reported hashrate is usually higher than effective
                reported_hashrate = effective_hashrate * np.random.uniform(1.02, 1.08)
                
                data.append({
                    'worker_id': worker_id,
                    'miner_id': miner_id,  # For easier joining
                    'timestamp': timestamp,
                    'effective_hashrate_th_s': effective_hashrate,
                    'hashrate_th_s': reported_hashrate,
                    'earnings_usd_24h': earnings_usd_24h,
                    'primary_coin': primary_coin,
                    'merge_mining_enabled': np.random.choice([True, False], p=[0.3, 0.7])
                })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} pool performance records")
        return df
    
    def generate_market_data(self) -> pd.DataFrame:
        """
        Generate mock cryptocurrency market data.
        
        Returns:
            DataFrame with mock market data
        """
        # Market data might be more frequent
        market_interval_minutes = self.interval_minutes // 2
        num_intervals = (self.days * 24 * 60) // market_interval_minutes
        
        data = []
        
        for coin_id in self.coin_ids:
            # Base parameters for this coin
            if coin_id == "BTC":
                base_price = np.random.uniform(50000, 60000)
                base_difficulty = 30e12
                base_block_reward = 6.25
            elif coin_id == "ETH":
                base_price = np.random.uniform(2000, 3000)
                base_difficulty = 10e15
                base_block_reward = 2.0
            else:  # LTC or others
                base_price = np.random.uniform(100, 200)
                base_difficulty = 15e6
                base_block_reward = 12.5
            
            # Market cap is roughly proportional to price
            base_market_cap = base_price * np.random.uniform(18e6, 20e6)
            
            # Generate price series with realistic patterns
            end_time = datetime.now()
            start_time = end_time - timedelta(days=self.days)
            
            # Create a random walk for price
            price_changes = np.random.normal(0, 0.01, num_intervals)  # Daily volatility around 1%
            price_factors = np.cumprod(1 + price_changes)
            
            for i in range(num_intervals):
                timestamp = start_time + timedelta(minutes=i * market_interval_minutes)
                
                # Calculate price with trend and random walk
                trend_factor = 1.0 + 0.0001 * i  # Slight upward trend
                price = base_price * price_factors[i] * trend_factor
                
                # Market cap follows price with some noise
                market_cap = base_market_cap * price_factors[i] * np.random.normal(1.0, 0.005)
                
                # Volume has daily patterns
                hour_factor = 1.0 + 0.2 * np.sin(timestamp.hour * np.pi / 12)  # Higher volume during certain hours
                volume = market_cap * 0.03 * hour_factor * np.random.normal(1.0, 0.1)
                
                # Difficulty increases slowly over time
                difficulty_factor = 1.0 + 0.0005 * i
                difficulty = base_difficulty * difficulty_factor
                
                # Calculate 24h price change
                if i >= (24 * 60) // market_interval_minutes:
                    price_24h_ago = base_price * price_factors[i - (24 * 60) // market_interval_minutes]
                    price_change_24h = ((price / price_24h_ago) - 1) * 100
                else:
                    price_change_24h = 0
                
                # Transaction fees as percentage of block reward
                tx_fees_percent = np.random.uniform(5, 15)
                
                data.append({
                    'coin_id': coin_id,
                    'timestamp': timestamp,
                    'price_usd': price,
                    'market_cap_usd': market_cap,
                    'volume_24h_usd': volume,
                    'price_change_24h_percent': price_change_24h,
                    'network_difficulty': difficulty,
                    'block_reward_usd': base_block_reward * price,
                    'transaction_fees_percent': tx_fees_percent
                })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} market data records")
        return df
    
    def generate_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate all mock datasets.
        
        Returns:
            Tuple of (miner_telemetry, pool_performance, market_data) DataFrames
        """
        miner_telemetry = self.generate_miner_telemetry()
        pool_performance = self.generate_pool_performance()
        market_data = self.generate_market_data()
        
        return miner_telemetry, pool_performance, market_data


# If run directly, generate and save mock data
if __name__ == "__main__":
    import os
    from pathlib import Path
    
    # Create data directory if it doesn't exist
    data_dir = Path(__file__).parent / "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate mock data
    generator = MockDataGenerator(num_miners=5, num_coins=3, days=7, interval_minutes=60)
    miner_data, pool_data, market_data = generator.generate_all_data()
    
    # Save to CSV files
    miner_data.to_csv(data_dir / "mock_miner_telemetry.csv", index=False)
    pool_data.to_csv(data_dir / "mock_pool_performance.csv", index=False)
    market_data.to_csv(data_dir / "mock_market_data.csv", index=False)
    
    logger.info(f"Generated mock data and saved to {data_dir}")
    
    # Test feature engineering pipeline
    pipeline = FeatureEngineeringPipeline()
    
    processed_miner = pipeline.process_miner_telemetry(miner_data)
    processed_pool = pipeline.process_pool_performance(pool_data)
    processed_market = pipeline.process_market_data(market_data)
    
    combined_features = pipeline.combine_features(processed_miner, processed_pool, processed_market)
    
    # Save processed features
    combined_features.to_csv(data_dir / "processed_features.csv", index=False)
    
    logger.info(f"Processed features and saved to {data_dir / 'processed_features.csv'}")
    logger.info(f"Feature count: {len(combined_features.columns)}")
