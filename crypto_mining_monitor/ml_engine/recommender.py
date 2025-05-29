"""
Recommendation generation logic that converts model outputs into actionable recommendations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import json
import uuid

from ml_engine.config import RECOMMENDATION_CONFIG
from ml_engine.models.profit_model import ProfitPredictionModel
from ml_engine.models.power_optimizer import PowerOptimizationModel
from ml_engine.models.hashrate_tuner import HashrateTuner
from ml_engine.models.overclocker import IntelligentOverclocker
from ml_engine.utils.logging_config import logger
from ml_engine.utils.validation import handle_missing_data, ValidationError


class RecommendationEngine:
    """
    Engine for generating actionable recommendations based on ML model outputs.
    """
    
    def __init__(self, profit_model_path: Optional[str] = None, power_model_path: Optional[str] = None, 
                 hashrate_tuner_path: Optional[str] = None, overclocker_path: Optional[str] = None):
        """
        Initialize the recommendation engine.
        
        Args:
            profit_model_path: Path to the trained profit prediction model
            power_model_path: Path to the trained power optimization model
            hashrate_tuner_path: Path to the trained hashrate tuning model
            overclocker_path: Path to the trained intelligent overclocking model
        """
        self.profit_model = None
        self.power_model = None
        self.hashrate_tuner = None
        self.overclocker = None
        
        # Load models if paths are provided
        if profit_model_path:
            self.load_profit_model(profit_model_path)
        
        if power_model_path:
            self.load_power_model(power_model_path)
            
        if hashrate_tuner_path:
            self.load_hashrate_tuner(hashrate_tuner_path)
            
        if overclocker_path:
            self.load_overclocker(overclocker_path)
        
        # Load configuration
        self.config = RECOMMENDATION_CONFIG
        
        # Initialize recommendation history
        self.recommendation_history = {}
        
        logger.info("Recommendation engine initialized")
    
    def load_profit_model(self, model_path: str) -> None:
        """
        Load the profit prediction model.
        
        Args:
            model_path: Path to the trained model file
        """
        try:
            self.profit_model = ProfitPredictionModel()
            self.profit_model.load(model_path)
            logger.info(f"Profit prediction model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading profit model: {str(e)}")
            raise
    
    def load_power_model(self, model_path: str) -> None:
        """
        Load the power optimization model.
        
        Args:
            model_path: Path to the trained model file
        """
        try:
            self.power_model = PowerOptimizationModel()
            self.power_model.load(model_path)
            logger.info(f"Power optimization model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading power model: {str(e)}")
            raise
            
    def load_hashrate_tuner(self, model_path: str) -> None:
        """
        Load the dynamic hashrate tuning model.
        
        Args:
            model_path: Path to the trained model file
        """
        try:
            self.hashrate_tuner = HashrateTuner()
            self.hashrate_tuner.load(model_path)
            logger.info(f"Hashrate tuning model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading hashrate tuning model: {str(e)}")
            raise
            
    def load_overclocker(self, model_path: str) -> None:
        """
        Load the intelligent overclocking model.
        
        Args:
            model_path: Path to the trained model file
        """
        try:
            self.overclocker = IntelligentOverclocker()
            self.overclocker.load(model_path)
            logger.info(f"Intelligent overclocking model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading overclocking model: {str(e)}")
            raise
    
    def generate_coin_switching_recommendations(self, 
                                               miner_data: pd.DataFrame, 
                                               pool_data: pd.DataFrame, 
                                               market_data: pd.DataFrame,
                                               user_preferences: Optional[Dict] = None) -> List[Dict]:
        """
        Generate coin switching recommendations.
        
        Args:
            miner_data: DataFrame with miner telemetry data
            pool_data: DataFrame with pool performance data
            market_data: DataFrame with market data
            user_preferences: Optional dictionary with user preferences
            
        Returns:
            List of recommendation dictionaries
        """
        try:
            # Validate inputs
            if miner_data.empty:
                logger.warning("Empty miner data provided, no recommendations possible")
                return []
            
            if pool_data.empty:
                logger.warning("Empty pool data provided, no recommendations possible")
                return []
            
            if market_data.empty:
                logger.warning("Empty market data provided, no recommendations possible")
                return []
            
            # Check if profit model is loaded
            if self.profit_model is None:
                logger.error("Profit prediction model not loaded")
                raise ValueError("Profit prediction model not loaded")
            
            # Default user preferences if not provided
            if user_preferences is None:
                user_preferences = {
                    "min_improvement_threshold": self.config["coin_switching"]["min_improvement_threshold"],
                    "confidence_threshold": self.config["coin_switching"]["confidence_threshold"],
                    "cooldown_period_hours": self.config["coin_switching"]["cooldown_period_hours"]
                }
            
            recommendations = []
            
            # Group data by miner
            for miner_id, miner_group in miner_data.groupby('miner_id'):
                try:
                    # Get the latest data for this miner
                    latest_miner_data = miner_group.sort_values('timestamp').iloc[-1]
                    
                    # Find corresponding pool data
                    # Assuming worker_id maps to miner_id (in a real system, you'd have a mapping table)
                    worker_id = f"worker_{miner_id.split('_')[1]}"
                    miner_pool_data = pool_data[pool_data['worker_id'] == worker_id]
                    
                    if miner_pool_data.empty:
                        logger.debug(f"No pool data found for miner {miner_id}")
                        continue  # Skip if no pool data for this miner
                    
                    latest_pool_data = miner_pool_data.sort_values('timestamp').iloc[-1]
                    
                    # Get current coin being mined
                    current_coin = latest_pool_data['primary_coin']
                    
                    # Check if we've recently made a recommendation for this miner
                    if miner_id in self.recommendation_history:
                        last_recommendation = self.recommendation_history[miner_id].get('coin_switching')
                        if last_recommendation:
                            last_time = datetime.fromisoformat(last_recommendation['timestamp'])
                            cooldown_hours = user_preferences['cooldown_period_hours']
                            if datetime.now() - last_time < timedelta(hours=cooldown_hours):
                                logger.debug(f"Skipping miner {miner_id} due to cooldown period")
                                # Skip if within cooldown period
                                continue
                    
                    # Prepare feature sets for each potential coin
                    coin_predictions = []
                    feature_sets = []
                    
                    # Prepare a list of coins to evaluate
                    coins_to_evaluate = [coin for coin in market_data['coin_id'].unique() if coin != current_coin]
                    
                    # Prepare feature sets for batch prediction
                    for coin_id in coins_to_evaluate:
                        # Get latest market data for this coin
                        coin_market_data = market_data[market_data['coin_id'] == coin_id]
                        if coin_market_data.empty:
                            logger.debug(f"No market data found for coin {coin_id}")
                            continue
                        
                        latest_coin_data = coin_market_data.sort_values('timestamp').iloc[-1]
                        
                        # Combine features for prediction
                        prediction_features = pd.DataFrame([{
                            **latest_miner_data.to_dict(),
                            **latest_pool_data.to_dict(),
                            **latest_coin_data.to_dict()
                        }])
                        
                        feature_sets.append((coin_id, prediction_features))
                    
                    if not feature_sets:
                        logger.debug(f"No valid alternative coins found for miner {miner_id}")
                        continue
                    
                    # Get baseline profitability for current coin
                    current_coin_data = market_data[market_data['coin_id'] == current_coin]
                    if current_coin_data.empty:
                        logger.debug(f"No market data found for current coin {current_coin}")
                        continue
                    
                    latest_current_coin_data = current_coin_data.sort_values('timestamp').iloc[-1]
                    
                    baseline_features = pd.DataFrame([{
                        **latest_miner_data.to_dict(),
                        **latest_pool_data.to_dict(),
                        **latest_current_coin_data.to_dict()
                    }])
                    
                    # Handle missing values
                    baseline_features = handle_missing_data(baseline_features, strategy='conservative')
                    
                    try:
                        baseline_profitability = self.profit_model.predict(baseline_features)[0]
                    except Exception as e:
                        logger.error(f"Baseline prediction failed for miner {miner_id}: {str(e)}")
                        continue
                    
                    # Make predictions for alternative coins
                    for coin_id, features in feature_sets:
                        # Handle missing values
                        features = handle_missing_data(features, strategy='conservative')
                        
                        try:
                            predicted_profitability = self.profit_model.predict(features)[0]
                            
                            # Calculate confidence based on feature quality
                            # In a real implementation, this would be model-specific
                            missing_rate = features.isna().sum().sum() / (features.shape[0] * features.shape[1])
                            confidence = 0.9 - missing_rate * 0.5  # Simple confidence calculation
                            
                            coin_predictions.append({
                                'coin_id': coin_id,
                                'predicted_profitability': predicted_profitability,
                                'confidence': confidence
                            })
                        except Exception as e:
                            logger.warning(f"Prediction failed for coin {coin_id}: {str(e)}")
                            continue
                    
                    if not coin_predictions:
                        logger.debug(f"No valid predictions for miner {miner_id}")
                        continue  # Skip if no predictions were made
                    
                    # Find the most profitable coin
                    best_coin = max(coin_predictions, key=lambda x: x['predicted_profitability'])
                    
                    # Calculate improvement percentage
                    improvement = (best_coin['predicted_profitability'] - baseline_profitability) / baseline_profitability
                    
                    # Check if improvement meets threshold and confidence is sufficient
                    if (improvement > user_preferences['min_improvement_threshold'] and 
                        best_coin['confidence'] >= user_preferences['confidence_threshold']):
                        
                        # Generate recommendation
                        recommendation = {
                            'id': str(uuid.uuid4()),
                            'type': 'coin_switching',
                            'miner_id': miner_id,
                            'current_coin': current_coin,
                            'recommended_coin': best_coin['coin_id'],
                            'current_profitability': baseline_profitability,
                            'predicted_profitability': best_coin['predicted_profitability'],
                            'improvement_percent': improvement * 100,
                            'confidence': best_coin['confidence'],
                            'reasoning': self._generate_coin_switching_reasoning(
                                current_coin, best_coin['coin_id'], improvement
                            ),
                            'implementation_steps': self._generate_coin_switching_steps(
                                miner_id, best_coin['coin_id']
                            ),
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        # Store in history
                        if miner_id not in self.recommendation_history:
                            self.recommendation_history[miner_id] = {}
                        self.recommendation_history[miner_id]['coin_switching'] = recommendation
                        
                        recommendations.append(recommendation)
                        logger.info(f"Generated coin switching recommendation for miner {miner_id}: {current_coin} -> {best_coin['coin_id']} with {improvement:.2%} improvement")
                
                except Exception as e:
                    logger.error(f"Error generating coin switching recommendation for miner {miner_id}: {str(e)}")
                    continue
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating coin switching recommendations: {str(e)}")
            raise
    
    def generate_power_optimization_recommendations(self,
                                                  miner_data: pd.DataFrame,
                                                  pool_data: pd.DataFrame,
                                                  market_data: pd.DataFrame,
                                                  user_preferences: Optional[Dict] = None) -> List[Dict]:
        """
        Generate power optimization recommendations.
        
        Args:
            miner_data: DataFrame with miner telemetry data
            pool_data: DataFrame with pool performance data
            market_data: DataFrame with market data
            user_preferences: Optional dictionary with user preferences
            
        Returns:
            List of recommendation dictionaries
        """
        try:
            # Validate inputs
            if miner_data.empty:
                logger.warning("Empty miner data provided, no recommendations possible")
                return []
            
            if pool_data.empty:
                logger.warning("Empty pool data provided, no recommendations possible")
                return []
            
            if market_data.empty:
                logger.warning("Empty market data provided, no recommendations possible")
                return []
            
            # Check if power model is loaded
            if self.power_model is None:
                logger.error("Power optimization model not loaded")
                raise ValueError("Power optimization model not loaded")
            
            # Default user preferences if not provided
            if user_preferences is None:
                user_preferences = {
                    "min_efficiency_improvement": self.config["power_optimization"]["min_efficiency_improvement"],
                    "max_hashrate_reduction": self.config["power_optimization"]["max_hashrate_reduction"],
                    "confidence_threshold": self.config["power_optimization"]["confidence_threshold"],
                    "cooldown_period_hours": self.config["power_optimization"]["cooldown_period_hours"]
                }
            
            recommendations = []
            
            # Group data by miner
            for miner_id, miner_group in miner_data.groupby('miner_id'):
                try:
                    # Get the latest data for this miner
                    latest_miner_data = miner_group.sort_values('timestamp').iloc[-1]
                    
                    # Find corresponding pool data
                    # Assuming worker_id maps to miner_id (in a real system, you'd have a mapping table)
                    worker_id = f"worker_{miner_id.split('_')[1]}"
                    miner_pool_data = pool_data[pool_data['worker_id'] == worker_id]
                    
                    if miner_pool_data.empty:
                        logger.debug(f"No pool data found for miner {miner_id}")
                        continue  # Skip if no pool data for this miner
                    
                    latest_pool_data = miner_pool_data.sort_values('timestamp').iloc[-1]
                    
                    # Get current coin being mined
                    current_coin = latest_pool_data['primary_coin']
                    
                    # Get market data for current coin
                    coin_market_data = market_data[market_data['coin_id'] == current_coin]
                    if coin_market_data.empty:
                        logger.debug(f"No market data found for coin {current_coin}")
                        continue  # Skip if no market data for this coin
                    
                    latest_market_data = coin_market_data.sort_values('timestamp').iloc[-1]
                    
                    # Check if we've recently made a recommendation for this miner
                    if miner_id in self.recommendation_history:
                        last_recommendation = self.recommendation_history[miner_id].get('power_optimization')
                        if last_recommendation:
                            last_time = datetime.fromisoformat(last_recommendation['timestamp'])
                            cooldown_hours = user_preferences['cooldown_period_hours']
                            if datetime.now() - last_time < timedelta(hours=cooldown_hours):
                                logger.debug(f"Skipping miner {miner_id} due to cooldown period")
                                # Skip if within cooldown period
                                continue
                    
                    # Combine features for optimization
                    optimization_features = pd.DataFrame([{
                        **latest_miner_data.to_dict(),
                        **latest_pool_data.to_dict(),
                        **latest_market_data.to_dict()
                    }])
                    
                    # Handle missing values
                    optimization_features = handle_missing_data(optimization_features, strategy='conservative')
                    
                    # Run power optimization
                    try:
                        optimization_results = self.power_model.optimize(optimization_features)
                        
                        # Check if improvement meets thresholds
                        efficiency_improvement = optimization_results['improvements']['efficiency_improvement']
                        hashrate_impact = optimization_results['improvements']['hashrate_impact']
                        confidence = optimization_results['confidence']
                        
                        if (efficiency_improvement > user_preferences['min_efficiency_improvement'] and 
                            hashrate_impact > -user_preferences['max_hashrate_reduction'] and
                            confidence >= user_preferences['confidence_threshold']):
                            
                            # Calculate net profitability impact
                            # Simple model: efficiency improvement minus hashrate reduction
                            net_profit_impact = efficiency_improvement + hashrate_impact
                            
                            # Generate recommendation
                            recommendation = {
                                'id': str(uuid.uuid4()),
                                'type': 'power_optimization',
                                'miner_id': miner_id,
                                'current_power': optimization_results['original_performance']['power_consumption_w'],
                                'recommended_power': optimization_results['optimal_settings']['absolute_power_w'],
                                'power_reduction_percent': (1 - optimization_results['optimal_settings']['power_limit_factor']) * 100,
                                'efficiency_improvement_percent': efficiency_improvement * 100,
                                'hashrate_impact_percent': hashrate_impact * 100,
                                'temperature_impact': optimization_results['improvements']['temperature_impact'],
                                'net_profitability_impact_percent': net_profit_impact * 100,
                                'confidence': confidence,
                                'reasoning': self._generate_power_optimization_reasoning(
                                    optimization_results
                                ),
                                'implementation_steps': self._generate_power_optimization_steps(
                                    miner_id, optimization_results
                                ),
                                'timestamp': datetime.now().isoformat()
                            }
                            
                            # Store in history
                            if miner_id not in self.recommendation_history:
                                self.recommendation_history[miner_id] = {}
                            self.recommendation_history[miner_id]['power_optimization'] = recommendation
                            
                            recommendations.append(recommendation)
                            logger.info(f"Generated power optimization recommendation for miner {miner_id}: {optimization_results['original_performance']['power_consumption_w']:.0f}W -> {optimization_results['optimal_settings']['absolute_power_w']:.0f}W with {efficiency_improvement:.2%} efficiency improvement")
                    
                    except Exception as e:
                        logger.error(f"Optimization failed for miner {miner_id}: {str(e)}")
                        continue
                
                except Exception as e:
                    logger.error(f"Error generating power optimization recommendation for miner {miner_id}: {str(e)}")
                    continue
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating power optimization recommendations: {str(e)}")
            raise
    
    def generate_all_recommendations(self,
                                    miner_data: pd.DataFrame,
                                    pool_data: pd.DataFrame,
                                    market_data: pd.DataFrame,
                                    user_preferences: Optional[Dict] = None) -> Dict[str, List[Dict]]:
        """
        Generate all types of recommendations.
        
        Args:
            miner_data: DataFrame with miner telemetry data
            pool_data: DataFrame with pool performance data
            market_data: DataFrame with market data
            user_preferences: Optional dictionary with user preferences
            
        Returns:
            Dictionary mapping recommendation types to lists of recommendations
        """
        all_recommendations = {}
        
        # Generate coin switching recommendations if profit model is loaded
        if self.profit_model is not None:
            try:
                coin_recommendations = self.generate_coin_switching_recommendations(
                    miner_data, pool_data, market_data, user_preferences
                )
                all_recommendations['coin_switching'] = coin_recommendations
                logger.info(f"Generated {len(coin_recommendations)} coin switching recommendations")
            except Exception as e:
                logger.error(f"Failed to generate coin switching recommendations: {str(e)}")
                all_recommendations['coin_switching'] = []
        
        # Generate power optimization recommendations if power model is loaded
        if self.power_model is not None:
            try:
                power_recommendations = self.generate_power_optimization_recommendations(
                    miner_data, pool_data, market_data, user_preferences
                )
                all_recommendations['power_optimization'] = power_recommendations
                logger.info(f"Generated {len(power_recommendations)} power optimization recommendations")
            except Exception as e:
                logger.error(f"Failed to generate power optimization recommendations: {str(e)}")
                all_recommendations['power_optimization'] = []
        
        # Generate dynamic hashrate tuning recommendations if hashrate tuner is loaded
        if self.hashrate_tuner is not None:
            try:
                hashrate_recommendations = self.generate_dynamic_hashrate_recommendations(
                    miner_data, pool_data, market_data, None, user_preferences
                )
                all_recommendations['dynamic_hashrate_tuning'] = hashrate_recommendations
                logger.info(f"Generated {len(hashrate_recommendations)} dynamic hashrate tuning recommendations")
            except Exception as e:
                logger.error(f"Failed to generate dynamic hashrate tuning recommendations: {str(e)}")
                all_recommendations['dynamic_hashrate_tuning'] = []
        
        # Generate intelligent overclocking recommendations if overclocker is loaded
        if self.overclocker is not None:
            try:
                overclocking_recommendations = self.generate_intelligent_overclocking_recommendations(
                    miner_data, pool_data, market_data, user_preferences
                )
                all_recommendations['intelligent_overclocking'] = overclocking_recommendations
                logger.info(f"Generated {len(overclocking_recommendations)} intelligent overclocking recommendations")
            except Exception as e:
                logger.error(f"Failed to generate intelligent overclocking recommendations: {str(e)}")
                all_recommendations['intelligent_overclocking'] = []
        
        return all_recommendations
    
    def batch_generate_recommendations(self, miners: List[Dict], user_preferences: Optional[Dict] = None) -> Dict[str, List[Dict]]:
        """
        Generate recommendations for multiple miners in batch.
        
        Args:
            miners: List of miner data dictionaries
            user_preferences: Optional dictionary with user preferences
            
        Returns:
            Dictionary mapping recommendation types to lists of recommendations
        """
        # Validate inputs
        if not miners:
            logger.warning("No miners provided for batch recommendation generation")
            return {}
        
        # Convert miners to DataFrame
        miner_data = pd.DataFrame(miners)
        
        # Get unique miner IDs
        miner_ids = miner_data['miner_id'].unique().tolist()
        logger.info(f"Generating batch recommendations for {len(miner_ids)} miners")
        
        # Fetch all required data in batch
        # In a real implementation, this would query a database or API
        # For this example, we'll use the miner data directly
        
        # Generate recommendations
        return self.generate_all_recommendations(miner_data, miner_data, miner_data, user_preferences)
    
    def _generate_coin_switching_reasoning(self, current_coin: str, recommended_coin: str, improvement: float) -> str:
        """Generate reasoning for coin switching recommendation."""
        reasons = [
            f"Switching from {current_coin} to {recommended_coin} is projected to increase profitability by {improvement:.1%}.",
            f"Current market conditions favor {recommended_coin} mining over {current_coin}.",
            f"The price trend and network difficulty of {recommended_coin} create a favorable mining opportunity."
        ]
        return " ".join(reasons)
    
    def _generate_coin_switching_steps(self, miner_id: str, recommended_coin: str) -> List[str]:
        """Generate implementation steps for coin switching."""
        steps = [
            f"Log in to your mining pool account.",
            f"Navigate to the worker settings for miner {miner_id}.",
            f"Change the primary mining coin to {recommended_coin}.",
            f"Save the new settings and wait for the changes to take effect (typically 5-10 minutes).",
            f"Monitor the miner's performance for the next few hours to ensure stability."
        ]
        return steps
    
    def _generate_power_optimization_reasoning(self, optimization_results: Dict) -> str:
        """Generate reasoning for power optimization recommendation."""
        power_limit = optimization_results['optimal_settings']['power_limit_factor']
        efficiency_improvement = optimization_results['improvements']['efficiency_improvement']
        hashrate_impact = optimization_results['improvements']['hashrate_impact']
        temperature_impact = optimization_results['improvements']['temperature_impact']
        
        reasons = [
            f"Reducing power to {power_limit:.1%} of current setting improves energy efficiency by {efficiency_improvement:.1%}.",
            f"This optimization will result in a {hashrate_impact:.1%} change in hashrate.",
            f"The miner's temperature is expected to decrease by {abs(temperature_impact):.1f}°C, reducing thermal stress."
        ]
        
        if efficiency_improvement > abs(hashrate_impact):
            reasons.append("The efficiency gains outweigh the minor hashrate reduction, resulting in better overall profitability.")
        
        return " ".join(reasons)
    
    def _generate_power_optimization_steps(self, miner_id: str, optimization_results: Dict) -> List[str]:
        """Generate implementation steps for power optimization."""
        power_limit = optimization_results['optimal_settings']['power_limit_factor']
        absolute_power = optimization_results['optimal_settings']['absolute_power_w']
        
        steps = [
            f"Log in to your miner's administration interface.",
            f"Navigate to the power settings for miner {miner_id}.",
            f"Reduce the power limit to {absolute_power:.0f}W (approximately {power_limit:.0%} of current setting).",
            f"Save the new settings and monitor the miner for stability over the next 30 minutes.",
            f"If the miner shows any instability, gradually increase the power limit until stability is restored."
        ]
        return steps
        
    def generate_dynamic_hashrate_recommendations(self,
                                                miner_data: pd.DataFrame,
                                                pool_data: pd.DataFrame,
                                                market_data: pd.DataFrame,
                                                energy_data: Optional[pd.DataFrame] = None,
                                                user_preferences: Optional[Dict] = None) -> List[Dict]:
        """
        Generate dynamic hashrate tuning recommendations.
        
        Args:
            miner_data: DataFrame with miner telemetry data
            pool_data: DataFrame with pool performance data
            market_data: DataFrame with market data
            energy_data: Optional DataFrame with energy pricing data
            user_preferences: Optional dictionary with user preferences
            
        Returns:
            List of recommendation dictionaries
        """
        try:
            # Validate inputs
            if miner_data.empty:
                logger.warning("Empty miner data provided, no recommendations possible")
                return []
            
            if pool_data.empty:
                logger.warning("Empty pool data provided, no recommendations possible")
                return []
            
            if market_data.empty:
                logger.warning("Empty market data provided, no recommendations possible")
                return []
            
            # Check if hashrate tuner model is loaded
            if self.hashrate_tuner is None:
                logger.error("Hashrate tuning model not loaded")
                raise ValueError("Hashrate tuning model not loaded")
            
            # Default user preferences if not provided
            if user_preferences is None:
                user_preferences = {
                    "min_profit_improvement": self.config.get("dynamic_hashrate_tuning", {}).get("min_profit_improvement", 0.03),
                    "confidence_threshold": self.config.get("dynamic_hashrate_tuning", {}).get("confidence_threshold", 0.7),
                    "cooldown_period_hours": self.config.get("dynamic_hashrate_tuning", {}).get("cooldown_period_hours", 6),
                    "max_hashrate_reduction": self.config.get("dynamic_hashrate_tuning", {}).get("max_hashrate_reduction", 0.3)
                }
            
            # Create default energy data if not provided
            if energy_data is None or energy_data.empty:
                # Create a simple energy pricing dataframe with constant price
                timestamps = miner_data['timestamp'].unique()
                energy_data = pd.DataFrame({
                    'timestamp': timestamps,
                    'energy_price': [0.12] * len(timestamps),  # Default price: $0.12 per kWh
                    'grid_load_percent': [70] * len(timestamps),
                    'renewable_percent': [20] * len(timestamps)
                })
            
            recommendations = []
            
            # Group data by miner
            for miner_id, miner_group in miner_data.groupby('miner_id'):
                try:
                    # Get the latest data for this miner
                    latest_miner_data = miner_group.sort_values('timestamp').iloc[-1]
                    
                    # Find corresponding pool data
                    # Assuming worker_id maps to miner_id (in a real system, you'd have a mapping table)
                    worker_id = f"worker_{miner_id.split('_')[1]}"
                    miner_pool_data = pool_data[pool_data['worker_id'] == worker_id]
                    
                    if miner_pool_data.empty:
                        logger.debug(f"No pool data found for miner {miner_id}")
                        continue  # Skip if no pool data for this miner
                    
                    latest_pool_data = miner_pool_data.sort_values('timestamp').iloc[-1]
                    
                    # Get current coin being mined
                    current_coin = latest_pool_data['primary_coin']
                    
                    # Get market data for current coin
                    coin_market_data = market_data[market_data['coin_id'] == current_coin]
                    if coin_market_data.empty:
                        logger.debug(f"No market data found for coin {current_coin}")
                        continue  # Skip if no market data for this coin
                    
                    latest_market_data = coin_market_data.sort_values('timestamp').iloc[-1]
                    
                    # Get latest energy data
                    latest_energy_data = energy_data.sort_values('timestamp').iloc[-1]
                    
                    # Check if we've recently made a recommendation for this miner
                    if miner_id in self.recommendation_history:
                        last_recommendation = self.recommendation_history[miner_id].get('dynamic_hashrate_tuning')
                        if last_recommendation:
                            last_time = datetime.fromisoformat(last_recommendation['timestamp'])
                            cooldown_hours = user_preferences['cooldown_period_hours']
                            if datetime.now() - last_time < timedelta(hours=cooldown_hours):
                                logger.debug(f"Skipping miner {miner_id} due to cooldown period")
                                # Skip if within cooldown period
                                continue
                    
                    # Combine features for prediction
                    prediction_features = pd.DataFrame([{
                        **latest_miner_data.to_dict(),
                        **latest_pool_data.to_dict(),
                        **latest_market_data.to_dict(),
                        **latest_energy_data.to_dict()
                    }])
                    
                    # Handle missing values
                    prediction_features = handle_missing_data(prediction_features, strategy='conservative')
                    
                    # Generate hashrate tuning recommendation
                    try:
                        tuning_result = self.hashrate_tuner.predict(prediction_features)
                        
                        # Get current hashrate
                        current_hashrate_percent = 100.0  # Assume current is 100%
                        recommended_hashrate_percent = tuning_result['optimal_hashrate_percent']
                        
                        # Calculate hashrate change
                        hashrate_change = (recommended_hashrate_percent - current_hashrate_percent) / 100.0
                        
                        # Check if the recommendation meets thresholds
                        expected_performance = tuning_result['expected_performance']
                        profit_improvement = expected_performance['daily_profit_usd'] / (latest_miner_data['hashrate_th_s'] * 0.15)  # Rough estimate of current profit
                        profit_improvement = profit_improvement - 1.0  # Convert to percentage change
                        
                        # Skip if the improvement is too small or the hashrate reduction is too large
                        if (profit_improvement < user_preferences['min_profit_improvement'] or 
                            hashrate_change < -user_preferences['max_hashrate_reduction'] or
                            tuning_result['confidence'] < user_preferences['confidence_threshold']):
                            logger.debug(f"Skipping hashrate recommendation for {miner_id}: insufficient improvement or excessive reduction")
                            continue
                        
                        # Generate recommendation
                        recommendation = {
                            'id': str(uuid.uuid4()),
                            'type': 'dynamic_hashrate_tuning',
                            'miner_id': miner_id,
                            'current_hashrate_percent': current_hashrate_percent,
                            'recommended_hashrate_percent': recommended_hashrate_percent,
                            'hashrate_change_percent': hashrate_change * 100,
                            'expected_daily_profit_usd': expected_performance['daily_profit_usd'],
                            'expected_daily_energy_cost_usd': expected_performance['daily_energy_cost_usd'],
                            'profit_improvement_percent': profit_improvement * 100,
                            'energy_efficiency_improvement_percent': (expected_performance['energy_efficiency_th_w'] / (latest_miner_data['hashrate_th_s'] / latest_miner_data['power_consumption_w']) - 1) * 100,
                            'confidence': tuning_result['confidence'],
                            'reasoning': self._generate_hashrate_tuning_reasoning(
                                current_hashrate_percent, recommended_hashrate_percent, 
                                profit_improvement, expected_performance
                            ),
                            'implementation_steps': self._generate_hashrate_tuning_steps(
                                miner_id, recommended_hashrate_percent
                            ),
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        # Store in history
                        if miner_id not in self.recommendation_history:
                            self.recommendation_history[miner_id] = {}
                        self.recommendation_history[miner_id]['dynamic_hashrate_tuning'] = recommendation
                        
                        recommendations.append(recommendation)
                        logger.info(f"Generated hashrate tuning recommendation for miner {miner_id}: {current_hashrate_percent:.1f}% -> {recommended_hashrate_percent:.1f}% with {profit_improvement:.2%} profit improvement")
                    
                    except Exception as e:
                        logger.error(f"Error generating hashrate tuning recommendation for miner {miner_id}: {str(e)}")
                        continue
                
                except Exception as e:
                    logger.error(f"Error processing miner {miner_id} for hashrate tuning: {str(e)}")
                    continue
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating hashrate tuning recommendations: {str(e)}")
            raise
    
    def generate_intelligent_overclocking_recommendations(self,
                                                        miner_data: pd.DataFrame,
                                                        pool_data: pd.DataFrame,
                                                        market_data: pd.DataFrame,
                                                        user_preferences: Optional[Dict] = None) -> List[Dict]:
        """
        Generate intelligent overclocking recommendations.
        
        Args:
            miner_data: DataFrame with miner telemetry data
            pool_data: DataFrame with pool performance data
            market_data: DataFrame with market data
            user_preferences: Optional dictionary with user preferences
            
        Returns:
            List of recommendation dictionaries
        """
        try:
            # Validate inputs
            if miner_data.empty:
                logger.warning("Empty miner data provided, no recommendations possible")
                return []
            
            if pool_data.empty:
                logger.warning("Empty pool data provided, no recommendations possible")
                return []
            
            if market_data.empty:
                logger.warning("Empty market data provided, no recommendations possible")
                return []
            
            # Check if overclocker model is loaded
            if self.overclocker is None:
                logger.error("Intelligent overclocking model not loaded")
                raise ValueError("Intelligent overclocking model not loaded")
            
            # Default user preferences if not provided
            if user_preferences is None:
                user_preferences = {
                    "min_hashrate_improvement": self.config.get("intelligent_overclocking", {}).get("min_hashrate_improvement", 0.05),
                    "min_efficiency_improvement": self.config.get("intelligent_overclocking", {}).get("min_efficiency_improvement", 0.03),
                    "max_temperature_increase": self.config.get("intelligent_overclocking", {}).get("max_temperature_increase", 5),
                    "min_stability_score": self.config.get("intelligent_overclocking", {}).get("min_stability_score", 0.8),
                    "confidence_threshold": self.config.get("intelligent_overclocking", {}).get("confidence_threshold", 0.7),
                    "cooldown_period_hours": self.config.get("intelligent_overclocking", {}).get("cooldown_period_hours", 24),
                    "optimization_goal": self.config.get("intelligent_overclocking", {}).get("optimization_goal", "balanced")
                }
            
            recommendations = []
            
            # Group data by miner
            for miner_id, miner_group in miner_data.groupby('miner_id'):
                try:
                    # Get the latest data for this miner
                    latest_miner_data = miner_group.sort_values('timestamp').iloc[-1]
                    
                    # Find corresponding pool data
                    # Assuming worker_id maps to miner_id (in a real system, you'd have a mapping table)
                    worker_id = f"worker_{miner_id.split('_')[1]}"
                    miner_pool_data = pool_data[pool_data['worker_id'] == worker_id]
                    
                    if miner_pool_data.empty:
                        logger.debug(f"No pool data found for miner {miner_id}")
                        continue  # Skip if no pool data for this miner
                    
                    latest_pool_data = miner_pool_data.sort_values('timestamp').iloc[-1]
                    
                    # Get current coin being mined
                    current_coin = latest_pool_data['primary_coin']
                    
                    # Determine mining algorithm based on coin
                    # In a real implementation, you would have a mapping of coins to algorithms
                    algorithm = 'SHA-256' if current_coin in ['BTC', 'BCH'] else 'Ethash' if current_coin in ['ETH', 'ETC'] else 'Scrypt'
                    
                    # Check if we've recently made a recommendation for this miner
                    if miner_id in self.recommendation_history:
                        last_recommendation = self.recommendation_history[miner_id].get('intelligent_overclocking')
                        if last_recommendation:
                            last_time = datetime.fromisoformat(last_recommendation['timestamp'])
                            cooldown_hours = user_preferences['cooldown_period_hours']
                            if datetime.now() - last_time < timedelta(hours=cooldown_hours):
                                logger.debug(f"Skipping miner {miner_id} due to cooldown period")
                                # Skip if within cooldown period
                                continue
                    
                    # Determine hardware type
                    # In a real implementation, you would extract this from the miner data
                    # For now, we'll use a simple mapping based on miner_id
                    hardware_type = 'Antminer S19' if int(miner_id.split('_')[1]) % 3 == 0 else 'Antminer S19 Pro' if int(miner_id.split('_')[1]) % 3 == 1 else 'Whatsminer M30S'
                    
                    # Create feature vector for overclocking
                    # In a real implementation, you would extract actual overclocking parameters
                    # For now, we'll create a simple feature vector with default values
                    overclocking_features = pd.DataFrame({
                        'core_clock_offset': [0],
                        'memory_clock_offset': [0],
                        'power_limit_percent': [100],
                        'core_voltage_offset': [0]
                    })
                    
                    # Generate overclocking recommendation
                    try:
                        optimization_goal = user_preferences.get('optimization_goal', 'balanced')
                        overclocking_result = self.overclocker.optimize(
                            overclocking_features,
                            hardware_type=hardware_type,
                            algorithm=algorithm,
                            optimization_goal=optimization_goal
                        )
                        
                        # Check if the recommendation meets thresholds
                        hashrate_improvement = overclocking_result['improvements']['hashrate_improvement']
                        efficiency_improvement = overclocking_result['improvements']['efficiency_improvement']
                        temperature_change = overclocking_result['improvements']['temperature_change']
                        stability_score = overclocking_result['predicted_performance']['stability_score']
                        confidence = overclocking_result['confidence']
                        
                        # Skip if the improvements don't meet thresholds
                        if (hashrate_improvement < user_preferences['min_hashrate_improvement'] or
                            efficiency_improvement < user_preferences['min_efficiency_improvement'] or
                            temperature_change > user_preferences['max_temperature_increase'] or
                            stability_score < user_preferences['min_stability_score'] or
                            confidence < user_preferences['confidence_threshold']):
                            logger.debug(f"Skipping overclocking recommendation for {miner_id}: thresholds not met")
                            continue
                        
                        # Generate recommendation
                        recommendation = {
                            'id': str(uuid.uuid4()),
                            'type': 'intelligent_overclocking',
                            'miner_id': miner_id,
                            'hardware_type': hardware_type,
                            'algorithm': algorithm,
                            'profile_name': overclocking_result['profile_name'],
                            'core_clock_offset': overclocking_result['optimal_parameters']['core_clock_offset'],
                            'memory_clock_offset': overclocking_result['optimal_parameters']['memory_clock_offset'],
                            'power_limit_percent': overclocking_result['optimal_parameters']['power_limit_percent'],
                            'core_voltage_offset': overclocking_result['optimal_parameters']['core_voltage_offset'],
                            'hashrate_improvement_percent': hashrate_improvement * 100,
                            'efficiency_improvement_percent': efficiency_improvement * 100,
                            'power_change_percent': overclocking_result['improvements']['power_change'] * 100,
                            'temperature_change': temperature_change,
                            'stability_score': stability_score,
                            'confidence': confidence,
                            'reasoning': self._generate_overclocking_reasoning(
                                hardware_type, algorithm, overclocking_result
                            ),
                            'implementation_steps': self._generate_overclocking_steps(
                                miner_id, hardware_type, overclocking_result['optimal_parameters']
                            ),
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        # Store in history
                        if miner_id not in self.recommendation_history:
                            self.recommendation_history[miner_id] = {}
                        self.recommendation_history[miner_id]['intelligent_overclocking'] = recommendation
                        
                        recommendations.append(recommendation)
                        logger.info(f"Generated overclocking recommendation for miner {miner_id}: {hardware_type} with {hashrate_improvement:.2%} hashrate improvement and {efficiency_improvement:.2%} efficiency improvement")
                    
                    except Exception as e:
                        logger.error(f"Error generating overclocking recommendation for miner {miner_id}: {str(e)}")
                        continue
                
                except Exception as e:
                    logger.error(f"Error processing miner {miner_id} for overclocking: {str(e)}")
                    continue
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating intelligent overclocking recommendations: {str(e)}")
            raise
    
    def _generate_hashrate_tuning_reasoning(self, current_hashrate: float, recommended_hashrate: float, 
                                          profit_improvement: float, performance: Dict) -> str:
        """Generate reasoning for hashrate tuning recommendation."""
        if recommended_hashrate < current_hashrate:
            reasons = [
                f"Reducing hashrate from {current_hashrate:.1f}% to {recommended_hashrate:.1f}% will optimize your mining efficiency.",
                f"At current energy prices and market conditions, running at {recommended_hashrate:.1f}% power will increase your daily profit by {profit_improvement:.1%}.",
                f"This adjustment will reduce your energy costs by approximately ${performance['daily_energy_cost_usd']:.2f} per day while maintaining optimal mining performance.",
                f"The energy efficiency of your miner will improve by {(performance['energy_efficiency_th_w'] / (performance['hashrate_th_s'] / performance['power_consumption_w']) - 1):.1%}."
            ]
        else:
            reasons = [
                f"Increasing hashrate from {current_hashrate:.1f}% to {recommended_hashrate:.1f}% will maximize your mining profitability.",
                f"Current market conditions favor higher hashrate operation, with a projected profit increase of {profit_improvement:.1%}.",
                f"While energy consumption will increase slightly, the additional revenue outweighs the costs by ${performance['daily_profit_usd'] - performance['daily_energy_cost_usd']:.2f} per day.",
                f"This adjustment is optimal based on current network difficulty, coin price, and energy costs."
            ]
        return " ".join(reasons)
    
    def _generate_hashrate_tuning_steps(self, miner_id: str, recommended_hashrate: float) -> List[str]:
        """Generate implementation steps for hashrate tuning."""
        steps = [
            f"Log in to your miner's administration interface for {miner_id}.",
            f"Navigate to the 'Performance Settings' or 'Power Management' section.",
            f"Set the power limit or hashrate target to {recommended_hashrate:.1f}% of maximum.",
            f"Save the new settings and monitor the miner for stability over the next 30 minutes.",
            f"Verify that the actual hashrate and power consumption have adjusted to the new settings.",
            f"Monitor mining performance for the next 24 hours to confirm the expected improvements."
        ]
        return steps
    
    def _generate_overclocking_reasoning(self, hardware_type: str, algorithm: str, result: Dict) -> str:
        """Generate reasoning for overclocking recommendation."""
        improvements = result['improvements']
        performance = result['predicted_performance']
        baseline = result['baseline_performance']
        
        reasons = [
            f"This {result['profile_name']} profile is optimized specifically for {hardware_type} running the {algorithm} algorithm.",
            f"The recommended settings will increase your hashrate by {improvements['hashrate_improvement']:.1%} while improving energy efficiency by {improvements['efficiency_improvement']:.1%}.",
            f"Power consumption will {'increase' if improvements['power_change'] > 0 else 'decrease'} by {abs(improvements['power_change'] * 100):.1f}%.",
            f"Expected temperature change of {improvements['temperature_change']:.1f}°C is within safe operating limits for this hardware.",
            f"The stability score of {performance['stability_score']:.2f} indicates a low risk of instability or hardware issues with these settings."
        ]
        return " ".join(reasons)
    
    def _generate_overclocking_steps(self, miner_id: str, hardware_type: str, parameters: Dict) -> List[str]:
        """Generate implementation steps for overclocking."""
        steps = [
            f"Log in to your miner's administration interface for {miner_id}.",
            f"Navigate to the 'Advanced Settings' or 'Overclocking' section.",
            f"Set the following parameters:",
            f"  - Core Clock Offset: {parameters['core_clock_offset']} MHz",
            f"  - Memory Clock Offset: {parameters['memory_clock_offset']} MHz",
            f"  - Power Limit: {parameters['power_limit_percent']}%",
            f"  - Core Voltage Offset: {parameters['core_voltage_offset']} mV",
            f"Save the new settings and restart the miner if required.",
            f"Monitor the miner closely for the first 30 minutes to ensure stability.",
            f"If you observe any instability (crashes, rejected shares, or temperature spikes), revert to default settings immediately.",
            f"Once stable, monitor mining performance for 24-48 hours to confirm the expected improvements."
        ]
        return steps


# If run directly, test the recommendation engine with mock data
if __name__ == "__main__":
    from ml_engine.feature_engineering import MockDataGenerator, FeatureEngineeringPipeline
    from ml_engine.models.profit_model import ProfitPredictionModel
    from ml_engine.models.power_optimizer import PowerOptimizationModel
    
    # Generate mock data
    print("Generating mock data...")
    generator = MockDataGenerator(num_miners=5, num_coins=3, days=7, interval_minutes=60)
    miner_data, pool_data, market_data = generator.generate_all_data()
    
    # Process features
    print("Processing features...")
    pipeline = FeatureEngineeringPipeline()
    processed_miner = pipeline.process_miner_telemetry(miner_data)
    processed_pool = pipeline.process_pool_performance(pool_data)
    processed_market = pipeline.process_market_data(market_data)
    combined_features = pipeline.combine_features(processed_miner, processed_pool, processed_market)
    
    # Train profit model
    print("Training profit prediction model...")
    profit_model = ProfitPredictionModel()
    
    # Prepare training data for profit model
    target = combined_features['earnings_per_th_usd']
    exclude_cols = ['earnings_per_th_usd', 'worker_id', 'miner_id', 'timestamp', 'coin_id']
    feature_cols = [col for col in combined_features.columns if col not in exclude_cols]
    features = combined_features[feature_cols].fillna(0)
    
    profit_model.train(features, target)
    profit_model_path = profit_model.save()
    print(f"Profit model saved to: {profit_model_path}")
    
    # Train power optimization model
    print("Training power optimization model...")
    power_model = PowerOptimizationModel()
    
    # Prepare training data for power model
    efficiency = combined_features['efficiency_j_th']
    hashrate = combined_features['hashrate_th_s']
    temperature = combined_features['avg_chip_temp_c']
    error_rate = pd.Series(np.random.uniform(0, 0.02, size=len(combined_features)))
    
    power_model.train(features, efficiency, hashrate, temperature, error_rate)
    power_model_path = power_model.save()
    print(f"Power model saved to: {power_model_path}")
    
    # Initialize recommendation engine
    print("Initializing recommendation engine...")
    recommender = RecommendationEngine(profit_model_path, power_model_path)
    
    # Generate recommendations
    print("Generating recommendations...")
    recommendations = recommender.generate_all_recommendations(
        processed_miner, processed_pool, processed_market
    )
    
    # Print recommendations
    print("\nCoin Switching Recommendations:")
    if 'coin_switching' in recommendations and recommendations['coin_switching']:
        for i, rec in enumerate(recommendations['coin_switching']):
            print(f"\nRecommendation {i+1}:")
            print(f"  Miner: {rec['miner_id']}")
            print(f"  Switch from {rec['current_coin']} to {rec['recommended_coin']}")
            print(f"  Expected improvement: {rec['improvement_percent']:.2f}%")
            print(f"  Confidence: {rec['confidence']:.2f}")
            print(f"  Reasoning: {rec['reasoning']}")
            print("  Implementation steps:")
            for j, step in enumerate(rec['implementation_steps']):
                print(f"    {j+1}. {step}")
    else:
        print("  No coin switching recommendations generated")
    
    print("\nPower Optimization Recommendations:")
    if 'power_optimization' in recommendations and recommendations['power_optimization']:
        for i, rec in enumerate(recommendations['power_optimization']):
            print(f"\nRecommendation {i+1}:")
            print(f"  Miner: {rec['miner_id']}")
            print(f"  Reduce power from {rec['current_power']:.0f}W to {rec['recommended_power']:.0f}W ({rec['power_reduction_percent']:.2f}% reduction)")
            print(f"  Efficiency improvement: {rec['efficiency_improvement_percent']:.2f}%")
            print(f"  Hashrate impact: {rec['hashrate_impact_percent']:.2f}%")
            print(f"  Net profitability impact: {rec['net_profitability_impact_percent']:.2f}%")
            print(f"  Confidence: {rec['confidence']:.2f}")
            print(f"  Reasoning: {rec['reasoning']}")
            print("  Implementation steps:")
            for j, step in enumerate(rec['implementation_steps']):
                print(f"    {j+1}. {step}")
    else:
        print("  No power optimization recommendations generated")
