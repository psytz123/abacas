
"""
Power optimization model using Bayesian Optimization.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Callable
from datetime import datetime
from skopt import gp_minimize, Optimizer
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_CONFIG, MODEL_DIR


class PowerOptimizationModel:
    """
    Bayesian Optimization model for optimizing miner power settings.
    """
    
    def __init__(self):
        """Initialize the power optimization model."""
        self.config = MODEL_CONFIG["power_optimization"]
        self.surrogate_model = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.optimizer = None
        self.best_params = None
        self.param_bounds = None
    
    def train(self, features: pd.DataFrame, efficiency: pd.Series, 
              hashrate: pd.Series, temperature: pd.Series,
              error_rate: Optional[pd.Series] = None) -> Dict:
        """
        Train the power optimization model.
        
        Args:
            features: DataFrame containing feature data
            efficiency: Series containing efficiency values (J/TH)
            hashrate: Series containing hashrate values (TH/s)
            temperature: Series containing temperature values (Celsius)
            error_rate: Optional series containing error rates
            
        Returns:
            Dictionary with training results and metrics
        """
        # Store feature names for inference
        self.feature_names = features.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(features)
        
        # Train a surrogate model to predict efficiency from features
        # This will be used by the Bayesian optimizer to evaluate points
        self.surrogate_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.surrogate_model.fit(X_scaled, efficiency)
        
        # Define parameter bounds for optimization
        self.param_bounds = {
            'power_limit': self.config['search_bounds']['power_limit'],
            'frequency': self.config['search_bounds']['frequency'],
            'voltage': self.config['search_bounds']['voltage']
        }
        
        # We'll use scikit-optimize's gp_minimize function directly during optimization
        # No need to initialize an optimizer object here
        
        # Train additional models for constraints
        self.hashrate_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.hashrate_model.fit(X_scaled, hashrate)
        
        self.temperature_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.temperature_model.fit(X_scaled, temperature)
        
        if error_rate is not None:
            self.error_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.error_model.fit(X_scaled, error_rate)
        else:
            self.error_model = None
        
        # Calculate model performance metrics
        efficiency_pred = self.surrogate_model.predict(X_scaled)
        mse = ((efficiency_pred - efficiency) ** 2).mean()
        rmse = np.sqrt(mse)
        mae = np.abs(efficiency_pred - efficiency).mean()
        
        # Return training results
        return {
            "model_type": "bayesian_optimization",
            "feature_count": len(self.feature_names),
            "training_samples": len(features),
            "metrics": {
                "efficiency_mse": mse,
                "efficiency_rmse": rmse,
                "efficiency_mae": mae
            },
            "feature_importance": dict(zip(
                self.feature_names, 
                self.surrogate_model.feature_importances_.tolist()
            ))
        }
    
    def optimize(self, features: pd.DataFrame, n_calls: int = 15) -> Dict:
        """
        Find optimal power settings for given features using scikit-optimize.
        
        Args:
            features: DataFrame containing feature data for a single miner
            n_calls: Number of optimization iterations
            
        Returns:
            Dictionary with optimal parameters and predicted performance
        """
        if self.surrogate_model is None:
            raise ValueError("Model has not been trained yet")
        
        if len(features) != 1:
            raise ValueError("Features should contain data for a single miner (one row)")
        
        # Ensure features have the correct columns in the correct order
        if not all(col in features.columns for col in self.feature_names):
            missing_cols = set(self.feature_names) - set(features.columns)
            raise ValueError(f"Missing features: {missing_cols}")
        
        # Select and order features correctly
        features = features[self.feature_names]
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Create a copy of the features that we can modify with different power settings
        base_features = features_scaled.copy()
        
        # Find the indices of power-related features in the feature list
        try:
            power_idx = self.feature_names.index('power_consumption_w')
        except ValueError:
            power_idx = None
            
        try:
            freq_idx = self.feature_names.index('frequency')
        except ValueError:
            freq_idx = None
            
        try:
            voltage_idx = self.feature_names.index('voltage')
        except ValueError:
            voltage_idx = None
        
        # Define the search space for scikit-optimize
        space = [
            Real(self.param_bounds['power_limit'][0], self.param_bounds['power_limit'][1], name='power_limit'),
            Real(self.param_bounds['frequency'][0], self.param_bounds['frequency'][1], name='frequency'),
            Real(self.param_bounds['voltage'][0], self.param_bounds['voltage'][1], name='voltage')
        ]
        
        # Define the objective function for optimization
        @use_named_args(space)
        def objective_function(power_limit, frequency, voltage):
            # Create a modified feature vector with the new power settings
            modified_features = base_features.copy()
            
            if power_idx is not None:
                # Adjust the power consumption feature
                power_mean = self.scaler.mean_[power_idx]
                power_scale = self.scaler.scale_[power_idx]
                original_power = (base_features[0, power_idx] * power_scale) + power_mean
                new_power = original_power * power_limit
                modified_features[0, power_idx] = (new_power - power_mean) / power_scale
            
            if freq_idx is not None:
                # Adjust the frequency feature
                freq_mean = self.scaler.mean_[freq_idx]
                freq_scale = self.scaler.scale_[freq_idx]
                original_freq = (base_features[0, freq_idx] * freq_scale) + freq_mean
                new_freq = original_freq * frequency
                modified_features[0, freq_idx] = (new_freq - freq_mean) / freq_scale
            
            if voltage_idx is not None:
                # Adjust the voltage feature
                voltage_mean = self.scaler.mean_[voltage_idx]
                voltage_scale = self.scaler.scale_[voltage_idx]
                original_voltage = (base_features[0, voltage_idx] * voltage_scale) + voltage_mean
                new_voltage = original_voltage * voltage
                modified_features[0, voltage_idx] = (new_voltage - voltage_mean) / voltage_scale
            
            # Predict efficiency with the modified features
            efficiency = self.surrogate_model.predict(modified_features)[0]
            
            # Predict hashrate, temperature, and error rate for constraint checking
            hashrate = self.hashrate_model.predict(modified_features)[0]
            temperature = self.temperature_model.predict(modified_features)[0]
            
            if self.error_model is not None:
                error_rate = self.error_model.predict(modified_features)[0]
            else:
                error_rate = 0
            
            # Apply constraints
            constraints_violated = False
            
            # Check temperature constraint
            if temperature > self.config['constraints']['max_temperature']:
                constraints_violated = True
            
            # Check hashrate constraint (assuming hashrate is in TH/s)
            hashrate_mean = self.scaler.mean_[self.feature_names.index('hashrate_th_s')] if 'hashrate_th_s' in self.feature_names else 100
            min_hashrate = hashrate_mean * self.config['constraints']['min_hashrate']
            if hashrate < min_hashrate:
                constraints_violated = True
            
            # Check error rate constraint
            if error_rate > self.config['constraints']['max_error_rate']:
                constraints_violated = True
            
            # Return a heavily penalized value if constraints are violated
            if constraints_violated:
                return 1000  # High value for minimization
            
            # Return the predicted efficiency (lower is better)
            return -efficiency  # Negative because we want to maximize efficiency (minimize J/TH)
        
        # Run the optimization
        result = gp_minimize(
            objective_function,
            space,
            n_calls=n_calls,
            random_state=42,
            n_random_starts=5,
            verbose=False
        )
        
        # Extract the best parameters
        best_params = {
            'power_limit': result.x[0],
            'frequency': result.x[1],
            'voltage': result.x[2]
        }
        
        # Evaluate the best parameters to get predicted performance
        modified_features = base_features.copy()
        
        if power_idx is not None:
            power_mean = self.scaler.mean_[power_idx]
            power_scale = self.scaler.scale_[power_idx]
            original_power = (base_features[0, power_idx] * power_scale) + power_mean
            new_power = original_power * best_params['power_limit']
            modified_features[0, power_idx] = (new_power - power_mean) / power_scale
        
        if freq_idx is not None:
            freq_mean = self.scaler.mean_[freq_idx]
            freq_scale = self.scaler.scale_[freq_idx]
            original_freq = (base_features[0, freq_idx] * freq_scale) + freq_mean
            new_freq = original_freq * best_params['frequency']
            modified_features[0, freq_idx] = (new_freq - freq_mean) / freq_scale
        
        if voltage_idx is not None:
            voltage_mean = self.scaler.mean_[voltage_idx]
            voltage_scale = self.scaler.scale_[voltage_idx]
            original_voltage = (base_features[0, voltage_idx] * voltage_scale) + voltage_mean
            new_voltage = original_voltage * best_params['voltage']
            modified_features[0, voltage_idx] = (new_voltage - voltage_mean) / voltage_scale
        
        # Predict performance metrics with the optimal settings
        predicted_efficiency = self.surrogate_model.predict(modified_features)[0]
        predicted_hashrate = self.hashrate_model.predict(modified_features)[0]
        predicted_temperature = self.temperature_model.predict(modified_features)[0]
        
        if self.error_model is not None:
            predicted_error_rate = self.error_model.predict(modified_features)[0]
        else:
            predicted_error_rate = 0
        
        # Calculate original performance for comparison
        original_efficiency = self.surrogate_model.predict(base_features)[0]
        original_hashrate = self.hashrate_model.predict(base_features)[0]
        original_temperature = self.temperature_model.predict(base_features)[0]
        
        # Calculate improvements
        efficiency_improvement = (original_efficiency - predicted_efficiency) / original_efficiency
        hashrate_impact = (predicted_hashrate - original_hashrate) / original_hashrate
        temperature_impact = (predicted_temperature - original_temperature)
        
        # Calculate absolute values for the optimal settings
        if power_idx is not None:
            power_mean = self.scaler.mean_[power_idx]
            power_scale = self.scaler.scale_[power_idx]
            original_power = (base_features[0, power_idx] * power_scale) + power_mean
            optimal_power = original_power * best_params['power_limit']
        else:
            original_power = None
            optimal_power = None
        
        # Return the optimization results
        return {
            "optimal_settings": {
                "power_limit_factor": best_params['power_limit'],
                "frequency_factor": best_params['frequency'],
                "voltage_factor": best_params['voltage'],
                "absolute_power_w": optimal_power
            },
            "predicted_performance": {
                "efficiency_j_th": predicted_efficiency,
                "hashrate_th_s": predicted_hashrate,
                "temperature_c": predicted_temperature,
                "error_rate": predicted_error_rate
            },
            "improvements": {
                "efficiency_improvement": efficiency_improvement,
                "hashrate_impact": hashrate_impact,
                "temperature_impact": temperature_impact
            },
            "original_performance": {
                "efficiency_j_th": original_efficiency,
                "hashrate_th_s": original_hashrate,
                "temperature_c": original_temperature,
                "power_consumption_w": original_power
            },
            "confidence": 0.8  # Fixed confidence for now, could be calculated based on model uncertainty
        }
    
    def save(self, filename: Optional[str] = None) -> str:
        """
        Save the trained model to disk.
        
        Args:
            filename: Name of the file to save the model to (without path)
                If None, a default name will be generated
                
        Returns:
            Path to the saved model file
        """
        if self.surrogate_model is None:
            raise ValueError("No trained model to save")
        
        # Create model directory if it doesn't exist
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Generate default filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"power_optimizer_{timestamp}.joblib"
        
        # Ensure filename has .joblib extension
        if not filename.endswith(".joblib"):
            filename += ".joblib"
        
        # Full path to save the model
        filepath = os.path.join(MODEL_DIR, filename)
        
        # Save the model and metadata
        model_data = {
            "surrogate_model": self.surrogate_model,
            "hashrate_model": self.hashrate_model,
            "temperature_model": self.temperature_model,
            "error_model": self.error_model,
            "feature_names": self.feature_names,
            "scaler": self.scaler,
            "param_bounds": self.param_bounds,
            "config": self.config,
            "timestamp": datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        
        return filepath
    
    def load(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model file
        """
        # Load the model data
        model_data = joblib.load(filepath)
        
        # Restore model components
        self.surrogate_model = model_data["surrogate_model"]
        self.hashrate_model = model_data["hashrate_model"]
        self.temperature_model = model_data["temperature_model"]
        self.error_model = model_data.get("error_model")
        self.feature_names = model_data["feature_names"]
        self.scaler = model_data["scaler"]
        self.param_bounds = model_data.get("param_bounds", self.param_bounds)
        self.config = model_data.get("config", self.config)


# If run directly, train and test the model with mock data
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add parent directory to path to import feature_engineering
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from ml_engine.feature_engineering import MockDataGenerator, FeatureEngineeringPipeline
    
    # Generate mock data (smaller dataset for faster training)
    print("Generating mock data...")
    generator = MockDataGenerator(num_miners=3, num_coins=2, days=3, interval_minutes=60)
    miner_data, pool_data, market_data = generator.generate_all_data()
    
    # Process features
    print("Processing features...")
    pipeline = FeatureEngineeringPipeline()
    processed_miner = pipeline.process_miner_telemetry(miner_data)
    processed_pool = pipeline.process_pool_performance(pool_data)
    processed_market = pipeline.process_market_data(market_data)
    combined_features = pipeline.combine_features(processed_miner, processed_pool, processed_market)
    
    # Prepare training data
    print("Preparing training data...")
    # Use efficiency_j_th as the target for optimization
    efficiency = combined_features['efficiency_j_th']
    hashrate = combined_features['hashrate_th_s']
    
    # Create a mock temperature if missing
    if 'avg_chip_temp_c' not in combined_features.columns:
        print("Adding mock temperature data since avg_chip_temp_c is missing")
        temperature = pd.Series(np.random.uniform(60, 70, size=len(combined_features)))
    else:
        temperature = combined_features['avg_chip_temp_c']
    
    # Create a mock error rate
    error_rate = pd.Series(np.random.uniform(0, 0.02, size=len(combined_features)))
    
    # Select features for training (exclude targets and any leaky features)
    exclude_cols = ['efficiency_j_th', 'worker_id', 'miner_id', 'timestamp', 'coin_id']
    feature_cols = [col for col in combined_features.columns if col not in exclude_cols]
    features = combined_features[feature_cols]
    
    # Handle missing values (simple imputation with mean)
    features = features.fillna(features.mean())
    
    # Train the model
    print("Training power optimization model...")
    model = PowerOptimizationModel()
    results = model.train(features, efficiency, hashrate, temperature, error_rate)
    
    # Print results
    print("\nTraining Results:")
    print(f"Model type: {results['model_type']}")
    print(f"Features: {results['feature_count']}")
    print(f"Training samples: {results['training_samples']}")
    print("\nMetrics:")
    for metric, value in results['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nTop 10 Feature Importance:")
    importance = results['feature_importance']
    sorted_importance = {k: v for k, v in sorted(importance.items(), key=lambda item: item[1], reverse=True)}
    for i, (feature, score) in enumerate(list(sorted_importance.items())[:10]):
        print(f"  {i+1}. {feature}: {score:.4f}")
    
    # Save the model
    model_path = model.save()
    print(f"\nModel saved to: {model_path}")
    
    # Test optimization
    print("\nTesting power optimization...")
    # Select a single miner's data for optimization
    test_miner = features.iloc[[0]]
    
    # Run optimization
    optimization_results = model.optimize(test_miner)
    
    print("\nOptimization Results:")
    print("Optimal Settings:")
    for param, value in optimization_results['optimal_settings'].items():
        if value is not None:
            print(f"  {param}: {value:.4f}")
    
    print("\nPredicted Performance:")
    for metric, value in optimization_results['predicted_performance'].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nImprovements:")
    for metric, value in optimization_results['improvements'].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nOriginal Performance:")
    for metric, value in optimization_results['original_performance'].items():
        if value is not None:
            print(f"  {metric}: {value:.4f}")
    
    print(f"\nConfidence: {optimization_results['confidence']:.2f}")
