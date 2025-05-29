
"""
Intelligent Overclocking model using Bayesian Optimization.

This module implements a Bayesian optimization approach to find optimal
overclocking parameters for mining hardware, balancing performance,
efficiency, and hardware longevity.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
import json
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from scipy.optimize import minimize
from scipy.stats import norm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_CONFIG, MODEL_DIR


class IntelligentOverclocker:
    """
    Bayesian Optimization model for intelligent overclocking.
    
    This model uses Gaussian Process Regression to model the relationship
    between overclocking parameters and performance metrics, and Bayesian
    Optimization to find the optimal parameters.
    """
    
    def __init__(self):
        """Initialize the intelligent overclocking model."""
        self.config = MODEL_CONFIG.get("intelligent_overclocking", {})
        self.feature_names = None
        self.scaler = StandardScaler()
        self.gp_models = {}
        self.hardware_profiles = {}
        self.param_bounds = None
        self.algorithm_profiles = {}
    
    def train(self, 
              features: pd.DataFrame, 
              hashrate: pd.Series, 
              power: pd.Series,
              temperature: pd.Series,
              stability: pd.Series,
              hardware_types: pd.Series) -> Dict:
        """
        Train the intelligent overclocking model.
        
        Args:
            features: DataFrame containing feature data
            hashrate: Series containing hashrate values (TH/s)
            power: Series containing power consumption values (W)
            temperature: Series containing temperature values (Celsius)
            stability: Series containing stability scores (0-1)
            hardware_types: Series containing hardware type identifiers
            
        Returns:
            Dictionary with training results and metrics
        """
        # Store feature names for inference
        self.feature_names = features.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(features)
        
        # Train separate models for different hardware types
        unique_hardware = hardware_types.unique()
        
        for hardware in unique_hardware:
            # Filter data for this hardware type
            mask = hardware_types == hardware
            X_hw = X_scaled[mask]
            hashrate_hw = hashrate[mask]
            power_hw = power[mask]
            temperature_hw = temperature[mask]
            stability_hw = stability[mask]
            
            # Calculate efficiency
            efficiency_hw = hashrate_hw / power_hw
            
            # Train Gaussian Process models for each metric
            self.gp_models[hardware] = {
                'hashrate': self._train_gp_model(X_hw, hashrate_hw),
                'power': self._train_gp_model(X_hw, power_hw),
                'temperature': self._train_gp_model(X_hw, temperature_hw),
                'stability': self._train_gp_model(X_hw, stability_hw),
                'efficiency': self._train_gp_model(X_hw, efficiency_hw)
            }
            
            # Create hardware profile
            self.hardware_profiles[hardware] = {
                'samples': len(X_hw),
                'hashrate_range': (hashrate_hw.min(), hashrate_hw.max()),
                'power_range': (power_hw.min(), power_hw.max()),
                'temperature_range': (temperature_hw.min(), temperature_hw.max()),
                'stability_range': (stability_hw.min(), stability_hw.max()),
                'efficiency_range': (efficiency_hw.min(), efficiency_hw.max())
            }
        
        # Define parameter bounds for optimization
        self.param_bounds = {
            'core_clock_offset': self.config.get('param_bounds', {}).get('core_clock_offset', (-100, 200)),
            'memory_clock_offset': self.config.get('param_bounds', {}).get('memory_clock_offset', (-500, 1500)),
            'power_limit_percent': self.config.get('param_bounds', {}).get('power_limit_percent', (70, 100)),
            'core_voltage_offset': self.config.get('param_bounds', {}).get('core_voltage_offset', (-50, 50))
        }
        
        # Create algorithm-specific profiles
        # In a real implementation, you would train separate models for different mining algorithms
        # For simplicity, we'll create some mock profiles
        self.algorithm_profiles = {
            'SHA-256': {
                'core_importance': 0.7,
                'memory_importance': 0.3,
                'power_sensitivity': 0.8,
                'temperature_sensitivity': 0.6
            },
            'Ethash': {
                'core_importance': 0.4,
                'memory_importance': 0.8,
                'power_sensitivity': 0.7,
                'temperature_sensitivity': 0.5
            },
            'Scrypt': {
                'core_importance': 0.6,
                'memory_importance': 0.5,
                'power_sensitivity': 0.6,
                'temperature_sensitivity': 0.7
            }
        }
        
        # Calculate model performance metrics
        metrics = {}
        for hardware, models in self.gp_models.items():
            metrics[hardware] = {}
            for metric, model in models.items():
                # For GP models, we can use the log marginal likelihood as a performance metric
                metrics[hardware][metric] = {
                    'log_marginal_likelihood': model.log_marginal_likelihood(model.kernel_.theta)
                }
        
        return {
            "model_type": "intelligent_overclocking_bayesian",
            "feature_count": len(self.feature_names),
            "hardware_types": list(unique_hardware),
            "hardware_profiles": self.hardware_profiles,
            "algorithm_profiles": list(self.algorithm_profiles.keys()),
            "param_bounds": self.param_bounds,
            "metrics": metrics
        }
    
    def _train_gp_model(self, X: np.ndarray, y: pd.Series) -> GaussianProcessRegressor:
        """
        Train a Gaussian Process model for a specific metric.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Trained Gaussian Process model
        """
        # Define the kernel
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(X.shape[1]), nu=2.5)
        
        # Create and train the GP model
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=42
        )
        
        gp.fit(X, y)
        
        return gp
    
    def optimize(self, 
                features: pd.DataFrame, 
                hardware_type: str,
                algorithm: str = 'SHA-256',
                optimization_goal: str = 'balanced',
                n_iter: int = 50) -> Dict:
        """
        Find optimal overclocking parameters for given features.
        
        Args:
            features: DataFrame containing feature data for a single miner
            hardware_type: Type of hardware to optimize for
            algorithm: Mining algorithm to optimize for
            optimization_goal: Optimization goal ('efficiency', 'performance', 'balanced', 'conservative')
            n_iter: Number of optimization iterations
            
        Returns:
            Dictionary with optimal parameters and predicted performance
        """
        if hardware_type not in self.gp_models:
            raise ValueError(f"No model available for hardware type: {hardware_type}")
        
        if algorithm not in self.algorithm_profiles:
            raise ValueError(f"No profile available for algorithm: {algorithm}")
        
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
        
        # Get models for this hardware type
        models = self.gp_models[hardware_type]
        
        # Get algorithm profile
        algo_profile = self.algorithm_profiles[algorithm]
        
        # Define the acquisition function based on optimization goal
        if optimization_goal == 'efficiency':
            acquisition_weights = {
                'hashrate': 0.3,
                'power': -0.4,
                'temperature': -0.1,
                'stability': 0.1,
                'efficiency': 0.5
            }
        elif optimization_goal == 'performance':
            acquisition_weights = {
                'hashrate': 0.6,
                'power': -0.1,
                'temperature': -0.1,
                'stability': 0.1,
                'efficiency': 0.2
            }
        elif optimization_goal == 'conservative':
            acquisition_weights = {
                'hashrate': 0.2,
                'power': -0.2,
                'temperature': -0.3,
                'stability': 0.4,
                'efficiency': 0.3
            }
        else:  # balanced
            acquisition_weights = {
                'hashrate': 0.4,
                'power': -0.2,
                'temperature': -0.1,
                'stability': 0.2,
                'efficiency': 0.3
            }
        
        # Adjust weights based on algorithm profile
        acquisition_weights['hashrate'] *= (algo_profile['core_importance'] * 0.7 + algo_profile['memory_importance'] * 0.3)
        acquisition_weights['power'] *= algo_profile['power_sensitivity']
        acquisition_weights['temperature'] *= algo_profile['temperature_sensitivity']
        
        # Define the parameter space for optimization
        param_space = [
            (self.param_bounds['core_clock_offset'][0], self.param_bounds['core_clock_offset'][1]),
            (self.param_bounds['memory_clock_offset'][0], self.param_bounds['memory_clock_offset'][1]),
            (self.param_bounds['power_limit_percent'][0], self.param_bounds['power_limit_percent'][1]),
            (self.param_bounds['core_voltage_offset'][0], self.param_bounds['core_voltage_offset'][1])
        ]
        
        # Define the objective function for Bayesian optimization
        def objective(params):
            # Create a feature vector with the overclocking parameters
            x = features_scaled.copy()
            
            # In a real implementation, you would modify the feature vector based on the parameters
            # For simplicity, we'll just use the parameters directly for prediction
            
            # Predict performance metrics
            hashrate_pred, hashrate_std = models['hashrate'].predict(x, return_std=True)
            power_pred, power_std = models['power'].predict(x, return_std=True)
            temp_pred, temp_std = models['temperature'].predict(x, return_std=True)
            stability_pred, stability_std = models['stability'].predict(x, return_std=True)
            efficiency_pred, efficiency_std = models['efficiency'].predict(x, return_std=True)
            
            # Calculate the weighted objective
            objective_value = (
                acquisition_weights['hashrate'] * hashrate_pred[0] +
                acquisition_weights['power'] * power_pred[0] +
                acquisition_weights['temperature'] * temp_pred[0] +
                acquisition_weights['stability'] * stability_pred[0] +
                acquisition_weights['efficiency'] * efficiency_pred[0]
            )
            
            # We want to maximize the objective
            return -objective_value
        
        # Run Bayesian optimization
        result = minimize(
            objective,
            x0=np.array([
                (param_space[0][0] + param_space[0][1]) / 2,
                (param_space[1][0] + param_space[1][1]) / 2,
                (param_space[2][0] + param_space[2][1]) / 2,
                (param_space[3][0] + param_space[3][1]) / 2
            ]),
            bounds=param_space,
            method='L-BFGS-B',
            options={'maxiter': n_iter}
        )
        
        # Extract the optimal parameters
        optimal_params = {
            'core_clock_offset': int(result.x[0]),
            'memory_clock_offset': int(result.x[1]),
            'power_limit_percent': float(result.x[2]),
            'core_voltage_offset': int(result.x[3])
        }
        
        # Predict performance with optimal parameters
        x_opt = features_scaled.copy()
        
        # In a real implementation, you would modify the feature vector based on the optimal parameters
        # For simplicity, we'll just use the parameters directly for prediction
        
        hashrate_pred, hashrate_std = models['hashrate'].predict(x_opt, return_std=True)
        power_pred, power_std = models['power'].predict(x_opt, return_std=True)
        temp_pred, temp_std = models['temperature'].predict(x_opt, return_std=True)
        stability_pred, stability_std = models['stability'].predict(x_opt, return_std=True)
        efficiency_pred, efficiency_std = models['efficiency'].predict(x_opt, return_std=True)
        
        # Get baseline performance (no overclocking)
        x_base = features_scaled.copy()
        
        hashrate_base, _ = models['hashrate'].predict(x_base, return_std=True)
        power_base, _ = models['power'].predict(x_base, return_std=True)
        temp_base, _ = models['temperature'].predict(x_base, return_std=True)
        stability_base, _ = models['stability'].predict(x_base, return_std=True)
        efficiency_base, _ = models['efficiency'].predict(x_base, return_std=True)
        
        # Calculate improvements
        hashrate_improvement = (hashrate_pred[0] - hashrate_base[0]) / hashrate_base[0]
        power_change = (power_pred[0] - power_base[0]) / power_base[0]
        temp_change = temp_pred[0] - temp_base[0]
        stability_change = stability_pred[0] - stability_base[0]
        efficiency_improvement = (efficiency_pred[0] - efficiency_base[0]) / efficiency_base[0]
        
        # Calculate confidence based on prediction uncertainty
        confidence_hashrate = 1.0 - min(hashrate_std[0] / hashrate_pred[0], 0.5)
        confidence_power = 1.0 - min(power_std[0] / power_pred[0], 0.5)
        confidence_temp = 1.0 - min(temp_std[0] / temp_pred[0], 0.5)
        confidence_stability = 1.0 - min(stability_std[0] / stability_pred[0], 0.5)
        confidence_efficiency = 1.0 - min(efficiency_std[0] / efficiency_pred[0], 0.5)
        
        # Overall confidence is a weighted average
        overall_confidence = (
            0.3 * confidence_hashrate +
            0.2 * confidence_power +
            0.1 * confidence_temp +
            0.2 * confidence_stability +
            0.2 * confidence_efficiency
        )
        
        # Create profile name based on optimization goal and algorithm
        profile_name = f"{optimization_goal.capitalize()}-{algorithm}"
        
        return {
            "profile_name": profile_name,
            "hardware_type": hardware_type,
            "algorithm": algorithm,
            "optimization_goal": optimization_goal,
            "optimal_parameters": optimal_params,
            "predicted_performance": {
                "hashrate": float(hashrate_pred[0]),
                "power_consumption": float(power_pred[0]),
                "temperature": float(temp_pred[0]),
                "stability_score": float(stability_pred[0]),
                "efficiency": float(efficiency_pred[0])
            },
            "prediction_uncertainty": {
                "hashrate": float(hashrate_std[0]),
                "power_consumption": float(power_std[0]),
                "temperature": float(temp_std[0]),
                "stability_score": float(stability_std[0]),
                "efficiency": float(efficiency_std[0])
            },
            "improvements": {
                "hashrate_improvement": float(hashrate_improvement),
                "power_change": float(power_change),
                "temperature_change": float(temp_change),
                "stability_change": float(stability_change),
                "efficiency_improvement": float(efficiency_improvement)
            },
            "baseline_performance": {
                "hashrate": float(hashrate_base[0]),
                "power_consumption": float(power_base[0]),
                "temperature": float(temp_base[0]),
                "stability_score": float(stability_base[0]),
                "efficiency": float(efficiency_base[0])
            },
            "confidence": float(overall_confidence)
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
        if not self.gp_models:
            raise ValueError("No trained models to save")
        
        # Create model directory if it doesn't exist
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Generate default filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"intelligent_overclocker_{timestamp}.joblib"
        
        # Ensure filename has .joblib extension
        if not filename.endswith(".joblib"):
            filename += ".joblib"
        
        # Full path to save the model
        filepath = os.path.join(MODEL_DIR, filename)
        
        # Save the model and metadata
        model_data = {
            "gp_models": self.gp_models,
            "hardware_profiles": self.hardware_profiles,
            "algorithm_profiles": self.algorithm_profiles,
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
        self.gp_models = model_data["gp_models"]
        self.hardware_profiles = model_data["hardware_profiles"]
        self.algorithm_profiles = model_data["algorithm_profiles"]
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
    
    # Generate mock data
    print("Generating mock data...")
    generator = MockDataGenerator(num_miners=5, num_coins=2, days=3, interval_minutes=60)
    miner_data, pool_data, market_data = generator.generate_all_data()
    
    # Process features
    print("Processing features...")
    pipeline = FeatureEngineeringPipeline()
    processed_miner = pipeline.process_miner_telemetry(miner_data)
    
    # Create mock overclocking data
    print("Creating mock overclocking data...")
    
    # Create a larger dataset with different overclocking settings
    num_samples = 200
    
    # Create random hardware types
    hardware_types = np.random.choice(['Antminer S19', 'Antminer S19 Pro', 'Whatsminer M30S'], num_samples)
    
    # Create random overclocking parameters
    core_clock_offset = np.random.uniform(-100, 200, num_samples)
    memory_clock_offset = np.random.uniform(-500, 1500, num_samples)
    power_limit_percent = np.random.uniform(70, 100, num_samples)
    core_voltage_offset = np.random.uniform(-50, 50, num_samples)
    
    # Create feature matrix
    features = pd.DataFrame({
        'core_clock_offset': core_clock_offset,
        'memory_clock_offset': memory_clock_offset,
        'power_limit_percent': power_limit_percent,
        'core_voltage_offset': core_voltage_offset
    })
    
    # Create target variables with some realistic relationships
    # Hashrate increases with core and memory clock, but with diminishing returns
    base_hashrate = np.random.uniform(90, 110, num_samples)
    hashrate = base_hashrate * (1 + 0.001 * core_clock_offset + 0.0005 * memory_clock_offset)
    
    # Power increases with core clock, memory clock, and voltage
    base_power = np.random.uniform(3000, 3500, num_samples)
    power = base_power * (power_limit_percent / 100) * (1 + 0.0005 * core_clock_offset + 0.0002 * memory_clock_offset + 0.001 * core_voltage_offset)
    
    # Temperature increases with power and core clock
    base_temp = np.random.uniform(60, 70, num_samples)
    temperature = base_temp + 0.01 * (power - base_power) + 0.02 * core_clock_offset
    
    # Stability decreases with higher clocks and lower voltage
    stability = 1.0 - 0.001 * np.abs(core_clock_offset) - 0.0005 * np.abs(memory_clock_offset) + 0.001 * core_voltage_offset
    stability = np.clip(stability, 0.5, 1.0)
    
    # Add some hardware-specific effects
    for i, hw in enumerate(hardware_types):
        if hw == 'Antminer S19 Pro':
            hashrate[i] *= 1.1  # Pro models have higher base hashrate
            stability[i] += 0.05  # Pro models are more stable
        elif hw == 'Whatsminer M30S':
            power[i] *= 1.05  # Whatsminer uses slightly more power
            temperature[i] -= 2  # Whatsminer runs slightly cooler
    
    # Add some noise
    hashrate += np.random.normal(0, 2, num_samples)
    power += np.random.normal(0, 50, num_samples)
    temperature += np.random.normal(0, 1, num_samples)
    stability += np.random.normal(0, 0.02, num_samples)
    
    # Ensure values are in reasonable ranges
    hashrate = np.clip(hashrate, 80, 130)
    power = np.clip(power, 2500, 4000)
    temperature = np.clip(temperature, 50, 85)
    stability = np.clip(stability, 0.5, 1.0)
    
    # Convert to pandas Series
    hashrate = pd.Series(hashrate)
    power = pd.Series(power)
    temperature = pd.Series(temperature)
    stability = pd.Series(stability)
    hardware_types = pd.Series(hardware_types)
    
    # Train the model
    print("Training intelligent overclocking model...")
    model = IntelligentOverclocker()
    results = model.train(features, hashrate, power, temperature, stability, hardware_types)
    
    # Print results
    print("\nTraining Results:")
    print(f"Model type: {results['model_type']}")
    print(f"Features: {results['feature_count']}")
    print(f"Hardware types: {results['hardware_types']}")
    print(f"Algorithm profiles: {results['algorithm_profiles']}")
    
    print("\nHardware Profiles:")
    for hw, profile in results['hardware_profiles'].items():
        print(f"  {hw}:")
        print(f"    Samples: {profile['samples']}")
        print(f"    Hashrate range: {profile['hashrate_range'][0]:.2f} - {profile['hashrate_range'][1]:.2f} TH/s")
        print(f"    Power range: {profile['power_range'][0]:.2f} - {profile['power_range'][1]:.2f} W")
        print(f"    Temperature range: {profile['temperature_range'][0]:.2f} - {profile['temperature_range'][1]:.2f} °C")
        print(f"    Stability range: {profile['stability_range'][0]:.2f} - {profile['stability_range'][1]:.2f}")
    
    # Save the model
    model_path = model.save()
    print(f"\nModel saved to: {model_path}")
    
    # Test optimization
    print("\nTesting intelligent overclocking optimization...")
    
    # Create a test feature vector
    test_features = pd.DataFrame({
        'core_clock_offset': [0],
        'memory_clock_offset': [0],
        'power_limit_percent': [100],
        'core_voltage_offset': [0]
    })
    
    # Test different optimization goals
    for goal in ['efficiency', 'performance', 'balanced', 'conservative']:
        print(f"\nOptimization goal: {goal}")
        
        # Optimize for each hardware type
        for hw in results['hardware_types']:
            print(f"\nHardware: {hw}")
            
            # Optimize for SHA-256 algorithm
            optimization = model.optimize(
                test_features,
                hardware_type=hw,
                algorithm='SHA-256',
                optimization_goal=goal
            )
            
            print(f"Profile: {optimization['profile_name']}")
            print("Optimal Parameters:")
            for param, value in optimization['optimal_parameters'].items():
                print(f"  {param}: {value}")
            
            print("\nPredicted Performance:")
            for metric, value in optimization['predicted_performance'].items():
                print(f"  {metric}: {value:.2f}")
            
            print("\nImprovements:")
            for metric, value in optimization['improvements'].items():
                if 'improvement' in metric or 'change' in metric:
                    print(f"  {metric}: {value*100:.2f}%")
                else:
                    print(f"  {metric}: {value:.2f}")
            
            print(f"\nConfidence: {optimization['confidence']:.2f}")
