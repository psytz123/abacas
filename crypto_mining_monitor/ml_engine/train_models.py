
"""
Script to train and save the ML models using mock data.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime

from ml_engine.feature_engineering import MockDataGenerator
from ml_engine.models.profit_model import ProfitPredictionModel
from ml_engine.models.power_optimizer import PowerOptimizationModel
from ml_engine.config import MODEL_DIR

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

def generate_mock_training_data():
    """Generate mock data for training the models."""
    print("Generating mock training data...")
    
    # Generate mock data
    generator = MockDataGenerator(num_miners=5, num_coins=3, days=7, interval_minutes=60)
    miner_data, pool_data, market_data = generator.generate_all_data()
    
    # Create a combined dataset with features suitable for training
    # For simplicity, we'll create a structured dataset directly
    
    # Number of samples
    n_samples = 1000
    
    # Generate features
    data = {
        # Miner telemetry features
        'hashrate_th_s': np.random.uniform(80, 120, n_samples),
        'power_consumption_w': np.random.uniform(3000, 3500, n_samples),
        'efficiency_j_th': np.random.uniform(30, 40, n_samples),
        'avg_chip_temp_c': np.random.uniform(60, 75, n_samples),
        'fan_speed_percent': np.random.uniform(60, 90, n_samples),
        
        # Pool performance features
        'effective_hashrate_th_s': np.random.uniform(75, 115, n_samples),
        'earnings_usd_24h': np.random.uniform(10, 20, n_samples),
        
        # Market data features
        'price_usd': np.random.uniform(40000, 60000, n_samples),
        'price_change_24h_percent': np.random.uniform(-5, 5, n_samples),
        'network_difficulty': np.random.uniform(25e12, 35e12, n_samples),
        'block_reward_usd': np.random.uniform(300000, 400000, n_samples),
        'transaction_fees_percent': np.random.uniform(5, 15, n_samples),
        
        # Derived features
        'hashrate_variance_24h': np.random.uniform(0, 5, n_samples),
        'temp_stability_24h': np.random.uniform(0, 3, n_samples),
        'price_volatility_24h': np.random.uniform(0.01, 0.05, n_samples),
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Calculate target variables
    df['earnings_per_th_usd'] = df['earnings_usd_24h'] / df['hashrate_th_s']
    
    # Add some noise and correlations to make the data more realistic
    df['earnings_per_th_usd'] *= (1 - 0.1 * df['network_difficulty'] / df['network_difficulty'].mean())
    df['earnings_per_th_usd'] *= (1 + 0.05 * df['price_change_24h_percent'] / 100)
    
    # Add error rate for power optimization
    df['error_rate'] = np.random.uniform(0, 0.02, n_samples)
    
    print(f"Generated {n_samples} samples with {len(df.columns)} features")
    return df

def train_profit_model(data):
    """Train and save the profit prediction model."""
    print("\nTraining profit prediction model...")
    
    # Prepare training data
    target = data['earnings_per_th_usd']
    
    # Select features for training (exclude targets and IDs)
    exclude_cols = ['earnings_per_th_usd', 'error_rate']
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    features = data[feature_cols]
    
    # Train the model
    model = ProfitPredictionModel()
    results = model.train(features, target)
    
    # Print results
    print("\nTraining Results:")
    print(f"Model type: {results['model_type']}")
    print(f"Features: {results['feature_count']}")
    print(f"Training samples: {results['training_samples']}")
    print(f"Test samples: {results['test_samples']}")
    print("\nMetrics:")
    for metric, value in results['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    # Save the model
    model_path = model.save()
    print(f"\nProfit model saved to: {model_path}")
    
    return model_path

def train_power_model(data):
    """Train and save the power optimization model."""
    print("\nTraining power optimization model...")
    
    # Prepare training data
    efficiency = data['efficiency_j_th']
    hashrate = data['hashrate_th_s']
    temperature = data['avg_chip_temp_c']
    error_rate = data['error_rate']
    
    # Select features for training (exclude targets)
    exclude_cols = ['efficiency_j_th', 'error_rate']
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    features = data[feature_cols]
    
    # Train the model
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
    
    # Save the model
    model_path = model.save()
    print(f"\nPower optimization model saved to: {model_path}")
    
    return model_path

def test_models(profit_model_path, power_model_path, test_data):
    """Test the trained models with a few samples."""
    print("\nTesting trained models...")
    
    # Load models
    profit_model = ProfitPredictionModel()
    profit_model.load(profit_model_path)
    
    power_model = PowerOptimizationModel()
    power_model.load(power_model_path)
    
    # Test profit prediction
    print("\nTesting profit prediction:")
    test_samples = test_data.sample(3)
    exclude_cols = ['earnings_per_th_usd', 'error_rate']
    feature_cols = [col for col in test_data.columns if col not in exclude_cols]
    
    predictions = profit_model.predict(test_samples[feature_cols])
    
    for i, pred in enumerate(predictions):
        actual = test_samples['earnings_per_th_usd'].iloc[i]
        print(f"  Sample {i+1}: Predicted: ${pred:.4f}, Actual: ${actual:.4f}")
    
    # Test power optimization
    print("\nTesting power optimization:")
    test_sample = test_data.sample(1)
    
    try:
        optimization_results = power_model.optimize(test_sample[feature_cols])
        
        print("Optimization Results:")
        print(f"  Current power: {test_sample['power_consumption_w'].iloc[0]:.0f}W")
        print(f"  Recommended power: {optimization_results['optimal_settings']['absolute_power_w']:.0f}W")
        print(f"  Efficiency improvement: {optimization_results['improvements']['efficiency_improvement']:.2%}")
        print(f"  Hashrate impact: {optimization_results['improvements']['hashrate_impact']:.2%}")
    except Exception as e:
        print(f"  Error during power optimization: {e}")

if __name__ == "__main__":
    # Generate mock training data
    training_data = generate_mock_training_data()
    
    # Train and save models
    profit_model_path = train_profit_model(training_data)
    power_model_path = train_power_model(training_data)
    
    # Test the models
    test_models(profit_model_path, power_model_path, training_data)
    
    print("\nModel training and testing complete!")
