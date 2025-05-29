
"""
Dynamic Hashrate Tuning model using Reinforcement Learning.

This module implements a reinforcement learning-based approach to dynamically
adjust mining hashrate based on market conditions, energy prices, and network metrics.
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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_CONFIG, MODEL_DIR


class HashrateTuner:
    """
    Reinforcement Learning model for dynamic hashrate tuning.
    
    This model uses a Deep Q-Network (DQN) approach to learn optimal hashrate
    settings based on market conditions, energy prices, and network metrics.
    """
    
    def __init__(self):
        """Initialize the hashrate tuning model."""
        self.config = MODEL_CONFIG.get("hashrate_tuning", {})
        self.feature_names = None
        self.scaler = StandardScaler()
        self.model = None
        self.target_model = None
        self.action_space = None
        self.state_dim = None
        self.action_dim = None
        self.gamma = self.config.get("gamma", 0.95)
        self.epsilon = self.config.get("epsilon", 1.0)
        self.epsilon_min = self.config.get("epsilon_min", 0.01)
        self.epsilon_decay = self.config.get("epsilon_decay", 0.995)
        self.learning_rate = self.config.get("learning_rate", 0.001)
        self.batch_size = self.config.get("batch_size", 32)
        self.memory = []
        self.max_memory_size = self.config.get("max_memory_size", 2000)
        self.update_target_frequency = self.config.get("update_target_frequency", 100)
        self.training_step = 0
    
    def build_model(self, state_dim: int, action_dim: int) -> keras.Model:
        """
        Build the DQN model.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            
        Returns:
            Keras model for DQN
        """
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(state_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(action_dim, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def train(self, 
              market_data: pd.DataFrame, 
              miner_data: pd.DataFrame, 
              energy_data: pd.DataFrame,
              network_data: pd.DataFrame) -> Dict:
        """
        Train the hashrate tuning model.
        
        Args:
            market_data: DataFrame containing market price and volume data
            miner_data: DataFrame containing miner telemetry data
            energy_data: DataFrame containing energy pricing data
            network_data: DataFrame containing network difficulty and hashrate data
            
        Returns:
            Dictionary with training results and metrics
        """
        # Prepare and combine features
        combined_features = self._prepare_features(market_data, miner_data, energy_data, network_data)
        
        # Store feature names for inference
        self.feature_names = combined_features.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(combined_features)
        
        # Define state and action space
        self.state_dim = X_scaled.shape[1]
        
        # Define action space: discrete hashrate levels (e.g., 70%, 75%, ..., 100%)
        self.action_space = np.linspace(
            self.config.get("min_hashrate_percent", 70),
            self.config.get("max_hashrate_percent", 100),
            self.config.get("hashrate_levels", 7)
        ) / 100.0
        self.action_dim = len(self.action_space)
        
        # Build models
        self.model = self.build_model(self.state_dim, self.action_dim)
        self.target_model = self.build_model(self.state_dim, self.action_dim)
        self.target_model.set_weights(self.model.get_weights())
        
        # Generate training episodes using historical data
        episodes = self._generate_episodes(combined_features, X_scaled)
        
        # Train the model using experience replay
        metrics = self._train_with_episodes(episodes)
        
        return {
            "model_type": "hashrate_tuning_dqn",
            "feature_count": len(self.feature_names),
            "training_samples": len(combined_features),
            "episodes": len(episodes),
            "metrics": metrics,
            "action_space": self.action_space.tolist(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim
        }
    
    def _prepare_features(self, 
                         market_data: pd.DataFrame, 
                         miner_data: pd.DataFrame, 
                         energy_data: pd.DataFrame,
                         network_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and combine features from different data sources.
        
        Args:
            market_data: DataFrame containing market price and volume data
            miner_data: DataFrame containing miner telemetry data
            energy_data: DataFrame containing energy pricing data
            network_data: DataFrame containing network difficulty and hashrate data
            
        Returns:
            Combined DataFrame with all features
        """
        # Align timestamps across all dataframes
        # For simplicity, we'll use inner join, but in production you might want
        # to use more sophisticated time-based alignment
        
        # Resample all dataframes to a common frequency if needed
        market_features = market_data.copy()
        miner_features = miner_data.copy()
        energy_features = energy_data.copy()
        network_features = network_data.copy()
        
        # Create time-based features
        for df in [market_features, miner_features, energy_features, network_features]:
            if 'timestamp' in df.columns:
                df['hour_of_day'] = pd.to_datetime(df['timestamp']).dt.hour
                df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        # Create market-specific features
        if 'price_usd' in market_features.columns:
            market_features['price_change_1h'] = market_features.groupby('coin_id')['price_usd'].pct_change(1)
            market_features['price_change_6h'] = market_features.groupby('coin_id')['price_usd'].pct_change(6)
            market_features['price_change_24h'] = market_features.groupby('coin_id')['price_usd'].pct_change(24)
            market_features['price_volatility'] = market_features.groupby('coin_id')['price_usd'].rolling(24).std().reset_index(0, drop=True)
        
        # Create energy-specific features
        if 'energy_price' in energy_features.columns:
            energy_features['energy_price_change_1h'] = energy_features['energy_price'].pct_change(1)
            energy_features['energy_price_change_24h'] = energy_features['energy_price'].pct_change(24)
            energy_features['energy_price_volatility'] = energy_features['energy_price'].rolling(24).std().reset_index(0, drop=True)
        
        # Create network-specific features
        if 'network_difficulty' in network_features.columns:
            network_features['difficulty_change_24h'] = network_features.groupby('coin_id')['network_difficulty'].pct_change(24)
        
        # Create miner-specific features
        if 'hashrate_th_s' in miner_features.columns and 'power_consumption_w' in miner_features.columns:
            miner_features['efficiency_j_th'] = miner_features['power_consumption_w'] / miner_features['hashrate_th_s']
        
        # Combine all features
        # In a real implementation, you would need to handle the joining more carefully
        # For simplicity, we'll just concatenate the features
        combined_features = pd.concat([
            market_features.drop(['timestamp', 'coin_id'], axis=1, errors='ignore'),
            miner_features.drop(['timestamp', 'miner_id'], axis=1, errors='ignore'),
            energy_features.drop(['timestamp'], axis=1, errors='ignore'),
            network_features.drop(['timestamp', 'coin_id'], axis=1, errors='ignore')
        ], axis=1)
        
        # Handle missing values
        combined_features = combined_features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return combined_features
    
    def _generate_episodes(self, features: pd.DataFrame, scaled_features: np.ndarray) -> List[List[Tuple]]:
        """
        Generate training episodes from historical data.
        
        Args:
            features: DataFrame with original features
            scaled_features: Scaled feature array
            
        Returns:
            List of episodes, where each episode is a list of (state, action, reward, next_state, done) tuples
        """
        episodes = []
        
        # For simplicity, we'll create episodes by sliding windows through the data
        # In a real implementation, you would use more sophisticated methods
        
        window_size = self.config.get("episode_length", 24)
        stride = self.config.get("episode_stride", 12)
        
        for start_idx in range(0, len(features) - window_size - 1, stride):
            episode = []
            
            for i in range(start_idx, start_idx + window_size):
                state = scaled_features[i]
                
                # Simulate taking an action
                # In a real implementation, you would use a more sophisticated reward function
                action_idx = np.random.randint(0, self.action_dim)
                action = self.action_space[action_idx]
                
                # Calculate reward based on profitability
                reward = self._calculate_reward(features.iloc[i], action)
                
                next_state = scaled_features[i + 1]
                done = (i == start_idx + window_size - 1)
                
                episode.append((state, action_idx, reward, next_state, done))
            
            episodes.append(episode)
        
        return episodes
    
    def _calculate_reward(self, features: pd.Series, hashrate_percent: float) -> float:
        """
        Calculate reward for a given state and action.
        
        Args:
            features: Feature series for the current state
            hashrate_percent: Hashrate percentage (action)
            
        Returns:
            Reward value
        """
        # Extract relevant features
        price = features.get('price_usd', 50000)
        energy_price = features.get('energy_price', 0.12)  # $/kWh
        network_difficulty = features.get('network_difficulty', 30000000000000)
        block_reward = features.get('block_reward_usd', 300000)
        
        # Get miner specs
        base_hashrate = features.get('hashrate_th_s', 100)
        base_power = features.get('power_consumption_w', 3400)
        
        # Calculate adjusted hashrate and power
        adjusted_hashrate = base_hashrate * hashrate_percent
        
        # Power scales non-linearly with hashrate
        # This is a simplified model - in reality, the relationship is more complex
        power_factor = 0.7 + 0.3 * hashrate_percent  # Power scales less than linearly with hashrate
        adjusted_power = base_power * power_factor
        
        # Calculate mining revenue (simplified)
        # Revenue = (your_hashrate / network_hashrate) * block_reward * blocks_per_day
        # For simplicity, we'll use a constant for blocks_per_day and estimate network_hashrate from difficulty
        blocks_per_day = 144  # Bitcoin averages 144 blocks per day
        network_hashrate = network_difficulty / 600 / 1e12  # Rough estimate, in TH/s
        daily_revenue = (adjusted_hashrate / network_hashrate) * block_reward * blocks_per_day
        
        # Calculate energy cost
        daily_energy_cost = (adjusted_power / 1000) * 24 * energy_price
        
        # Calculate profit
        daily_profit = daily_revenue - daily_energy_cost
        
        # Calculate efficiency (profit per watt)
        efficiency = daily_profit / adjusted_power if adjusted_power > 0 else 0
        
        # Reward is a combination of profit and efficiency
        # We want to maximize profit while also considering efficiency
        reward = daily_profit * 0.7 + efficiency * adjusted_power * 0.3
        
        return reward
    
    def _train_with_episodes(self, episodes: List[List[Tuple]]) -> Dict:
        """
        Train the model using experience replay from episodes.
        
        Args:
            episodes: List of episodes
            
        Returns:
            Dictionary with training metrics
        """
        # Initialize metrics
        losses = []
        rewards = []
        
        # Flatten episodes into memory
        for episode in episodes:
            for transition in episode:
                self.remember(*transition)
        
        # Training iterations
        num_iterations = self.config.get("training_iterations", 1000)
        
        for i in range(num_iterations):
            # Sample batch from memory
            if len(self.memory) < self.batch_size:
                continue
                
            batch = self._sample_batch()
            
            states = np.array([b[0] for b in batch])
            actions = np.array([b[1] for b in batch])
            rewards = np.array([b[2] for b in batch])
            next_states = np.array([b[3] for b in batch])
            dones = np.array([b[4] for b in batch])
            
            # Current Q values
            current_q = self.model.predict(states, verbose=0)
            
            # Target Q values
            target_q = current_q.copy()
            
            # Update target Q values using Bellman equation
            next_q_values = self.target_model.predict(next_states, verbose=0)
            max_next_q = np.max(next_q_values, axis=1)
            
            for j in range(self.batch_size):
                if dones[j]:
                    target_q[j, actions[j]] = rewards[j]
                else:
                    target_q[j, actions[j]] = rewards[j] + self.gamma * max_next_q[j]
            
            # Train the model
            history = self.model.fit(states, target_q, epochs=1, verbose=0)
            losses.append(history.history['loss'][0])
            
            # Update target model periodically
            self.training_step += 1
            if self.training_step % self.update_target_frequency == 0:
                self.target_model.set_weights(self.model.get_weights())
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        
        return {
            "mean_loss": np.mean(losses),
            "final_loss": losses[-1] if losses else None,
            "mean_reward": np.mean(rewards) if rewards else None,
            "final_epsilon": self.epsilon
        }
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store transition in memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        if len(self.memory) >= self.max_memory_size:
            self.memory.pop(0)
        
        self.memory.append((state, action, reward, next_state, done))
    
    def _sample_batch(self):
        """
        Sample a batch from memory.
        
        Returns:
            Batch of transitions
        """
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        return [self.memory[i] for i in indices]
    
    def predict(self, features: pd.DataFrame) -> Dict:
        """
        Predict optimal hashrate settings for given features.
        
        Args:
            features: DataFrame containing feature data for a single miner
            
        Returns:
            Dictionary with optimal hashrate settings and expected performance
        """
        if self.model is None:
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
        
        # Get Q-values for all actions
        q_values = self.model.predict(features_scaled, verbose=0)[0]
        
        # Get the best action
        best_action_idx = np.argmax(q_values)
        best_hashrate_percent = self.action_space[best_action_idx]
        
        # Calculate expected performance for all actions
        performance = {}
        for i, hashrate_percent in enumerate(self.action_space):
            # Calculate expected performance metrics
            perf = self._calculate_performance(features.iloc[0], hashrate_percent)
            performance[f"{int(hashrate_percent * 100)}%"] = perf
        
        # Get the best performance
        best_performance = performance[f"{int(best_hashrate_percent * 100)}%"]
        
        # Calculate confidence based on Q-value distribution
        q_range = np.max(q_values) - np.min(q_values) if len(q_values) > 1 else 1
        q_normalized = (q_values - np.min(q_values)) / q_range if q_range > 0 else q_values
        confidence = q_normalized[best_action_idx]
        
        return {
            "optimal_hashrate_percent": best_hashrate_percent * 100,
            "confidence": float(confidence),
            "expected_performance": best_performance,
            "all_options": performance,
            "q_values": q_values.tolist()
        }
    
    def _calculate_performance(self, features: pd.Series, hashrate_percent: float) -> Dict:
        """
        Calculate expected performance metrics for a given hashrate setting.
        
        Args:
            features: Feature series for the current state
            hashrate_percent: Hashrate percentage
            
        Returns:
            Dictionary with performance metrics
        """
        # Extract relevant features
        price = features.get('price_usd', 50000)
        energy_price = features.get('energy_price', 0.12)  # $/kWh
        network_difficulty = features.get('network_difficulty', 30000000000000)
        block_reward = features.get('block_reward_usd', 300000)
        
        # Get miner specs
        base_hashrate = features.get('hashrate_th_s', 100)
        base_power = features.get('power_consumption_w', 3400)
        
        # Calculate adjusted hashrate and power
        adjusted_hashrate = base_hashrate * hashrate_percent
        
        # Power scales non-linearly with hashrate
        power_factor = 0.7 + 0.3 * hashrate_percent  # Power scales less than linearly with hashrate
        adjusted_power = base_power * power_factor
        
        # Calculate mining revenue (simplified)
        blocks_per_day = 144  # Bitcoin averages 144 blocks per day
        network_hashrate = network_difficulty / 600 / 1e12  # Rough estimate, in TH/s
        daily_revenue = (adjusted_hashrate / network_hashrate) * block_reward * blocks_per_day
        
        # Calculate energy cost
        daily_energy_cost = (adjusted_power / 1000) * 24 * energy_price
        
        # Calculate profit
        daily_profit = daily_revenue - daily_energy_cost
        
        # Calculate efficiency metrics
        energy_efficiency = adjusted_hashrate / adjusted_power  # TH/s per watt
        cost_efficiency = daily_profit / daily_energy_cost if daily_energy_cost > 0 else 0  # Profit per dollar spent on energy
        
        return {
            "hashrate_th_s": adjusted_hashrate,
            "power_consumption_w": adjusted_power,
            "daily_revenue_usd": daily_revenue,
            "daily_energy_cost_usd": daily_energy_cost,
            "daily_profit_usd": daily_profit,
            "energy_efficiency_th_w": energy_efficiency,
            "cost_efficiency": cost_efficiency
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
        if self.model is None:
            raise ValueError("No trained model to save")
        
        # Create model directory if it doesn't exist
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Generate default filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hashrate_tuner_{timestamp}.joblib"
        
        # Ensure filename has .joblib extension
        if not filename.endswith(".joblib"):
            filename += ".joblib"
        
        # Full path to save the model
        filepath = os.path.join(MODEL_DIR, filename)
        
        # Save the Keras model separately
        model_weights_path = os.path.join(MODEL_DIR, f"hashrate_tuner_weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5")
        self.model.save_weights(model_weights_path)
        
        # Save the model and metadata
        model_data = {
            "feature_names": self.feature_names,
            "scaler": self.scaler,
            "action_space": self.action_space,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "learning_rate": self.learning_rate,
            "model_weights_path": model_weights_path,
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
        self.feature_names = model_data["feature_names"]
        self.scaler = model_data["scaler"]
        self.action_space = model_data["action_space"]
        self.state_dim = model_data["state_dim"]
        self.action_dim = model_data["action_dim"]
        self.gamma = model_data.get("gamma", self.gamma)
        self.epsilon = model_data.get("epsilon", self.epsilon)
        self.epsilon_min = model_data.get("epsilon_min", self.epsilon_min)
        self.epsilon_decay = model_data.get("epsilon_decay", self.epsilon_decay)
        self.learning_rate = model_data.get("learning_rate", self.learning_rate)
        self.config = model_data.get("config", self.config)
        
        # Rebuild the model
        self.model = self.build_model(self.state_dim, self.action_dim)
        self.target_model = self.build_model(self.state_dim, self.action_dim)
        
        # Load model weights
        model_weights_path = model_data.get("model_weights_path")
        if model_weights_path and os.path.exists(model_weights_path):
            self.model.load_weights(model_weights_path)
            self.target_model.load_weights(model_weights_path)
        else:
            raise ValueError(f"Model weights file not found: {model_weights_path}")


# If run directly, train and test the model with mock data
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add parent directory to path to import feature_engineering
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from ml_engine.feature_engineering import MockDataGenerator, FeatureEngineeringPipeline
    
    # Generate mock data
    print("Generating mock data...")
    generator = MockDataGenerator(num_miners=3, num_coins=2, days=3, interval_minutes=60)
    miner_data, pool_data, market_data = generator.generate_all_data()
    
    # Create mock energy data
    print("Creating mock energy data...")
    timestamps = miner_data['timestamp'].unique()
    energy_data = pd.DataFrame({
        'timestamp': timestamps,
        'energy_price': np.random.uniform(0.08, 0.16, size=len(timestamps)),
        'grid_load_percent': np.random.uniform(50, 90, size=len(timestamps)),
        'renewable_percent': np.random.uniform(10, 40, size=len(timestamps))
    })
    
    # Process features
    print("Processing features...")
    pipeline = FeatureEngineeringPipeline()
    processed_miner = pipeline.process_miner_telemetry(miner_data)
    processed_market = pipeline.process_market_data(market_data)
    
    # Train the model
    print("Training hashrate tuning model...")
    model = HashrateTuner()
    results = model.train(processed_market, processed_miner, energy_data, processed_market)
    
    # Print results
    print("\nTraining Results:")
    print(f"Model type: {results['model_type']}")
    print(f"Features: {results['feature_count']}")
    print(f"Training samples: {results['training_samples']}")
    print(f"Episodes: {results['episodes']}")
    print("\nMetrics:")
    for metric, value in results['metrics'].items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    # Save the model
    model_path = model.save()
    print(f"\nModel saved to: {model_path}")
    
    # Test prediction
    print("\nTesting hashrate tuning prediction...")
    test_features = processed_miner.iloc[[0]].copy()
    for col in model.feature_names:
        if col not in test_features.columns:
            if col in processed_market.columns:
                test_features[col] = processed_market.iloc[0][col]
            else:
                test_features[col] = 0
    
    prediction = model.predict(test_features)
    
    print("\nPrediction Results:")
    print(f"Optimal hashrate: {prediction['optimal_hashrate_percent']:.1f}%")
    print(f"Confidence: {prediction['confidence']:.4f}")
    print("\nExpected Performance:")
    for metric, value in prediction['expected_performance'].items():
        print(f"  {metric}: {value:.4f}")
