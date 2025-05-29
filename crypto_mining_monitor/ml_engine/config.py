"""
Configuration settings for the ML recommendation engine.
"""

from ml_engine.utils.constants import MODEL_DIR

# Model configuration
MODEL_CONFIG = {
    "profit_prediction": {
        "algorithm": "xgboost",
        "hyperparameters": {
            "objective": "reg:squarederror",
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0,
            "alpha": 0.1,
            "lambda": 1,
            "n_estimators": 200,
            "early_stopping_rounds": 20,
            "seed": 42
        },
        "features": [
            # Miner telemetry features
            "hashrate_th_s", "power_consumption_w", "efficiency_j_th",
            "avg_chip_temp_c", "fan_speed_percent",
            
            # Pool performance features
            "effective_hashrate_th_s", "earnings_usd_24h", "earnings_per_th_usd",
            
            # Market data features
            "price_usd", "price_change_24h_percent", "network_difficulty",
            "block_reward_usd", "transaction_fees_percent"
        ],
        "target": "predicted_earnings_per_th_usd"
    },
    "power_optimization": {
        "algorithm": "bayesian_optimization",
        "hyperparameters": {
            "acquisition_function": "expected_improvement",
            "alpha": 0.0001,
            "n_initial_points": 10,
            "noise": "gaussian",
            "normalize_y": True,
            "kernel": "matern",
            "n_restarts_optimizer": 5,
            "random_state": 42
        },
        "search_bounds": {
            "power_limit": [0.7, 1.0],  # Percentage of rated power
            "frequency": [0.8, 1.1],    # Percentage of rated frequency
            "voltage": [0.9, 1.05]      # Percentage of rated voltage
        },
        "constraints": {
            "max_temperature": 75,       # Celsius
            "min_hashrate": 0.8,         # Percentage of rated hashrate
            "max_error_rate": 0.02       # Maximum acceptable error rate
        },
        "target": "efficiency_j_th"
    },
    "hashrate_tuning": {
        "algorithm": "deep_q_learning",
        "hyperparameters": {
            "gamma": 0.95,               # Discount factor
            "epsilon": 1.0,              # Initial exploration rate
            "epsilon_min": 0.01,         # Minimum exploration rate
            "epsilon_decay": 0.995,      # Exploration rate decay
            "learning_rate": 0.001,      # Learning rate
            "batch_size": 32,            # Batch size for training
            "max_memory_size": 2000,     # Maximum memory size for experience replay
            "update_target_frequency": 100  # Frequency to update target network
        },
        "min_hashrate_percent": 70,      # Minimum hashrate percentage
        "max_hashrate_percent": 100,     # Maximum hashrate percentage
        "hashrate_levels": 7,            # Number of discrete hashrate levels
        "episode_length": 24,            # Length of training episodes
        "episode_stride": 12,            # Stride between episodes
        "training_iterations": 1000      # Number of training iterations
    },
    "intelligent_overclocking": {
        "algorithm": "bayesian_optimization",
        "hyperparameters": {
            "acquisition_function": "expected_improvement",
            "alpha": 1e-6,
            "normalize_y": True,
            "n_restarts_optimizer": 5,
            "random_state": 42
        },
        "param_bounds": {
            "core_clock_offset": [-100, 200],     # MHz
            "memory_clock_offset": [-500, 1500],  # MHz
            "power_limit_percent": [70, 100],     # Percentage
            "core_voltage_offset": [-50, 50]      # mV
        },
        "optimization_goals": ["efficiency", "performance", "balanced", "conservative"],
        "default_goal": "balanced"
    }
}

# API configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": True,
    "reload": True
}

# Feature engineering configuration
FEATURE_CONFIG = {
    "time_windows": [
        {"name": "1h", "minutes": 60},
        {"name": "6h", "minutes": 360},
        {"name": "24h", "minutes": 1440},
        {"name": "7d", "minutes": 10080}
    ],
    "aggregation_functions": ["mean", "std", "min", "max"],
    "electricity_cost_per_kwh": 0.10,
    "baseline_percentage": 0.1,
    "trend_analysis": {
        "polynomial_degree": 3,
        "recent_percentage": 0.3
    }
}

# Recommendation configuration
RECOMMENDATION_CONFIG = {
    "coin_switching": {
        "min_improvement_threshold": 0.05,  # 5% minimum improvement to recommend switch
        "confidence_threshold": 0.7,        # 70% confidence required
        "cooldown_period_hours": 12         # Minimum hours between switches
    },
    "power_optimization": {
        "min_efficiency_improvement": 0.03,  # 3% minimum efficiency improvement
        "max_hashrate_reduction": 0.05,      # 5% maximum hashrate reduction
        "confidence_threshold": 0.75,
        "cooldown_period_hours": 24
    },
    "dynamic_hashrate_tuning": {
        "min_profit_improvement": 0.03,      # 3% minimum profit improvement
        "confidence_threshold": 0.7,         # 70% confidence required
        "cooldown_period_hours": 6,          # Minimum hours between adjustments
        "max_hashrate_reduction": 0.3        # 30% maximum hashrate reduction
    },
    "intelligent_overclocking": {
        "min_hashrate_improvement": 0.05,    # 5% minimum hashrate improvement
        "min_efficiency_improvement": 0.03,  # 3% minimum efficiency improvement
        "max_temperature_increase": 5,       # 5°C maximum temperature increase
        "min_stability_score": 0.8,          # 80% minimum stability score
        "confidence_threshold": 0.7,         # 70% confidence required
        "cooldown_period_hours": 24,         # Minimum hours between adjustments
        "optimization_goal": "balanced"      # Default optimization goal
    }
}
