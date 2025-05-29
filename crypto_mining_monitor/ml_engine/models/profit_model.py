"""
Profit prediction model using gradient boosting.
"""

import os
import joblib
import numpy as np
import pandas as pd
import json
import sys
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score
from scipy.stats import skew, kurtosis
import xgboost as xgb

from ml_engine.utils.constants import MODEL_DIR
from ml_engine.utils.logging_config import logger
from ml_engine.utils.validation import ValidationError, validate_dataframe, handle_missing_data
from ml_engine.config import MODEL_CONFIG


class ProfitPredictionModel:
    """
    Gradient boosting model for predicting mining profitability.
    """
    
    def __init__(self):
        """Initialize the profit prediction model."""
        self.config = MODEL_CONFIG["profit_prediction"]
        self.model = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.feature_selector = None
    
    def train(self, features: pd.DataFrame, target: pd.Series, 
              test_size: float = 0.2, random_state: int = 42,
              use_feature_selection: bool = True,
              use_cv: bool = True) -> Dict:
        """
        Train the profit prediction model.
        
        Args:
            features: DataFrame containing feature data
            target: Series containing target values (predicted profitability)
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            use_feature_selection: Whether to use feature selection
            use_cv: Whether to use cross-validation
            
        Returns:
            Dictionary with training results and metrics
        """
        # Validate inputs
        try:
            validate_dataframe(features, features.columns.tolist(), "Features")
            if target.isna().any():
                raise ValidationError("Target contains NaN values")
        except ValidationError as e:
            logger.error(f"Validation error during training: {str(e)}")
            raise
        
        # Handle missing values
        features = handle_missing_data(features, strategy='conservative')
        
        # Store feature names for inference
        self.feature_names = features.columns.tolist()
        
        # Initialize metrics tracking
        cv_scores = {
            'mse': [],
            'rmse': [],
            'mae': [],
            'r2': [],
            'directional_accuracy': []
        }
        
        if use_cv:
            # Use time-series cross-validation
            logger.info("Training with time-series cross-validation")
            tscv = TimeSeriesSplit(n_splits=5)
            
            for train_idx, test_idx in tscv.split(features):
                # Split data
                X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
                y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
                
                # Feature selection if enabled
                if use_feature_selection:
                    X_train, X_test = self._perform_feature_selection(X_train, X_test, y_train)
                
                # Scale features
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                
                # Train model
                self.model = xgb.XGBRegressor(**self.config["hyperparameters"])
                self.model.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_test_scaled, y_test)],
                    early_stopping_rounds=self.config["hyperparameters"].get("early_stopping_rounds", 10),
                    verbose=False
                )
                
                # Evaluate
                y_pred = self.model.predict(X_test_scaled)
                
                # Calculate metrics
                mse = ((y_pred - y_test) ** 2).mean()
                rmse = np.sqrt(mse)
                mae = np.abs(y_pred - y_test).mean()
                r2 = r2_score(y_test, y_pred)
                
                # Calculate directional accuracy
                if len(y_test) > 1:
                    y_test_direction = np.diff(y_test) > 0
                    y_pred_direction = np.diff(y_pred) > 0
                    directional_accuracy = np.mean(y_test_direction == y_pred_direction)
                else:
                    directional_accuracy = np.nan
                
                # Store metrics
                cv_scores['mse'].append(mse)
                cv_scores['rmse'].append(rmse)
                cv_scores['mae'].append(mae)
                cv_scores['r2'].append(r2)
                cv_scores['directional_accuracy'].append(directional_accuracy)
            
            # Calculate average metrics
            avg_metrics = {
                'mse': np.mean(cv_scores['mse']),
                'rmse': np.mean(cv_scores['rmse']),
                'mae': np.mean(cv_scores['mae']),
                'r2': np.mean(cv_scores['r2']),
                'directional_accuracy': np.mean([x for x in cv_scores['directional_accuracy'] if not np.isnan(x)])
            }
            
            # Calculate standard deviation (for confidence intervals)
            std_metrics = {
                'mse_std': np.std(cv_scores['mse']),
                'rmse_std': np.std(cv_scores['rmse']),
                'mae_std': np.std(cv_scores['mae']),
                'r2_std': np.std(cv_scores['r2']),
                'directional_accuracy_std': np.std([x for x in cv_scores['directional_accuracy'] if not np.isnan(x)])
            }
            
            # Train final model on all data
            if use_feature_selection:
                features = self._perform_feature_selection(features, None, target)[0]
            
            X_scaled = self.scaler.fit_transform(features)
            self.model = xgb.XGBRegressor(**self.config["hyperparameters"])
            self.model.fit(X_scaled, target)
            
            # Update feature names if feature selection was used
            if use_feature_selection:
                self.feature_names = features.columns.tolist()
            
            # Get feature importance
            feature_importance = self.model.feature_importances_
            
            # Return results
            return {
                "model_type": "xgboost",
                "feature_count": len(self.feature_names),
                "training_samples": len(features),
                "test_samples": len(features) // 5,  # Approximate based on CV
                "metrics": avg_metrics,
                "metrics_std": std_metrics,
                "feature_importance": dict(zip(self.feature_names, feature_importance.tolist()))
            }
        else:
            # Use simple train-test split
            logger.info("Training with simple train-test split")
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=test_size, random_state=random_state
            )
            
            # Feature selection if enabled
            if use_feature_selection:
                X_train, X_test = self._perform_feature_selection(X_train, X_test, y_train)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Initialize and train the model
            if self.config["algorithm"].lower() == "xgboost":
                self.model = xgb.XGBRegressor(**self.config["hyperparameters"])
                
                # Train the model
                self.model.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_test_scaled, y_test)],
                    early_stopping_rounds=self.config["hyperparameters"].get("early_stopping_rounds", 10),
                    verbose=False
                )
                
                # Update feature names if feature selection was used
                if use_feature_selection:
                    self.feature_names = X_train.columns.tolist()
                
                # Get feature importance
                feature_importance = self.model.feature_importances_
                
                # Make predictions on test set
                y_pred = self.model.predict(X_test_scaled)
                
                # Calculate metrics
                mse = ((y_pred - y_test) ** 2).mean()
                rmse = np.sqrt(mse)
                mae = np.abs(y_pred - y_test).mean()
                r2 = r2_score(y_test, y_pred)
                
                # Calculate directional accuracy (correct prediction of up/down movement)
                if len(y_test) > 1:
                    y_test_direction = np.diff(y_test) > 0
                    y_pred_direction = np.diff(y_pred) > 0
                    directional_accuracy = np.mean(y_test_direction == y_pred_direction)
                else:
                    directional_accuracy = np.nan
                
                # Return training results
                return {
                    "model_type": "xgboost",
                    "feature_count": len(self.feature_names),
                    "training_samples": len(X_train),
                    "test_samples": len(X_test),
                    "metrics": {
                        "mse": mse,
                        "rmse": rmse,
                        "mae": mae,
                        "r2": r2,
                        "directional_accuracy": directional_accuracy
                    },
                    "feature_importance": dict(zip(self.feature_names, feature_importance.tolist()))
                }
            else:
                raise ValueError(f"Unsupported algorithm: {self.config['algorithm']}")
    
    def _perform_feature_selection(self, X_train, X_test, y_train):
        """
        Perform feature selection using SelectFromModel.
        
        Args:
            X_train: Training features
            X_test: Test features (can be None)
            y_train: Training target
            
        Returns:
            Tuple of (X_train_selected, X_test_selected)
        """
        logger.info("Performing feature selection")
        
        # Initialize feature selector
        self.feature_selector = SelectFromModel(
            estimator=xgb.XGBRegressor(
                objective='reg:squarederror',
                learning_rate=0.1,
                max_depth=5,
                n_estimators=100,
                random_state=42
            ),
            threshold='median'
        )
        
        # Fit and transform training data
        X_train_selected = pd.DataFrame(
            self.feature_selector.fit_transform(X_train, y_train),
            columns=X_train.columns[self.feature_selector.get_support()]
        )
        
        # Log selected features
        selected_features = X_train.columns[self.feature_selector.get_support()].tolist()
        logger.info(f"Selected {len(selected_features)} features: {selected_features}")
        
        # Transform test data if provided
        X_test_selected = None
        if X_test is not None:
            X_test_selected = pd.DataFrame(
                self.feature_selector.transform(X_test),
                columns=X_train.columns[self.feature_selector.get_support()]
            )
        
        return X_train_selected, X_test_selected
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Make profitability predictions.
        
        Args:
            features: DataFrame containing feature data
            
        Returns:
            Array of predicted profitability values
            
        Raises:
            ValidationError: If features are invalid
            ValueError: If model is not trained
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet")
        
        try:
            # Validate input
            if not isinstance(features, pd.DataFrame):
                raise ValidationError(f"Features must be a DataFrame, got {type(features)}")
            
            # Ensure features have the correct columns
            if not all(col in features.columns for col in self.feature_names):
                missing_cols = set(self.feature_names) - set(features.columns)
                raise ValidationError(f"Missing features: {missing_cols}")
            
            # Select and order features correctly
            features = features[self.feature_names]
            
            # Handle missing values
            features = handle_missing_data(features, strategy='conservative')
            
            # Apply feature selection if it was used during training
            if self.feature_selector is not None:
                features = pd.DataFrame(
                    self.feature_selector.transform(features),
                    columns=features.columns[self.feature_selector.get_support()]
                )
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make predictions
            predictions = self.model.predict(features_scaled)
            
            return predictions
        
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def batch_predict_profitability(self, features_list: List[pd.DataFrame]) -> np.ndarray:
        """
        Make batch predictions for multiple feature sets.
        
        Args:
            features_list: List of DataFrames with feature data
            
        Returns:
            Array of predicted profitability values
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet")
        
        # Combine features into a single DataFrame
        combined_features = pd.concat(features_list, ignore_index=True)
        
        # Make predictions
        predictions = self.predict(combined_features)
        
        return predictions
    
    def save(self, filename: Optional[str] = None, metadata: Optional[Dict] = None) -> str:
        """
        Save the trained model to disk.
        
        Args:
            filename: Name of the file to save the model to (without path)
                If None, a default name will be generated
            metadata: Additional metadata to save with the model
                
        Returns:
            Path to the saved model file
        """
        if self.model is None:
            raise ValueError("No trained model to save")
        
        # Create model directory if it doesn't exist
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Generate version if not provided in filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"v{timestamp}"
        
        # Generate default filename if not provided
        if filename is None:
            filename = f"profit_model_{version}.joblib"
        
        # Ensure filename has .joblib extension
        if not filename.endswith(".joblib"):
            filename += ".joblib"
        
        # Full path to save the model
        filepath = os.path.join(MODEL_DIR, filename)
        
        # Prepare metadata
        default_metadata = {
            "model_type": "profit_prediction",
            "algorithm": self.config["algorithm"],
            "feature_count": len(self.feature_names),
            "hyperparameters": self.config["hyperparameters"],
            "feature_names": self.feature_names,
            "created_at": datetime.now().isoformat(),
            "created_by": os.getenv("USER", "unknown"),
            "python_version": sys.version,
            "library_versions": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit-learn": sklearn.__version__,
                "xgboost": xgb.__version__
            }
        }
        
        # Merge with provided metadata
        if metadata:
            default_metadata.update(metadata)
        
        # Save model and metadata
        model_data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "scaler": self.scaler,
            "feature_selector": self.feature_selector,
            "config": self.config,
            "metadata": default_metadata
        }
        
        joblib.dump(model_data, filepath)
        
        # Save metadata separately for easy access
        metadata_path = os.path.join(MODEL_DIR, f"profit_model_{version}_metadata.json")
        with open(metadata_path, 'w') as f:
            # Convert numpy types to Python native types for JSON serialization
            metadata_json = {}
            for k, v in default_metadata.items():
                if isinstance(v, dict):
                    metadata_json[k] = {kk: vv if not isinstance(vv, np.generic) else vv.item() for kk, vv in v.items()}
                elif isinstance(v, list):
                    metadata_json[k] = [vv if not isinstance(vv, np.generic) else vv.item() for vv in v]
                elif isinstance(v, np.generic):
                    metadata_json[k] = v.item()
                else:
                    metadata_json[k] = v
            
            json.dump(metadata_json, f, indent=2)
        
        logger.info(f"Model saved to: {filepath}")
        logger.info(f"Metadata saved to: {metadata_path}")
        
        return filepath
    
    def load(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model file
        """
        try:
            # Load the model data
            model_data = joblib.load(filepath)
            
            # Restore model components
            self.model = model_data["model"]
            self.feature_names = model_data["feature_names"]
            self.scaler = model_data["scaler"]
            self.feature_selector = model_data.get("feature_selector")
            self.config = model_data.get("config", self.config)
            
            logger.info(f"Model loaded from: {filepath}")
            logger.info(f"Model has {len(self.feature_names)} features")
        
        except Exception as e:
            logger.error(f"Error loading model from {filepath}: {str(e)}")
            raise
    
    def evaluate_model(self, features: pd.DataFrame, target: pd.Series) -> Dict:
        """
        Comprehensive model evaluation.
        
        Args:
            features: DataFrame with feature data
            target: Series with target values
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Ensure model is trained
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Handle missing values
        features = handle_missing_data(features, strategy='conservative')
        
        # Apply feature selection if it was used during training
        if self.feature_selector is not None:
            features = pd.DataFrame(
                self.feature_selector.transform(features),
                columns=features.columns[self.feature_selector.get_support()]
            )
        
        # Scale features
        X_scaled = self.scaler.transform(features)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Basic metrics
        metrics = {
            'mse': ((predictions - target) ** 2).mean(),
            'rmse': np.sqrt(((predictions - target) ** 2).mean()),
            'mae': np.abs(predictions - target).mean(),
            'r2': r2_score(target, predictions)
        }
        
        # Directional accuracy
        if len(target) > 1:
            target_direction = np.diff(target) > 0
            pred_direction = np.diff(predictions) > 0
            metrics['directional_accuracy'] = np.mean(target_direction == pred_direction)
        
        # Residual analysis
        residuals = target - predictions
        metrics['residual_mean'] = residuals.mean()
        metrics['residual_std'] = residuals.std()
        
        # Check for bias
        metrics['bias'] = residuals.mean() / target.mean() if target.mean() != 0 else 0
        
        # Error distribution
        metrics['error_skew'] = skew(residuals)
        metrics['error_kurtosis'] = kurtosis(residuals)
        
        # Prediction intervals (bootstrap method)
        n_bootstraps = 100
        bootstrap_predictions = []
        
        for _ in range(n_bootstraps):
            # Sample with replacement
            idx = np.random.choice(len(features), len(features), replace=True)
            X_boot = features.iloc[idx]
            y_boot = target.iloc[idx]
            
            # Train model on bootstrap sample
            model_boot = xgb.XGBRegressor(**self.config["hyperparameters"])
            X_boot_scaled = self.scaler.transform(X_boot)
            model_boot.fit(X_boot_scaled, y_boot)
            
            # Predict on original data
            boot_preds = model_boot.predict(X_scaled)
            bootstrap_predictions.append(boot_preds)
        
        # Calculate prediction intervals
        bootstrap_predictions = np.array(bootstrap_predictions)
        lower_bound = np.percentile(bootstrap_predictions, 2.5, axis=0)
        upper_bound = np.percentile(bootstrap_predictions, 97.5, axis=0)
        
        # Calculate interval coverage
        coverage = np.mean((target >= lower_bound) & (target <= upper_bound))
        metrics['prediction_interval_coverage'] = coverage
        
        return metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the trained model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None or self.feature_names is None:
            raise ValueError("Model has not been trained or loaded yet")
        
        # Get feature importance
        importance = self.model.feature_importances_
        
        # Create dictionary mapping feature names to importance scores
        importance_dict = dict(zip(self.feature_names, importance))
        
        # Sort by importance (descending)
        importance_dict = {k: v for k, v in sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)}
        
        return importance_dict
    
    def optimize_model(self, optimization_type='pruning'):
        """
        Optimize the trained model for inference performance.
        
        Args:
            optimization_type: Type of optimization ('pruning', 'quantization')
            
        Returns:
            True if optimization was performed, False otherwise
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        if optimization_type == 'pruning':
            # For XGBoost, prune the model
            if isinstance(self.model, xgb.XGBRegressor):
                # Get feature importance
                importance = self.model.feature_importances_
                
                # Identify features with low importance
                threshold = 0.01  # 1% importance threshold
                low_importance = importance < threshold
                
                if any(low_importance):
                    # Get feature names with low importance
                    low_imp_features = [
                        self.feature_names[i] for i in range(len(self.feature_names))
                        if low_importance[i]
                    ]
                    
                    logger.info(f"Pruning {sum(low_importance)} low importance features: {low_imp_features}")
                    
                    # Create new feature list without low importance features
                    self.feature_names = [
                        self.feature_names[i] for i in range(len(self.feature_names))
                        if not low_importance[i]
                    ]
                    
                    # Note: In a real implementation, you would retrain the model with the pruned feature set
                    # This is a simplified approach
                    logger.info("Model pruned - note that retraining with pruned feature set is recommended")
                    
                    return True
            
            return False
        
        elif optimization_type == 'quantization':
            # Model quantization for reduced memory footprint
            # This is a placeholder - actual implementation would depend on the model type
            logger.info("Model quantization not implemented for this model type")
            return False


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
    
    # Check if earnings_per_th_usd exists, otherwise create a synthetic target
    if 'earnings_per_th_usd' in combined_features.columns:
        target = combined_features['earnings_per_th_usd']
    else:
        # Create a synthetic target based on available data
        print("Creating synthetic target for training...")
        if 'earnings_usd_24h' in combined_features.columns and 'hashrate_th_s' in combined_features.columns:
            target = combined_features['earnings_usd_24h'] / combined_features['hashrate_th_s']
        else:
            # Generate random values as a fallback
            target = pd.Series(np.random.uniform(0.1, 0.5, size=len(combined_features)))
    
    # Select features for training (exclude the target and any leaky features)
    exclude_cols = ['earnings_per_th_usd', 'worker_id', 'miner_id', 'timestamp', 'coin_id', 'earnings_usd_24h']
    feature_cols = [col for col in combined_features.columns if col not in exclude_cols]
    features = combined_features[feature_cols]
    
    # Handle missing values (simple imputation with mean)
    features = features.fillna(features.mean())
    
    # Train the model
    print("Training profit prediction model...")
    model = ProfitPredictionModel()
    results = model.train(features, target, use_feature_selection=True, use_cv=True)
    
    # Print results
    print("\nTraining Results:")
    print(f"Model type: {results['model_type']}")
    print(f"Features: {results['feature_count']}")
    print(f"Training samples: {results['training_samples']}")
    print(f"Test samples: {results['test_samples']}")
    print("\nMetrics:")
    for metric, value in results['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nTop 10 Feature Importance:")
    importance = model.get_feature_importance()
    for i, (feature, score) in enumerate(list(importance.items())[:10]):
        print(f"  {i+1}. {feature}: {score:.4f}")
    
    # Save the model
    model_path = model.save()
    print(f"\nModel saved to: {model_path}")
    
    # Test loading and prediction
    print("\nTesting model loading and prediction...")
    new_model = ProfitPredictionModel()
    new_model.load(model_path)
    
    # Make predictions on a few samples
    test_samples = features.head(5)
    predictions = new_model.predict(test_samples)
    
    print("\nSample Predictions:")
    for i, pred in enumerate(predictions):
        print(f"  Sample {i+1}: Predicted earnings per TH: ${pred:.4f}")
