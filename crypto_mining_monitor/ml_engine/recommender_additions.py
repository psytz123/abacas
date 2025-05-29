
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
            f"Power consumption will {('increase by ' + str(abs(improvements['power_change'] * 100):.1f) + '%') if improvements['power_change'] > 0 else ('decrease by ' + str(abs(improvements['power_change'] * 100):.1f) + '%')}.",
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
