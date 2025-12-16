"""Feature Engineering Pipeline

Transforms raw supply chain data into features for machine learning.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering for supply chain risk prediction."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def create_time_features(self, df):
        """Create time-based features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional time features
        """
        df = df.copy()
        
        # Lag features for delivery delays
        df['delay_lag_1'] = df['delivery_delay_days'].shift(1).fillna(0)
        df['delay_lag_3'] = df['delivery_delay_days'].shift(3).fillna(0)
        df['delay_lag_7'] = df['delivery_delay_days'].shift(7).fillna(0)
        
        # Rolling statistics
        df['delay_rolling_mean_7'] = df['delivery_delay_days'].rolling(window=7, min_periods=1).mean()
        df['delay_rolling_std_7'] = df['delivery_delay_days'].rolling(window=7, min_periods=1).std().fillna(0)
        df['quality_rolling_mean_14'] = df['quality_score'].rolling(window=14, min_periods=1).mean()
        
        return df
    
    def create_interaction_features(self, df):
        """Create feature interactions.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        df = df.copy()
        
        # Key interactions
        df['delay_quality_interaction'] = df['delivery_delay_days'] * (100 - df['quality_score'])
        df['risk_weather_interaction'] = df['geopolitical_risk'] * df['weather_disruption']
        df['demand_leadtime_ratio'] = df['seasonal_demand_factor'] / (df['lead_time_days'] + 1)
        
        return df
    
    def engineer_features(self, df):
        """Apply complete feature engineering pipeline.
        
        Args:
            df: Raw input DataFrame
            
        Returns:
            Engineered feature DataFrame
        """
        logger.info("Starting feature engineering...")
        
        df = self.create_time_features(df)
        df = self.create_interaction_features(df)
        
        # Store feature names (excluding target)
        self.feature_names = [col for col in df.columns if col != 'high_risk']
        
        logger.info(f"Feature engineering complete. Total features: {len(self.feature_names)}")
        
        return df
    
    def scale_features(self, X_train, X_test=None):
        """Scale features using StandardScaler.
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            
        Returns:
            Scaled features
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
