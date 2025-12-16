"""Synthetic Supply Chain Data Generator

Generates realistic supply chain risk scenarios for model training and testing.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SupplyChainDataGenerator:
    """Generate synthetic supply chain operational data."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_data(self, n_records=10000):
        """Generate synthetic supply chain dataset.
        
        Args:
            n_records: Number of records to generate
            
        Returns:
            DataFrame with supply chain features and risk labels
        """
        logger.info(f"Generating {n_records} synthetic supply chain records...")
        
        # Generate base features
        data = {
            'supplier_id': np.random.randint(1000, 9999, n_records),
            'delivery_delay_days': np.random.exponential(scale=2.5, size=n_records),
            'quality_score': np.random.normal(loc=85, scale=10, size=n_records).clip(0, 100),
            'seasonal_demand_factor': np.random.uniform(0.7, 1.3, n_records),
            'weather_disruption': np.random.binomial(1, 0.15, n_records),
            'geopolitical_risk': np.random.uniform(0, 1, n_records),
            'lead_time_days': np.random.normal(loc=14, scale=5, size=n_records).clip(1, 60),
            'order_quantity': np.random.lognormal(mean=6, sigma=1, size=n_records)
        }
        
        df = pd.DataFrame(data)
        
        # Generate risk score based on multiple factors
        risk_score = (
            df['delivery_delay_days'] * 0.3 +
            (100 - df['quality_score']) * 0.25 +
            df['weather_disruption'] * 10 +
            df['geopolitical_risk'] * 15 +
            df['lead_time_days'] * 0.2
        )
        
        # Binary classification: high_risk
        df['high_risk'] = (risk_score > risk_score.median()).astype(int)
        
        logger.info(f"Generated data shape: {df.shape}")
        logger.info(f"High risk percentage: {df['high_risk'].mean():.2%}")
        
        return df
    
    def save_data(self, df, output_path):
        """Save generated data to file.
        
        Args:
            df: DataFrame to save
            output_path: Path to save CSV file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic supply chain data')
    parser.add_argument('--records', type=int, default=10000, help='Number of records to generate')
    parser.add_argument('--output', type=str, default='data/raw/supply_chain_data.csv', help='Output file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    generator = SupplyChainDataGenerator(random_state=args.seed)
    df = generator.generate_data(n_records=args.records)
    generator.save_data(df, args.output)
    
    logger.info("Data generation complete!")


if __name__ == "__main__":
    main()
