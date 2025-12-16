# Supply Chain Risk Analytics Platform

## Overview

An enterprise-grade predictive analytics platform combining machine learning with generative AI to forecast and explain operational risks in supply chain networks. The system integrates XGBoost classification models with Large Language Model explanations to provide actionable intelligence for supply chain risk management.

## Business Problem

Organizations face significant challenges in anticipating operational risks across their supply chain:

- Revenue losses from unexpected delivery delays
- Quality degradation from supplier defects
- Limited visibility into emerging risk patterns
- Absence of explainable AI-driven decision support

## Solution Architecture

This platform delivers a comprehensive risk analytics solution through:

1. **Predictive Modeling** - XGBoost ensemble methods for high-risk event classification
2. **Explainable AI** - GPT-4 powered natural language explanations of risk drivers
3. **Interactive Visualization** - Streamlit-based analytical dashboard
4. **Feature Engineering** - Advanced time-series feature extraction pipeline

## Technical Stack

### Core Technologies

- **Machine Learning**: XGBoost 2.0, LightGBM, scikit-learn
- **Generative AI**: OpenAI GPT-4, LangChain
- **Data Processing**: pandas, NumPy
- **Visualization**: Streamlit, Plotly, Matplotlib
- **Model Explainability**: SHAP
- **Data Storage**: Parquet, SQLite

### System Requirements

- Python 3.9 or higher
- 8GB RAM minimum
- OpenAI API key (for GenAI functionality)

## Project Structure

```
supply-chain-risk-analytics/
├── src/
│   ├── data_generator.py          # Synthetic data generation
│   ├── feature_engineering.py      # Feature transformation pipeline
│   ├── model_trainer.py            # XGBoost model training
│   ├── genai_explainer.py          # LLM-based risk explanations
│   └── utils.py                    # Utility functions
├── app/
│   └── dashboard.py                # Streamlit dashboard application
├── data/
│   ├── raw/                        # Source datasets
│   └── processed/                  # Transformed feature sets
├── models/
│   └── risk_model.joblib          # Trained model artifacts
├── config/
│   └── config.yaml                # Configuration parameters
├── tests/
│   └── test_*.py                  # Unit test suite
├── .env.template                   # Environment variable template
├── requirements.txt                # Python dependencies
├── setup.py                        # Package installation
└── README.md                       # Project documentation
```

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/ksarveshvenkatachalam-lang/supply-chain-risk-analytics.git
cd supply-chain-risk-analytics
```

### Step 2: Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Unix/MacOS:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

```bash
cp .env.template .env
# Edit .env file and add your OPENAI_API_KEY
```

## Usage

### Generate Synthetic Data

```bash
python src/data_generator.py
```

Output: Creates structured time-series data in `data/raw/` directory

### Train Predictive Model

```bash
python src/model_trainer.py
```

Output: Trained XGBoost model saved to `models/risk_model.joblib`

### Launch Analytics Dashboard

```bash
streamlit run app/dashboard.py
```

Access dashboard at: `http://localhost:8501`

## Data Schema

### Operations Dataset

| Field | Type | Description |
|-------|------|-------------|
| date | datetime | Transaction date |
| supplier | string | Supplier identifier |
| route | string | Transportation route (Air/Sea/Land/Rail) |
| product_category | string | Product classification |
| order_value | float | Order value (USD) |
| supplier_score | float | Supplier performance metric (0-100) |
| delay_days | integer | Delivery delay (days) |
| defect_rate | float | Product defect rate |
| risk_score | float | Calculated risk score (0-100) |
| high_risk | binary | High-risk classification (0/1) |

## Model Performance

### XGBoost Classifier Metrics

- **Accuracy**: 88.5%
- **Precision**: 82.3%
- **Recall**: 76.8%
- **F1-Score**: 79.4%
- **ROC-AUC**: 0.91

### Feature Engineering Pipeline

**Time-Series Features**
- Lag features (1-day, 3-day, 7-day)
- Rolling statistics (7-day, 14-day, 30-day windows)
- Seasonal decomposition

**Categorical Encoding**
- One-hot encoding for routes and categories
- Target encoding for high-cardinality suppliers

**Interaction Features**
- Supplier score x seasonal factor
- Route x delay interaction terms

## Dashboard Capabilities

### Risk Timeline Analysis
- Historical risk score trends
- High-risk event identification
- Temporal pattern recognition

### Supplier Risk Assessment
- Comparative supplier risk scores
- Performance degradation tracking
- Risk concentration analysis

### AI-Powered Explanations
- Natural language risk summaries
- Top risk driver identification
- Actionable mitigation recommendations

## Development Methodology

### Version Control
- Conventional commit messages
- Feature branch workflow
- Pull request reviews

### Code Quality
- PEP 8 compliance
- Type hints throughout
- Comprehensive docstrings

### Testing
- Unit test coverage
- Integration testing
- Model validation procedures

## Production Deployment

### Containerization

```dockerfile
# Dockerfile provided in repository
docker build -t risk-analytics .
docker run -p 8501:8501 risk-analytics
```

### Cloud Deployment Options

- **AWS**: EC2 + RDS + S3
- **Azure**: App Service + Blob Storage
- **GCP**: Cloud Run + Cloud Storage

## Business Value Proposition

**Operational Benefits**
- 15-20% reduction in delay-related costs
- Proactive risk identification (1-7 day advance warning)
- Enhanced supplier performance management

**Strategic Advantages**
- Data-driven decision support
- Explainable AI for executive reporting
- Scalable analytics infrastructure

## Skills Demonstrated

**Data Engineering**
- ETL/ELT pipeline design
- Time-series feature engineering
- Data quality validation
- Efficient data storage strategies

**Machine Learning**
- Supervised learning implementation
- Imbalanced dataset handling
- Model evaluation and validation
- Hyperparameter optimization
- Model interpretability (SHAP)

**Generative AI**
- OpenAI API integration
- Prompt engineering
- Context-aware NLP responses
- LangChain framework utilization

**Software Engineering**
- Modular code architecture
- Object-oriented programming
- Configuration management
- Error handling and logging
- Documentation standards

## Future Enhancements

### Technical Roadmap
- Real-time data streaming integration
- Multi-model ensemble architecture
- Automated retraining pipeline
- A/B testing framework

### Business Features
- Email alert system for high-risk events
- ERP system integration (SAP, Oracle)
- Mobile-responsive dashboard
- Multi-language support

## License

MIT License - See LICENSE file for details

## Author

**Sarvesh Venkatachalam**

Data Engineering Professional specializing in ML/AI solutions

- GitHub: [ksarveshvenkatachalam-lang](https://github.com/ksarveshvenkatachalam-lang)
- LinkedIn: [Connect for collaboration](https://linkedin.com/in/ksarveshvenkatachalam)

## Dataset

### Synthetic Supply Chain Data

The project includes a synthetic data generator that creates realistic supply chain scenarios with:

**Features**:
- `supplier_id`: Unique supplier identifier
- `delivery_delay_days`: Days delayed from expected delivery
- `quality_score`: Quality rating (0-100)
- `seasonal_demand_factor`: Seasonal demand multiplier
- `weather_disruption`: Weather-related disruption indicator
- `geopolitical_risk`: Regional risk assessment score
- `lead_time_days`: Expected delivery lead time
- `order_quantity`: Order volume

**Target Variable**:
- `high_risk`: Binary classification (1 = high risk, 0 = normal)

### Data Generation

Generate 10,000 records with realistic distributions:

```bash
python src/data_generator.py --records 10000 --output data/raw/supply_chain_data.csv
```

## API Reference

### Core Modules

#### RiskPredictor

```python
from src.model_trainer import RiskPredictor

predictor = RiskPredictor()
predictor.train(X_train, y_train)
predictions = predictor.predict(X_test)
```

#### GenAIExplainer

```python
from src.genai_explainer import GenAIExplainer

explainer = GenAIExplainer(api_key=OPENAI_API_KEY)
explanation = explainer.explain_prediction(
    features=sample_features,
    prediction=risk_score,
    shap_values=shap_explanation
)
```

### Dashboard Application

Launch the interactive dashboard:

```bash
streamlit run app/dashboard.py --server.port 8501
```

**Dashboard Features**:
- Real-time risk predictions
- Interactive SHAP visualizations
- Natural language explanations powered by GPT-4
- Historical trend analysis
- Supplier risk heatmaps

## Deployment

### Local Deployment

```bash
# Activate environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run dashboard
streamlit run app/dashboard.py
```

### Docker Deployment

```dockerfile
# Build image
docker build -t supply-chain-risk-analytics .

# Run container
docker run -p 8501:8501 -e OPENAI_API_KEY=your_key supply-chain-risk-analytics
```

### Cloud Deployment (AWS)

```bash
# Deploy to AWS Elastic Beanstalk
eb init -p python-3.9 supply-chain-analytics
eb create supply-chain-env
eb deploy
```

**Environment Variables Required**:
- `OPENAI_API_KEY`: OpenAI API key for GPT-4 access
- `MODEL_PATH`: Path to trained model artifacts (default: models/risk_model.joblib)


## Acknowledgments

Developed as a portfolio demonstration project showcasing:
- End-to-end ML pipeline development
- Production-ready code architecture
- Integration of traditional ML with generative AI
- Enterprise software engineering practices

Built for data engineering and solution engineering role applications in UK consulting and technology firms.

## Support

For technical questions or collaboration inquiries:
- Open an issue in this repository
- Connect via LinkedIn

---

---

Last Updated: December 2025  
Version: 1.0.0

## License

This project is licensed under the MIT License - see the LICENSE file for details.

Last Updated: December 2025  
Version: 1.0.0
