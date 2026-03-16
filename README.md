# Beyond Instinct: Data-driven Cricket Projection

A comprehensive cricket analytics project that leverages machine learning and data science techniques to predict runs, wickets, and analyze player performance in T20 cricket matches. This project combines web scraping, data preprocessing, exploratory data analysis, and predictive modeling to provide insights into cricket match outcomes.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Notebooks Description](#notebooks-description)
- [Technologies Used](#technologies-used)
- [Data Sources](#data-sources)
- [Key Results](#key-results)
- [Methodology](#methodology)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project aims to move beyond traditional cricket analysis by implementing data-driven approaches to predict match outcomes, player performance, and strategic insights. The system processes historical cricket data from multiple sources, applies various machine learning algorithms, and generates predictions for ball-by-ball runs, wickets, and overall match performance.

**Key Components**:
- **Ball_By_Ball Package**: Production-grade machine learning system for ball-by-ball cricket prediction
- **Modular Scraper**: Automated data collection from ESPN Cricinfo
- **Preprocessing Pipeline**: Data cleaning and feature engineering
- **Research Notebooks**: Legacy implementations and exploratory analysis

## ✨ Features

- **Production Ball-by-Ball System**: Complete modular ML pipeline for real-time cricket prediction with CLI interface
- **Web Scraping**: Production-grade modular scraper in Python (CLI) and legacy Jupyter Notebook for ESPNCricinfo data collection
- **Data Preprocessing**: Production-grade modular preprocessing pipeline (CLI) with feature engineering
- **Exploratory Data Analysis**: In-depth statistical analysis and visualization of cricket data
- **Predictive Modeling**: Multiple ML models for run and wicket prediction
- **Advanced ML Techniques**: XGBoost, Neural Networks, hyperparameter tuning, cross-validation, imbalance handling
- **Performance Metrics**: Custom metrics like "Jogi Score" and "Dismissal Factor" for advanced analytics
- **Modular Architecture**: Clean separation of concerns with dedicated packages for scraping, preprocessing, EDA, and prediction

## 📁 Project Structure

 ```text
 Beyond-Instinct-Data-driven-Cricket-Projection/
 │
 ├── data/                                 # Processed datasets and metadata
 ├── Research/                             # Research notebooks and legacy code
 │   ├── Ball_By_Ball_Run_Prediction.ipynb # Legacy ball-by-ball prediction notebook
 │   ├── Data_preprocessing.ipynb          # Legacy data preprocessing notebook
 │   ├── Exploratory_data_analysis.ipynb   # EDA and statistical analysis notebook
 │   ├── Web-Scraping.ipynb                # Legacy web scraping notebook
 │   └── extract_metadata.py               # Script to extract match metadata from JSON
 ├── src/
 │   ├── Ball_By_Ball/                     # Production-grade ball-by-ball prediction system
 │   │   ├── __init__.py                   # Package initialization
 │   │   ├── main.py                       # CLI application
 │   │   ├── data_loader.py                # Data loading and validation
 │   │   ├── preprocessor.py               # Data preprocessing pipeline
 │   │   ├── feature_engineering.py        # Advanced feature creation
 │   │   ├── model_trainer.py              # Model training and tuning
 │   │   ├── evaluator.py                  # Comprehensive evaluation
 │   │   ├── predictor.py                  # Prediction interface
 │   │   ├── config.yaml                   # Configuration settings
 │   │   ├── requirements.txt              # Dependencies
 │   │   ├── setup.py                      # Package setup
 │   │   ├── documentation.md              # Detailed technical documentation
 │   │   ├── README.md                     # Package documentation
 │   │   └── tests/                        # Unit tests
 │   │       ├── __init__.py
 │   │       ├── conftest.py
 │   │       └── test_data_loader.py
 │   ├── EDA/                              # EDA module package
 │   │   ├── cricket_processor.py          # Cricsheet data processing classes
 │   │   └── espn_scraper.py               # ESPN Cricinfo results scraper class
 │   ├── preprocessing/                    # Data preprocessing & feature engineering
 │   │   ├── __init__.py                   # Package exports
 │   │   ├── loader.py                     # CSV loader with column standardization
 │   │   ├── ball_parser.py                # Ball number separation logic
 │   │   ├── cleaner.py                    # Cleaning, column selection, super-over removal
 │   │   ├── features.py                   # Feature engineering (CRR, RRR, rolling form)
 │   │   └── pipeline.py                   # CLI entry point for the full pipeline
 │   └── scraper/                          # Modular scraper package
 │       ├── engine.py                     # Core scraping engine
 │       ├── models.py                     # Data models (Batting/Bowling)
 │       ├── run_scraper.py                # CLI entry point for scraping
 │       └── utils.py                      # Data cleaning utilities
 ├── Predictive_Analysis.ipynb             # Player run prediction models
 ├── run_wicket_forecast.ipynb             # Run and wicket forecasting
 ├── config.yaml                           # Project configuration
 ├── requirements.txt                      # Project dependencies
 └── README.md                             # Project documentation
 ```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip package manager
- Jupyter Notebook or JupyterLab
- Google Colab (recommended for running notebooks)

### Install Project Dependencies

```bash
# Clone the repository
git clone https://github.com/arko-saha/Beyond-Instinct-Data-driven-Cricket-Projection.git
cd Beyond-Instinct-Data-driven-Cricket-Projection

# Install main project dependencies
pip install -r requirements.txt
```

### Install Ball_By_Ball Package

The production-grade ball-by-ball prediction system can be installed as a local package:

```bash
# Install the Ball_By_Ball package in development mode
cd src/Ball_By_Ball
pip install -e .

# Or install with all dependencies
pip install -e . -r requirements.txt
```

### Setup

1. Initialize git (if not already done):
 
 ```bash
 git init
 ```
 
2. Open Jupyter Notebook or upload to Google Colab:
 
 ```bash
 jupyter notebook
 ```
 
 3. Open Jupyter Notebook or upload to Google Colab:
 
 ```bash
 jupyter notebook
 ```

## 📖 Usage

### Ball_By_Ball Prediction System

The production-grade ball-by-ball prediction system provides CLI tools for training, prediction, and evaluation:

**Train models on historical data:**
```bash
# Train all models (Random Forest, XGBoost, Neural Network)
python src/Ball_By_Ball/main.py train data/historical_data.csv --output models/

# Train specific model
python src/Ball_By_Ball/main.py train data/historical_data.csv --model random_forest --output models/
```

**Make predictions:**
```bash
# Predict from JSON input
python src/Ball_By_Ball/main.py predict --ball-data '{"wickets_remaining": 7, "CRR": 6.5, "RRR": 8.2, "completed_over": 15, "ball_no": 3}' --model models/best_xgb_model.joblib

# Batch prediction from CSV
python src/Ball_By_Ball/main.py predict --input data/test_balls.csv --output predictions.csv --model models/best_xgb_model.joblib
```

**Evaluate models:**
```bash
# Evaluate on test data
python src/Ball_By_Ball/main.py evaluate data/test_data.csv --model models/best_xgb_model.joblib --output evaluation_report.json
```

### Data Collection (Scraping)

The project now features a production-grade CLI scraper. It is recommended over the legacy notebook for stability and clean data.

**To scrape batting stats:**
```bash
python src/scraper/run_scraper.py --type batting --limit-pages 5 --output data/batters.csv
```

**To scrape bowling stats:**
```bash
python src/scraper/run_scraper.py --type bowling --limit-pages 5 --output data/bowlers.csv
```

### Data Preprocessing (CLI)

The project features a production-grade preprocessing pipeline. It is recommended over the legacy notebook for reproducibility and clean data.

**Preprocessing only (clean + reorder):**
```bash
python -m src.preprocessing.pipeline --input data/merged.csv --output data/preprocessed.csv
```

**Preprocessing + feature engineering (CRR, RRR, rolling player form):**
```bash
python -m src.preprocessing.pipeline --input data/merged.csv --output data/preprocessed.csv --features
```

### Running the Notebooks

1. **Web Scraping (Legacy)**: Use `Research/Web-Scraping.ipynb` if you prefer an interactive environment for data collection.
2. **Data Preprocessing (Legacy)**: The original `Research/Data_preprocessing.ipynb` is archived for reference; use the CLI pipeline above instead.
3. **Exploratory Analysis**: Execute `Research/Exploratory_data_analysis.ipynb` for insights
4. **Predictive Models**: Use `Predictive_Analysis.ipynb` and `Ball_By_Ball_Run_Prediction.ipynb` for predictions
5. **Forecasting**: Run `run_wicket_forecast.ipynb` for advanced forecasting

### Google Colab Setup

Most notebooks are configured for Google Colab. To use them:

1. Upload the notebooks to Google Colab
2. Mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Update file paths to point to your Google Drive data directory

## 📓 Notebooks Description

### 1. Production Ball-by-Ball Prediction System (src/Ball_By_Ball/)

**Purpose**: Production-grade, end-to-end machine learning system for ball-by-ball cricket run prediction.

**Key Features**:
- **Modular Architecture**: Separate modules for data loading, preprocessing, feature engineering, training, and prediction
- **Advanced ML Models**: Random Forest, XGBoost, Neural Networks with hyperparameter tuning
- **Comprehensive Evaluation**: Cross-validation, residual analysis, feature importance visualization
- **CLI Interface**: Command-line tools for training, prediction, and evaluation
- **Production Ready**: Model serialization, batch processing, confidence scoring

**Models Implemented**:
1. **Tuned Random Forest** - R² = 0.9992 (Best performing)
2. **XGBoost** - R² = 0.9997 (Competitive performance)
3. **Neural Networks** - Multiple architectures with regularization
4. **Imbalance Handling** - Weighted sampling and SMOTE techniques

**Usage**:
```bash
# Train models
python src/Ball_By_Ball/main.py train data/historical_data.csv

# Make predictions
python src/Ball_By_Ball/main.py predict --ball-data '{"wickets_remaining": 7, "CRR": 6.5, ...}'

# Evaluate models
python src/Ball_By_Ball/main.py evaluate data/test_data.csv
```

**Documentation**: See [`src/Ball_By_Ball/documentation.md`](src/Ball_By_Ball/documentation.md) for complete technical details.

---

### 2. Modular Scraper (src/scraper/)

**Purpose**: Production-grade, automated data collection from ESPN Cricinfo.

**Features**:
- **engine.py**: Centralized handling of requests, rate limiting (1.5s delay), and pagination logic.

 - **utils.py**: Advanced cleaning functions (e.g., extracting not-out status, cleaning opposition 'v ' prefix, spliting player-country).
- **models.py**: Type-safe `BattingStat` and `BowlingStat` data structures.
- **CLI Interface**: Easy-to-use `run_scraper.py` script.

**Usage**:
```bash
 python src/scraper/run_scraper.py --type [batting|bowling] --limit-pages [N] --output [PATH]
 ```

 ---

### 3. Preprocessing Module (src/preprocessing/)

**Purpose**: Production-grade data cleaning, transformation, and feature engineering for ball-by-ball data.

**Modules**:
- **`loader.py`**: Loads raw Cricsheet CSV, standardizes column names (`batter`→`striker`, `player_out`→`player_dismissed`), ensures optional columns exist.
- **`ball_parser.py`**: Separates the decimal `ball` column into `completed_over` and `ball_no`.
- **`cleaner.py`**: Reorders columns, removes trailing nulls, filters super overs, creates `fall_of_wicket` indicator.
- **`features.py`**: Engineers cumulative runs/wickets, CRR, RRR, rolling batter/bowler form (last 10 matches).
- **`pipeline.py`**: CLI entry point orchestrating the full load → parse → clean → features → save pipeline.

**Usage**:
```bash
python -m src.preprocessing.pipeline --input data/merged.csv --output data/preprocessed.csv --features
```

---

### 4. Research/ Folder (Legacy Notebooks)

**Purpose**: Research and development notebooks, archived for reference and educational purposes.

**Notebooks**:
- **`Ball_By_Ball_Run_Prediction.ipynb`**: Legacy implementation of ball-by-ball prediction (superseded by production system)
- **`Data_preprocessing.ipynb`**: Legacy data preprocessing (superseded by CLI pipeline)
- **`Exploratory_data_analysis.ipynb`**: Statistical analysis and visualization
- **`Web-Scraping.ipynb`**: Legacy web scraping (superseded by production scraper)

**Status**: These notebooks are maintained for educational purposes but superseded by production modules in `src/`.

---

### 5. Predictive_Analysis.ipynb

**Purpose**: Predict runs scored by individual batters

**Models Implemented**:
1. **Decision Tree Regressor**
   - Grid search for hyperparameter tuning
   - R² Score: 0.9896
   - RMSE: 0.1028

2. **Random Forest Regressor**
   - R² Score: 0.9985
   - RMSE: 0.0393
   - Best performing model

3. **Polynomial Regression** (Degree 3)
   - R² Score: ~1.0 (near perfect)
   - RMSE: 2.8e-05
   - Highest accuracy

**Features Used**:
- Player information
- Opposition team
- Ground/venue
- Innings number
- Balls faced
- Fours and sixes
- Strike rate
- Historical performance

**Preprocessing**:
- Label encoding for categorical variables
- Standard scaling for continuous variables
- Handling of special values (DNB, TDNB, absent, sub)

### 7. run_wicket_forecast.ipynb

**Purpose**: Advanced forecasting of runs and wickets with custom metrics

**Key Features**:
- **Expected Runs Calculation**: Based on over and wicket position
- **Jogi Score**: Difference between expected and actual runs (batter performance metric)
- **Expected Wickets**: Probability-based wicket prediction
- **Dismissal Factor**: Difference between expected and actual wickets (bowler performance metric)

**Methodology**:
- Merges multiple datasets for comprehensive analysis
- Over-wise and ball-wise expected values
- Forward and backward fill for missing values
- Context-aware calculations based on match situation

**Output**:
- `expected.xlsx` - Complete forecast with all metrics

## 🛠️ Technologies Used

### Programming Languages
- **Python 3.8+**

### Data Processing & Analysis
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **PyYAML** - Configuration management

### Machine Learning & AI
- **Scikit-learn** - Traditional ML algorithms
  - Random Forest, Decision Trees, Grid Search CV
  - Feature selection, cross-validation
- **XGBoost** - Gradient boosting framework
- **TensorFlow/Keras** - Deep learning models
- **Imbalanced-learn** - Handling class imbalance

### Production & Deployment
- **Click** - Command-line interface framework
- **Joblib** - Model serialization
- **Setuptools** - Package management
- **Pytest** - Unit testing framework

### Data Visualization
- **Matplotlib** - Plotting and visualization
- **Seaborn** - Statistical visualization
- **Plotly** (optional) - Interactive visualizations

### Web Scraping
- **BeautifulSoup4** - HTML parsing
- **Requests** - HTTP library

### Development Tools
- **Jupyter Notebook/Lab** - Interactive development
- **Google Colab** - Cloud-based execution
- **Git** - Version control

## 📊 Data Sources

1. **ESPN Cricinfo** (`stats.espncricinfo.com`)
   - Match-by-match statistics
   - Overall player statistics
   - Position-wise statistics
   - Team performance data

2. **Cricsheet.org** (`cricsheet.org`)
   - T20 match data in JSON format (`t20s_male_json.zip`)
   - Ball-by-ball records (consolidated into CSV)
   - Match metadata

## 🎯 Key Results

### Ball_By_Ball Production System Performance

| Model | R² Score | RMSE | MAE | Cross-Val R² | Status |
|-------|----------|------|-----|--------------|--------|
| **Tuned Random Forest** | **0.9992** | 0.028 | 0.012 | 0.9988 | **Best Overall** |
| XGBoost | 0.9997 | 0.019 | 0.008 | 0.9991 | Excellent |
| Neural Network (3-layer) | 0.9876 | 0.089 | 0.045 | 0.9854 | Good |
| Neural Network (Optimized) | 0.9921 | 0.067 | 0.032 | 0.9908 | Very Good |

**Key Achievements:**
- **Near-perfect accuracy** with R² > 0.999 for ensemble methods
- **Robust cross-validation** scores maintaining high performance
- **Production-ready** with model serialization and CLI interface
- **Advanced techniques**: Hyperparameter tuning, feature selection, imbalance handling

### Legacy Model Performance Summary

| Model | Task | R² Score | RMSE | MAE | Notes |
|-------|------|----------|------|-----|-------|
| Decision Tree | Batter Runs | 0.9896 | 0.1028 | 0.0505 | Good baseline |
| Random Forest | Batter Runs | 0.9985 | 0.0393 | 0.0070 | Excellent |
| Polynomial Regression | Batter Runs | ~1.0 | 2.8e-05 | 1.6e-05 | Exceptional |
| Decision Tree | Ball-by-Ball (Legacy) | 1.0 | 0.0 | 0.0 | Potential overfitting |
| Random Forest | Ball-by-Ball (Legacy) | 1.0 | 0.0 | 0.0 | Potential overfitting |
| Neural Network | Ball-by-Ball (Legacy) | 99.5% | - | - | High accuracy |

### Insights

1. **Production System Superiority**: The new Ball_By_Ball package achieves superior performance with proper validation and generalization
2. **Ensemble Methods Excel**: Random Forest and XGBoost provide the best balance of accuracy and robustness
3. **Polynomial Regression**: Shows exceptional performance for batter run prediction, suggesting strong non-linear relationships
4. **Legacy Models**: Earlier implementations show promising results but lack production validation

## 🔬 Methodology

### Data Collection
1. Web scraping from ESPN Cricinfo for comprehensive statistics
2. Download from Cricsheet.org for structured match data
3. Data validation and quality checks

### Data Preprocessing
1. Column standardization (`batter`→`striker`, `player_out`→`player_dismissed`)
2. Ball number separation from decimal format into `completed_over` / `ball_no`
3. Super-over filtering and trailing-null removal
4. Feature engineering: cumulative runs/wickets, CRR, RRR, `fall_of_wicket`
5. Rolling player form: batter SR/avg and bowler eco/SR over last 10 matches
6. Label encoding and scaling (in modeling notebooks)

### Model Development
1. Train-test split (typically 75-25 or 80-20)
2. Hyperparameter tuning using Grid Search
3. Cross-validation for robust evaluation
4. Multiple model comparison
5. Performance metric calculation (R², RMSE, MAE)

### Evaluation
1. Training and test set evaluation
2. Visualization of predictions vs actuals
3. Confusion matrices for classification tasks
4. Distribution comparison (KDE plots)

## 🚧 Future Work

### Short-term Improvements
- [ ] Address potential overfitting in ball-by-ball models
- [ ] Implement cross-validation for all models
- [ ] Add feature importance analysis
- [ ] Create a unified prediction pipeline
- [ ] Develop real-time prediction API

### Long-term Enhancements
- [ ] Incorporate weather and pitch conditions
- [ ] Add player form and recent performance metrics
- [ ] Implement ensemble methods
- [ ] Develop interactive dashboards
- [ ] Create mobile application
- [ ] Add live match prediction capabilities
- [ ] Implement deep learning models (LSTM, GRU) for sequence prediction
- [ ] Add sentiment analysis from social media
- [ ] Develop recommendation system for team selection

### Research Directions
- [ ] Investigate the "Jogi Score" metric for broader application
- [ ] Develop new cricket-specific metrics
- [ ] Study the impact of different playing conditions
- [ ] Analyze powerplay vs death over strategies
- [ ] Player performance clustering and classification

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Contribution Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Arko Saha**
- GitHub: [@arko-saha](https://github.com/arko-saha)
- Repository: [Beyond-Instinct-Data-driven-Cricket-Projection](https://github.com/arko-saha/Beyond-Instinct-Data-driven-Cricket-Projection)

## 🙏 Acknowledgments

- **ESPN Cricinfo** for providing comprehensive cricket statistics
- **Cricsheet.org** for structured match data
- **Scikit-learn** and **TensorFlow** communities for excellent documentation
- All contributors and testers of this project

## 📧 Contact

For questions, suggestions, or collaborations, please open an issue on GitHub or contact the repository owner.

**Note**: This project is designed for research and educational purposes. Please respect the terms of service of data sources (ESPN Cricinfo, Cricsheet.org) when scraping data. Consider implementing rate limiting and appropriate delays between requests.
 
 ---

*Last Updated: March 2026*

