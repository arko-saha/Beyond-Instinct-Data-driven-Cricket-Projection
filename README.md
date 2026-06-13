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
  │   ├── run_wicket_forecast.ipynb         # Main Overhauled Stochastic Forecast & Decision Analytics notebook
  │   ├── Ball_By_Ball_Run_Prediction.ipynb # Legacy ball-by-ball prediction notebook
  │   ├── Data_preprocessing.ipynb          # Legacy data preprocessing notebook
  │   ├── Exploratory_data_analysis.ipynb   # EDA and statistical analysis notebook
  │   ├── Web-Scraping.ipynb                # Legacy web scraping notebook
  │   └── data_extraction.ipynb             # Legacy data extraction notebook
  ├── src/
  │   ├── Ball_By_Ball/                     # Production-grade ball-by-ball prediction system
  │   │   ├── __init__.py                   # Package initialization
  │   │   ├── main.py                       # CLI application
  │   │   └── ...                           # (See package docs)
  │   ├── EDA/                              # EDA module package
  │   │   ├── cricket_processor.py          # Cricsheet data processing classes
  │   │   └── espn_scraper.py               # ESPN Cricinfo results scraper class
  │   ├── preprocessing/                    # Data preprocessing & feature engineering
  │   │   ├── __init__.py                   # Package exports
  │   │   ├── loader.py                     # CSV loader with column standardization
  │   │   └── ...                           # (See preprocessing docs)
  │   ├── scraper/                          # Modular scraper package
  │   │   ├── engine.py                     # Core scraping engine
  │   │   └── ...                           # (See scraper docs)
  │   └── forecast/                         # Stochastic Forecast & Decision Analytics Package (New)
  │       ├── __init__.py                   # Package initialization
  │       ├── config.py                     # Centralized project configuration
  │       ├── data_pipeline.py              # Chronological splits, states, and empirical baselines
  │       ├── skill_model.py                # Log-odds/logit skill interaction model
  │       ├── simulator.py                  # Parallel vectorised Monte Carlo simulation engine
  │       ├── backtester.py                 # Multi-match historical backtester
  │       ├── calibration.py                # Conformal calibration for target prediction intervals
  │       ├── optimizer.py                  # Tactical optimizers (Bowling lineups and batting orders)
  │       └── dashboard.py                  # High-quality visualization and charting suite
  ├── tests/                                # Forecasting engine test suite
  │   ├── test_phase1_pipeline.py           # Pipeline integrity tests
  │   ├── test_phase2_skill_model.py        # Logit/sigmoid skill math tests
  │   ├── test_phase2_realdata.py           # Real-data skill profile checks
  │   ├── test_phase3_simulator.py          # Monte Carlo simulator path tests
  │   ├── test_phase4_backtest.py           # Backtesting and coverage validation
  │   ├── test_phase5_optimizer.py          # Lineup optimization and constraints tests
  │   └── test_phase6_dashboard.py          # Visualization rendering tests
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

### 7. run_wicket_forecast.ipynb (Stochastic Forecast Engine)

**Purpose**: Transitioned from a deterministic expected-value calculator to a full stochastic Monte Carlo simulation engine and tactical optimizer (Phases 1–6).

**Core Phases**:
1. **Data Pipeline Integrity**: Implements chronological splits (strictly avoiding target/lookahead leakage), match-state feature engineering (runs, wickets, balls remaining, asking rates), and 4-level fallback empirical lookups.
2. **Log-Odds Skill Interaction Model**: Models match-up adjusted wicket probabilities via logit/sigmoid space, and run expectations via log-ratio multipliers, resolving the physical limits of linear additivity.
3. **Monte Carlo Simulation Engine**: Parallel vectorised simulation of 10,000 ball-by-ball paths, outputting full outcome distributions rather than point estimates.
4. **Historical Backtester & Calibration**: Backtests on hundreds of historical matches. Applies conformal calibration to map simulated intervals to an exact target coverage level (e.g. conformal factor alpha = 2.09 ensures 83.5% actual coverage against an 80% target).
5. **Tactical Decision Optimizer**:
   - *Bowling Optimizer*: Evaluates valid bowling assignments over the death overs using beam search to maximize win probability or minimize runs under standard T20 constraints.
   - *Batting Optimizer*: Examines batting order permutations to maximize expected runs.
   - *What-If Scenarios*: Implements immutable match-state overrides to compare alternative tactical pathways.
6. **Visualization Dashboard**: Renders publication-quality graphics:
   - Score fan charts (P10–P90 percentiles).
   - Rolling win probability timelines.
   - Strategy comparisons (expected runs and collapse probabilities).
   - Player career XP leaderboards.
   - Calibration curves and bias heatmaps.

**Outputs**:
- Serialized simulation lookups, calibrated models, and backtest logs under `models/`.
- Graphical dashboard plots saved as high-resolution PNGs (e.g., `models/phase6_backtest_dashboard.png`).

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

### Stochastic Forecast Engine Performance & Conformal Calibration

| Model / Strategy | Raw Coverage (80% Target) | Conformal Coverage (80% Target) | MAE (Runs) | Brier Score (Win/Loss) |
|------------------|---------------------------|----------------------------------|------------|-------------------------|
| **Baseline Model (No Skill)** | 48.9% | 80.2% | 24.1 | 0.165 |
| **Full Model (Skill Adjusted)**| 49.8% | **83.5%** | **23.7** | **0.147** |

**Key Findings & Achievements:**
- **Raw Coverage Deficit Identified**: Running pure Monte Carlo forecasts results in only ~49% coverage of actual scores within the P10–P90 band. This is due to real-world boundary dynamics, tail-ender batting collapses, and second-innings chasing team target biases.
- **Conformal Calibration Success**: By learning a conformal expansion factor ($\alpha = 2.09$) on hold-out calibration sets, the engine successfully scales the prediction intervals to guarantee an **83.5% coverage rate** on unseen test data, meeting the $\ge 80\%$ target.
- **Skill Model Edge**: Incorporating player-specific log-odds and log-ratio skill residuals improves the Brier score (0.147 vs 0.165) and reduces Mean Absolute Error (MAE) from 24.1 to 23.7 runs compared to a situation-only baseline.
- **Beam Search Optimization**: The Bowling Optimizer successfully generates and evaluates alternative death-over strategies in under 3 seconds using beam search, enforcing standard T20 constraints (no consecutive overs, bowler allocation limits).

### Insights

1. **Stochastic Over Deterministic**: Point predictions of cricket scores are highly unreliable; modeling the full probability distribution via Monte Carlo simulations is necessary to assess tactical risk (e.g. collapse probability).
2. **Conformal Methods are Critical**: Standard simulation variance consistently underestimates real-world outcome variance. Applying conformal calibration is essential before presenting P10–P90 bands to decision-makers.
3. **Production System Superiority**: The new Ball_By_Ball package achieves superior performance with proper validation and generalization.
4. **Ensemble Methods Excel**: Random Forest and XGBoost provide the best balance of accuracy and robustness.
5. **Polynomial Regression**: Shows exceptional performance for batter run prediction, suggesting strong non-linear relationships.
6. **Legacy Models**: Earlier implementations show promising results but lack production validation.

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
- [ ] Investigate the "batting performance metric" for broader application
- [ ] Develop new cricket-specific metrics
- [ ] Study the impact of different playing conditions
- [ ] Analyze powerplay vs death over strategies
- [ ] Player performance clustering and classification

## 👤 Author

**Arko Saha**
- GitHub: [@arko-saha](https://github.com/arko-saha)
- Repository: [Beyond-Instinct-Data-driven-Cricket-Projection](https://github.com/arko-saha/Beyond-Instinct-Data-driven-Cricket-Projection)

## 🙏 Acknowledgments

- **ESPN Cricinfo** for providing comprehensive cricket statistics
- **Cricsheet.org** for structured match data
- **Scikit-learn** and **TensorFlow** communities for excellent documentation

## 📧 Contact

For questions, suggestions, or collaborations, please open an issue on GitHub or contact the repository owner.

**Note**: This project is designed for research and educational purposes. Please respect the terms of service of data sources (ESPN Cricinfo, Cricsheet.org) when scraping data. Consider implementing rate limiting and appropriate delays between requests.
 
 ---

*Last Updated: June 2026*

