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

## ✨ Features

- **Web Scraping**: Automated data collection from ESPN Cricinfo for comprehensive cricket statistics
- **Data Preprocessing**: Robust data cleaning and transformation pipelines
- **Exploratory Data Analysis**: In-depth statistical analysis and visualization of cricket data
- **Predictive Modeling**: Multiple ML models for run and wicket prediction
- **Ball-by-Ball Analysis**: Real-time prediction capabilities for individual deliveries
- **Performance Metrics**: Custom metrics like "Jogi Score" and "Dismissal Factor" for advanced analytics

## 📁 Project Structure

```
Beyond-Instinct-Data-driven-Cricket-Projection/
│
├── Web-Scraping.ipynb                    # Web scraping from ESPN Cricinfo
├── Data_preprocessing.ipynb              # Data cleaning and preprocessing
├── Exploratory_data_analysis.ipynb       # EDA and statistical analysis
├── Predictive_Analysis.ipynb            # Player run prediction models
├── Ball_By_Ball_Run_Prediction.ipynb    # Ball-by-ball run prediction
├── run_wicket_forecast.ipynb             # Run and wicket forecasting
└── README.md                             # Project documentation
```

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- Jupyter Notebook or JupyterLab
- Google Colab (recommended for running notebooks)

### Required Libraries

```bash
pip install pandas numpy scikit-learn tensorflow keras matplotlib seaborn beautifulsoup4 requests openpyxl
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/arko-saha/Beyond-Instinct-Data-driven-Cricket-Projection.git
cd Beyond-Instinct-Data-driven-Cricket-Projection
```

2. Initialize git (if not already done):
```bash
git init
```

3. Open Jupyter Notebook or upload to Google Colab:
```bash
jupyter notebook
```

## 📖 Usage

### Running the Notebooks

1. **Web Scraping**: Start with `Web-Scraping.ipynb` to collect data from ESPN Cricinfo
2. **Data Preprocessing**: Run `Data_preprocessing.ipynb` to clean and prepare the data
3. **Exploratory Analysis**: Execute `Exploratory_data_analysis.ipynb` for insights
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

### 1. Web-Scraping.ipynb

**Purpose**: Automated data collection from ESPN Cricinfo

**Features**:
- Scrapes match-by-match batting statistics
- Scrapes match-by-match bowling statistics
- Collects overall batting and bowling statistics
- Position-wise statistics (Upper Order, Middle Order, Lower Order for batting; Opening, First Change, Second Change, Others for bowling)
- Handles pagination automatically
- Data cleaning and formatting
- Exports to Excel format

**Output Files**:
- `batters.xlsx` - Match-by-match batting data
- `bowlers.xlsx` - Match-by-match bowling data
- `Overall_batters.xlsx` - Overall batting statistics
- `Overall_bowlers.xlsx` - Overall bowling statistics
- `Upper_Order.xlsx`, `Middle_Order.xlsx`, `Lower_Order.xlsx` - Position-wise batting
- `Opening_Bowlers.xlsx`, `First_change_bowlers.xlsx`, `Second_change_bowlers.xlsx`, `Other_changes_bowlers.xlsx` - Position-wise bowling

### 2. Data_preprocessing.ipynb

**Purpose**: Data cleaning, transformation, and initial modeling

**Key Operations**:
- Separates ball numbers from over numbers
- Handles missing values and outliers
- Removes super overs (innings 3 and 4) to maintain data consistency
- Feature engineering (fall_of_wicket indicator)
- Polynomial regression implementation
- Data type optimization for memory efficiency

**Techniques**:
- Ball number extraction from decimal format
- Outlier detection and removal
- Data normalization
- Feature selection

### 3. Exploratory_data_analysis.ipynb

**Purpose**: Statistical analysis and visualization of cricket data

**Features**:
- Downloads T20 cricket data from Cricsheet.org
- Processes match metadata
- Team-specific analysis (e.g., Bangladesh performance)
- Strike rate analysis by bowling team
- Over-wise performance analysis (e.g., overs 17-20)
- Data visualization with seaborn and matplotlib
- Custom class-based data processing (`CricketDataProcessor`, `MatchData`)

**Analysis Includes**:
- Strike rate by bowling team
- Performance in death overs (17-20)
- Match result analysis
- Team performance trends

### 4. Predictive_Analysis.ipynb

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

### 5. Ball_By_Ball_Run_Prediction.ipynb

**Purpose**: Predict runs on a ball-by-ball basis during a match

**Models Implemented**:
1. **Decision Tree Regressor**
   - Grid search optimization
   - Perfect training accuracy (potential overfitting)
   - R² Score: 1.0

2. **Random Forest Regressor**
   - R² Score: 1.0
   - Excellent generalization

3. **Neural Network** (Deep Learning)
   - Architecture: 3 hidden layers (32, 64, 128 neurons)
   - Activation: ReLU
   - Output: Softmax (10 classes for runs 0-9)
   - Accuracy: 99.5%
   - Loss: Sparse Categorical Crossentropy

**Features**:
- Match ID
- Venue
- Innings
- Batting and bowling teams
- Striker and bowler
- Completed over and ball number
- Historical context (cumulative runs, fall of wicket)

**Data**:
- 418,509 ball-by-ball records
- Memory optimized with category data types
- Label encoded categorical features

### 6. run_wicket_forecast.ipynb

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
- **Python 3.7+**

### Data Processing
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing

### Machine Learning
- **Scikit-learn** - Traditional ML algorithms
  - Decision Tree Regressor
  - Random Forest Regressor
  - Polynomial Regression
  - Grid Search CV
- **TensorFlow/Keras** - Deep learning models

### Data Visualization
- **Matplotlib** - Plotting and visualization
- **Seaborn** - Statistical visualization

### Web Scraping
- **BeautifulSoup4** - HTML parsing
- **Requests** - HTTP library

### Data Sources
- **ESPN Cricinfo** - Player and match statistics
- **Cricsheet.org** - Ball-by-ball match data

## 📊 Data Sources

1. **ESPN Cricinfo** (`stats.espncricinfo.com`)
   - Match-by-match statistics
   - Overall player statistics
   - Position-wise statistics
   - Team performance data

2. **Cricsheet.org** (`cricsheet.org`)
   - T20 match data in CSV format
   - Ball-by-ball records
   - Match metadata

## 🎯 Key Results

### Model Performance Summary

| Model | Task | R² Score | RMSE | MAE |
|-------|------|----------|------|-----|
| Decision Tree | Batter Runs | 0.9896 | 0.1028 | 0.0505 |
| Random Forest | Batter Runs | 0.9985 | 0.0393 | 0.0070 |
| Polynomial Regression | Batter Runs | ~1.0 | 2.8e-05 | 1.6e-05 |
| Decision Tree | Ball-by-Ball | 1.0 | 0.0 | 0.0 |
| Random Forest | Ball-by-Ball | 1.0 | 0.0 | 0.0 |
| Neural Network | Ball-by-Ball | 99.5% | - | - |

### Insights

1. **Polynomial Regression** shows exceptional performance for batter run prediction, suggesting strong non-linear relationships in the data
2. **Random Forest** provides excellent balance between accuracy and interpretability
3. **Neural Networks** demonstrate high accuracy for ball-by-ball classification
4. **Ball-by-ball models** achieve perfect scores, indicating potential data leakage or overfitting - requires further investigation

## 🔬 Methodology

### Data Collection
1. Web scraping from ESPN Cricinfo for comprehensive statistics
2. Download from Cricsheet.org for structured match data
3. Data validation and quality checks

### Data Preprocessing
1. Missing value handling
2. Outlier detection and removal
3. Feature engineering (over/ball separation, wicket indicators)
4. Data type optimization
5. Label encoding and scaling

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

---

**Note**: This project is designed for research and educational purposes. Please respect the terms of service of data sources (ESPN Cricinfo, Cricsheet.org) when scraping data. Consider implementing rate limiting and appropriate delays between requests.

---

*Last Updated: 2024*

