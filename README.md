# Wine Quality Analysis & Prediction

A comprehensive machine learning project that analyzes wine quality using Classification and Regression Trees (CARTs) and Random Forests to predict quality classifications based on physicochemical properties of Portuguese Vinho Verde wines.

## Overview

This project performs exploratory data analysis (EDA) and builds predictive models to classify wine quality as "high" (≥6) or "low" (<6) using various chemical properties. The analysis covers both red and white wines from the UCI Wine Quality dataset (Cortez et al., 2009), employing decision trees and random forest algorithms to identify key quality indicators. The Random Forest classifier achieved an impressive **82.8% accuracy**, significantly outperforming traditional single decision trees.

## Table of Contents

- [Features](#features)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [Visualizations](#visualizations)
- [Key Findings](#key-findings)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [References](#references)

## Features

- **Comprehensive EDA**: Distribution analysis, correlation studies, and missing value detection
- **Binary Classification**: Predicts wine quality as high (≥6) or low (<6)
- **Regression Analysis**: Predicts exact quality scores (0-10 scale)
- **Multiple Models**: Decision trees with CP-based pruning and random forest ensembles (10,000 trees)
- **Rich Visualizations**: Correlation heatmaps, density plots, scatterplot matrices, MDS plots, and variable importance analysis
- **Model Evaluation**: Comprehensive metrics including accuracy, precision, recall, F1-score, specificity, balanced accuracy, and RMSE
- **Class Imbalance Handling**: Addressed through ensemble methods and evaluated using balanced accuracy metrics

## Dataset

The project uses the Wine Quality dataset from the UCI Machine Learning Repository, containing physicochemical properties of Portuguese Vinho Verde wines.

**Dataset Statistics:**
- Total samples: 6,497 wines (after merging red and white)
- Red wines: 1,599 samples
- White wines: 4,898 samples
- Training set: 4,497 samples
- Test set: 2,000 samples

**Features (11 input variables):**
- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol content

**Target Variable:**
- Quality (score between 3-9 in practice, 0-10 theoretical scale)
- Binary classification: "low" (<6) vs "high" (≥6)

**Wine Type:**
- Red wine (encoded as 0)
- White wine (encoded as 1)

**Data Quality:**
- No missing values detected
- Some duplicate entries present
- Class imbalance: Higher proportion of "high" quality wines
- White wines show larger proportion of high-quality ratings than red wines

## Requirements

```r
R (>= 4.0.0)
ggplot2
gridExtra
naniar
dplyr
tibble
GGally
rpart
ggparty
caret
randomForest
corrplot
readxl
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/kodiakthebear/wine-quality-analysis.git
cd wine-quality-analysis
```

2. Install required R packages:
```r
install.packages(c("ggplot2", "gridExtra", "naniar", "dplyr", 
                   "tibble", "GGally", "rpart", "ggparty", 
                   "caret", "randomForest", "corrplot", "readxl"))
```

3. Download the Wine Quality dataset from [UCI Machine Learning Repository](https://doi.org/10.24432/C56S3T) and update the file path in the script:
```r
rohil <- read.csv("path/to/your/BA2.csv")
```

## Usage

Run the complete analysis:

```r
source("winemodel_r.R")
```

Or execute specific sections:

```r
# Load data and perform EDA
wine$quality_label <- ifelse(wine$quality>=6, "high","low")
wine$quality_label <- as.factor(wine$quality_label)

# Build decision tree model with optimal parameters
winetree <- rpart(quality_label ~ . - quality - wine_type, 
                  data = wine, 
                  control = rpart.control(minsplit = 10, minbucket = 3, 
                                         cp = 0.01, maxdepth = 10), 
                  method = "class")

# Prune tree using optimal CP
opt = which.min(winetree$cptable[,"xerror"])
cp = winetree$cptable[opt, "CP"]
pruned_winetree = prune(winetree, cp)

# Build random forest classifier
set.seed(42)
ind = sample(1:nrow(wine), size=2000)
test_wine = wine[ind,]
train_wine = wine[-ind,]

winerf <- randomForest(factor(quality_label) ~ .-quality, 
                       data = train_wine, 
                       ntree=10000, 
                       mtry=3, 
                       proximity=TRUE)
```

## Models

### 1. Decision Tree Classifier (CART)

**Configuration:**
- Minimum split: 10
- Minimum bucket: 3
- Complexity parameter (CP): 0.01
- Maximum depth: 10
- Pruning: CP-based using cross-validation error minimization

**Model Selection:**
- Used `printcp()` to identify optimal CP value
- Selected model with minimum cross-validation error (xerror = 0.72148)
- Final model: 2 splits, 3 leaf nodes

**Decision Structure:**
1. **First split**: Alcohol content (primary predictor)
2. **Second split**: Volatile acidity (for lower alcohol wines)

**Key Insights:**
- High alcohol content strongly indicates high quality
- For lower alcohol wines, low volatile acidity indicates higher quality
- Aligns with wine science literature on balanced acidity and aroma preservation

### 2. Random Forest Classifier

**Configuration:**
- Number of trees: 10,000 (computationally expensive but optimal)
- Variables per split (mtry): 3 (optimized via 5-fold cross-validation)
- Train/test split: 4,497 / 2,000 samples
- Proximity calculation: Enabled
- Bootstrap sampling: Yes (bagging applied)

**Optimization Process:**
- Performed 5-fold cross-validation to determine optimal mtry
- Tested class weights and undersampling to address class imbalance
- Error rate plateaus as ntree approaches 10,000

**Features:**
- Variable importance analysis via Mean Decrease in Gini
- MDS (Multidimensional Scaling) visualization for cluster analysis
- Out-of-bag (OOB) error estimation: 17.75%

### 3. Random Forest Regressor (Experimental)

**Configuration:**
- Number of trees: 1,000
- Predicts exact quality scores (1-10 scale)
- Evaluated using RMSE and prediction accuracy

**Performance:**
- Accuracy: 77.8%
- RMSE: Available in model output
- Result: Underperformed compared to classification approach; rejected as primary model

## Results

### Decision Tree Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 73.5% |
| **Kappa** | 0.42 |
| **Balanced Accuracy** | 70.9% |
| **Precision** | 78.15% |
| **Recall (Sensitivity)** | 80.77% |
| **Specificity** | 61.03% |
| **F1-Score** | 79.44% |

**Interpretation:** The decision tree achieves reasonable accuracy and performs well at identifying high-quality wines. However, lower specificity (61.03%) indicates difficulty correctly classifying low-quality wines, likely due to class imbalance.

### Random Forest Performance (Classification) ⭐ Best Model

| Metric | Value |
|--------|-------|
| **Accuracy** | 82.8% |
| **Kappa** | 0.63 |
| **Balanced Accuracy** | 80.71% |
| **Precision** | 84.10% |
| **Recall (Sensitivity)** | 89.24% |
| **Specificity** | 72.19% |
| **OOB Error Rate** | 17.75% |

**Interpretation:** The Random Forest significantly outperforms the single decision tree across all metrics. The high recall (89.24%) indicates excellent detection of high-quality wines, while improved specificity (72.19%) shows better handling of low-quality wines. The balanced accuracy of 80.71% demonstrates superior performance despite class imbalance.

### Random Forest Performance (Regression)

| Metric | Value |
|--------|-------|
| **Accuracy** | 77.8% |
| **RMSE** | See model output |

**Interpretation:** The regression approach underperformed compared to the classification model, confirming that binary classification is more suitable for this dataset's ordinal quality ratings.

### Model Comparison Summary

The Random Forest classifier emerged as the superior model:

- **+9.3 percentage points** improvement in accuracy over decision tree
- **+11.16 percentage points** improvement in specificity
- **+8.47 percentage points** improvement in recall
- **+0.21** increase in Kappa statistic

The ensemble approach successfully mitigated overfitting, captured complex non-linear relationships, and handled class imbalance more effectively than the single decision tree.

## Visualizations

The project generates multiple comprehensive visualizations:

### 1. Distribution Analysis
- Histograms for all 11 physicochemical properties
- Density plots comparing red and white wine quality distributions
- Boxplots showing quality rating distributions and class imbalance

### 2. Correlation Analysis
- **Correlation heatmap**: Reveals multicollinearity among features
  - Strong correlations observed between density, alcohol, and residual sugar
  - Fixed acidity correlates with citric acid and pH
- **Scatterplot matrices**: Generated for red wine, white wine, and combined datasets
  - Shows pairwise relationships between all variables
  - Early indication of alcohol, density, and volatile acidity as key predictors

### 3. Quality Relationships
- Alcohol content vs. quality (color-coded by wine type)
- Volatile acidity vs. quality (color-coded by wine type)
- Density vs. alcohol content (classified by quality label)
- Volatile acidity vs. alcohol content (classified by quality label)
- Density vs. volatile acidity (classified by quality label)

### 4. Model Outputs

**Decision Tree:**
- Tree structure diagrams showing 2 splits and 3 leaf nodes
- CP table for pruning optimization
- Decision path visualization using `ggparty`

**Random Forest:**
- **Error rate plot**: Shows convergence as ntree increases to 10,000
- **Variable importance plot**: Ranks features by Mean Decrease in Gini
  - Top 3: Alcohol, Volatile Acidity, Density
- **MDS plot**: 2D visualization of high-dimensional classification patterns
- **Prediction accuracy visualization**: Actual vs. predicted quality scores (for regression model)

## Key Findings

### Primary Drivers of Wine Quality

1. **Alcohol Content** (Most Important)
   - Higher alcohol content strongly correlates with higher quality ratings
   - Primary split variable in decision tree
   - Top ranked variable in Random Forest importance analysis

2. **Volatile Acidity** (Second Most Important)
   - Lower volatile acidity indicates higher quality
   - Affects wine's aroma, mouthfeel, and taste
   - Critical for wines with lower alcohol content
   - Aligns with wine science literature on balanced acidity (Vilela-Moura et al., 2010)

3. **Density** (Third Most Important)
   - Identified by Random Forest but not single decision tree
   - Lower density wines tend to receive higher ratings
   - Correlated with alcohol content and residual sugar

### Dataset Characteristics

- **White wine dominance**: 4,898 white wines vs. 1,599 red wines (75% white)
- **Class imbalance**: Higher proportion of "high" quality wines than "low" quality
- **Quality distribution**: White wines show larger proportion of high-quality ratings than red wines
- **Multicollinearity**: Strong correlations exist between multiple features, particularly density-alcohol-residual sugar

### Model Insights

- **Random Forests outperform single trees**: 82.8% vs. 73.5% accuracy
- **Ensemble methods handle imbalance better**: Balanced accuracy improved by ~10 percentage points
- **Variable discovery**: Random Forests identified density as important, missed by decision tree
- **Generalization**: OOB error rate of 17.75% indicates good generalization to unseen data
- **Computational trade-off**: 10,000 trees provides optimal performance but is computationally expensive

### Wine Science Validation

The model's findings align with established wine science:
- Volatile acids significantly impact aroma and taste, key factors in subjective wine classification (Vilela-Moura et al., 2010)
- Balanced acidity while preserving characteristic bouquet is essential (Crespo et al., 2023)
- Alcohol content influences mouthfeel and overall wine perception (Gambetta et al., 2016)

## Limitations

### Dataset Constraints
1. **Geographic limitation**: Dataset restricted to Portuguese Vinho Verde wines, potentially limiting generalizability to other wine regions and varieties
2. **Class imbalance**: Disproportionate representation of high-quality wines affects model performance on minority class
3. **Wine type imbalance**: White wines dominate dataset (75%), potentially biasing results

### Model Limitations
1. **Binary classification**: Grouping wines into "high" vs. "low" removes granularity and nuanced quality distinctions (e.g., difference between quality 6 and 9)
2. **Multicollinearity**: Strong correlations between features (alcohol-density, fixed acidity-pH) may influence model stability
3. **Interpretability**: Random Forest, while highly accurate, lacks the interpretability of decision trees
4. **Computational expense**: 10,000-tree Random Forest requires significant computational resources
5. **Quality variable exclusion**: Had to exclude numeric quality from models to prevent overfitting, limiting direct score prediction

### Evaluation Constraints
1. **Specificity**: Both models show lower specificity than sensitivity, indicating difficulty identifying low-quality wines
2. **Class weight adjustments ineffective**: Attempts to address class imbalance via class weights and undersampling did not improve performance
3. **Regression underperformance**: Direct quality score prediction achieved only 77.8% accuracy

## Future Work

### Dataset Enhancement
- **Expand dataset**: Include diverse wine types, regions, and vintages to improve generalizability
- **Collect more low-quality samples**: Address class imbalance by gathering additional minority class data
- **Add temporal data**: Include vintage year and aging information
- **Incorporate expert ratings**: Cross-validate physicochemical predictions with professional sommelier assessments

### Feature Engineering
- **Principal Component Analysis (PCA)**: Reduce redundancy among correlated variables
- **Feature interactions**: Engineer interaction terms (e.g., alcohol × volatile acidity)
- **Polynomial features**: Capture non-linear relationships more explicitly
- **Domain-specific ratios**: Create ratios based on wine science (e.g., acid balance indices)

### Advanced Modeling
- **Neural Networks**: Explore deep learning approaches for capturing complex patterns
- **Gradient Boosting**: Test XGBoost, LightGBM, or CatBoost for improved performance
- **Regularized regression**: Apply Ridge, Lasso, or Elastic-Net regression for feature selection
- **Support Vector Machines (SVM)**: Evaluate kernel-based classification
- **SMOTE**: Apply Synthetic Minority Over-sampling Technique to address class imbalance
- **Stacking ensembles**: Combine multiple model types for improved predictions

### Multi-class Classification
- **Granular quality levels**: Implement 3-5 class system instead of binary (e.g., low/medium/high or excellent/good/average/poor/bad)
- **Ordinal regression**: Preserve ordering information in quality ratings
- **Threshold optimization**: Fine-tune classification boundaries for each quality level

### Interpretability and Deployment
- **SHAP values**: Implement Shapley Additive Explanations for model interpretation
- **LIME**: Use Local Interpretable Model-agnostic Explanations
- **Feature importance analysis**: Conduct deeper investigation of variable interactions
- **Production deployment**: Create API endpoint for real-time quality prediction
- **Web interface**: Build user-friendly application for winemakers and enthusiasts

## Project Structure

```
wine-quality-analysis/
│
├── winemodel_r.R          # Main analysis script
├── BA2.csv                # Combined wine dataset (not included)
├── README.md              # This file
├── MRT_Wine_Qual_report.pdf  # Detailed technical report
└── outputs/               # Generated plots and results
    ├── distributions/     # Histograms and density plots
    ├── correlations/      # Heatmaps and scatterplot matrices
    ├── models/            # Decision tree and RF visualizations
    └── performance/       # Confusion matrices and metrics
```

## Contributing

Contributions are welcome! This project would benefit from:
- Implementation of additional algorithms (neural networks, gradient boosting)
- Class imbalance solutions (SMOTE, advanced resampling)
- Feature engineering experiments
- Multi-class classification approaches
- Expanded datasets from different wine regions

To contribute:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## References

1. Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Wine Quality [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56S3T

2. Cortez, P., Cerdeira, A.L., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems*, 47, 547-553.

3. Breiman, L. (2001). Random forests. *Machine Learning*, 45, 5-32.

4. Hastie, T., Tibshirani, R., Friedman, J. H., & Friedman, J. H. (2009). *The elements of statistical learning: data mining, inference, and prediction* (Vol. 2, pp. 1-758). New York: Springer.

5. Vilela-Moura, A., Côrte-Real, M., Sousa, M.J., et al. (2010). The impact of acetate metabolism on yeast fermentative performance and wine quality: reduction of volatile acidity of grape musts and wines. *Applied Microbiology and Biotechnology*, 89(2), 271–280.

6. Gambetta, J.M., Jeffery, D.W., Schmidtke, L.M., et al. (2016). Relating Expert Quality Ratings of Australian Chardonnay Wines to Volatile Composition and Production Method. *American Journal of Enology and Viticulture*, 68(1), 39–48.

7. Crespo, J., García, M., Arroyo, T., et al. (2023). Influence of Native Saccharomyces cerevisiae Strains on Malvasia aromatica Wines. *Frontiers in Bioscience-Elite*, 15(3), 18.

8. Supriatna, D.J.I., Saputra, H., & Hasan, K. (2023). Enhancing the Red Wine Quality Classification Using Ensemble Voting Classifiers. *Infolitika Journal of Data Science*, 1(2), 42–47.

## License

This project is available for academic and educational purposes. Please cite the original dataset (Cortez et al., 2009) and this repository in any derivative works.

## Contact

**Author**: Mukund Ranjan Tiwari (kodiakthebear)

---

**Academic Note**: This project was developed as part of a statistical machine learning course. Model predictions should not be used as the sole basis for commercial wine quality assessment. Professional wine evaluation requires trained sensory panels and expert judgment alongside analytical measurements.
