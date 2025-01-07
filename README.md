# Loan-Default-Risk-Analysis-and-Prediction
This project provides a comprehensive analysis of loan default risk using the German Credit Dataset from the UCI Repository. It marks the first project in my MSc Artificial Intelligence studies. The project is organised into three key phases: data preprocessing, exploratory data analysis (EDA), and machine learning model development. It also incorporates concepts such as Principal Component Analysis (PCA) for feature reduction and Markov Decision Processes (MDP) for simple decision optimisation simulation. Below is an overview of the process:

## Phase 1: Data Inspection and Preprocessing
- Data Cleaning: Addressed missing values, encoded categorical features with OrdinalEncoder and LabelEncoder, and handled outliers.
- Feature Engineering: Created additional features to improve predictive performance.
 
## Phase 2: Exploratory Data Analysis (EDA)
- Statistical Insights: Conducted thorough EDA to understand data distributions and relationships with loan default.
- Correlation Analysis: Used Pearson’s correlation coefficient to explore linear relationships between important features.
- Hypothesis Testing: Applied Chi-Squared and T-tests to examine feature dependencies and group differences.

## Phase 3: Machine Learning Model Development
- Model Building: Developed a Decision Tree Classifier to predict loan defaults.
- Feature Reduction: Implemented PCA to reduce dimensionality while preserving variance.
- Model Optimisation: Tuned model hyperparameters with GridSearchCV for improved performance.
- Decision Optimisation: Applied a Markov Decision Process (MDP) to simulate a simple decision-making strategies based on previous loan payments.

## Tools and Libraries Used
- Data Analysis: pandas, numpy
- Data Visualisation: matplotlib, seaborn
- Preprocessing and Encoding: OrdinalEncoder, LabelEncoder, RobustScaler, StandardScaler, MinMaxScaler
- Statistical Analysis: scipy.stats (Chi-Squared, T-tests, Pearson Correlation)
- Model Building and Evaluation: DecisionTreeClassifier, train_test_split, accuracy_score, precision_score, recall_score, f1_score
- Feature Reduction: PCA
- Optimisation: GridSearchCV

## Model Performance
- Achieved an average performance of 0.60 across metrics including model accuracy, precision, recall, and F1 score.
  
## Project Outcomes
This project highlights a structured approach to analysing and predicting loan default risks, effectively combining data science practices with core AI techniques. The integration of PCA for feature reduction and MDP for decision optimisation demonstrates an application of advanced methodologies to address real-world financial challenges. By systematically processing, analysing, and modelling the data, the project delivers meaningful insights into the factors influencing loan default. It also provides a robust framework for credit risk analysis, with practical implications for improving decision-making in financial systems.
