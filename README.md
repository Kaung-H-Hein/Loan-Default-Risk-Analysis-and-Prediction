# Loan-Default-Risk-Analysis-and-Prediction
This project provides a comprehensive analysis of loan default risk using the German Credit Dataset from the UCI Repository. It marks the first project in my MSc Artificial Intelligence studies. The project is organised into three key phases: data preprocessing, exploratory data analysis (EDA), and machine learning model development. It also incorporates advanced techniques such as Principal Component Analysis (PCA) for feature reduction and Markov Decision Processes (MDP) for simple decision optimisation. Here's an overview of the process:

## Phase 1: Data Inspection and Preprocessing
- Data Cleaning: Addressed missing values, encoded categorical features with OrdinalEncoder and LabelEncoder, and handled outliers.
- Feature Engineering: Created additional features to improve predictive performance.
- Scaling: Normalised numerical data using scaling techniques like RobustScaler, StandardScaler, and MinMaxScaler.
 
## Phase 2: Exploratory Data Analysis (EDA)
- Statistical Insights: Conducted thorough EDA to understand data distributions and relationships with loan default.
- Correlation Analysis: Used Pearson’s correlation coefficient to explore numerical feature relationships.
- Hypothesis Testing: Applied Chi-Squared and T-tests to examine feature dependencies and group differences.

## Phase 3: Machine Learning Model Development
- Model Building: Developed a Decision Tree Classifier to predict loan defaults.
- Feature Reduction: Implemented PCA to reduce dimensionality while preserving variance.
- Model Optimisation: Tuned model hyperparameters with GridSearchCV for improved performance.
- Decision Optimisation: Applied a Markov Decision Process (MDP) to optimise decision-making strategies based on loan default outcomes.

## Tools and Libraries Used
- Data Analysis: pandas, numpy
- Data Visualisation: matplotlib, seaborn
- Preprocessing and Encoding: OrdinalEncoder, LabelEncoder, RobustScaler, StandardScaler, MinMaxScaler
- Statistical Analysis: scipy.stats (Chi-Squared, T-tests, Pearson Correlation)
- Model Building and Evaluation: DecisionTreeClassifier, train_test_split, accuracy_score, precision_score, recall_score, f1_score
- Feature Reduction: PCA
- Optimisation: GridSearchCV

## Project Outcomes
This project demonstrates a systematic approach to analysing and predicting loan default risks. It integrates data science techniques with foundational AI concepts like PCA, and MDP, providing a thorough analysis and actionable insights for decision-making. The methodology offers practical applications in the financial domain, showcasing a robust pipeline for credit risk analysis and decision optimisation.
