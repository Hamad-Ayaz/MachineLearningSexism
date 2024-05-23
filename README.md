# Explainable Detection of Online Sexism (EDOS)

## Overview
This project develops machine learning models to detect and explain sexist content online, as part of the SemEval 2023 - Task 10 competition. The goal is to identify whether posts are sexist or non-sexist, enhancing interpretability and trust in automated moderation tools.

## Technologies Used
- **Programming Languages**: Python
- **Libraries and Frameworks**: pandas, numpy, scikit-learn, XGBoost, nltk
- **Tools**: TfidfVectorizer, CountVectorizer, Logistic Regression, SVC, GridSearchCV
- **Environment**: Jupyter Notebook

## Features
- **Binary Sexism Detection**: Classifies posts as sexist or not.
- **Feature Engineering**: Implements techniques such as TF-IDF and Count Vectorization.
- **Model Evaluation**: Uses precision, accuracy, and F1-score metrics for comparison.

## Installation
```bash
pip install -r requirements.txt
```

## How to Run
```bash
python run_model.py
```

## Data
- The project uses a subset of the original EDOS dataset with 5,000 labeled samples. Preprocessing and vectorization are applied to prepare data for modeling.

## Model Training and Evaluation
- Details on the three models used:
  - Logistic Regression
  - Support Vector Classifier (SVC)
  - XGBoost
Performance is evaluated against a test set to measure generalization.

## Results
- Provide a summary of the best performing model and its metrics, highlighting aspects like weighted F1-score, precision, and recall.
