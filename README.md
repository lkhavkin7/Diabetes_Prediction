# Diabetes_Prediction

## Project Overview

This project focuses on predicting the onset of diabetes in Pima Indian women using diagnostic measurements from the Pima Indians Diabetes Dataset. It is designed for the biology/healthcare industry, emphasizing preventive medicine and risk assessment. The analysis includes extensive Exploratory Data Analysis (EDA), data cleaning to handle invalid zeros, custom feature engineering (e.g., BMI categories and age groups), and machine learning models implemented with Scikit-Learn to compare performance.

## Key highlights:

Dataset: Pima Indians Diabetes Database from Kaggle (768 samples, 9 features including glucose levels, BMI, age, etc.).
Techniques: EDA with visualizations (histograms, correlation heatmaps, boxplots, pairplots), data imputation using group medians, one-hot encoding for categorical features.
Models: Logistic Regression, Random Forest, SVM with hyperparameter tuning via GridSearchCV on Random Forest.
Metrics: Accuracy, Precision, Recall, F1 Score, ROC AUC.
Originality: Custom age grouping and BMI categorization for deeper insights; comparison of models with a focus on imbalanced class handling implicitly through stratification.
This project demonstrates skills in data preprocessing, visualization, and supervised classification, making it a strong addition to a resume for roles in healthcare analytics or bioinformatics.

# Requirements
Python 3.x
Jupyter Notebook
Libraries:
numpy
pandas
matplotlib
seaborn
scikit-learn

pip install numpy pandas matplotlib seaborn scikit-learn

## Dataset
Download the dataset from Kaggle: Pima Indians Diabetes Database. Place diabetes.csv in the project directory.

## How to Run
Open the Jupyter Notebook diabetes_prediction.ipynb in Jupyter Lab or Notebook.
Ensure the dataset file is in the same directory.
Run all cells sequentially. The notebook is self-contained with imports, data loading, EDA, cleaning, feature engineering, model training, and evaluation.
Outputs include visualizations and a results table for model comparisons.
File Structure
diabetes_prediction.ipynb: Main Jupyter Notebook with all code and explanations.
diabetes.csv: Dataset file (download separately).
README.md: This file.
Results Example
Models are evaluated on a test set (20% split). Example metrics (may vary slightly):

Logistic Regression: F1 ~0.65
Random Forest: F1 ~0.70 (improved with tuning)
SVM: F1 ~0.68
Feature importance and confusion matrices are visualized for interpretability.

Limitations and Future Work
Dataset is small; consider augmentation or external data for robustness.
Explore deep learning models (e.g., via TensorFlow) for comparison.
Deploy as a web app using Streamlit for interactive predictions.****
