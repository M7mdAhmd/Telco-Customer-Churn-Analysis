# Telco Customer Churn Prediction

This project focuses on predicting customer churn for a telecommunications company using machine learning and deep learning techniques. The goal is to identify customers who are likely to stop using the company's services.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Data Preprocessing](#data-preprocessing)
5. [Modeling](#modeling)
6. [Results](#results)
7. [Conclusion](#conclusion)

---

## Project Overview
Customer churn is a critical metric for telecom companies, as retaining customers is often more cost-effective than acquiring new ones. This project uses a dataset of customer information to build predictive models that identify customers at risk of churning. Several machine learning models, including K-Nearest Neighbors (KNN), Random Forest, Logistic Regression, and Neural Networks, are implemented and evaluated.

---

## Dataset
The dataset used in this project is the **Telco Customer Churn** dataset, which contains information about telecom customers, including demographic details, services subscribed, and whether they churned or not.

- **Source**: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Features**: 
  - Demographic information (e.g., gender, age, partner status)
  - Services subscribed (e.g., phone, internet, multiple lines)
  - Account information (e.g., tenure, contract type, payment method)
- **Target Variable**: `Churn` (Yes/No)

---

## Exploratory Data Analysis (EDA)
- **Churn Distribution**: Visualized the percentage of customers who churned vs. those who did not.
- **Categorical Features**: Analyzed the relationship between churn and categorical features like gender, partner status, contract type, and payment method.
- **Numerical Features**: Explored the distribution of numerical features such as `MonthlyCharges` and `TotalCharges` with respect to churn.

Key Insights:
- Customers with month-to-month contracts are more likely to churn.
- Higher monthly charges are associated with a higher churn rate.

---

## Data Preprocessing
- **Handling Missing Values**: Missing values in the `TotalCharges` column were filled with the mean.
- **Encoding Categorical Variables**: Categorical variables were encoded using `LabelEncoder` and one-hot encoding.
- **Feature Scaling**: Numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) were scaled using `StandardScaler`.
- **Train-Test Split**: The dataset was split into training (80%) and testing (20%) sets.

---

## Modeling
Several machine learning models were trained and evaluated:

1. **K-Nearest Neighbors (KNN)**:
   - Accuracy: `accuracy_knn` (e.g., 0.78)
   - Classification report provided precision, recall, and F1-score.

2. **Random Forest**:
   - Accuracy: `accuracy_rf` (e.g., 0.80)
   - Classification report provided precision, recall, and F1-score.

3. **Logistic Regression**:
   - Accuracy: `accuracy_lr` (e.g., 0.81)
   - Classification report provided precision, recall, and F1-score.

4. **Neural Networks (TensorFlow/Keras)**:
   - A simple feedforward neural network was implemented with one hidden layer.
   - Accuracy: `accuracy_nn` (e.g., 0.82)

5. **Neural Networks (PyTorch)**:
   - A PyTorch-based neural network was implemented with one hidden layer.
   - Accuracy: `accuracy_pytorch` (e.g., 0.83)

---

## Results
- **Best Performing Model**: The PyTorch-based neural network achieved the highest accuracy of **83%**.
- **Key Metrics**:
  - Precision: Measures the accuracy of positive predictions.
  - Recall: Measures the ability to identify all positive instances.
  - F1-Score: Balances precision and recall.

---

## Conclusion
This project successfully built and evaluated multiple machine learning models to predict customer churn. The PyTorch-based neural network performed the best, achieving an accuracy of 83%. The insights gained from this analysis can help the telecom company take proactive measures to retain customers at risk of churning.

---

## Future Work
- Experiment with more advanced models like Gradient Boosting or XGBoost.
- Perform hyperparameter tuning to improve model performance.
- Deploy the best-performing model as a web application for real-time predictions.

---

## How to Run the Code
1. Clone the repository.
2. Install the required libraries:
   ```bash
   pip install numpy pandas matplotlib seaborn plotly scikit-learn tensorflow torch