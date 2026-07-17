# Customer Churn Prediction and Retention Analytics Dashboard

## Project Overview

Customer churn is a major challenge for businesses because losing existing customers affects revenue, customer lifetime value, and long-term business growth. This project develops a machine learning based customer churn prediction system that identifies customers who are likely to leave and provides insights into the factors influencing customer retention.

The project uses customer demographic information, service usage patterns, billing information, subscription details, and customer satisfaction data to predict churn probability.

A Logistic Regression machine learning model was developed and integrated into an interactive Streamlit dashboard that allows users to analyse customer behaviour, identify high-risk customers, and generate churn predictions.

---

# Business Problem

Businesses often know when customers have already left, but the greater challenge is identifying customers who are likely to leave before churn happens.

This project provides a predictive solution that helps organisations:

- Identify customers at risk of leaving
- Understand the major causes of churn
- Prioritise retention efforts
- Make data-driven customer management decisions

---

# Project Objectives

The objectives of this project are:

1. Analyse customer behaviour patterns associated with churn.
2. Identify important factors influencing customer retention.
3. Develop a machine learning model for churn prediction.
4. Generate churn probability scores for individual customers.
5. Build an interactive dashboard for business decision-making.

---

# Dataset Description

The dataset contains 7,043 customer records with information relating to:

## Customer Demographics

- Gender
- Age
- Senior citizen status
- Marital status
- Dependents

## Customer Services

- Phone service
- Multiple lines
- Internet service
- Internet type
- Online security
- Online backup
- Device protection
- Premium technical support
- Streaming services

## Financial Information

- Monthly charges
- Total charges
- Total revenue
- Refunds
- Additional charges

## Customer Behaviour

- Tenure duration
- Number of referrals
- Satisfaction score
- Contract type
- Payment method

## Target Variable

The target variable is:

`Churn Value`

Where:

```
0 = Customer Stayed

1 = Customer Churned
```

---

# Data Preparation

The following steps were performed during data preparation:

## Data Cleaning

- Checked missing values
- Reviewed dataset structure
- Removed unnecessary variables
- Prepared data for modelling

## Feature Engineering

Categorical variables were transformed into numerical values using encoding techniques.

Examples:

- Contract type
- Internet type
- Payment method
- Customer service subscriptions

## Feature Scaling

Numerical variables were standardised using feature scaling before model training.

## Data Splitting

The dataset was divided into training and testing datasets to evaluate model performance on unseen data.

---

# Exploratory Data Analysis

Several analyses were performed to understand customer churn behaviour.

## Customer Satisfaction

Customer satisfaction was identified as one of the strongest indicators of churn.

Customers with lower satisfaction scores showed a higher tendency to leave.

Business implication:

Improving customer experience and service quality can improve retention.

---

## Contract Type

Customers using shorter contract periods showed higher churn probability compared with customers on longer contracts.

Business implication:

Businesses can encourage longer commitments through loyalty benefits and improved customer value.

---

## Monthly Charges

Customers with higher monthly charges showed increased churn probability.

Business implication:

Businesses should ensure pricing reflects customer value and service quality.

---

## Additional Services

Customers using services such as online security and technical support showed lower churn tendencies.

Business implication:

Value-added services can strengthen customer retention.

---

# Machine Learning Model

## Model Used

The final machine learning model selected was:

**Logistic Regression**

Logistic Regression was selected because it provides:

- Strong classification performance
- Churn probability estimation
- Easy interpretation of important factors affecting churn

---

# Model Performance

The Logistic Regression model achieved the following results:

| Metric | Score |
|---|---|
| Accuracy | 96.1% |
| Precision | 95% |
| Recall | 90% |
| F1 Score | 92% |

Confusion Matrix:

```
[[1018, 17],
 [38, 336]]
```

The model successfully identified customers likely to churn while maintaining strong overall prediction accuracy.

---

# Model Insights

The model coefficients were analysed to understand the factors affecting churn.

## Factors Reducing Churn Risk

Important retention factors include:

- Higher satisfaction score
- Customer referrals
- Online security subscription
- Longer contract periods
- Having dependents

---

## Factors Increasing Churn Risk

Important churn factors include:

- Higher monthly charges
- Lower customer satisfaction
- Short-term contracts
- Some service subscriptions

---

# Dashboard Features

The Streamlit dashboard contains the following sections:

## Executive Overview

Displays:

- Total customers
- Churned customers
- Churn rate
- Retention rate

---

## Churn Analysis

Provides visual analysis of:

- Churn by contract type
- Churn by internet type
- Satisfaction score differences between churned and retained customers

---

## Model Insights

Displays:

- Important churn factors
- Features affecting customer retention

---

## High Risk Customers

Allows users to:

- Identify customers with high churn probability
- Search customer records
- Download customer risk reports

---

## Prediction Tool

Allows users to enter customer details and receive:

- Churn probability
- Customer risk classification

Example:

```
Churn Probability: 82%

Risk Level: High Risk
```

---

# Technologies Used

## Programming Language

- Python

## Data Analysis

- Pandas
- NumPy

## Machine Learning

- Scikit-learn

## Visualisation

- Matplotlib
- Seaborn
- Plotly

## Dashboard Development

- Streamlit

---

# Project Structure

```
customer_churn/

│
├── app.py
│
├── README.md
│
├── data/
│   └── customer_churn_cleaned.csv
│
├── models/
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   └── features.pkl
│
└── notebooks/
    └── customer_churn_analysis.ipynb
```

---

# How To Run The Project

Install required packages:

```
pip install -r requirements.txt
```

Run the dashboard:

```
streamlit run app.py
```

---

# Business Recommendations

Based on the analysis, businesses should:

1. Monitor customer satisfaction regularly.
2. Provide proactive support to high-risk customers.
3. Encourage longer contract commitments.
4. Review pricing strategies for customers with high monthly charges.
5. Promote value-added services that improve customer loyalty.

---

# Conclusion

This project demonstrates how machine learning can transform customer data into actionable retention strategies.

By combining predictive modelling with an interactive dashboard, businesses can identify customers at risk of churn, understand customer behaviour, and develop targeted strategies to improve retention.