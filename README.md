# Dataset Description

## Overview
This project uses a public loan dataset for predicting borrower default risk. The dataset contains historical loan application and repayment-related information, designed for binary classification tasks.

- Target Variable: `is_default`
- Task Type: Classification
- Objective: Predict whether a borrower will default on a loan

---

## Dataset Size

- Total Records: ~800,000+
- Time Range: 2012 - 2018
- Features: 45+

---

## Target Variable

| Column Name | Description |
|------------|-------------|
| is_default | 0 = Non-default, 1 = Default |

---

## Example Features

| Feature Name | Description |
|-------------|-------------|
| loan_amnt | Loan amount |
| term | Loan term |
| interest_rate | Interest rate |
| installment | Installment payment |
| grade | Credit grade |
| annual_income | Annual income |
| dti | Debt-to-income ratio |
| fico_range_low | Lower FICO score |
| fico_range_high | Upper FICO score |
| open_acc | Number of open credit lines |
| revol_bal | Revolving balance |
| total_acc | Total credit accounts |

---

## Data Preprocessing

The following preprocessing steps were applied:

- Removed missing values
- Encoded categorical variables
- Selected important features using Random Forest
- Standardized numerical features
- Time-based train/test split

---

## Validation Strategy

To simulate real-world financial forecasting, a rolling time-window validation method was used:

- Train on 2012 → Test on 2013
- Train on 2013 → Test on 2014
- ...
- Train on 2017 → Test on 2018

This avoids data leakage and better reflects production scenarios.

---

## Notes

The raw dataset is not uploaded in this repository due to file size and potential licensing restrictions.

Users may replace the dataset path with their own local copy.

Example:

```python
DATA_PATH = r"D:\Desktop\数据集.csv"
