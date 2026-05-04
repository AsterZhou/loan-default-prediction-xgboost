# -*- coding: utf-8 -*-
"""
Created on Sat May  2 20:30:47 2026

@author: Baize
"""

# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, Perceptron, RidgeClassifier, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

from xgboost import XGBClassifier


DATA_PATH = r"D:\Desktop\数据集.csv"


COLUMN_NAMES = [
    "loan_amnt", "term", "interest_rate", "installment", "grade", "sub_grade",
    "employment_title", "employment_length", "home_ownership", "annual_income",
    "verification_status", "issue_date", "is_default", "purpose", "post_code",
    "region_code", "dti", "delinquency_2years", "fico_range_low",
    "fico_range_high", "open_acc", "pub_rec", "pub_rec_bankruptcies",
    "revol_bal", "revol_util", "total_acc", "initial_list_status",
    "application_type", "earlies_credit_line", "title", "policy_code",
    "n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9",
    "n10", "n11", "n12", "n13", "n14"
]


SELECTED_FEATURES = [
    "issue_date",
    "interest_rate",
    "n5",
    "n10",
    "open_acc",
    "fico_range_high",
    "fico_range_low",
    "n7",
    "revol_bal",
    "revol_util",
    "installment",
    "annual_income",
    "loan_amnt",
    "total_acc",
    "n6",
    "n8",
    "grade",
    "dti",
    "is_default"
]


def load_data(path):
    data = pd.read_csv(path)
    data = data.dropna().copy()

    if "id" in data.columns:
        data = data.drop(columns=["id"])

    data.columns = COLUMN_NAMES
    return data


def encode_features(data):
    data = data.copy()

    grade_map = {
        "A": 7,
        "B": 6,
        "C": 5,
        "D": 4,
        "E": 3,
        "F": 2,
        "G": 1
    }

    data["grade"] = data["grade"].map(grade_map)

    data["employment_length"] = (
        data["employment_length"]
        .replace({"< 1 year": "1 year", "10+ years": "10 years"})
        .str.split(" ", expand=True)[0]
        .astype("int64")
    )

    return data


def feature_importance_analysis(data):
    data = data.copy()

    y = data["is_default"]
    X = data.drop(columns=["is_default"])

    drop_columns = [
        "employment_title",
        "verification_status",
        "issue_date",
        "post_code",
        "region_code",
        "initial_list_status",
        "earlies_credit_line",
        "title",
        "sub_grade"
    ]

    X = X.drop(columns=[col for col in drop_columns if col in X.columns])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rfc = RandomForestClassifier(random_state=42, n_estimators=100)
    rfc.fit(X_scaled, y)

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": rfc.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print("\n===== Random Forest Feature Importance =====")
    print(importance_df)

    return importance_df


def prepare_selected_data(data):
    data = data.copy()
    data = data[SELECTED_FEATURES]
    return data


def split_by_year(data):
    data = data.copy()

    data["issue_date"] = pd.to_datetime(data["issue_date"], errors="coerce")
    data = data.dropna(subset=["issue_date"])

    data["year"] = data["issue_date"].dt.year
    data = data.drop(columns=["issue_date"])

    year_data = {}

    for year in range(2012, 2019):
        temp = data.loc[data["year"] == year].copy()
        temp = temp.drop(columns=["year"])

        print(f"{year} 年数据量：{len(temp)}")

        if len(temp) == 0:
            continue

        y = temp["is_default"]
        X = temp.drop(columns=["is_default"])

        year_data[year] = {
            "X": X,
            "y": y
        }

    return year_data

def train_and_predict_by_window(model, year_data, use_scaler=True):
    all_true = []
    all_pred = []
    all_score = []

    for train_year in range(2012, 2017):
        test_year = train_year + 1

        if train_year not in year_data or test_year not in year_data:
            print(f"跳过 {train_year} -> {test_year}，因为其中某一年没有数据")
            continue

        X_train = year_data[train_year]["X"]
        y_train = year_data[train_year]["y"]

        X_test = year_data[test_year]["X"]
        y_test = year_data[test_year]["y"]

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"跳过 {train_year} -> {test_year}，因为训练集或测试集为空")
            continue

        if use_scaler:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            y_score = y_pred

        all_true.append(y_test)
        all_pred.append(y_pred)
        all_score.append(y_score)

    if len(all_true) == 0:
        raise ValueError("所有年份窗口都没有成功训练，请检查 issue_date 的年份范围。")

    y_true = np.hstack(all_true)
    y_pred = np.hstack(all_pred)
    y_score = np.hstack(all_score)

    return y_true, y_pred, y_score
def evaluate_model(name, y_true, y_pred, y_score):
    print(f"\n===== {name} =====")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, zero_division=0))
    print("F1:", f1_score(y_true, y_pred, zero_division=0))
    print("AUC:", roc_auc_score(y_true, y_score))

    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    return {
        "name": name,
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc_score(y_true, y_score)
    }


def plot_roc_curves(roc_results):
    plt.figure(figsize=(12, 8))

    for result in roc_results:
        plt.plot(
            result["fpr"],
            result["tpr"],
            linewidth=2,
            label=f'{result["name"]} AUC={result["auc"]:.4f}'
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()


def final_test_on_2018(year_data):
    X_train = pd.concat([year_data[year]["X"] for year in range(2012, 2018)], axis=0)
    y_train = pd.concat([year_data[year]["y"] for year in range(2012, 2018)], axis=0)

    X_test = year_data[2018]["X"]
    y_test = year_data[2018]["y"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBClassifier(
        learning_rate=0.05,
        n_estimators=400,
        max_depth=4,
        min_child_weight=5,
        seed=0,
        subsample=0.7,
        colsample_bytree=0.77,
        gamma=0,
        reg_alpha=5,
        reg_lambda=0.01,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_score = model.predict_proba(X_test_scaled)[:, 1]

    print("\n===== Final XGBoost Test on 2018 =====")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, zero_division=0))
    print("F1:", f1_score(y_test, y_pred, zero_division=0))
    print("AUC:", roc_auc_score(y_test, y_score))


def main():
    data = load_data(DATA_PATH)
    data = encode_features(data)

    feature_importance_analysis(data)

    selected_data = prepare_selected_data(data)
    year_data = split_by_year(selected_data)

    models = [
        ("LogisticRegression", LogisticRegression(max_iter=1000)),
        ("PassiveAggressiveClassifier", PassiveAggressiveClassifier(random_state=42)),
        ("Perceptron", Perceptron(random_state=42)),
        ("RidgeClassifier", RidgeClassifier()),
        ("SGDClassifier", SGDClassifier(random_state=42)),
        ("XGBClassifier", XGBClassifier(eval_metric="logloss", random_state=42))
    ]

    roc_results = []

    for name, model in models:
        y_true, y_pred, y_score = train_and_predict_by_window(model, year_data, use_scaler=True)
        result = evaluate_model(name, y_true, y_pred, y_score)
        roc_results.append(result)

    plot_roc_curves(roc_results)

    final_test_on_2018(year_data)


if __name__ == "__main__":
    main()
