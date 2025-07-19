# 📉 Customer Churn Prediction

This project is focused on predicting whether a customer will **churn (leave)** a service or continue using it. By leveraging historical customer data, the model aims to help businesses retain valuable customers and reduce churn rates.

---

## 🔍 Problem Statement

Customer churn is a major concern for businesses. Identifying at-risk customers can help target retention strategies. This project uses **machine learning** to predict churn based on customer demographics, services subscribed, usage patterns, and other factors.

---

## 📂 Dataset

- **Source:** Telco Customer Churn Dataset
- **Format:** `.xlsx` (can be `.csv`)
- **Features Include:**
  - Customer ID
  - Gender
  - SeniorCitizen
  - Tenure
  - InternetService
  - Contract Type
  - MonthlyCharges
  - TotalCharges
  - Churn (Target)

---

## 🧠 Technologies Used

- **Python**
- **Pandas, NumPy**
- **Matplotlib, Seaborn** (for EDA)
- **Scikit-learn** (Logistic Regression, Evaluation)
- **Streamlit** (for web app deployment)

---

## 🛠️ Project Structure

```bash
├── app.py                  # Streamlit App
├── churn_model.pkl         # Trained model
├── requirements.txt        # Required libraries
├── data/
│   └── Telco_customer_churn.xlsx
├── notebooks/
│   └── Churn_EDA_Model.ipynb
└── README.md               # Project overview
