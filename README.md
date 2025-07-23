# 💼 LoanWise — Machine Learning-Powered Loan Approval Predictor

**LoanWise** is a Flask-based machine learning web application that predicts whether a loan will be approved based on user-submitted financial information. It combines a simple user interface with a trained logistic regression model for fast and accurate predictions.

---

## 🚀 Features

- 🧠 ML-backed predictions using Logistic Regression
- 📊 Input-based prediction with feature preprocessing
- 🎨 Responsive and styled HTML/CSS interface
- 🧰 Flask backend for real-time serving
- 📁 Organized project structure with templates and static assets

---

## 📂 Project Structure

LoanWise-ML-Loan-Approval-Platform/
│
├── flask_app/                      # Flask web app
│   ├── app                         # Entry point (app.py)
│   ├── models                      # Saved ML models (.pkl files)
│   ├── static                      # Static files (e.g., CSS)
│   └── templates                   # HTML templates (e.g., index.html)
│
├── notebooks/                     # Jupyter notebooks for EDA and model training
│   ├── Question4.ipynb            # Main analysis notebook
│   └── 1.ipynb                    # Additional notebook (possibly drafts or initial exploration)
│
├── Portfolio/                     # Basic frontend portfolio (optional)
│   ├── index.html                 # Standalone HTML page
│   ├── css/                       # Styling for portfolio
│   └── assets/                    # Images and static assets
│
├── requirements.txt               # Python dependencies
└── README.md                      # Project overview and instructions

---

## 🧪 ML Model

- ✅ Logistic Regression
- ✅ Feature scaling with `StandardScaler`
- ✅ Label encoding for categorical inputs
- ✅ Trained on historical loan applicant data

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.7+
- pip

### Clone the repository

```bash
git clone https://github.com/your-username/LoanWise.git
cd LoanWise/FlaskLoan
