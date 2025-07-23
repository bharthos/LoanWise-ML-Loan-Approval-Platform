from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the saved model, scaler, feature names, and LabelEncoders
logistic_regression_model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")
label_encoders = joblib.load("label_encoders.pkl")  # Load saved LabelEncoders

@app.route('/')
def index():
    # Render the input form
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form
    try:
        input_data = {
            "person_age": float(form_data["person_age"]),
            "person_income": float(form_data["person_income"]),
            "person_emp_length": float(form_data["person_emp_length"]),
            "loan_amnt": float(form_data["loan_amnt"]),
            "loan_int_rate": float(form_data["loan_int_rate"]),
            "loan_percent_income": float(form_data["loan_percent_income"]),
            "cb_person_cred_hist_length": float(form_data["cb_person_cred_hist_length"]),
            "person_home_ownership": form_data["person_home_ownership"],
            "loan_intent": form_data["loan_intent"],
            "loan_grade": form_data["loan_grade"],
            "cb_person_default_on_file": form_data["cb_person_default_on_file"],
        }

        input_df = pd.DataFrame([input_data])

        # Apply Label Encoding to categorical variables
        categorical_columns = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
        for col in categorical_columns:
            input_df[col] = label_encoders[col].transform(input_df[col])

        # Align the input DataFrame with the expected feature order
        input_df = input_df.reindex(columns=feature_names, fill_value=0)
        print("Input DataFrame (aligned):", input_df)

        # Scale the features
        scaled_features = scaler.transform(input_df)
        print("Scaled Features:", scaled_features)

        # Predict using the Logistic Regression model
        probabilities = logistic_regression_model.predict_proba(scaled_features)
        prediction = logistic_regression_model.predict(scaled_features)
        print("Prediction Probabilities:", probabilities)
        print("Prediction Class:", prediction)

        predicted_class = int(prediction[0])  # Binary classification (0 or 1)

        # Prepare result
        result = "Approved" if predicted_class == 1 else "Denied"
        confidence = probabilities[0][predicted_class]

        return render_template('index.html', prediction=result, confidence=f"{confidence:.2f}")
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
