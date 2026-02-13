# Python libraries
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model, scaler, and selected features
model = joblib.load('final_model/heart_disease_model.pkl')
scaler = joblib.load('final_model/scaler.pkl')
selected_features = joblib.load('final_model/selected_features.pkl')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form', methods=['GET', 'POST'])
def form():
    result = None
    if request.method == 'POST':
        try:
            # Extract input values for the selected features
            input_data = [float(request.form[feature]) for feature in selected_features]
            
            # Scale the input data
            scaled_data = scaler.transform([input_data])
            
            # Make a prediction
            prediction = model.predict(scaled_data)[0]
            
            # Generate a result based on the prediction
            result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
        except KeyError as e:
            result = f"Missing input for feature: {e.args[0]}"
        except Exception as e:
            result = f"Error: {str(e)}"
    
    return render_template('form.html', features=selected_features, result=result)



if __name__ == '__main__':
    app.run(debug=True)
