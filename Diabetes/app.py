from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler (replace with your actual filepaths)
model_path = "D:\Lavanya\diabetes\diabetes.pkl"
scaler_path = "D:\Lavanya\diabetes\scaled_data_diabetes.pkl"

try:
  with open(model_path, 'rb') as f:
    model = pickle.load(f)

  with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

except FileNotFoundError:
  print(f"Error: Model or scaler file not found at {model_path} or {scaler_path}")
  exit()


@app.route('/')
def fun():

  return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

  if request.method == 'POST':
    try:
      # Get input values from the form
      features = [float(request.form[feature]) for feature in request.form]

      # Reshape input data
      input_data = np.array(features).reshape(1, -1)

      # Standardize the input data (if used during model training)
      input_data_scaled = scaler.transform(input_data)

      # Make prediction
      prediction = model.predict(input_data_scaled)

      # Determine the predicted class (assuming binary classification)
      if prediction[0] == 0:
          predicted_class = "Non-Diabetic"
      else:
          predicted_class = "Diabetic"

      # Return the prediction result
      return render_template('index.html', prediction_text=predicted_class)

    except ValueError:
      return render_template('index.html', prediction_text="Invalid input. Please enter numerical values.")

if __name__ == '__main__':
    app.run(debug=True)