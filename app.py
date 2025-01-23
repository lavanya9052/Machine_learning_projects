import flask
from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model and scaler (Assuming files are in the same directory)
with open('D:\Lavanya\ml_tasks\iris .pkl', 'rb') as f:
    model = pickle.load(f)
with open('D:\Lavanya\ml_tasks\scaling_data .pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def fun():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        sepal_length = float(request.form['sepal-length'])
        sepal_width = float(request.form['sepal-width'])
        petal_length = float(request.form['petal-length'])
        petal_width = float(request.form['petal-width'])

        # Prepare input data
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Standardize input data
        scaled_input = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(scaled_input)[0]

        # Map prediction to class labels
        if prediction == 0:
            predicted_class = 'setosa'
        elif prediction == 1:
            predicted_class = 'versicolor'
        else:
            predicted_class = 'virginica'

        return render_template('index.html', prediction_text=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)