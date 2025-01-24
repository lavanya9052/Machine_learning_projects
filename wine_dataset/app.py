from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model and scaler
with open('D:\Lavanya\wine_dataset\wine.pkl', 'rb') as f:
    model = pickle.load(f)

with open('D:\Lavanya\wine_dataset\scaled_data.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        alcohol_5th = float(request.form['alcohol_5th'])
        malic_acid_5th = float(request.form['malic_acid_5th'])
        alcalinity_of_ash_5th = float(request.form['alcalinity_of_ash_5th'])
        total_phenols_5th = float(request.form['total_phenols_5th'])
        flavanoids_5th = float(request.form['flavanoids_5th'])
        proanthocyanins_5th = float(request.form['proanthocyanins_5th'])
        color_intensity_5th = float(request.form['color_intensity_5th'])
        od280_od315_of_diluted_wines_5th = float(request.form['od280_od315_of_diluted_wines_5th'])
        proline_5th = float(request.form['proline_5th'])

        # Create input array
        input_data = np.array([[alcohol_5th, malic_acid_5th, alcalinity_of_ash_5th,
                                total_phenols_5th, flavanoids_5th, proanthocyanins_5th,
                                color_intensity_5th, od280_od315_of_diluted_wines_5th,
                                proline_5th]])

        # Standardize the input data
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)

        # Determine the predicted class
        predicted_class = "Class " + str(prediction[0])

        # Return the prediction result
        return render_template('index.html', prediction_text=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)