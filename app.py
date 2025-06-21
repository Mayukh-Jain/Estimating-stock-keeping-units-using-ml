import numpy as np
import pickle
from flask import Flask, request, render_template


import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

model = pickle.load(open("sales_demand_forecasting.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        total_price = float(request.form['total_price'])
        base_price = float(request.form['base_price'])
        is_featured = float(request.form['is_featured_sku'])
        is_display = float(request.form['is_display_sku'])

        input_data = np.array([[total_price, base_price, is_featured, is_display]])
        prediction = model.predict(input_data)

        return render_template('result.html', prediction_text=f"Forecasted Demand: {round(prediction[0], 2)} units")
    except Exception as e:
        return render_template('result.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
