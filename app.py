import numpy as np
import pickle
from flask import Flask, request, render_template
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Create Flask app
app = Flask(__name__)

# Load trained model
model = pickle.load(open("sales_demand_forecasting.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_features = [float(x) for x in request.form.values()]

        final_features = [np.array(input_features)]
        prediction = model.predict(final_features)
        output = 'Fraudulent Claim' if prediction[0] == 1 else 'Genuine Claim'
        return render_template('result.html', prediction_text=output)
    except Exception as e:
        return render_template('result.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
