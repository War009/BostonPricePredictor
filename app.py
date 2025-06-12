from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the saved model
model = joblib.load('boston_house_price_model.pkl')

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from form
        features = [float(request.form.get(feat)) for feat in [
            'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
            'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
        ]]
        input_array = np.array([features])  # shape (1, 13)
        prediction = model.predict(input_array)[0]
        price = round(prediction, 2)

        return render_template('index.html', prediction_text=f"Estimated House Price: ${price * 1000:,.2f}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
  