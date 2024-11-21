# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler  # Import if you're using scaling
import joblib  # Alternative to pickle for model loading

app = Flask(__name__)


try:
    # Try loading with joblib first (more reliable for sklearn models)
    model = joblib.load('model.pkl')

except:
    try:
        # Fallback to pickle if joblib fails
        model = pickle.load(open('model.pkl', 'rb'))
        # scaler = pickle.load(open('scaler.pkl', 'rb'))  # Uncomment if you have a scaler
    except Exception as e:
        print(f"Error loading model: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form
        features = [float(x) for x in request.form.values()]
        
        # Convert features to numpy array and reshape
        final_features = np.array(features).reshape(1, -1)
        

        if isinstance(model, np.ndarray):
            # Manual prediction for linear regression
            prediction = np.dot(final_features, model)
        else:
            # For sklearn models
            prediction = model.predict(final_features)
        
        # Round the prediction to 2 decimal places
        output = round(float(prediction[0]), 2)
        
        return render_template('index.html', 
                             prediction_text=f'Predicted Gold Price: ${output}')
                             
    except Exception as e:
        return render_template('index.html', 
                             prediction_text=f'Error in prediction: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)