from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load the trained model
model_path = 'best_combined_model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file {model_path} does not exist. Please check the file path and name.")
model = load_model(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = np.array(data['input']).reshape(1, 200, 13)  # Adjust the shape based on your model's expected input
    prediction = model.predict(input_data)
    output = prediction[0][0]
    return jsonify({'prediction': output})

if __name__ == '__main__':
    app.run(debug=True)
