# from app import create_app

# app = create_app()

# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask

# app = Flask(__name__)

# @app.route("/")
# def local():
#     return "Welcome to Benedict's server"

# if __name__ == "__main__":
#     app.run(debug=True, port=5001)







# Load the saved model
# with open('SalesNN1.pkl', 'rb') as f:
#     saved_model = pickle.load(f)

# # Assign loaded weights and biases
#     model = pickle.load(f)
# weights_input_hidden = saved_model["weights_input_hidden"]
# bias_hidden = saved_model["bias_hidden"]
# weights_hidden_output = saved_model["weights_hidden_output"]
# bias_output = saved_model["bias_output"]

# print("Model loaded successfully!")

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the saved model and scaler
with open('instances/SalesNN1.pkl', 'rb') as f:
    model, scaler = pickle.load(f)

    if model and scaler:
        print("Model and Scaler loaded successfully")
    else:
        print("Failed to load model or scaler")



@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON
        data = request.json
        if not data:
            return jsonify({'error': 'Invalid or missing JSON data'}), 400
        
        print(f"Received data: {data}")

        # Extract and scale input data
        input_data = np.array([
            data['TV Ad Budget'], 
            data['Radio Ad Budget'], 
            data['Newspaper Ad Budget']
        ]).reshape(1, -1)
        input_data_scaled = scaler.transform(input_data)

        # Manual forward propagation
        weights_input_hidden = model['weights_input_hidden']
        bias_hidden = model['bias_hidden']
        weights_hidden_output = model['weights_hidden_output']
        bias_output = model['bias_output']

        # Compute hidden layer activations
        z_hidden = np.dot(input_data_scaled, weights_input_hidden) + bias_hidden
        a_hidden = np.maximum(0, z_hidden)  # ReLU activation

        # Compute output layer activations
        z_output = np.dot(a_hidden, weights_hidden_output) + bias_output
        predicted_sales = z_output.item()  # Convert to scalar
        
        print(f"Predicted Sales: {predicted_sales}")
        return jsonify({'predicted_sales': predicted_sales})

    except Exception as e:
        # Log error and respond with a message
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction', 'details': str(e)}), 500

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.json
#         input_data = np.array([
#             data['TV Ad Budget'], data['Radio Ad Budget'], data['Newspaper Ad Budget']
#         ]).reshape(1, -1)

#         # Scale the input data
#         input_data_scaled = scaler.transform(input_data)

#         # Make prediction using the model
#         prediction = model.predict(input_data_scaled).item()
        
#         return jsonify({'predicted_sales': prediction})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
# def predict():
#     data = request.json
#     input_data = np.array([
#         data['TV Ad Budget'], data['Radio Ad Budget'], data['Newspaper Ad Budget']
#     ]).reshape(1, -1)

#     # Scale the input data
#     input_data_scaled = scaler.transform(input_data)

#     # Use the custom predict function
#     predicted_sales = model["predict"](input_data_scaled).item()
    
#     return jsonify({'predicted_sales': predicted_sales})




@app.route("/")
def main():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)