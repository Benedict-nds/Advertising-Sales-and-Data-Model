# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression  # Replace with your actual model
# import pickle

# # Example data (replace with your actual dataset)
# X = np.array([[230.1, 37.8, 69.2],
#               [44.5, 39.3, 45.1],
#               [17.2, 45.9, 69.3]])  # Feature data: TV, Radio, Newspaper budgets
# y = np.array([22.1, 10.4, 9.3])     # Target data: Sales

# # Initialize and train the scaler and model
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# model = LinearRegression()  # Replace with your actual model
# model.fit(X_scaled, y)

# # Save the model and scaler to a file
# with open('Backend/instances/SalesNN1.pkl', 'wb') as f:
#     pickle.dump((model, scaler), f)

# print("Model and scaler saved successfully!")
# #################################################################################################################

# import pandas as pd
# import numpy as np
# df = pd.read_excel("C:\\Users\\Benedict HA\\Desktop\\Ml data\\AdvertisingBudgetandSales.xlsx")
# df

# df.head()
# df.info()
# df.columns

# df.corr()
# x = df[['TV Ad Budget', 'Radio Ad Budget', 'Newspaper Ad Budget']]
# y = df['Sales'].values.reshape(-1, 1)
# # Normalize features
# x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

# # Define the neural network architecture
# input_dim = x.shape[1]  # Number of features (3)
# hidden_dim = 4          # Number of neurons in the hidden layer
# output_dim = 1          # Output (Sales prediction)

# # Initialize weights and biases
# weights_input_hidden = np.random.randn(input_dim, hidden_dim) * 0.1
# bias_hidden = np.zeros((1, hidden_dim))
# weights_hidden_output = np.random.randn(hidden_dim, output_dim) * 0.1
# bias_output = np.zeros((1, output_dim))

# # Activation function (ReLU for hidden layer and linear for output)
# def relu(z):
#     return np.maximum(0, z)

# def relu_derivative(z):
#     return (z > 0).astype(float)

# # Loss function (Mean Squared Error)
# def mean_squared_error(y_true, y_pred):
#     return np.mean((y_true - y_pred)**2)
# # Step 3: Training parameters
# learning_rate = 0.01
# epochs = 3800

# import pickle

# # Loss tracking
# losses = []

# # Training loop
# for epoch in range(epochs):
#     # Forward propagation
#     z_hidden = np.dot(x, weights_input_hidden) + bias_hidden  # Hidden layer linear transform
#     a_hidden = relu(z_hidden)  # Apply activation function
#     z_output = np.dot(a_hidden, weights_hidden_output) + bias_output  # Output layer linear transform
#     y_pred = z_output  # Linear activation for output
    
#     # Compute loss
#     loss = mean_squared_error(y, y_pred)
#     losses.append(loss)
    
#     # Backward propagation
#     dloss_dz_output = (y_pred - y) / y.shape[0]  # Derivative of loss w.r.t z_output
#     dloss_dw_hidden_output = np.dot(a_hidden.T, dloss_dz_output)  # Derivative w.r.t weights
#     dloss_db_output = np.sum(dloss_dz_output, axis=0, keepdims=True)  # Derivative w.r.t bias
    
#     dloss_da_hidden = np.dot(dloss_dz_output, weights_hidden_output.T)  # Loss gradient w.r.t a_hidden
#     dloss_dz_hidden = dloss_da_hidden * relu_derivative(z_hidden)  # Apply ReLU derivative
#     dloss_dw_input_hidden = np.dot(x.T, dloss_dz_hidden)  # Derivative w.r.t weights
#     dloss_db_hidden = np.sum(dloss_dz_hidden, axis=0, keepdims=True)  # Derivative w.r.t bias
    
#     # Update weights and biases (Gradient Descent)
#     weights_hidden_output -= learning_rate * dloss_dw_hidden_output
#     bias_output -= learning_rate * dloss_db_output
#     weights_input_hidden -= learning_rate * dloss_dw_input_hidden
#     bias_hidden -= learning_rate * dloss_db_hidden
    
#     # Print loss every 100 epochs
#     if epoch % 100 == 0:
#         print(f"Epoch {epoch}, Loss: {loss:.4f}")

# # Save the model
# model = {
#     "weights_input_hidden": weights_input_hidden,
#     "bias_hidden": bias_hidden,
#     "weights_hidden_output": weights_hidden_output,
#     "bias_output": bias_output
# }

# with open('SalesNN1.pkl', 'wb') as f:
#     pickle.dump(model, f)

# print("\nModel saved as 'SalesNN1.pkl'")

# # Evaluation
# print("\nFinal Evaluation")
# print(f"Final Loss: {loss:.4f}")
# print("Predicted vs Actual values (first 5 examples):")
# print(np.hstack((y_pred[:5], y[:5])))  # Compare predictions with actual targets


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

# Step 1: Load the dataset
df = pd.read_excel('AdvertisingBudgetandSales.xlsx')

# Step 2: Preprocess the data
x = df[['TV Ad Budget', 'Radio Ad Budget', 'Newspaper Ad Budget']].values
y = df['Sales'].values.reshape(-1, 1)

# Initialize and fit the scaler
scaler = StandardScaler()
x = scaler.fit_transform(x)  # Normalize features

# Step 3: Define the neural network architecture
input_dim = x.shape[1]  # Number of features (3)
hidden_dim = 4          # Number of neurons in the hidden layer
output_dim = 1          # Output (Sales prediction)

# Initialize weights and biases
weights_input_hidden = np.random.randn(input_dim, hidden_dim) * 0.1
bias_hidden = np.zeros((1, hidden_dim))
weights_hidden_output = np.random.randn(hidden_dim, output_dim) * 0.1
bias_output = np.zeros((1, output_dim))

# Activation function (ReLU for hidden layer and linear for output)
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

# Loss function (Mean Squared Error)
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Step 4: Training parameters
learning_rate = 0.01
epochs = 3800

# Loss tracking
losses = []

# Training loop
for epoch in range(epochs):
    # Forward propagation
    z_hidden = np.dot(x, weights_input_hidden) + bias_hidden  # Hidden layer linear transform
    a_hidden = relu(z_hidden)  # Apply activation function
    z_output = np.dot(a_hidden, weights_hidden_output) + bias_output  # Output layer linear transform
    y_pred = z_output  # Linear activation for output
    
    # Compute loss
    loss = mean_squared_error(y, y_pred)
    losses.append(loss)
    
    # Backward propagation
    dloss_dz_output = (y_pred - y) / y.shape[0]  # Derivative of loss w.r.t z_output
    dloss_dw_hidden_output = np.dot(a_hidden.T, dloss_dz_output)  # Derivative w.r.t weights
    dloss_db_output = np.sum(dloss_dz_output, axis=0, keepdims=True)  # Derivative w.r.t bias
    
    dloss_da_hidden = np.dot(dloss_dz_output, weights_hidden_output.T)  # Loss gradient w.r.t a_hidden
    dloss_dz_hidden = dloss_da_hidden * relu_derivative(z_hidden)  # Apply ReLU derivative
    dloss_dw_input_hidden = np.dot(x.T, dloss_dz_hidden)  # Derivative w.r.t weights
    dloss_db_hidden = np.sum(dloss_dz_hidden, axis=0, keepdims=True)  # Derivative w.r.t bias
    
    # Update weights and biases (Gradient Descent)
    weights_hidden_output -= learning_rate * dloss_dw_hidden_output
    bias_output -= learning_rate * dloss_db_output
    weights_input_hidden -= learning_rate * dloss_dw_input_hidden
    bias_hidden -= learning_rate * dloss_db_hidden
    
    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Step 5: Save the trained model and scaler
model = {
    "weights_input_hidden": weights_input_hidden,
    "bias_hidden": bias_hidden,
    "weights_hidden_output": weights_hidden_output,
    "bias_output": bias_output
}

# Save both the model and scaler together
with open('instances/SalesNN1.pkl', 'wb') as f:
    pickle.dump((model, scaler), f)

print("\nModel and scaler saved as 'SalesNN1.pkl'")

# Final evaluation
print("\nFinal Evaluation")
print(f"Final Loss: {loss:.4f}")
print("Predicted vs Actual values (first 5 examples):")
print(np.hstack((y_pred[:5], y[:5])))  # Compare predictions with actual targets