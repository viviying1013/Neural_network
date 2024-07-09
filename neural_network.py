
#import library
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score #R² is a measure of how well the model fits the data. MSE is a measure of how far a regression line is from the data points 


np.random.seed(0) # Setting the random seed to a specific value like 0 ensures that the random numbers generated will be the same every time the code is run. This is useful for reproducibility in data analysis or machine learning tasks.


"""
assume 3 layers : h(g(x)) = h(tanh(x+2)^2)  with non-liner function : Y = a + bX + cX^2
3 layer (input layer): x = x + 2 這層只是將輸入值加2。
2 layer (hidden layer): g = tanh(x) 這層對前一層的輸出應用雙曲正切函數。
1 layer (output layer): h = g^2 這層將前一層的輸出平方。
"""

# Define dataset
def true_function(X, a, b, c):
    return a + b * X + c * X**2

## load data
X = np.linspace(-5, 5, 100).reshape(-1, 1) #generate 100 equally spaced numbers between -5 and 5 # two dimensional array of shape (100, 1)
a, b, c = 2, -1, 0.5
Y = true_function(X, a, b, c)
Y_noisy = Y + np.random.normal(0, 0.5, Y.shape) #add noise to the target variable

# 1.define layer(activation function)
def layer1(x):
    return x + 2

def layer2(x):
    return np.tanh(x)

def layer3(x):
    return x**2

# define neural network(define the composite function).定義神經網絡 
"""
神經網路（Neural Network）的運作方式可以看作是數學上的複合函數（Composite Function）
一個神經網路由多個層（Layers）組成，每一層都包含一個或多個神經元（Neurons）。每個神經元都會對其輸入數據進行一些數學運算
（例如，加權和和偏置），然後通過一個激活函數（Activation Function）來決定其輸出。
"""
def neural_network(x):
    return layer3(layer2(layer1(x)))

## count each layer's output.計算每一層的輸出
output1 = layer1(X)
output2 = layer2(output1)
output3 = layer3(output2)
predictions = neural_network(X) #predictions = composite 預測也等同組合

# count the MSE(mean squared error) of each layer and the final output .計算每一層的MSE
mse_y3 = mean_squared_error(Y, output1) #y3=output1=x+2
mse_y2 = mean_squared_error(Y, output2)
mse_y1 = mean_squared_error(Y, output3)
mse_predictions = mean_squared_error(Y, predictions) #mse_predictions = mse_composite

# count the R² score of each layer and the final output .計算每一層的R²
r2_y3 = r2_score(Y, output1)
r2_y2 = r2_score(Y, output2)
r2_y1 = r2_score(Y, output3)
r2_predictions = r2_score(Y_noisy, predictions) 

# 2.define model and train regression model. 創建和訓練線性回歸模型 ：能夠從數據中學習，調整其參數以最小化誤差。
model = LinearRegression() 
model.fit(X, Y_noisy) # it the model to the input data `X` and target data `Y_noisy`. 將模型訓練在輸入數據 `X` 和目標數據 `Y_noisy` 上。

# 3.define neural network model
model_nn = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,), bias_initializer='zeros'),
    tf.keras.layers.Dense(64, activation='relu', bias_initializer='zeros'),
    tf.keras.layers.Dense(1, bias_initializer='zeros')
])

# compile and fit : 使用 TensorFlow 創建了一個神經網絡模型 `model_nn`
model_nn.compile(optimizer='adam', loss='mse')
model_nn.fit(X, Y_noisy, epochs=100) # equivalent of training

# make predictions
Y_pred_linear = model.predict(X) # use the linear model to predict the values of `X`
Y_pred_nn = model_nn.predict(X) # use the neural network model to predict the values of `X`

# evaluate the models
mse_linear = mean_squared_error(Y_noisy, Y_pred_linear)
r2_linear = r2_score(Y_noisy, Y_pred_linear)
mse_nn = mean_squared_error(Y_noisy, Y_pred_nn)
r2_nn = r2_score(Y_noisy, Y_pred_nn)

# plot
plt.figure(figsize=(12, 8))## create figure

## plot noisy data & true function & linear regression & neural network
plt.scatter(X, Y_noisy, color='blue', alpha=0.5, label='Noisy data')
plt.plot(X, Y, color='red', label='True function')
plt.plot(X, Y_pred_linear, color='green', label='Linear regression')
plt.plot(X, Y_pred_nn, color='orange', label='Neural network')

## plot each layer's output(使用循環繪製每一層的輸出和對應的MSE)
colors = ['blue', 'green', 'purple']
layer_names = ['x + 2', 'tanh(x)', 'x^2']
outputs = [output1, output2, output3]
mse_scores = [mse_y3, mse_y2, mse_y1]
r2_predictions=[r2_y3, r2_y2, r2_y1]
for i, (output, mse) in enumerate(zip(outputs, mse_scores)):
    plt.plot(X, output, color=colors[i], label=f'Layer {i+1}: {layer_names[i]}, (MSE = {mse:.4f}), (R² = {r2_predictions[i]:.4f})')

## plot composite function output and MSE
plt.plot(X, predictions, label=f'Composite: (tanh(x+2))^2, MSE: {mse_predictions:.4f}(R² = {r2_predictions[i]:.4f})', linewidth=2)

##set title and labels
plt.legend() # 顯示圖例 
plt.title('Three-Layer Activation Function')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

# print
for i, (output, mse) in enumerate(zip(outputs, mse_scores), 1):
    print(f'Layer {i}: {layer_names[i-1]}, MSE: {mse:.4f}') #every layer's MSE
print(f'Composite function output, MSE: {mse_predictions:.4f}') # composite function's MSE
print(f"Coefficients: {model.coef_}") # linear regression's coefficients
print(f"Intercept: {model.intercept_}") # linear regression's intercept
print(f"Final R² score: {r2_linear:.2f}") #lineear regression's MSE
print(f"Mean squared error(MSE): {mse_nn:.2f}") # neural network's MSE
print(f"Final R² score: {r2_nn:.2f}") #neural network's R²
