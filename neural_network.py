
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error #R² is a measure of how well the model fits the data. MSE is a measure of how far a regression line is from the data points 


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

# load data
X = np.linspace(-5, 5, 100) # 生成等差數列，範圍從 -5 到 5，共 100 個點
a, b, c = 2, -1, 0.5  # 設置參數
Y = true_function(X, a, b, c)
Y_noisy = Y + np.random.normal(0, 0.5, Y.shape)  # 添加一些噪聲

# define layer(activation function)
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
 

# plot
## create figure
plt.figure(figsize=(12, 8))

## plot noisy data and true function
plt.scatter(X, Y_noisy, alpha=0.5, label='Noisy Data')
plt.plot(X, Y, color='red', label='True Function')

## plot each layer's output(使用循環繪製每一層的輸出和對應的MSE)
"""
if didn't use For loop的方式繪製:
    plt.plot(x, y3, label=f'Layer 3: x + 2, MSE: {mse_y3:.4f}')
    plt.plot(x, y2, label=f'Layer 2: tanh(x), MSE: {mse_y2:.4f}')
    plt.plot(x, y1, label=f'Layer 1: x^2, MSE: {mse_y1:.4f}')

for i, (output, mse) in enumerate(zip(outputs, mse_list)) 的意思: refer to ST.py
"""
colors = ['blue', 'green', 'purple']
layer_names = ['x + 2', 'tanh(x)', 'x^2']
outputs = [output1, output2, output3]
mse_scores = [mse_y3, mse_y2, mse_y1]
r2_predictions=[r2_y3, r2_y2, r2_y1]

for i, (output, mse) in enumerate(zip(outputs, mse_scores)):
    plt.plot(X, output, color=colors[i], label=f'Layer {i+1}: {layer_names[i]} (MSE = {mse:.4f})(R² = {r2_predictions[i]:.4f})')

## plot composite function output and MSE
plt.plot(X, predictions, label=f'Composite: (tanh(x+2))^2, MSE: {mse_predictions:.4f}(R² = {r2_predictions[i]:.4f})', linewidth=2)

##set title and labels
plt.legend() # 顯示圖例 
plt.title('Three-Layer Activation Function')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

# print results
for i, (output, mse) in enumerate(zip(outputs, mse_scores), 1):
    print(f'Layer {i}: {layer_names[i-1]}, MSE: {mse:.4f}')
print(f'Composite function output, MSE: {mse_predictions:.4f}')
print(f"Final R² score: {r2_predictions[i]:.4f}")
