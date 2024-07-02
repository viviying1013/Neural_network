import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

np.random.seed(0) # Setting the random seed to a specific value like 0 ensures that the random numbers generated will be the same every time the code is run. This is useful for reproducibility in data analysis or machine learning tasks.

"""
assume 3 layers : h(g(x)) = h(tanh(x+2)^2)  
with non-liner function : Y = a + bX + cX^2 Traget：To fitting this function.
3 layer (input layer): x = x + 2 這層只是將輸入值加2。
2 layer (hidden layer): g = tanh(x) 這層對前一層的輸出應用雙曲正切函數。
1 layer (output layer): h = g^2 這層將前一層的輸出平方。
"""

# define true_function.定義真實函數
def true_function(X, a, b, c):
    return a + b * X + c * X**2

## 生成數據
X = np.linspace(-5, 5, 100)
a, b, c = 2, -1, 0.5
Y = true_function(X, a, b, c)
Y_noisy = Y + np.random.normal(0, 0.5, Y.shape)

# define layer.定義層
def layer1(x):
    return x + 2

def layer2(x):
    return np.tanh(x)

def layer3(x):
    return x**2

# define neural network(conect each layer).定義神經網絡 
def neural_network(x):
    return layer3(layer2(layer1(x)))

## count each layer's output.計算每一層的輸出
output1 = layer1(X)
output2 = layer2(output1)
output3 = layer3(output2)

## count each layer's R².計算每一層輸出的R²
r2_1 = r2_score(Y_noisy, output1)
r2_2 = r2_score(Y_noisy, output2)
r2_3 = r2_score(Y_noisy, output3)

## count final prediction and R².計算最終預測和R²
predictions = neural_network(X)
r2 = r2_score(Y_noisy, predictions)


# polt
## create figure
plt.figure(figsize=(12, 8))

## plot noisy data and true function
plt.scatter(X, Y_noisy, alpha=0.5, label='Noisy Data')
plt.plot(X, Y, color='red', label='True Function')

## 繪製每一層的輸出(使用循環繪製每一層的輸出和對應的 R² 值)
colors = ['blue', 'green', 'purple']
layer_names = ['x + 2', 'tanh(x)', 'x^2']
outputs = [output1, output2, output3]
r2_scores = [r2_1, r2_2, r2_3]

for i, (output, r2) in enumerate(zip(outputs, r2_scores)):
    plt.plot(X, output, color=colors[i], label=f'Layer {i+1}: {layer_names[i]} (R² = {r2:.4f})')

## plot the final neural network
plt.plot(X, predictions, color='orange', label=f'Neural Network (R² = {r2:.4f})')

## set title and labels
plt.title('Neural Network Approximation with h(g(x)) = (tanh(x + 2))^2')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

print(f"Final R² score: {r2:.4f}")
