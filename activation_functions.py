# improt library
import numpy as np
import matplotlib.pyplot as plt


"""
assume 3 layers : h(g(x)) = h(tanh(x+2)^2) 
3 layer (input layer): x = x + 2 這層只是將輸入值加2。
2 layer (hidden layer): g = tanh(x) 這層對前一層的輸出應用雙曲正切函數。
1 layer (output layer): h = g^2 這層將前一層的輸出平方。
"""
# define actication function
def layer_activation3(x):
    return x + 2

def layer_activation2(x):
    return np.tanh(x)

def layer_activation1(x):
    return x**2

# define composite function : shows h(g(x))
def composite_function(x):
    return layer_activation3(layer_activation2(layer_activation1(x)))

# 生成等差數列，範圍從 -5 到 5，共 100 個點
x = np.linspace(-5, 5, 100)

# claculate the output of each layer and the final output 計算各層的輸出和最終輸出
y3 = layer_activation3(x)
y2 = layer_activation2(y3)
y1 = layer_activation1(y2)
y_composite = composite_function(x)

# plot
plt.figure(figsize=(12, 8))
plt.plot(x, y3, label='Layer 3: x + 2')
plt.plot(x, y2, label='Layer 2: tanh(x)')
plt.plot(x, y1, label='Layer 1: x^2')
plt.plot(x, y_composite, label='Composite: (tanh(x+2))^2', linewidth=2)
plt.legend()
plt.title('Three-Layer Activation Function')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
