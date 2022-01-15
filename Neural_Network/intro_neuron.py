from random import sample
import numpy as np
from matplotlib import pyplot as plt
import math 

np.random.seed(0)
def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

#100 feature sets of 3 classes
X, y = spiral_data(100,3)
# print(X)
# print(y)
# plt.scatter(X[:, 0], X[:, 1])#, c=y, s=40, cmap=plt.cm.Spectral)
# plt.show()

# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap="brg")
# plt.show()


class Layer_Dense:
    #inputs is the size of each feature (this case 4), neurons is how many features there are (3)
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1, n_neurons)) 
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        numerator = np.exp(inputs - np.max(np.array(inputs)))
        demoninator = np.sum(np.exp(inputs - np.max(inputs)), axis = 1, keepdims=True)
        self.output = np.divide(numerator, demoninator)

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_Crossentropy(Loss):
    def forward(self, y_pred, y_true):
        clipped_y = np.clip(y_pred, np.exp(-7), 1- np.exp(-7)) 
        if len(y_true.shape) == 1:
            confidence = clipped_y[range(len(clipped_y)),y_true]
        elif len(y_true.shape == 2):
            confidence = np.sum(clipped_y*y_true, axis =1)
        return -np.log(confidence)


dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss_function = Loss_Crossentropy()
loss = loss_function.calculate(activation2.output, y)

print(loss)

# print(activation2.output[:5])
# print(layer1.output)

# activation1 = Activation_ReLU()
# activation1.forward(layer1.output)
# print(activation1.output)


#Creating two output layers
# X = [[1, 2, 3, 2.5], 
#           [2.0, 5.0, -1.0, 2.0], 
#             [-1.5, 2.7, 3.3,-0.8]]

# layer1 = Layer_Dense(4,5)
# layer2 = Layer_Dense(5,2)
# layer1.forward(X)
# layer2.forward(layer1.output)

# weights = [[0.2 , 0.8, -0.5, 1.0],
#             [0.5 , -0.91, 0.26, -0.5],
#             [-0.26 , -0.27, 0.17, 0.87]]

# biases = [2, 3, 0.5]

# weights2 = [[0.1 , -0.14, 0.5],
#             [-0.5 , 0.12, -0.33],
#             [-0.44 , 0.73, -0.13]]


# biases2 = [-1,2, -0.5]


# layer1_output = np.dot(np.array(inputs), np.array(weights).T, ) + biases
# layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2




# inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
# output = []

# for i in inputs:
#     if i > 0:
#         output.append(i)
#     elif i <= 0:
#         output.append(0)

# print(output)

#cross-entropy loss
# softmax_output = [0.7, 0.1, 0.2]
# target_output = [1,0,0]
# loss = -(math.log(softmax_output[0]*target_output[0] + softmax_output[1]*target_output[1] + softmax_output[2]*target_output[2]))
# print(loss)
