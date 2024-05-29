#giving 4 inputs to 3 neurons for 3 outputs
import numpy as np
inputs = np.array([1,2,3,2.5])

weights = np.array([[0.2,0.8,-0.5,1],
            [0.5, -0.91,0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]])
biases = [2, 3, 0.5]

layer_outputs = []
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = np.dot(inputs, neuron_weights) + neuron_bias
    layer_outputs.append(neuron_output)

# or
# layer_outputs = np.dot(weights, inputs) + biases # matrix multiplication of 3x4 and 1x4 output will be 3x1

print(layer_outputs)