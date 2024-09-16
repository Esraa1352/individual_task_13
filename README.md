# Chapter 1: Using Neural Networks to Recognize Handwritten Digits
The human visual system is one of the most fascinating aspects of biology. To replicate this ability on computers, scientists came up with the idea of neural networks, a concept inspired by how the brain processes information. Neural networks are modeled after the brain’s neurons, and at the heart of these networks are perceptrons.

## Perceptrons:
A perceptron mimics a neuron in the brain. It takes multiple inputs, each with a weight, calculates a sum of these inputs, and produces an output. If this sum exceeds a certain threshold, the output is 1; otherwise, it’s 0. In this way, perceptrons are a way of weighing evidence to make decisions.

Perceptrons can be used to compute basic logical functions such as AND, OR, and NAND. In fact, the NAND gate is universal for computation, which means perceptrons can, in theory, perform any computational task by forming networks.

For example, we can design a network of perceptrons to mimic NAND gates, each with two inputs and weights of −2. This shows how powerful networks of perceptrons can be in computing even complex functions

## Sigmoid Neurons
While perceptrons can compute simple functions, they’re not very flexible. To improve the performance, scientists introduced sigmoid neurons. Instead of outputting only 0 or 1, sigmoid neurons use an activation function that outputs values between 0 and 1. This allows the network to handle probabilities and gives it more nuance when making decisions.
(the function is mentioned in the chapter )
فا

## Architecture of Neural Networks
A typical neural network has three types of layers:
Input layer: Where the data (like images or signals or numbers in our case) enters the network.
Output layer: Where the network produces its result.
Hidden layers: The intermediate layers that do most of the heavy computation. The design of these hidden layers often requires skill and experimentation.
Neural networks can be feedforward networks, where the output of one layer is passed as input to the next. In this way, information flows in one direction from the input to the output.

## Training the Model
To make a neural network useful, it needs to be trained. Training involves adjusting the weights and biases of the neurons so that the network can minimize errors when making predictions.
This process is done by something called the cost function, which measures how far off the predictions are from the actual values which measures the preformance of the model

## Gradient Descent and Backpropagation
Gradient descent is the algorithm used to minimize the cost function for it to be more accurate and make the predictions closer to the actual values by updating the weights and biases in a way that reduces the error. The network does this by calculating the gradient of the cost function and adjusting the weights accordingly.

To calculate these gradients efficiently, neural networks use a method called backpropagation. This technique sends the error backward through the network, layer by layer, allowing each neuron to update its weights and biases.

The learning rate controls how much the weights are updated with each step. If the learning rate is too large, the network might overshoot the optimal solution; if it's too small, training might take too long.

## Stochastic Gradient Descent (SGD)
A variation of gradient descent is stochastic gradient descent, which speeds up the learning process. Instead of using the entire dataset to calculate the gradient, SGD uses small, random subsets of data called mini-batches. This allows the network to make more frequent updates and often leads to faster convergence, helping avoid problems like getting stuck in local minima.



