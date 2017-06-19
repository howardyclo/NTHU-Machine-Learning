import numpy as np

""" Cost """
class Cost:
    def __call__(self, y_truth, y_pred):
        raise NotImplementedError

    def delta(self, y_truth, y_pred):
        raise NotImplementedError

class SquaredError(Cost):
    def __call__(self, y_truth, y_pred):
        return (y_pred - y_truth) ** 2 / 2

    def delta(self, y_truth, y_pred):
        return y_pred - y_truth

""" Regularizer """
class Regularizer:
    def __call__(self, weights):
        raise NotImplementedError

    def delta(self, weights):
        raise NotImplementedError

class L2(Regularizer):
    def __init__(self, reg_lambda=0.01):
        self.reg_lambda = reg_lambda

    def __call__(self, weights):
        return self.reg_lambda/2 * sum([np.sum(np.square(weight)) for weight in weights])

    def delta(self, weight):
        reg_lambda = np.ones_like(weight) * self.reg_lambda
        reg_lambda[0] = 0 # Bias do not have regularization terms
        return reg_lambda * weight

""" Activation """
class Activation:
    def __call__(self, x):
        raise NotImplementedError

    def delta(self, x, z, above):
        raise NotImplementedError

class Linear(Activation):
    def __call__(self, x):
        return x

    def delta(self, x, z, above):
        delta = np.ones(x.shape).astype(float)
        return delta * above

class Sigmoid(Activation):
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def delta(self, x, z, above):
        delta = z * (1 - z)
        return delta * above

""" Layer """
class Layer:
    def __init__(self, size, activation):
        self.size = size
        self.activation = activation()
        self.x = np.zeros(size)
        self.z = np.zeros(size)

    def apply(self, x):
        self.x = x
        self.z = self.activation(x)

    def delta(self, above):
        return self.activation.delta(self.x, self.z, above)

""" Network """
class Network:
    def __init__(self, layers):
        self.layers = layers
        self.sizes = tuple(layer.size for layer in self.layers)
        self.shapes = zip(self.sizes[:-1], self.sizes[1:])
        # +1 for bias term.
        self.shapes = [(x+1, y) for x, y in self.shapes]

    def feed(self, weights, data):
        self.layers[0].apply(data)
        for prev, weight, current in zip(self.layers[:-1], weights, self.layers[1:]):
            x = self.forward(weight, prev.z)
            current.apply(x)
        return self.layers[-1].z

    @staticmethod
    def forward(weight, x):
        # Insert bias input of one
        x = np.insert(x, 0, 1)
        z = x.dot(weight)
        return z

    @staticmethod
    def backward(weight, z):
        backward = z.dot(weight.T)
        # Don't expose the bias input of one.
        backward = backward[1:]
        return backward

""" Backprop """
class Gradient:
    def __init__(self, network, cost, regularizer):
        self.network = network
        self.cost = cost
        self.regularizer = regularizer

    def __call__(self, weights, x, y):
        raise NotImplementedError

class Backprop(Gradient):
    """ Ref: http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm """
    def __call__(self, weights, x, y):
        y_pred = self.network.feed(weights, x)
        delta_output = self._delta_output(y_pred, y)
        delta_layers = self._delta_layers(delta_output, weights)
        delta_weights = self._delta_weights(delta_layers, weights)
        return delta_weights

    def _delta_output(self, y_pred, y):
        """ The derivative with respect to the output layer is computed as the
            product of cost error derivative and local derivative at the layer.
        """
        delta_cost = self.cost.delta(y, y_pred)
        delta_output = self.network.layers[-1].delta(delta_cost)
        return delta_output

    def _delta_layers(self, delta_output, weights):
        """ Propagate backwards through the hidden layers but not the input
            layer. The current weight matrix is the one to the right of the
            current layer.
        """
        delta_layers = [delta_output]
        hidden = list(zip(weights[1:], self.network.layers[1:-1]))
        for weight, layer in reversed(hidden):
            delta_layer = self._delta_layer(layer, weight, delta_layers[-1])
            delta_layers.append(delta_layer)
        return reversed(delta_layers)

    def _delta_layer(self, layer, weight, above):
        """ The gradient at a layer is computed as `delta_layer` (the derivative of both the
            local activation) and `backward` (the weighted sum of the derivatives in the
            deeper layer).
        """
        backward = self.network.backward(weight, above)
        delta_layer = layer.delta(backward)
        return delta_layer

    def _delta_weights(self, delta_layers, weights):
        """ The gradient w.r.t the weights is computed as the gradient
            at the target neuron multiplied by the activation of the source neuron.
        """
        delta_weights = [np.zeros(shape) for shape in self.network.shapes]
        for i, (prev, delta_layer) in enumerate(zip(self.network.layers[:-1], delta_layers)):
            z = np.insert(prev.z, 0, 1)
            delta_weights[i] = np.outer(z, delta_layer) + self.regularizer.delta(weights[i])
        return delta_weights

""" Optimizer """
class GradientDecent:
    def __call__(self, weights, delta_weights, learning_rate=0.01):
        for i in range(len(weights)):
            weights[i] = weights[i] - learning_rate * delta_weights[i]
        return weights
