from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class Activation():
    # Create a new activation function object that include the function and its derivative
    def g(self, x):
        pass

    def g_derivative(self, x):
        pass

class Relu(Activation):
    # Create a new Relu activation function object that include the function and its derivative
    def g(self, x):
        return max(x, 0.0)

    def g_derivative(self, x):
        if x > 0:
            return 1.0
        else:
            return 0.0

    def __repr__(self):
        return "Relu()"

class Sigmoid(Activation):
    # Create a new Sigmoid activation function object that include the function and its derivative
    def g(self, x):
        return 1 / (1 + np.exp(-x))

    def g_derivative(self, x):
        return self.g(x) * (1 - self.g(x))

    def __repr__(self):
        return "Sigmoid()"

class Binary(Activation):
    # Create a new Binary activation function object that include the function and its derivative
    def g(self, x):
        if x > 0:
            return 1
        else:
            return 0

    def g_derivative(self, x):
        return 0

    def __repr__(self):
        return "Binary()"


class NeuralNetwork:
    # Create a new neural network object based on the given architecture
    # layer_activation - the activation function for the spesific layer
    # input_size - the number of dimensions of the input
    # batch_size - the number of samples that will be propagated through the network
    # learning_rate - defined how much to change the model in response to the estimated error
    # random_state - the number of the seed, it keeps the result the same on each execute
    def __init__(self, layer_activation, input_size, layers_size, batch_size, learning_rate, random_state=1729):
        assert(len(layer_activation) == len(layers_size))  # Each layer should have an activation function
        self.layer_activation = [None] + layer_activation

        assert(len(layers_size) > 0)
        self.layers_size = [input_size] + layers_size # Add the input size as fist layer in the neural network

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        np.random.seed(self.random_state)

        self.initialize_weight()


    @property
    def num_layers(self):
        return len(self.layers_size)

    def initialize_weight(self):
        # initialize_weight weight vector (W) and bias (b).

        self.W = [None]
        self.b = [None]

        for i in range(self.num_layers - 1):
            # self.layers_size[i] + 1 - in order to include the bias theta
            # np.randn - initialize with normal random matrix
            curr_W = np.random.randn(self.layers_size[i+1], self.layers_size[i])
            curr_b = np.random.randn(self.layers_size[i+1])
            self.W.append(curr_W)
            self.b.append(curr_b)


    def feed_forward(self, v_input):
        # Given a training sample (v_input), calculate the neutron values in each layer
        # :param v_input: vector of features
        # :return S: list of neutron values in each layer
        # :return h: list of values of neurons in the layers before the activation

        S = [np.copy(v_input)]
        h = [None] # to align the indexes
        for L in range(1, self.num_layers):
            h_L_next = self.W[L].dot(S[L-1]) + self.b[L]  # b is the bias list
            h.append(h_L_next)
            S_L_next = np.array([self.layer_activation[L].g(pre_neuron) for pre_neuron in h_L_next])
            S.append(S_L_next)

        return S, h


    # Run the network on all vectors in X and calculate the network output.
    def predict(self, v_input):
        # Calculater a predict value for a neutron
        # :param v_input: vector of features
        # :return: predict value for a neutron
        S, h = self.feed_forward(v_input)
        return S[-1]


    def error_function(self, y_pred, y_true):
        # Calculater the current error
        # :param y_pred: the current output of the system
        # :param y_true: the training sample, the true value
        # :return: the current error
        return 0.5 * (y_pred - y_true)**2.0


    def error_derivative(self, y_pred, y_true):
        return (y_pred - y_true)


    def backpropagation(self, batch_X, batch_y):
        # Test for errors working back from output nodes to input nodes
        # :param batch_X: list of training examples utilized in one iteration for the train set
        # :param batch_y: list of training examples utilized in one iteration for the test set
        # :return W_grad: list of The gradient of the weights for all the layers
        # :return b_grad: list of The gradient for the bias

        assert(len(X) == len(y))

        W_grad = [None]
        b_grad = [None]
        for L in range(1, self.num_layers):
            W_grad.append(np.zeros(self.W[L].shape))
            b_grad.append(np.zeros(self.b[L].shape))

        num_samples = len(batch_X)
        for k in range(num_samples):
            curr_x = batch_X[k]
            curr_y = batch_y[k]

            # S[L][m] := The output of the m-th neuron in the L-th layer
            # h[L][m] := The value of  the m-th neuron in the L-th layer before the activation
            S, h = self.feed_forward(curr_x)

            # backpropagation to calc deltas
            delta = [None for L in range(self.num_layers)]
            L = self.num_layers - 1

            # C(g(h_L_m),y) = C(S_L_m,y)
            # C(g(h_L_m),y)' = C'(S_L_m,y) * g'(h_L_m)
            delta[L] = np.array([self.layer_activation[L].g_derivative(h_L_m) for h_L_m in h[L]])
            delta[L] *= self.error_derivative(S[L], curr_y)
            # We just calculated the last delta, i.e. delta[self.num_layers - 1]
            # So the next delta we are going to calculate (since this is a backward loop) is self.num_layers - 2.
            for L in range(self.num_layers - 2, 0, -1):
                activation_derivative = [self.layer_activation[L].g_derivative(h_L_m) for h_L_m in h[L]]
                delta[L] = np.array(activation_derivative) * self.W[L+1].T.dot(delta[L+1])

            for L in range(1, self.num_layers):
                W_grad[L] += (1.0 / num_samples) * delta[L].dot(S[L].T)
                b_grad[L] += (1.0 / num_samples) * delta[L]

        return W_grad, b_grad


    def split_to_batches(self, X, y):
        # Split the dataset into batches
        # :param X: data of the training examples
        # :param y: targets of data the training examples
        # :return: list of batches
        num_batches = len(X) // self.batch_size
        if (len(X) %  self.batch_size != 0):
            num_batches+=1

        X_batches = np.array_split(X, num_batches)
        y_batches = np.array_split(y, num_batches)
        assert(len(X_batches) == len(y_batches))

        return list(zip(X_batches, y_batches))


    def fit(self, X_train, y_train, X_test, y_test, num_epochs):
        # Neural network training method
        # :param X_train: list of samples for the train dataset
        # :param y_train: list of targets for the train dataset
        # :param X_test: list of samples for the test dataset (for calculating statistics only)
        # :param y_test: list of targets for test dataset (for calculating statistics only)
        # :param num_epochs: the number of complete passes through the training dataset
        batches = self.split_to_batches(X_train, y_train)

        epoch = []  # Collected for statistics
        train_loss_graph = []  # Collected for statistics
        test_loss_graph = []  # Collected for statistics

        # calc current loss
        train_loss = self.loss(X_train, y_train)
        test_loss = self.loss(X_test, y_test)

        print(f"EPOCH ({0}-pre training): loss (Train)={train_loss}, loss (Test)={test_loss}")

        for i in range(1, num_epochs+1):
            for batch_X, batch_y in batches:
                W_grad, b_grad = self.backpropagation(batch_X, batch_y)
                for L in range(1, self.num_layers):
                    self.W[L] -= (self.learning_rate * W_grad[L])
                    self.b[L] -= (self.learning_rate * b_grad[L])

            # Calc current loss
            train_loss = self.loss(X_train, y_train)
            test_loss = self.loss(X_test, y_test)

            print(f"EPOCH ({i}): loss (Train)={train_loss}, loss (Test)={test_loss}")

            # Collect data for statistics graph
            epoch.append(i)
            train_loss_graph.append(train_loss)
            test_loss_graph.append(test_loss)

        statistics(epoch, train_loss_graph, test_loss_graph)

    def loss(self, inputs, true_output):
        # Calculate the loss
        # :param true_output: list of target values
        # :return total_loss: the network error rate in total
        total_loss = 0.0
        for curr_input, curr_true_output in zip(inputs, true_output):
            output_pred = self.predict(curr_input)
            total_loss += self.error_function(output_pred, curr_true_output)
        total_loss /= len(inputs)
        return total_loss



def statistics(epoch, train_loss_graph, test_loss_graph):
    # Present train_loss_graph and test_loss_graph in a graph "Error on train and test sets"
    # :param epoch: list of numbers, each present an epoch
    # :param train_loss_graph: list of loss values of the train set
    # :param test_loss_graph: list of loss values of the test set
    # :return total_loss: the network error rate in total

    epoch = np.array(epoch)
    train_loss_graph = np.array(train_loss_graph)
    test_loss_graph = np.array(test_loss_graph)

    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Error on train and test sets')

    plt.plot(epoch, train_loss_graph, label="train loss")
    plt.plot(epoch, test_loss_graph, label="test loss")
    plt.legend()
    plt.show()

# Load iris dataset
iris = load_iris()

# Place data into variables
X = iris['data']
y = iris['target']

p = X.shape[1] # Input_size for the ann


# Split the data Into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1729)


# Insert architecture parameter into the NeuralNetwork
ann_model = NeuralNetwork(layer_activation=[Sigmoid(), Relu(), Relu()], input_size=p,
                          layers_size=[50, 50, 1], batch_size=16, learning_rate=1e-3, random_state=17)

# ann_model = NeuralNetwork(layer_activation=[Binary(), Binary(), Binary()], input_size=p,
#                             layers_size=[50, 50, 1], batch_size=16, learning_rate=1e-3, random_state=17)


# Insert batches into the training method
# Note: (X_test, y_test) are passed to the training method only for calculating statistics.
ann_model.fit(X_train, y_train, X_test, y_test, num_epochs=1_000)


# Create prediction value for each table line (x) in the train dataset
train_pred = np.array([ann_model.predict(x) for x in X_train])
train_pred = np.round(train_pred).flatten()


# Calculate the precedent of train set accuracy, how many flowers the system got right out of the train datasets
train_accuracy = (train_pred == y_train).sum() / len(y_train)
mismatch_cond = (train_pred != y_train)
train_combined_outputs = np.array(list(zip(train_pred, y_train)))

print(f"Train set accuracy: {100 * train_accuracy}%")
print("Train set prediction mismatch:\n", train_combined_outputs[mismatch_cond])


# Calculate the precedent of test set accuracy, how many flowers the system got right out of the test datasets
test_pred = np.array([ann_model.predict(x) for x in X_test])
test_pred = np.round(test_pred).flatten()

test_accuracy = (test_pred == y_test).sum() / len(y_test)
mismatch_cond = (test_pred != y_test)
test_combined_outputs = np.array(list(zip(test_pred, y_test)))

print(f"Test set accuracy: {100 * test_accuracy}%")
print("Test set prediction mismatch:\n", test_combined_outputs[mismatch_cond])




