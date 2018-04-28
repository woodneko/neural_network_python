from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

nn_input_dim = 2
nn_output_dim = 2

# gradient descent parameters
epsilon = 0.01
reg_lambda = 0.01

# generate dataset
np.random.seed(0)
X, y = datasets.make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.spectral)
plt.title("origin data set")

num_examples = len(X)

def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

    

# evaluate the total loss on the datasets
def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)

    # calculating the loss 
    coorect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)

    # Add regulatization term to loss (optional)
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss

def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)

    return np.argmax(probs, axis=1)

def build_model(nn_hdim, num_passes=20000, print_loss=False):
    #init the parameters to random values
    np.random.seed(int(round(time.time())) & 0xFFFFFFFF)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    # ret value
    model = {}

    for i in xrange(0, num_passes):
        # forward propagation
        z1 = X.dot(W1) + b1 
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # backpropagation 
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Add regularization terms
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # Gradient descent parameters update
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2

        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # optionally print the loss
        # This is expensive because it uses the whole datasets, so we don't want to do it too often 
        if print_loss and i % 1000:
            print "Loss after iteration %i: %f" %(i, calculate_loss(model))

    return model

def plt_output(plt_num, index, nn_hdim, model, X, y):
    plt.subplot(plt_num,2,index)
    plt.title('hidden layer size %d' % nn_hdim)
    plot_decision_boundary(lambda x: predict(model, x), X, y) 


plt.figure(figsize=(16,32))
hidden_layer_dimensions=[3,4]
def do_python_method():
    plt_num = len(hidden_layer_dimensions)
    for i, nn_hdim in enumerate(hidden_layer_dimensions):
        model=build_model(nn_hdim)
        plt_output(plt_num, i+1, nn_hdim, model, X, y)

if __name__=='__main__':
    py_version = "2.01"
    py_description = "Auto generate code according to specify input file"

    parser = argparse.ArgumentParser(description=py_description)

    parser.add_argument('-c', '--clibrary'
            , nargs=1
            , type=str
            , default=None
            , help="input user generate implement")
    parser.add_argument('-p', '--python'
            , action='store_true'
            , help="operation use python only")

    args=parser.parse_args()

    if None == args.clibrary and None == args.python:
        parser.print_help()
        exit(-1)
      
    if None != args.python:
        do_python_method()

    # show plot
    plt.show()

