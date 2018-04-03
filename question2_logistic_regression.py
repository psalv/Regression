
import numpy as np
from matplotlib import pyplot as plt


def cost_function(X, y, theta):
    hx = sigmoid(theta, X)
    return np.sum(np.multiply(-1 * y, np.log10(hx)) - np.multiply(1 - y, np.log10(1 - hx))) / len(y)


def sigmoid(theta, X):
    return (1 + np.exp(-1 * np.matmul(X, theta))) ** -1


def gradient_descent(X, y, theta, alpha, threshold, costs):

    costs.append(cost_function(X, y, theta))
    m = np.size(X, 0)

    # Loop until we reach a step significantly small (convergence)
    while len(costs) < 2 or abs(costs[-2] - costs[-1]) > threshold:

        # Calculating the changes to theta
        theta_ = (alpha * (m ** -1) * np.sum(np.multiply(sigmoid(theta, X) - y, X), 0)).reshape(np.size(theta), 1)
        theta = theta - theta_

        # Saving the new cost
        costs.append(cost_function(X, y, theta))

    return theta


def train_logistic_regression():

    # Getting the data columns
    data = np.genfromtxt('breast-cancer-wisconsin.test.data.csv', delimiter=',')

    # Adding column of 1s for theta0
    X = np.transpose(np.concatenate([np.transpose(np.ones([len(data), 1])), np.transpose(data[:, 1:-1])]))
    y = data[:, -1:]

    # Changing y values to be 0 (benign) or 1 (malignant)
    for i in range(len(y)):
        if y[i] == 2:
            y[i] = 0
        else:
            y[i] = 1

    # Initializing thetas
    theta = np.zeros([np.shape(X)[1], 1])

    costs = []
    theta = gradient_descent(X, y, theta, 0.1, 10E-8, costs)
    print(theta)

    # # Plotting the learning curve
    # plt.plot(costs)
    # plt.ylabel("J(theta)")
    # plt.xlabel("Iteration")
    # plt.show()


train_logistic_regression()

