
import numpy as np


def cost_function(X, y, theta):

    # Calculating the cost of the function (sum of squared errors)
    return np.sum(((np.matmul(X, theta) - y) ** 2)) / (2 * np.size(X, 0))


def gradient_descent(X, y, theta, alpha, threshold, costs):

    costs.append(cost_function(X, y, theta))
    m = np.size(X, 0)

    # Loop until we reach a step significantly small (convergence)
    while len(costs) < 2 or abs(costs[-2] - costs[-1]) > threshold:

        # Calculating the changes to theta
        theta0 = alpha * (m ** -1) * np.sum(np.matmul(X, theta) - y)
        theta1 = alpha * (m ** -1) * np.sum(np.multiply(np.matmul(X, theta) - y, X[:, 1].reshape(len(X), 1)))

        # Updating theta
        theta = np.array(([theta[0][0] - theta0], [theta[1][0]] - theta1))

        # Saving the new cost
        costs.append(cost_function(X, y, theta))

    return theta


def train_linear_regression():

    # Getting the data columns
    data = np.genfromtxt('Assignment 4 - Question 1 data.csv', delimiter=',')

    # Add a column of ones for gradient descent
    X = np.transpose(np.array([np.ones([len(data) - 1, 1])[:, 0], data[1:, 0]]))
    y = data[1:, 1].reshape(len(data) - 1, 1)
    theta = np.array(([0], [0])).reshape(2, 1)

    costs = []
    theta = gradient_descent(X, y, theta, 0.01, 10E-12, costs)
    print(theta)


train_linear_regression()

