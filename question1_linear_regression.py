
import numpy as np
from matplotlib import pyplot as plt
from math import floor, ceil


def cost_function(X, y, theta):

    # Calculating the cost of the function (sum of squared errors)
    return sum_of_squared_errors(X, y, theta) / (2 * np.size(X, 0))


def sum_of_squared_errors(X, y, theta):
    return np.sum(((np.matmul(X, theta) - y) ** 2))


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
    theta = gradient_descent(X, y, theta, 0.001, 10E-12, costs)

    # # Plotting the learning curve
    # plt.plot(costs)
    # plt.ylabel("Cost")
    # plt.xlabel("Iteration")
    # plt.show()

    # Calculating the linear regression line of prediction
    x_range = np.array([floor(np.amin(data[1:, 0])), ceil(np.amax(data[1:, 0]))])
    x_pred = np.array([x_range[0] * theta[1] + theta[0], x_range[1] * theta[1] + theta[0]])

    # Plotting the data
    plt.scatter(data[1:, 0], data[1:, 1])
    plt.plot(x_range, x_pred)

    print("Sum of squared errors: ", sum_of_squared_errors(X, y, theta))
    print("Average sum of squared errors: ", costs[-1])

    plt.show()



train_linear_regression()

