import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from mpl_toolkits.mplot3d import Axes3D

def split_data(X, y, factor=0.3):
    """
    Splits data into testing and training
    :param X: Design Matrix
    :param y: Target Vector
    :param factor: Split factor
    :return: x_train, y_train, x_test, y_test
    """
    # split = round(X.shape[0] * (1-factor))
    # x_train = X[0:split]
    # y_train = y[0:split]
    # x_test = X[-(X.shape[0]-split):]
    # y_test = y[-(y.shape[0]-split):]
    # return x_train, x_test, y_train, y_test

    return train_test_split(X, y, test_size=factor)


def prepend_bias_term(X):
    """
    Prepends a column of ones to the design matrix (bias term)
    :param X: Design matrix
    :return: Column of ones prepended to design matrix
    """
    return np.column_stack((np.ones(X.shape[0]), X))


def normalize(X):
    """
    Scales all columns in X between 0 and 1
    :param X: Design matrix
    :return: Scaled Design matrix
    """
    max = np.max(X, axis=0)
    min = np.min(X, axis=0)
    return (X - min) / (max - min)


def normalise_data(x):
    # rescale data to lie between 0 and 1
    scale = x.max(axis=0)
    return x/scale


def l2_cost(predicted, actual):
    diff = predicted - actual
    squared = diff ** 2
    return np.mean(squared)


def predict(data, theta):
    # print(data.shape)
    # print(theta.shape)
    return np.dot(data, theta)


def gradient(X, predicted, actual):
    differences = predicted - actual

    # Multiply each column in X by differences - need to transpose
    # X = (m,n), differences = (m,1) --> cant multiply (tries to do dot)
    # X' = (n, m), differences = (m,1) --> dot product now is what we want
    x_t = X.T
    inner = np.dot(x_t, differences)

    # Finally multiply the gradients by 2/m
    return inner * 2 / X.shape[0]


def gradient_descent(x_train, y_train, learn=0.02, iterations=1000, threshold=0.1):
    theta = np.array([0.5, 0.8])
    loss = []
    for i in range(iterations):

        # Get new predictions
        predictions = predict(x_train, theta)

        # Get the loss assoc with these values of theta
        loss.append(l2_cost(predictions, y_train))

        # Check if within threshold

        # Get the gradient
        grad = gradient(x_train, predictions, y_train)

        # Update theta
        theta = theta - learn * grad

    # Plot the results
    plt.plot(loss)
    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    plt.show()

    min_loss = min(loss)
    print("Final loss: ", min_loss)

    return theta, min_loss

def main():
    # Load the data
    data = np.loadtxt('stock_prices.csv', usecols=(1, 2))
    X = np.array(data[:, 0])
    y = np.array(data[:, 1])

    # Make them 2d array for consitency
    X = np.reshape(X, (X.shape[0], 1))
    # y = np.reshape(y, (y.shape[0], 1))

    # Plot the data
    # fig, ax = plt.subplots(figsize=(12, 8))
    # ax.scatter(X, y, label="Data")
    # ax.set_xlabel("Amazon")
    # ax.set_ylabel("Google")
    # ax.set_title("Google vs Amazon stock price")
    # ax.set_xlim(xmin=0)
    # ax.set_ylim(ymin=0)
    # plt.show()

    # FIRST THING - Split data, don't leak any info about test set into model
    x_train, x_test, y_train, y_test = split_data(X, y)

    # Prepare data for training
    x_train = normalize(x_train)

    # # Prepend bias term after normalization (or else lots of nans)
    x_train = prepend_bias_term(x_train)

    # Test the functions
    # theta = np.array([300, 0.5])
    # predicted = predict(x_train, theta)
    # loss = l2_cost(predicted, y_train)
    # grad = gradient(x_train, predicted, y_train)

    # Print results
    # print(predicted)
    # print(loss)
    # print(grad)

    # Perform gradient descent
    theta, min_loss = gradient_descent(x_train, y_train, learn=0.1, iterations=100)
    print("Learned theta = ", theta)
    print("Minimum loss = ", min_loss)

    # Prepare test data
    x_test = normalize(x_test)
    x_test = prepend_bias_term(x_test)

    # Predict the test values
    predicted = predict(x_test, theta)
    loss = l2_cost(predicted, y_test)
    print("Loss on test = ", loss)

    # Plot the predictions vs the actual values
    plt.scatter(x_test[:, 1], predicted, label="Predicted")
    plt.scatter(x_test[:, 1], y_test, label="Actual")
    plt.xlabel("Amazon")
    plt.ylabel("Google")
    axes = plt.gca()
    axes.set_xlim(xmin=0)
    axes.set_ylim(ymin=0)
    plt.legend()
    plt.show()

    # Plot the cost function in 3d
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(1, 1, 1, projection='3d')
    n=100
    theta0, theta1 = np.meshgrid(np.linspace(-500, 500, n), np.linspace(-500, 500, n))
    cost = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            predicted = predict(x_test, [theta0[i, j], theta1[i, j]])
            cost[i,j] = l2_cost(predicted, y_test)

    ax3.plot_surface(theta0, theta1, cost)
    ax3.set_xlabel('theta0')
    ax3.set_ylabel('theta1')
    ax3.set_zlabel('J(theta)')
    plt.show()

# Run main
if __name__ == "__main__":
    main()