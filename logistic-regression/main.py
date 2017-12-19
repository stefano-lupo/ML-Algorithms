import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def split_data(X, y, factor=0.3):
    """
    Splits data into testing and training
    :param X: Design Matrix
    :param y: Target Vector
    :param factor: Split factor
    :return: x_train, x_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=factor, random_state=99)


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
    return (X - min) / (max - min), max-min


def normalise_data(x):
    # rescale data to lie between 0 and 1
    scale = x.max(axis=0)
    return x/scale


def normaliseData(x):
    # rescale data to lie between 0 and 1
    scale = x.max(axis=0)
    return (x/scale, scale)


def add_quadratic_feature(X):
    return np.column_stack((X, X[:, 0] ** 2))


def cost(X, theta, actual):
    return np.mean(np.log(1+np.exp(-actual * np.dot(X, theta))))


def predict(data, theta, degree=1):
    xtt = np.dot(data ** degree, theta)
    return np.where(xtt < 0, -1, 1)


def gradient(X, theta, actual):
    common = np.exp(-actual * np.dot(X, theta))

    right = common / (1 + common)

    most = -actual * right
    # print("Most: ", most.shape)
    out = []
    for i in range(X.shape[1]):
        tmp = X[:, i]
        # print("tmp: ", tmp.shape)
        ans = tmp * most
        out.append(np.mean(ans))

    # print("out:: ", out)

    return np.array(out)


def gradient_descent(x_train, y_train, learn=0.02, iterations=1000, threshold=0.1, degree=1):
    theta = np.array(np.random.rand(x_train.shape[1]))
    loss = []
    for i in range(iterations):

        # Get new predictions
        predictions = predict(x_train, theta, degree=degree)

        # Get the loss assoc with these values of theta
        loss.append(cost(x_train, theta, y_train))

        # Check if within threshold

        # Get the gradient
        grad = gradient(x_train, theta, y_train)

        # Update theta
        theta = theta - learn * grad

    return theta, np.min(loss), loss

def main():
    # Load the data
    data = np.loadtxt('health.csv')

    x_cols = list(range(data.shape[1]-1))
    print("Using columns ", x_cols, " for X columns")

    X = np.array(data[:, x_cols])
    y = np.array(data[:, -1])

    print("X Shape: ", X.shape)
    print("Y Shape: ", y.shape)

    # Plot the data
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    # ax1.scatter(X[:, 0], y, label="Data")
    # ax1.set_xlabel("Blood 1")
    # ax1.set_ylabel("Disease")
    # ax1.set_title("Blood 1 vs Disease")
    #
    # ax2.scatter(X[:, 0], y, label="Data")
    # ax2.set_xlabel("Blood 2")
    # ax2.set_ylabel("Disease")
    # ax2.set_title("Blood 2 vs Disease")

    positive = y >= 0
    negative = y < 0
    ax1.scatter(X[positive, 0], X[positive, 1], c='b', marker='o', label="Healthy")
    ax1.scatter(X[negative, 0], X[negative, 1], c='r', marker='x', label="Not Healthy")

    # FIRST THING - Split data, don't leak any info about test set into model
    x_train, x_test, y_train, y_test = split_data(X, y)
    # x_train = add_quadratic_feature(x_train)

    # Prepare data for training
    x_train, x_scale = normalize(x_train)
    y_train, y_scale = normalize(y_train)
    x_train = prepend_bias_term(x_train)

    print("X = ", x_train[0])
    print("Y = ", y_train[0])

    # Test the functions
    # theta = np.array([1, 2, 3])
    # example_x = np.array([[1, 1, 1], [-1, -1, -1]])
    # predictions = predict(example_x, theta)
    # print("Predictions: ", predictions)

    # theta = np.array([0, 0, 0, 0])
    # cst = cost(x_train, theta, y_train)
    # print("Cost :", cst)
    # print("Gradient: ", gradient(x_train, theta, y_train))

    # Perform gradient descent
    theta, min_loss, loss = gradient_descent(x_train, y_train, learn=0.5, iterations=3000, degree=1)
    print("Learned theta = ", theta)
    print("Minimum loss = ", min_loss)

    # Plot the results
    ax2.plot(loss)

    # Prepare test data
    # x_test = add_quadratic_feature(x_test)
    x_test, derp = normalize(x_test)
    y_test, derp = normalize(y_test)
    x_test = prepend_bias_term(x_test)

    test_cost = cost(x_test, theta, y_test)
    print("Cost on test data: ", test_cost)

    X, derp = normalize(X)

    # Plot the results
    ax3.scatter(X[positive, 0], X[positive, 1], c='b', marker='o', label="Healthy")
    ax3.scatter(X[negative, 0], X[negative, 1], c='r', marker='x', label="Not Healthy")

    # Plot Decision Boundary
    db_x = np.linspace(0, 1, 50)
    if len(theta) == 3:
        # Linear
        boundary = -(theta[0] + (theta[1] * db_x)) / theta[2]
        # boundary = -((theta[0] * db_x) + (theta[1] * db_x))
        ax3.scatter(db_x, boundary)
    else:
        z = 2
        # boundary = -(theta[0] + (theta[1] * db_x) + (theta[2] * db_x) + (theta[3] * np.square(db_x)))
        # ax3.scatter(db_x, boundary)

    ax3.legend()
    ax4.legend()
    plt.show()


# Run main
if __name__ == "__main__":
    main()