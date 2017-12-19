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
    return (X - min) / (max - min)


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

    # Plot the results
    plt.plot(loss)
    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    plt.show()
    min_loss = min(loss)

    return theta, min_loss

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
    ax1.scatter(X[:, 0], y, label="Data")
    ax1.set_xlabel("Blood 1")
    ax1.set_ylabel("Disease")
    ax1.set_title("Blood 1 vs Disease")

    ax2.scatter(X[:, 0], y, label="Data")
    ax2.set_xlabel("Blood 2")
    ax2.set_ylabel("Disease")
    ax2.set_title("Blood 2 vs Disease")

    positive = y >= 0
    negative = y < 0
    ax3.scatter(X[positive, 0], X[positive, 1], c='b', marker='o', label="Healthy")
    ax3.scatter(X[negative, 0], X[negative, 1], c='r', marker='x', label="Not Healthy")

    ax3.legend()
    plt.show()

    degree = 1

    # FIRST THING - Split data, don't leak any info about test set into model
    # x_train, x_test, y_train, y_test = split_data(X, y)
    x_train = X
    y_train = y

    # Prepare data for training
    # x_train = normalize(x_train)
    # y_train = normalize(y_train)

    x_train = add_quadratic_feature(x_train)
    x_train = prepend_bias_term(x_train)

    x_train = normalise_data(x_train)
    # y_train = normalise_data(y_train)

    print("X = ", x_train[0])
    print("Y = ", y_train[0])


    # Test the functions
    theta = np.array([1, 2, 3])
    example_x = np.array([[1, 1, 1], [-1, -1, -1]])
    predictions = predict(example_x, theta)
    print("Predictions: ", predictions)

    theta = np.array([0, 0, 0, 0])
    # print(predictions)
    cst = cost(x_train, theta, y_train)
    print("Cost :", cst)

    print("Gradient: ", gradient(x_train, theta, y_train))

    # Perform gradient descent
    theta, min_loss = gradient_descent(x_train, y_train, learn=0.1, iterations=3000, degree=1)
    print("Learned theta = ", theta)
    print("Minimum loss = ", min_loss)


    """
   # Prepare test data
   x_test = normalize(x_test)
   y_test = normalize(y_test)
   x_test = prepend_bias_term(x_test)

   # Plot the results
   axes = plt.gca()
   if x_train.shape[1] == 2:

       # Plot the training data and the fit
       fit_plot = np.column_stack((np.ones(50), np.linspace(np.min(x_train), np.max(x_train), 50)))
       fitted = predict(fit_plot, theta, degree=degree)
       plt.scatter(fit_plot[:, 1], fitted, label="Fitted")
       plt.scatter(x_train[:, 1], y_train, label="Predicted")
       plt.xlabel("x_train")
       plt.ylabel("y_train")
       axes.set_title("Fit to training data")
       plt.show()

       # Predict the test values
       predicted = predict(x_test, theta, degree=degree)
       loss = cost(predicted, y_test)
       print("Loss on test = ", loss)

       # Plot the predictions against the actual
       plt.scatter(x_test[:, 1], predicted, label="Predicted")
       plt.scatter(x_test[:, 1], y_test, label="Actual")
       plt.xlabel("X")
       plt.ylabel("Y")
       axes.set_title("Input vs Output")
   else:
       # Predict the test values
       predicted = predict(x_test, theta, degree=degree)
       loss = cost(predicted, y_test)
       print("Loss on test = ", loss)

       axes.set_title("Predicted vs Actual (should be straight line)")
       plt.scatter(predicted, y_test)
       plt.xlabel("Predicted")
       plt.ylabel("Actual")

   plt.legend()
   plt.show()


   # Plot the cost function in 3d
   # fig3 = plt.figure()
   # ax3 = fig3.add_subplot(1, 1, 1, projection='3d')
   # n=100
   # theta0, theta1 = np.meshgrid(np.linspace(-500, 500, n), np.linspace(-500, 500, n))
   # cost = np.empty((n, n))
   # for i in range(n):
   #     for j in range(n):
   #         predicted = predict(x_test, [theta0[i, j], theta1[i, j]])
   #         cost[i,j] = l2_cost(predicted, y_test)
   #
   # ax3.plot_surface(theta0, theta1, cost)
   # ax3.set_xlabel('theta0')
   # ax3.set_ylabel('theta1')
   # ax3.set_zlabel('J(theta)')
   # plt.show()
   """

# Run main
if __name__ == "__main__":
    main()