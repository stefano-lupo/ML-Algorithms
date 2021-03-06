import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


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


def normalize_data(x):
    # rescale data to lie between 0 and 1
    scale = x.max(axis=0)
    return x/scale, scale


def add_quadratic_feature(X):
    return np.column_stack((X, X[:, 0] ** 2))


def predict(data, theta, degree=1):
    xtt = np.dot(data ** degree, theta)
    return np.where(xtt < 0, -1, 1)


def cost(X, theta, actual, lamb=2):
    """
    J(Theta) = average( max(0, 1-(yi*thetaT*xi)) ) + lambda*thetaT*theta
    """
    reg = lamb * np.dot(theta, theta)
    lin_comb = np.dot(X, theta)
    lin_comb *= actual
    lin_comb = 1 - lin_comb
    max = np.maximum(np.zeros(X.shape[0]), lin_comb)

    return np.mean(max) + reg


def gradient(X, theta, actual, lamb=0):
    """
    2*lamb*theta_j - average(yi * xij if(yi*thetaT*xi <= 1) )
    :param X:
    :param theta:
    :param actual:
    :return:
    """

    reg = 2*lamb * theta        # Nx1
    lc = np.dot(X, theta)       # Mx1
    condition = actual * lc     # Mx1
    condition = condition <= 1  # Mx1
    left = (X.T * actual).T     # MxN
    inner = (left.T * condition).T  # MxN

    return reg - np.mean(inner, axis=0)     #Nx1


def gradient_descent(x_train, y_train, learn=0.02, iterations=1000, threshold=0.1, degree=1, lamb=0):
    print("Running Gradient Descent")
    theta = np.array(np.random.rand(x_train.shape[1]))
    loss = []
    for i in range(iterations):

        # Get new predictions
        predictions = predict(x_train, theta, degree=degree)

        # Get the loss assoc with these values of theta
        loss.append(cost(x_train, theta, y_train))

        # Check if within threshold

        # Get the gradient
        grad = gradient(x_train, theta, y_train, lamb=lamb)

        # Update theta
        theta = theta - learn * grad

    return theta, np.min(loss), loss


def train_test(X, y):
    # Split data
    x_train, x_test, y_train, y_test = split_data(X, y)

    # Adjust Training Data
    x_train = add_quadratic_feature(x_train)
    x_train, x_scale = normalize_data(x_train)
    x_train = prepend_bias_term(x_train)

    # Adjust Testing Data
    x_test = add_quadratic_feature(x_test)
    x_test, x_scale_test = normalize_data(x_test)
    x_test = prepend_bias_term(x_test)

    return x_train, y_train, x_test, y_test, x_scale


def training_only(X, y):
    x_train = add_quadratic_feature(X)
    x_train = prepend_bias_term(x_train)
    x_train, x_scale = normalize_data(x_train)
    y_train = y
    # y_train, y_scale = normalize(y_train)
    x_test = x_train
    y_test = y_train
    return x_train, y_train, x_test, y_test, x_scale

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

    x_train, y_train, x_test, y_test, x_scale = training_only(X, y)
    # x_train, y_train, x_test, y_test, x_scale = train_test(X, y)


    # Test the functions
    # theta = np.array([1, 2, 3])
    # example_x = np.array([[1, 1, 1], [-1, -1, -1]])
    # predictions = predict(example_x, theta)
    # print("Predictions: ", predictions)
    #
    # theta = np.array([0, 0, 0, 0])
    # cst = cost(x_train, theta, y_train)
    # print("Cost :", cst)
    # print("Gradient: ", gradient(x_train, theta, y_train, lamb=0))

    # Perform gradient descent
    theta, min_loss, loss = gradient_descent(x_train, y_train, learn=0.05, iterations=5000, lamb=1)
    print("Learned theta = ", theta)
    print("Minimum loss = ", min_loss)

    # Plot the results
    ax4.plot(loss)

    test_cost = cost(x_test, theta, y_test)
    print("Cost on test data: ", test_cost)

    predictions = predict(x_test, theta)
    print(classification_report(y_test, predictions))

    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy on Test: ", accuracy)

    # Plot Decision Boundary
    # Eg Linear: z = t0 + t1*x1 + t2*x2 where z is height (label)
    # Project onto plane --> let z = 0 and plot with x1 as x access, x2 as y access
    # 0 = t0 + t1*x1 + t2*x2 --> t2*x2 = -(t0 + t1*x1)
    # x2 = -(t0 + t1*x1) / t2 or x2 = -(t0 + t1*x1 + t3 * x1^2) / t2 for quadratic
    db_x = np.linspace(-1, 1, 50)
    if len(theta) == 3:
        # Linear
        x2 = -(theta[0] + ((theta[1] * db_x) / x_scale[0])) / (theta[2] / x_scale[1])
        # boundary = -((theta[0] * db_x) + (theta[1] * db_x))
        ax3.scatter(db_x, x2, label="Boundary", c='g')
    else:
        x2 = -(theta[0] + theta[1] * db_x + theta[3] * np.square(db_x)) / theta[2]
        ax3.scatter(db_x, x2, label="Boundary", c='g')

    ax3.legend()
    ax4.legend()
    plt.show()


# Run main
if __name__ == "__main__":
    main()