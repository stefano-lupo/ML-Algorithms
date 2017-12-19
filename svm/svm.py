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


def normaliseData(x):
    # rescale data to lie between 0 and 1
    scale = x.max(axis=0)
    return (x/scale, scale)


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


def plotDecisionBoundary(Xt,y,Xscale,theta):
    # plots the training data plus the decision boundary in the model
    fig, ax = plt.subplots(figsize=(12,8))
    # plot the data
    positive = y>0
    negative = y<0
    ax.scatter(Xt[positive,1]*Xscale[1], Xt[positive,2]*Xscale[2], c='b', marker='o', label='Healthy')
    ax.scatter(Xt[negative,1]*Xscale[1], Xt[negative,2]*Xscale[2], c='r', marker='x', label='Not Healthy')
    # calc the decision boundary
    x=np.linspace(Xt[:,2].min()*Xscale[2],Xt[:,2].max()*Xscale[2],50)
    if (len(theta) == 3):
        # linear boundary
        x2 = -(theta[0]/Xscale[0]+theta[1]*x/Xscale[1])/theta[2]*Xscale[2]
    else:
        # quadratic boundary
        x2 = -(theta[0]/Xscale[0]+theta[1]*x/Xscale[1]+theta[3]*np.square(x)/Xscale[3])/theta[2]*Xscale[2]
    # and plot it
    ax.plot(x,x2,label='Decision boundary')
    ax.legend()
    ax.set_xlabel('Test 1')
    ax.set_ylabel('Test 2')
    plt.show()


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
    # x_train, x_test, y_train, y_test = split_data(X, y)

    x_train = X
    y_train = y
    x_train = add_quadratic_feature(x_train)
    x_train = prepend_bias_term(x_train)
    # Prepare data for training
    # x_train, x_scale = normalize(x_train)
    x_train, x_scale = normaliseData(x_train)
    # y_train, y_scale = normalize(y_train)


    print("X = ", x_train[0])
    print("Y = ", y_train[0])

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
    theta, min_loss, loss = gradient_descent(x_train, y_train, learn=0.05, iterations=5000, degree=1, lamb=15)
    print("Learned theta = ", theta)
    print("Minimum loss = ", min_loss)

    # Plot the results
    ax2.plot(loss)
    plt.show()

    plotDecisionBoundary(x_train, y_train, x_scale, theta)

    input("Derp")

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
    # Eg Linear: z = t0 + t1*x1 + t2*x2 where z is height (label)
    # Project onto plane --> let z = 0 and plot with x1 as x access, x2 as y access
    # 0 = t0 + t1*x1 + t2*x2 --> t2*x2 = -(t0 + t1*x1)
    # x2 = -(t0 + t1*x1) / t2 or x2 = -(t0 + t1*x1 + t3 * x1^2) / t2 for quadratic
    db_x = np.linspace(0, 1, 50)
    if len(theta) == 3:
        # Linear
        x2 = -(theta[0] + (theta[1] * db_x) / x_scale[0]) / (theta[2]// x_scale[1])
        # boundary = -((theta[0] * db_x) + (theta[1] * db_x))
        ax3.scatter(db_x, x2)
    else:
        x2 = -(theta[0] + theta[1] * db_x + theta[3] * np.square(db_x)) / theta[2]
        ax3.scatter(db_x, x2)

    ax3.legend()
    ax4.legend()
    plt.show()


# Run main
if __name__ == "__main__":
    main()