import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn import svm


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
    return (x/scale, scale)


def add_quadratic_feature(X):
    return np.column_stack((X, X[:, 0] ** 2))


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
    # plt.show()


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


def plot_gaussian_boundary(Xt,y,Xscale,model):
    # plots the training data plus the decision boundary when using a kernel SVM
    fig, ax = plt.subplots(figsize=(12,8))
    # plot the data
    positive = y>0
    negative = y<0
    ax.scatter(Xt[positive,1]*Xscale[1], Xt[positive,2]*Xscale[2], c='b', marker='o', label='Healthy')
    ax.scatter(Xt[negative,1]*Xscale[1], Xt[negative,2]*Xscale[2], c='r', marker='x', label='Not Healthy')
    # calc the decision boundary
    x_min, x_max = Xt.min() - 0.1, Xt.max() + 0.1
    y_min, y_max = y.min() - 0.1, y.max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
    Z = model.predict(np.column_stack((np.ones(xx.ravel().shape),xx.ravel(), yy.ravel(), np.square(xx.ravel()))))
    Z = Z.reshape(xx.shape)
    ax.contour(xx*Xscale[1], yy*Xscale[2], Z)
    ax.legend()
    ax.set_xlabel('Test 1')
    ax.set_ylabel('Test 2')


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

    x_train, y_train, x_test, y_test, x_scale = training_only(X, y)
    # x_train, y_train, x_test, y_test, x_scale = train_test(X, y)

    model = svm.SVC(C=1, kernel="linear")
    model.fit(x_train, y_train)

    theta = np.concatenate((model.intercept_, (model.coef_.ravel())[1:4]))
    print("Learned Theta: ", theta)
    # Plot the results
    # ax2.plot(loss)
    # plt.show()

    predictions = model.predict(x_test)
    print(classification_report(y_test, predictions))

    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy on Test: ", accuracy)

    # Plot the results
    negative = y < 0
    positive = y > 0

    ax3.scatter(X[positive, 0], X[positive, 1], c='b', marker='o', label="Healthy")
    ax3.scatter(X[negative, 0], X[negative, 1], c='r', marker='x', label="Not Healthy")


    # Plot Decision Boundary
    # Eg Linear: z = t0 + t1*x1 + t2*x2 where z is height (label)
    # Project onto plane --> let z = 0 and plot with x1 as x access, x2 as y access
    # 0 = t0 + t1*x1 + t2*x2 --> t2*x2 = -(t0 + t1*x1)
    # x2 = -(t0 + t1*x1) / t2 or x2 = -(t0 + t1*x1 + t3 * x1^2) / t2 for quadratic
    db_x = np.linspace(-1, 1, 50)
    if len(theta) == 3:
        # Linear
        x2 = -(theta[0] + ((theta[1] * db_x) / x_scale[0]) + x_offset[0]) / ((theta[2] / x_scale[1]) + x_offset[1])
        # boundary = -((theta[0] * db_x) + (theta[1] * db_x))
        ax3.scatter(db_x, x2, label="Boundary", c='g')
    else:
        x2 = -(theta[0] + theta[1] * db_x + theta[3] * np.square(db_x)) / theta[2]
        ax3.scatter(db_x, x2, label="Boundary", c='g')

    ax3.legend()
    ax4.legend()
    plt.show()

    # Training with a Gaussian Kernel (RBF)
    model = svm.SVC(C=0.5, gamma=0.75, kernel='rbf')
    model.fit(x_train, y_train)
    plot_gaussian_boundary(x_train, y_train, x_scale, model)
    predictions = model.predict(x_train)
    accuracy = accuracy_score(y_train, predictions)
    print("Accuracy using gaussian kernel = ", accuracy)
    plt.show()


# Run main
if __name__ == "__main__":
    main()