import numpy as np
import matplotlib.pyplot as plt


# 感知机函数
def perceptron_sgd_plot(X, Y):
    w = np.zeros(len(X[0]))
    eta = 0.8
    epochs = 30
    errors = []
    for t in range(epochs):
        total_error = 0
        for i, x in enumerate(X):
            if (np.dot(X[i], w) * Y[i]) <= 0:
                total_error += (np.dot(X[i], w) * Y[i])
                w = w + eta * X[i] * Y[i]
        errors.append(total_error * -1)
    plt.plot(errors)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.show()
    return w


def main():
    np.random.seed(12)
    num_observations = 500

    x1 = np.random.multivariate_normal([0, 0], [[1, 0.75], [0.75, 1]], num_observations)
    x2 = np.random.multivariate_normal([1, 4], [[1, 0.75], [0.75, 1]], num_observations)
    X = np.vstack((x1, x2)).astype(np.float32)
    y = np.hstack((-np.ones(num_observations), np.ones(num_observations)))

    w = perceptron_sgd_plot(X, y)

    for d, sample in enumerate(X):
        if d < num_observations:
            plt.scatter(sample[0], sample[1], c='r', s=120, marker='_')
        else:
            plt.scatter(sample[0], sample[1], c='b', s=120, marker='+')

    x2 = [w[0], w[1], -w[1], w[0]]
    x3 = [w[0], w[1], w[1], -w[0]]

    x2x3 = np.array([x2, x3])
    X, Y, U, V = zip(*x2x3)
    ax = plt.gca()
    ax.quiver(X, Y, U, V, scale=1, color='blue')
    plt.show()


if __name__ == '__main__':
    main()
