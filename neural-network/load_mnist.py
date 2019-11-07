import sys
import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork

sys.path.append('mnist')
import mnist

train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()

n_train, w, h = train_images.shape
X_train = train_images.reshape((n_train, w * h))
y_train = train_labels

n_test, w, h = test_images.shape
X_test = test_images.reshape((n_test, w * h))
y_test = test_labels

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


def costplt1(nn):
    plt.plot(range(len(nn.cost_)), nn.cost_)
    plt.ylim([0, 2000])
    plt.ylabel('Cost')
    plt.xlabel('Epochs * 50')
    plt.tight_layout()
    plt.show()


def costplt2(nn):
    """代价函数图象"""
    batches = np.array_split(range(len(nn.cost_)), 1000)
    cost_ary = np.array(nn.cost_)
    cost_avgs = [np.mean(cost_ary[i]) for i in batches]

    plt.plot(range(len(cost_avgs)), cost_avgs, color='red')
    plt.ylim([0, 10000])
    plt.ylabel('Cost')
    plt.xlabel('Epochs')
    plt.tight_layout()
    plt.show()


def main():
    nn = NeuralNetwork(n_output=10,
                       n_features=X_train.shape[1],
                       n_hidden=50,
                       l2=0.1,
                       l1=0.0,
                       epochs=1000,
                       eta=0.001,
                       alpha=0.001,
                       decrease_const=0.00001,
                       shuffle=True,
                       minibatches=50,
                       random_state=1)
    nn.fit(X_train, y_train)
    costplt1(nn)
    costplt2(nn)
    y_train_pred = nn.predict(X_train)
    acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
    print('训练准确率: %.2f%%' % (acc * 100))

    y_test_pred = nn.predict(X_test)
    acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
    print('测试准确率: %.2f%%' % (acc * 100))

    # 错误样本
    miscl_img = X_test[y_test != y_test_pred][:25]
    correct_lab = y_test[y_test != y_test_pred][:25]
    miscl_lab = y_test_pred[y_test != y_test_pred][:25]
    fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True, )
    ax = ax.flatten()
    for i in range(25):
        img = miscl_img[i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
        ax[i].set_title('%d) t: %d p: %d' % (i + 1, correct_lab[i], miscl_lab[i]))
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()

    # 正确样本
    unmiscl_img = X_test[y_test == y_test_pred][:25]
    uncorrect_lab = y_test[y_test == y_test_pred][:25]
    unmiscl_lab = y_test_pred[y_test == y_test_pred][:25]
    fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True, )
    ax = ax.flatten()
    for i in range(25):
        img = unmiscl_img[i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
        ax[i].set_title('%d) t: %d p: %d' % (i + 1, uncorrect_lab[i], unmiscl_lab[i]))
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
