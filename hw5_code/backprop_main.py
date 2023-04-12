import backprop_data

import backprop_network
import matplotlib.pyplot as plt


def article_b():
    training_data, test_data = backprop_data.load(train_size=10000, test_size=5000)
    rates = [0.001, 0.01, 0.1, 1, 10, 100]
    epochs_num = 30
    epochs = list(range(epochs_num))
    rate_to_stats = {}
    for rate in rates:
        net = backprop_network.Network([784, 40, 10])
        stats = net.SGD(
            training_data,
            epochs=epochs_num,
            mini_batch_size=10,
            learning_rate=rate,
            test_data=test_data,
            calc_stats=True,
        )
        rate_to_stats[rate] = (stats[:, 0], stats[:, 1], stats[:, 2])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    for rate, (test_acc, train_acc, train_loss) in rate_to_stats.items():
        ax1.plot(epochs, test_acc, label=f"{rate=}")
        ax1.set_ylabel("test accuracy")
        ax2.plot(epochs, train_acc, label=f"{rate=}")
        ax2.set_ylabel("train accuracy")
        ax3.plot(epochs, train_loss, label=f"{rate=}")
        ax3.set_ylabel("train loss")
        for ax in (ax1, ax2, ax3):
            ax.set_xlabel("epochs")

    plt.legend(title="rates")
    plt.show()


def article_c():
    training_data, test_data = backprop_data.load(train_size=50000, test_size=10000)
    net = backprop_network.Network([784, 50, 10])
    net.SGD(
        training_data,
        epochs=100,
        mini_batch_size=10,
        learning_rate=0.1,
        test_data=test_data,
        calc_stats=False,
    )


if __name__ == "__main__":
    print("article c")
    article_c()
