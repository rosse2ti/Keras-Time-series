import matplotlib.pyplot as plt


def show_heatmap(data):
    plt.matshow(data.corr(numeric_only=True))
    plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.show()


def normalize(data):

    data_mean = data.mean()
    data_std = data.std()
    return ((data - data_mean) / data_std), data_mean, data_std


def deNormalize(data, dataMean, dataStd):
    data = [((i * dataStd) + dataMean) for i in data]
    return data


def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def showTrainingSet(y_train, targ, predictions, testTarg, testPredictions):
    plt.figure()
    plt.plot(
        range(len(y_train)),
        y_train,
        "r",
        label="Training Set",
        linewidth=1,
    )
    plt.plot(
        range(len(y_train) - 1, len(y_train) + len(targ) - 1),
        targ,
        "b",
        label="Validation Set",
        linewidth=1,
    )
    plt.plot(
        range(len(y_train) - 1, len(y_train) + len(predictions) - 1),
        predictions,
        "#be00ed",
        label="Predictions",
        linewidth=1,
    )
    plt.plot(
        range(
            len(y_train) + len(predictions) - 1,
            len(y_train) + len(predictions) + len(testPredictions) - 1,
        ),
        testPredictions,
        "#a4a61f",
        label="testPredictions",
        linewidth=1,
    )
    plt.plot(
        range(
            len(y_train) + len(predictions) - 1,
            len(y_train) + len(predictions) + len(testTarg) - 1,
        ),
        testTarg,
        "g",
        label="testTarg",
        linewidth=1,
    )
    plt.title("Datas")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
