from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def train_ann_model(epochs=5, batch_size=32, hidden_units=128, activation='relu', optimizer='adam'):
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    print("Training data shape:", X_train.shape)
    print("Training labels shape:", y_train.shape)
    print("Test data shape:", X_test.shape)
    print("Test labels shape:", y_test.shape)
    labels, count = np.unique(y_train, return_counts=True)  # Shows first digit
    print("Unique labels:", labels)
    print("Counts:", count)
    # img = Image.fromarray(X_train[0])
    # img.show()  # Displays the first digit image

    X_train = X_train / 255
    X_test = X_test / 255

    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(hidden_units, activation=activation))
    model.add(Dense(10, activation='softmax'))

    model.compile(
        optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy']
    )

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    test_loss, test_acc = model.evaluate(X_test, y_test)

    return model, history, X_train, y_train, X_test, y_test, test_loss, test_acc