from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Training data shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Test data shape:", X_test.shape)
print("Test labels shape:", y_test.shape)
labels, count = np.unique(y_train, return_counts=True)  # Shows first digit
print("Unique labels:", labels)
print("Counts:", count)
img = Image.fromarray(X_train[0])
img.show()  # Displays the first digit image

X_train = X_train/255
X_test = X_test/255

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(
optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']
)

history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

model.evaluate(X_test, y_test)

plt.figure(figsize=(10, 5))

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend() # hm ne jo labels likhe hn un ko show krta ha.R
plt.show()

