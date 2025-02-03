import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import to_categorical

import tensorflow as tf

# List available GPUs
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("Available GPUs:", tf.config.list_physical_devices('GPU'))

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build a simple neural network
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=1, batch_size=32, validation_split=0.2)

# Predict on the first 10 test images
predictions = model.predict(x_test[:10])
predicted_labels = predictions.argmax(axis=1)

# Plot the first 10 test images and their predicted labels
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
axes = axes.flatten()

for img, label, ax in zip(x_test[:10], predicted_labels, axes):
    ax.imshow(img, cmap="gray")
    ax.set_title(f"Predicted: {label}")
    ax.axis('off')

plt.tight_layout()
plt.show()
