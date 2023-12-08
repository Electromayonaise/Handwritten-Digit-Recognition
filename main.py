import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the data
dataset = tf.keras.datasets.mnist
# Split the data into training and testing
(x_train, y_train), (x_test, y_test) = dataset.load_data()
# Normalize the data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
# Build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(x_train, y_train, epochs=3)
# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(accuracy)
print(loss)
# Save the model
model.save('digits.model')
# Load the model
new_model = tf.keras.models.load_model('digits.model')

image_index = 0
while os.path.isfile(f"Digits/Digit{image_index}.png"):
    try:
        img = cv.imread(f"Digits/Digit{image_index}.png")[:, :, 0]
        img = np.invert(np.array([img]))
        prediction = new_model.predict(img)
        print(f"The number is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error")
    finally:
        image_index += 1

