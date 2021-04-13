# import tensorflow as tf
# print(tf.__version__)
#
# mnist = tf.keras.datasets.fashion_mnist
# (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
# import numpy as np
# import matplotlib.pyplot as plt
# plt.imshow(training_images[0])
# print(training_labels[0])
# print(training_images[0])
#
# training_images  = training_images / 255.0
# test_images = test_images / 255.0
#
# model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
#                                     tf.keras.layers.Dense(128, activation=tf.nn.relu),
#                                     tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
#
# model.compile(optimizer = tf.optimizers.Adam(),
#               loss = 'sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(training_images, training_labels, epochs=5)
# model.evaluate(test_images, test_labels)
# classifications = model.predict(test_images)
# print(classifications[1])

# import numpy as np
# import tensorflow as tf
#
# fashion_mnist= tf.keras.datasets.fashion_mnist
# (training_images, training_labels), (testing_images, testing_labels) = fashion_mnist.load_data()
#
# training_images = training_images/ 255
# testing_images = testing_images/ 255
#
# model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
#     tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)
# ])
#
# model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy',
#               metrics='accuracy')
#
# model.fit(training_images, training_labels, epochs = 5)
# model.evaluate(testing_images, testing_labels)
#
# classification = model.predict(testing_images)
# print(classification[0])

import numpy as np
import tensorflow as tf

fashion_mnist=tf.keras.datasets.fashion_mnist
(training_images, training_labels), (testing_images, testing_labels)=fashion_mnist.load_data()
# Rescale
training_images= training_images/ 255
testing_images = testing_images/ 255
# Model building
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)
])
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(training_images, training_labels, epochs=5)
model.evaluate(testing_images, testing_labels)

classification = model.predict(testing_images)
print(classification[0])

