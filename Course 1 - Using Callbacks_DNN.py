# import tensorflow as tf
#
# import tensorflow as tf
#
# class myCallback(tf.keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs={}):
#     if(logs.get('accuracy')>0.6):
#       print("\nReached 60% accuracy so cancelling training!")
#       self.model.stop_training = True
#
# mnist = tf.keras.datasets.fashion_mnist
#
# (x_train, y_train),(x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0
#
# callbacks = myCallback()
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(512, activation=tf.nn.relu),
#   tf.keras.layers.Dense(10, activation=tf.nn.softmax)
# ])
#
# model.compile(optimizer=tf.optimizers.Adam(),
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])

import tensorflow as tf
from tensorflow import keras

fashion_mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (testing_images, testing_labels) = fashion_mnist.load_data()

training_images = training_images / 255
testing_images = testing_images / 255


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 90:
            print('\nReached 60% accuracy so cancelling training')
            self.model.stop_training = True


callback = myCallback()


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = tf.keras.activations.relu),
    tf.keras.layers.Dense(10, activation = tf.keras.activations.softmax)
])

model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(training_images, training_labels, epochs=5, callbacks= [callback])

