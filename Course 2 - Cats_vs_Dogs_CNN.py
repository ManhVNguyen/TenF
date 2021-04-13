import os

base_dir = "C:\\Users\\manhnguyen\\PycharmProjects\\ManhOnThi\\data\\cats_and_dogs_filtered"

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')


train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')


# Dua tat ca labels nay vao trong 1 group list
train_cats_names = os.listdir(train_cats_dir)
train_dogs_names = os.listdir(train_dogs_dir)

validation_cats_names = os.listdir(validation_cats_dir)
validation_dogs_names = os.listdir(validation_dogs_dir)

print('total training cat images:', len(train_cats_names))
print('total training dog images:', len(train_dogs_names))
print('total validation cat images', len(validation_cats_names))
print('total validation dog images', len(validation_dogs_names))

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

nrows = 4
ncols = 4

pic_index = 0 # Index for iterating over image

fig = plt.gcf() # lam cho hien ra cai frame anh rong
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_cat_pix = [os.path.join(train_cats_dir, fname)
                for fname in train_cats_names[pic_index-8: pic_index]
                ]

next_dog_pix = [os.path.join(train_dogs_dir, fname)
                for fname in train_dogs_names[pic_index-8: pic_index]
                ]

for i, img_path in enumerate(next_cat_pix + next_dog_pix):
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis('Off')

    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()


import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              loss= 'binary_crossentropy',
              metrics='accuracy')

model.summary()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    target_size=(150,150),
                                                    class_mode = 'binary')

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                               target_size=(150,150),
                                                              class_mode='binary',
                                                              batch_size= 20)

history = model.fit_generator(train_generator,
          validation_data = validation_generator,
          steps_per_epoch= 100,
          epochs = 5,
          validation_steps = 50,
          verbose = 2)

from tensorflow.keras.preprocessing import image
test_image = "C:\\Users\\manhnguyen\\Desktop\\cat2.jpg"
import numpy as np

img = image.load_img(test_image, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])

classes = model.predict(images, batch_size=10)
print(classes[0])

if classes[0] > 0:
    print("This is a dog")
else:
    print("This is a cat")

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')

plt.figure()
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')



# VISUALIZE INTERMEDIATE REPRESENTATIONS




