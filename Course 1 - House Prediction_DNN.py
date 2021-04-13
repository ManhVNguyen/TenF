# import tensorflow as tf
# import numpy as np
# from tensorflow import keras
#
# def house_price(y_new):
#   xs = np.array([1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12], dtype = float)
#   ys = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4.5, 5, 5.5, 6, 6.5], dtype = float)
#   model = keras.Sequential([keras.layers.Dense(units = 1, input_shape = [1])])
#   model.compile(optimizer='sgd', loss = 'mean_squared_error')
#   model.fit(xs, ys, epochs = 2000)
#   return model.predict(y_new)
#
# prediction = house_price([7])
# print(prediction)
#
# ###############################################################################
# ###############################################################################
# ###############################################################################
# # Solution 2
# import tensorflow as tf
# import numpy as np
# from tensorflow import keras
#
# xs = np.array([1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12], dtype = float)
# ys = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4.5, 5, 5.5, 6, 6.5], dtype = float)
# model = keras.Sequential([keras.layers.Dense(units = 1, input_shape = [1])])
# model.compile(optimizer='sgd', loss = 'mean_squared_error')
# model.fit(xs, ys, epochs = 2000)
# prediction = model.predict([7])

import numpy as np
import tensorflow as tf
from tensorflow import keras


def house_price(y_new):
  xs=np.array([1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12], dtype=float)
  ys=np.array([1, 1.5, 2, 2.5, 3, 3.5, 4.5, 5, 5.5, 6, 6.5], dtype=float)
  model = tf.keras.Sequential([
   tf.keras.layers.Dense(units=1, input_shape=[1])
  ])
  model.compile(optimizer='sgd', loss='mean_squared_error')
  model.fit(xs, ys, epochs = 500)

  return model.predict(y_new)


prediction = house_price([7])
print(prediction)




