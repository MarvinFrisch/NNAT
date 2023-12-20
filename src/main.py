import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import matplotlib.pyplot as plt

initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)

model = Sequential()
model.add(Dense(3, input_shape=(1,), activation="sigmoid", kernel_initializer=initializer))
model.add(Dense(1, activation="sigmoid"))

model.compile()

input = [t*0.001 for t in range(0, 1000)]
output = model.predict([[i] for i in input])

plt.figure()
plt.plot(input, output)
plt.savefig("plot.png")
