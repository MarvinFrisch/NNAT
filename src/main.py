import tensorflow as tf
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import matplotlib.pyplot as plt

input = [t*0.001 for t in range(0, 1000)]

max_number_of_neurons = 20
number_of_passes = 20

for number_of_neurons in range(1,max_number_of_neurons+1):
    plt.figure()
    for pass_number in range(number_of_passes):
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=10.)

        model = Sequential()
        model.add(Dense(number_of_neurons, input_shape=(1,), activation="sigmoid", kernel_initializer=initializer))
        model.add(Dense(1, activation="sigmoid"))

        model.compile()

        output = model.predict([[i] for i in input])

        min_output = min(output)
        max_output= max(output)
        output = [(i-min_output)/(max_output-min_output) for i in output]

        plt.plot(input, output)
    plt.savefig(f"1_Hidden_with_{number_of_neurons}_neurons.png")
    plt.close()
