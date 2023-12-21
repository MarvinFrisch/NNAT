import tensorflow as tf
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import matplotlib.pyplot as plt

input = [t*0.001 for t in range(-1000, 1000)]

max_number_of_neurons = 5
max_exponent = 5

activation_function = "linear"

for exponent in range(max_exponent):
    print(f"x**{exponent}")
    true_output = [i**exponent for i in input]

    plt.figure()
    plt.xlim([-1.2,1.2])
    plt.ylim([-1.2,1.2])
    plt.plot(input, true_output, color='b', label="True")
    for number_of_neurons in range(1,max_number_of_neurons+1):
        print(f"    1-{number_of_neurons}-1")
        

        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)

        model = Sequential()
        model.add(Dense(number_of_neurons, input_shape=(1,), activation=activation_function, kernel_initializer=initializer))
        model.add(Dense(1, activation="tanh"))

        model.compile(loss='mse')

        model.fit(input, true_output, epochs=100, verbose=0)

        print(f"    Weigths: {model.layers[0].get_weights()[0]} , Bias: {model.layers[0].get_weights()[1]}")
        print(f"    Weigths: {model.layers[1].get_weights()[0]} , Bias: {model.layers[1].get_weights()[1]}")

        predicted_output = model.predict([[i] for i in input], verbose=0)

        plt.plot(input, predicted_output, label=f"1-{number_of_neurons}-1")

    plt.legend()
    plt.savefig(f"exponent_{exponent}.png")
    plt.close()
