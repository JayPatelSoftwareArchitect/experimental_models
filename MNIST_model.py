import numpy as np
import tensorflow as tf
import tensorflow.keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import pathlib
from NonLinearActivations import CustomNonlinearActivation
#This code snipet is for testing different non linear activation functions, and custom complex activation functions.
#For first training iteration, It will use default activation functions like relu and softmax, after that it will use my custom activation functions.
#Initial testing shows promise for good custom activation functions. Maybe i will write few papers on it.

#download mnist data set.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Rescale the images from [0,255] to the [0.0,1.0] range.
x_train, x_test = x_train/255.0, x_test/255.0


print("Number of original training examples:", len(x_train))
print("Number of original test examples:", len(x_test))

#A dynamic model test.
def test_model(activationfn,epochs=2):
    #sequential api
    model = tf.keras.Sequential([
        #Dense layer 1
        tf.keras.layers.Dense(units=20,activation=activationfn),
        #apply small dropout
        tf.keras.layers.Dropout(0.01),
        #Dense layer 2
        tf.keras.layers.Dense(units=20,activation=activationfn),
        #Flatten the last layer input
        tf.keras.layers.Flatten(),
        #apply output layer, Here we have 10 output labels ranging from 0-9 
        tf.keras.layers.Dense(units=10),
    ])
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.fit(x_train, y_train, epochs=epochs)
    model.evaluate(x_test,y_test)

# A helper function to iterate through activation functions.
def testActivations(epochs=2):
    custom_functions = CustomNonlinearActivation()
 
    activation_dict = [
                    #relu, only account values greater then 0.
                    tf.keras.activations.relu,
                    #softmax, probabilistic activation
                    tf.keras.activations.softmax,
                    #following are custom activation functions
                    #This functions performs well with almost same accuracy as relu
                    #log(1+a) + e^|ib| 
                    custom_functions.getActivationFunction('add_c3'),
                    #e^a - e^|ib|
                    custom_functions.getActivationFunction('sub_c2'),
                    #e^a * e^|ib|
                    custom_functions.getActivationFunction('mul_c2'),
                    #log(1+a) / e^|ib| 
                    custom_functions.getActivationFunction('div_c3'),
                    ]
    
    for index in range(len(activation_dict)):
        test_model(activation_dict[index], epochs=epochs)

#call dynamic model tests, here testactivations function will go through few activation functions and it will train a sample model. 
testActivations(epochs=10)

