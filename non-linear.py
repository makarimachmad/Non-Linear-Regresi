# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:01:23 2020

@author: FUJITSU
"""

import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf

from keras.models import Model
from tensorflow.keras.models import Sequential, load_model
from keras.layers import Input, Activation, Dense
from keras.optimizers import SGD



# ------  membuat dataset  -------
# Generate data from -20, -19.75, -19.5, .... , 20
train_x = np.arange(-20, 20, 0.25)

# Calculate Target : sqrt(2x^2 + 1)
train_y = np.sqrt((2*train_x**2)+1) #contoh ini menggunakan nilai 26
print(train_y)

# ------  membuat model  -------
# menggunakan SGD dan Mean Squared Error (MSE) sebagai loss functionnya
# Create Network
inputs = Input(shape=(1,))
h_layer = Dense(8, activation='relu')(inputs)
h_layer = Dense(4, activation='relu')(h_layer)
outputs = Dense(1, activation='linear')(h_layer)
model = Model(inputs=inputs, outputs=outputs)

# Optimizer / Update Rule
sgd = SGD(lr=0.001)
# Compile the model with Mean Squared Error Loss
model.compile(optimizer=sgd, loss='mse')


# ------  training data  ------
# Train the network and save the weights after training
model.fit(train_x, train_y, batch_size=20, epochs=10000, verbose=1)
model.save_weights('weights.h5')


# -----  prediksi  -------
# Predict training data
predict = model.predict(np.array([26]))
print('f(26) = ', predict)

predict_y = model.predict(train_x)

# Visualisasi perbandingan prediksi dengan nilai target
plt.plot(train_x, train_y, 'r')
plt.plot(train_x, predict_y, 'b')
plt.show()