# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 11:12:14 2018

@author: DANIEL MARTINEZ BIELOSTOTZKY -- github.com/Bielos
"""

from keras.models import Sequential
from keras import layers

def model_build(DIGITS, MAXLEN, chars, checkpoint=''):
    RNN = layers.LSTM
    HIDDEN_SIZE = 128
    LAYERS = 1
    model = Sequential()
    # "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
    # Note: In a situation where your input sequences have a variable length,
    # use input_shape=(None, num_feature).
    model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))
    # As the decoder RNN's input, repeatedly provide with the last output of
    # RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
    # length of output, e.g., when DIGITS=3, max output is 999+999=1998.
    model.add(layers.RepeatVector(DIGITS + 1))
    # The decoder RNN could be multiple layers stacked or a single layer.
    for _ in range(LAYERS):
        # By setting return_sequences to True, return not only the last output but
        # all the outputs so far in the form of (num_samples, timesteps,
        # output_dim). This is necessary as TimeDistributed in the below expects
        # the first dimension to be the timesteps.
        model.add(RNN(HIDDEN_SIZE, return_sequences=True))
    
    # Apply a dense layer to the every temporal slice of an input. For each of step
    # of the output sequence, decide which character should be chosen.
    model.add(layers.TimeDistributed(layers.Dense(len(chars), activation='softmax')))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    if checkpoint != '':
        model.load_weights(checkpoint)
    
    return model
