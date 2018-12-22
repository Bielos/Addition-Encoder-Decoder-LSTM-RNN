# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 11:20:23 2018

@author: DANIEL MARTINEZ BIELOSTOTZKY -- github.com/Bielos
"""
import numpy as np
from CharacterTable import CharacterTable
from model import model_build
from load_data import read_additions

# Parameters for the model and dataset.
TRAINING_SIZE = 400000
DIGITS = 4
REVERSE = True
BATCH_SIZE = 128

# Maximum length of input is 'int + int' (e.g., '345+678'). Maximum length of
# int is DIGITS.
MAXLEN = DIGITS + 1 + DIGITS

# All the numbers, plus sign and space for padding.
chars = '0123456789+ '
ctable = CharacterTable(chars)

x_train, y_train, x_val, y_val = read_additions(DIGITS, MAXLEN, chars, ctable)

print('Build model...')
model = model_build(DIGITS, MAXLEN, chars)
model.summary()

# Train the model each generation and show predictions against the validation
# dataset.
for iteration in range(1, 20):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=1,
              validation_data=(x_val, y_val))
    # Select 10 samples from the validation set at random so we can visualize
    # errors.
    for i in range(10):
        ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowx, verbose=0)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q[::-1] if REVERSE else q, end=' ')
        print('T', correct, end=' ')
        if correct == guess:
            print('☑', end=' ')
        else:
            print('☒', end=' ')
        print(guess)

#Save weights
#model.save_weights('addition_model.hdf5')