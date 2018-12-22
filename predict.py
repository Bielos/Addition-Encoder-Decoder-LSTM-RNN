# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 11:42:41 2018

@author: DANIEL MARTINEZ BIELOSTOTZKY -- github.com/Bielos
"""
import numpy as np
from CharacterTable import CharacterTable
from model import model_build

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

print('Load model...')
checkpoint = "addition_model.hdf5"
model = model_build(DIGITS, MAXLEN, chars, checkpoint=checkpoint)
model.summary()

# Predict addition provided by user
while(True):
    print('-' * 50)
    print('ATTENTION: ONLY {0} DIGITS MAX FOR EACH NUMBER, EXAMPLE: 4444+10'.format(DIGITS))
    sentence = input('sentence: ')
    _sentence = sentence + ' ' * (MAXLEN - len(sentence))
    _sentence = _sentence[::-1]
    x = ctable.encode(_sentence, MAXLEN)
    preds = model.predict_classes(np.array([x]), verbose=0)
    guess = ctable.decode(preds[0], calc_argmax=False)
    print(sentence + ' = ' + guess)
