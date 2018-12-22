# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 11:36:21 2018

@author: DANIEL MARTINEZ BIELOSTOTZKY -- github.com/Bielos
"""

import pandas as pd
import numpy as np

def read_additions(DIGITS, MAXLEN, chars, ctable):
    print('Reading data...')
    
    data = pd.read_csv('additions.csv', dtype={'question':str, 'answer':str})
    questions = data['question'].astype(str).values
    expected = data['answer'].astype(str).values
    
    print('Total addition questions:', len(questions))
    
    print('Vectorization...')
    x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
    y = np.zeros((len(questions), DIGITS + 1, len(chars)), dtype=np.bool)
    for i, sentence in enumerate(questions):
        x[i] = ctable.encode(sentence, MAXLEN)
    for i, sentence in enumerate(expected):
        y[i] = ctable.encode(sentence, DIGITS + 1)
    
    # Shuffle (x, y) in unison as the later parts of x will almost all be larger
    # digits.
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]
    
    # Explicitly set apart 10% for validation data that we never train over.
    split_at = len(x) - len(x) // 10
    (x_train, x_val) = x[:split_at], x[split_at:]
    (y_train, y_val) = y[:split_at], y[split_at:]
    
    print('Training Data:')
    print(x_train.shape)
    print(y_train.shape)
    
    print('Validation Data:')
    print(x_val.shape)
    print(y_val.shape)
    
    return x_train, y_train, x_val, y_val
