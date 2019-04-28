#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 08:46:12 2019

@author: luohaozheng
"""

import numpy as np
np.random.seed(45)
import tensorflow as tf
tf.set_random_seed(45)
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout
#from keras.layers.core import  Dropout, RepeatVector
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pickle

import heapq



    
def train(SEQUENCE_LENGTH,chars):    
    model = Sequential()
    model.add(LSTM(64, input_shape=(SEQUENCE_LENGTH, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(X, y, validation_split=0.15, batch_size=64, epochs=8, shuffle=True).history
    model.save('keras_model.h5')
    pickle.dump(history, open("history.p", "wb"))


def simulation():
    model = load_model('keras_model.h5')
    history = pickle.load(open("history.p", "rb"))
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left');
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('acc')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left');
    


def prepare_input(text):
    y = np.zeros((1, SEQUENCE_LENGTH, len(chars)))
    for t, char in enumerate(text):
        y[0, t, char_indices[char]] = 1.
        
    return y



def sample(pred, top_n=3):
    pred = np.asarray(pred).astype('float64')
    pred = np.log(pred)
    exp = np.exp(pred)
    pred = exp / np.sum(exp)
    
    return heapq.nlargest(top_n, range(len(pred)), pred.take)

def predict_completion(text):
    model = load_model('keras_model.h5')
    original_text = text
    generated = text
    completion = ''
    while True:
        x = prepare_input(text)
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, top_n=1)[0]
        next_char = indices_char[next_index]
        text = text[1:] + next_char
        completion += next_char
        
        if len(original_text + completion) + 2 > len(original_text) and next_char == ' ':
            return completion
        
def predict_completions(text, n=3):
    model = load_model('keras_model.h5')
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [indices_char[idx] + predict_completion(text[1:] + indices_char[idx]) for idx in next_indices]



path = '3student.txt'
text = open(path).read().lower()
print('corpus length:', len(text))
chars = sorted(list(set(text)))
char_indices=dict()
indices_char=dict()
for i, c in enumerate(chars):
    char_indices[c]= i
    indices_char[i]= c
print(f'unique chars: {len(chars)}')
    
SEQUENCE_LENGTH = 50
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - SEQUENCE_LENGTH, step):
    sentences.append(text[i: i + SEQUENCE_LENGTH])
    next_chars.append(text[i + SEQUENCE_LENGTH])
print(f'num training examples: {len(sentences)}')
X = np.zeros((len(sentences), SEQUENCE_LENGTH, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
    
train(SEQUENCE_LENGTH,chars)
simulation()
quotes = [
    "It is not a lack of love, but a lack of friendship that makes unhappy marriages.",
    "That which does not kill us makes us stronger.",
    "I'm not upset that you lied to me, I'm upset that from now on I can't believe you.",
    "And those who were seen dancing were thought to be insane by those who could not hear the music.",
    "It is hard enough to remember my opinions, without also remembering my reasons for them!"
]
for q in quotes:
    seq = q[:40].lower()
    print(seq)
    print(predict_completions(seq, 5))
    print()


