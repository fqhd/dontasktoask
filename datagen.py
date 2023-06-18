import tensorflow as tf
from tensorflow import keras
import numpy as np
import random

class DataGenerator(keras.utils.Sequence):
    def __init__(self, templates, messages, n_channels=5, sequence_length=50, batch_size=32, shuffle=True):
        self.n_channels = n_channels
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.batch_half_size = batch_size // 2
        self.shuffle=shuffle
        self.character_mappings = {
          'a': [0, 0, 0, 0, 1],
          'b': [0, 0, 0, 1, 0],
          'c': [0, 0, 0, 1, 1],
          'd': [0, 0, 1, 0, 0],
          'e': [0, 0, 1, 0, 1],
          'f': [0, 0, 1, 1, 0],
          'g': [0, 0, 1, 1, 1],
          'h': [0, 1, 0, 0, 0],
          'i': [0, 1, 0, 0, 1],
          'j': [0, 1, 0, 1, 0],
          'k': [0, 1, 0, 1, 1],
          'l': [0, 1, 1, 0, 0],
          'm': [0, 1, 1, 0, 1],
          'n': [0, 1, 1, 1, 0],
          'o': [0, 1, 1, 1, 1],
          'p': [1, 0, 0, 0, 0],
          'q': [1, 0, 0, 0, 1],
          'r': [1, 0, 0, 1, 0],
          's': [1, 0, 0, 1, 1],
          't': [1, 0, 1, 0, 0],
          'u': [1, 0, 1, 0, 1],
          'v': [1, 0, 1, 1, 0],
          'w': [1, 0, 1, 1, 1],
          'x': [1, 1, 0, 0, 0],
          'y': [1, 1, 0, 0, 1],
          'z': [1, 1, 0, 1, 0],
          ' ': [1, 1, 0, 1, 1],
        }
        with open('greetings.txt', 'r') as file:
            self.greetings = file.read().split('\n')
        with open('verbs.txt', 'r') as file:
            self.verbs = file.read().split('\n')
        with open('technologies.txt', 'r') as file:
            self.technologies = file.read().split('\n')
        with open('subjects.txt', 'r') as file:
            self.subjects = file.read().split('\n')
        with open(templates, 'r', encoding='utf-8') as file:
            self.templates = file.read().split('\n')
        with open(messages, 'r', encoding='utf-8') as file:
            self.messages = file.read().split('\n')
        self.on_epoch_end()

    def generate_message(self):
        msg_template = random.choice(self.templates)
        tokens = msg_template.split(' ')
        final = ''
        tech = random.choice(self.technologies)
        if random.randint(0, 2) == 0:
            final += random.choice(self.greetings) + ' '
        for t in tokens:
            if t == '<verb>':
                final += random.choice(self.verbs) + ' '
            elif t == '<subject>':
                final += random.choice(self.subjects) + ' '
            elif t == '<technology>':
                final += tech + ' '
            else:
                final += t + ' '
        return final[:-1]

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.messages))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def mutate(self, msg):
        final = ''
        for e in msg:
            r = random.randint(0, 74) # 1/75 chance of mutation
            if r == 0:
                continue
            elif r == 1:
                final += e + e
            else:
                final += e
        return final

    def vectorize_message(self, msg):
        vec = []
        for c in msg:
            if len(vec) == self.sequence_length * self.n_channels:
                return vec
            if self.character_mappings.get(c) is not None:
                vec += self.character_mappings[c]
        while len(vec) < self.sequence_length * self.n_channels:
            vec += [0, 0, 0, 0, 0]
        return vec

    def __data_generation(self, indexes):
        X = np.empty((self.batch_size, self.sequence_length * self.n_channels))

        zeros = np.zeros((self.batch_size // 2))
        ones = np.ones((self.batch_size // 2))

        y = np.concatenate((zeros, ones))
        np.random.shuffle(y)

        itr_fmsg = iter(indexes)
        for index, e in enumerate(y):
            msg = self.mutate(self.generate_message()) if e else self.messages[next(itr_fmsg)]
            X[index] = self.vectorize_message(msg)

        return X, y

    def __len__(self):
        return int((len(self.messages) * 2 / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_half_size:(index+1)*self.batch_half_size]

        X, y = self.__data_generation(indexes)

        return X, y

