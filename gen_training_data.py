import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import string

def get_character_embeddings():
    embeddings = []
    for i in range(128):
        curr = []
        for j in range(7):
            curr.append((i // pow(2, j)) % 2)
        embeddings.append(curr)
    return embeddings

def get_character_mappings():
    embeddings = get_character_embeddings()
    ascii_chars = string.ascii_letters + string.digits + string.punctuation + ' '
    char_mappings = {}
    for i in range(len(ascii_chars)):
        char_mappings[ascii_chars[i]] = embeddings[i]
    return char_mappings

class DataGenerator(keras.utils.Sequence):
    def __init__(self, templates, messages, sequence_length=50, batch_size=32, shuffle=True):
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.batch_half_size = batch_size // 2
        self.shuffle=shuffle
        self.character_mappings = get_character_mappings()
        with open('data/greetings.txt', 'r') as file:
            self.greetings = file.read().split('\n')
        with open('data/verbs.txt', 'r') as file:
            self.verbs = file.read().split('\n')
        with open('data/technologies.txt', 'r') as file:
            self.technologies = file.read().split('\n')
        with open('data/subjects.txt', 'r') as file:
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
        if random.randint(0, 2) == 0:
            final += random.choice(self.greetings)
            should_add_comma = random.randint(0, 4)
            if should_add_comma == 0:
                final += ','
            elif should_add_comma == 1:
                final += ', '
            elif should_add_comma == 2:
                final += ' ,'
            elif should_add_comma == 3:
                final += ' , '
            else:
                final += ' '
        for t in tokens:
            if t == '<verb>':
                final += random.choice(self.verbs) + ' '
            elif t == '<subject>':
                final += random.choice(self.subjects) + ' '
            elif t == '<technology>':
                final += random.choice(self.technologies) + ' '
            else:
                final += t + ' '
        return final[:-1]

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.messages))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def mutate_message(self, msg):
        final = ''
        for e in msg:
            r = random.randint(0, 199) # 1/75 chance of mutation
            if r == 0:
                continue
            elif r == 1:
                final += e + e
            elif r == 2:
                final += random.choice(string.ascii_letters[:26]) + e
            elif r == 3:
                final += e + random.choice(string.ascii_letters[:26])
            elif r == 4:
                final += random.choice(string.ascii_letters[:26])
            else:
                final += e
        return final

    def vectorize_message(self, msg):
        vec = []
        msg = msg.lower()
        for c in msg:
            if len(vec) == self.sequence_length * 7:
                return vec
            if self.character_mappings.get(c) is not None:
                vec += self.character_mappings[c]
        while len(vec) < self.sequence_length * 7:
            vec += [1, 1, 1, 1, 1, 1, 1]
        return vec

    def __data_generation(self, indexes):
        X = np.empty((self.batch_size, self.sequence_length * 7))

        zeros = np.zeros((self.batch_size // 2))
        ones = np.ones((self.batch_size // 2))

        y = np.concatenate((zeros, ones))
        np.random.shuffle(y)

        itr_fmsg = iter(indexes)
        for index, e in enumerate(y):
            msg = self.mutate_message(self.generate_message()) if e else self.messages[next(itr_fmsg)]
            X[index] = self.vectorize_message(msg)

        return X, y

    def __len__(self):
        return int((len(self.messages) * 2 / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_half_size:(index+1)*self.batch_half_size]

        X, y = self.__data_generation(indexes)

        return X, y
