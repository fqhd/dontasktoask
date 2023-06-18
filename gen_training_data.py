import numpy as np
import random
import string
import pickle

sequence_length = 50
n_channels = 5

character_mappings = {
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
    ' ': [1, 1, 0, 1, 1]
}

with open('data/greetings.txt', 'r') as file:
    greetings = file.read().split('\n')
with open('data/verbs.txt', 'r') as file:
    verbs = file.read().split('\n')
with open('data/technologies.txt', 'r') as file:
    technologies = file.read().split('\n')
with open('data/subjects.txt', 'r') as file:
    subjects = file.read().split('\n')
with open('data/templates.txt', 'r', encoding='utf-8') as file:
    templates = file.read().split('\n')
    random.shuffle(templates)
with open('data/messages.txt', 'r', encoding='utf-8') as file:
    messages = file.read().split('\n')
    random.shuffle(messages)

def generate_message(train):
    if train:
        msg_template = random.choice(templates[:-10])
    else:
        msg_template = random.choice(templates[-10:])
    tokens = msg_template.split(' ')
    final = ''
    tech = random.choice(technologies)
    if random.randint(0, 2) == 0:
        final += random.choice(greetings) + ' '
    for t in tokens:
        if t == '<verb>':
            final += random.choice(verbs) + ' '
        elif t == '<subject>':
            final += random.choice(subjects) + ' '
        elif t == '<technology>':
            final += tech + ' '
        else:
            final += t + ' '
    return final[:-1]

def mutate_message(msg):
    final = ''
    for e in msg:
        r = random.randint(0, 499) # 1/100 chance of mutation
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

def vectorize_message(msg):
    vec = []
    msg = msg.lower()
    for c in msg:
        if len(vec) == sequence_length * n_channels:
            return vec
        if c in character_mappings:
            vec += character_mappings[c]
    while len(vec) < sequence_length * n_channels:
        vec += [0, 0, 0, 0, 0]
    return vec


train_x = []
train_y = []
test_x = []
test_y = []

# Make training data
for msg in messages[:-500]:
    train_x.append(vectorize_message(msg))
    train_y.append(0)

for i in range(len(messages) - 500):
    train_x.append(vectorize_message(mutate_message(generate_message(train=True))))
    train_y.append(1)

# Make testing data

for msg in messages[-500:]:
    test_x.append(vectorize_message(msg))
    test_y.append(0)

for i in range(500):
    test_x.append(vectorize_message(mutate_message(generate_message(train=False))))
    test_y.append(1)

data = (np.array(train_x), np.array(train_y, dtype='int8')), (np.array(test_x), np.array(test_y, dtype='int8'))

with open('data/data.ds', 'wb') as f_out:
    pickle.dump(data, f_out)

