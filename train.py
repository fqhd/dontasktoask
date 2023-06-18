import tensorflow as tf
from tensorflow import keras
import numpy as np
import keras_tuner as kt
from datagen import DataGenerator
import matplotlib.pyplot as plt
import tensorflowjs as tfjs

sequence_length = 50
n_channels = 5
batch_size = 32

params = {
    'n_channels': n_channels,
    'sequence_length': sequence_length,
    'batch_size': batch_size,
    'shuffle': True
}

train_generator = DataGenerator(templates='train_templates.txt', messages='train_messages.txt', **params)
val_generator = DataGenerator(templates='val_templates.txt', messages='val_messages.txt', **params)

model = keras.Sequential()

model.add(keras.layers.Dense(400, activation='gelu', kernel_initializer='lecun_normal', bias_initializer='zeros', input_dim=sequence_length*n_channels))
model.add(keras.layers.Dense(60, activation='elu', kernel_initializer='lecun_normal', bias_initializer='zeros'))
model.add(keras.layers.Dense(20, activation='gelu', kernel_initializer='he_normal', bias_initializer='zeros'))
model.add(keras.layers.Dense(100, activation='elu', kernel_initializer='lecun_uniform', bias_initializer='zeros'))
model.add(keras.layers.Dense(1))

model.compile(
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=[keras.metrics.BinaryAccuracy()]
)

model.summary()

history = model.fit(train_generator, validation_data=val_generator, epochs=20)

h_dic = history.history

export_model = keras.Sequential([
    model,
    keras.layers.Activation('sigmoid')
])

export_model.compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-4)
)

plt.figure(figsize=(12, 3))

plt.subplot(1, 2, 1)
plt.plot(h_dic['loss'], label='Training Loss')
plt.plot(h_dic['val_loss'], label='Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(h_dic['binary_accuracy'], label='Training Accuracy')
plt.plot(h_dic['val_binary_accuracy'], label='Validation Accuracy')
plt.legend()

plt.show()

tfjs.converters.save_keras_model(export_model, 'model/')