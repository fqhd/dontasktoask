import tensorflow as tf
from tensorflow import keras
import numpy as np
import keras_tuner as kt
from datagen import DataGenerator

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

def model_builder(hp):
    model = keras.Sequential()
    for i in range(hp.Int('layers', 2, 6)):
        model.add(keras.layers.Dense(
            units=hp.Int('units_' + str(i), 10, 500, step=10),
            activation='relu',
            kernel_initializer='he_normal',
            bias_initializer='zeros'
        ))
    model.add(keras.layers.Dense(1))
    
    model.compile(
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=[keras.metrics.BinaryAccuracy()]
    )
    
    return model

tuner = kt.RandomSearch(
    model_builder,
    objective='val_binary_accuracy',
    max_trials=500,
    executions_per_trial=3,
    directory='kt_trials',
    project_name='dontasktoask'
)

tuner.search_space_summary()
tuner.search(train_generator, validation_data=val_generator, epochs=5)
