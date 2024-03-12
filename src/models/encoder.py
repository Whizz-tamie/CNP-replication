# encoder.py
import tensorflow as tf
from tensorflow.keras import layers

class Encoder(layers.Layer):
    def __init__(self, output_sizes):
        super(Encoder, self).__init__()
        self.output_sizes = output_sizes
        self.dense_layers = [layers.Dense(size, activation='relu') for size in output_sizes[:-1]]
        self.dense_layers.append(layers.Dense(output_sizes[-1]))

    def call(self, context_x, context_y):
        # `context_x` shape (batch_size, observation_points, x_dim)
        # `context_y` shape (batch_size, observation_points, y_dim)
        encoder_input = tf.concat([context_x, context_y], axis=-1)
        batch_size, num_context_points, filter_size = tf.shape(encoder_input)[0], tf.shape(encoder_input)[1], tf.shape(encoder_input)[2]

        hidden = tf.reshape(encoder_input, (batch_size * num_context_points, -1))
        
        for layer in self.dense_layers:
            hidden = layer(hidden)
        
        hidden = tf.reshape(hidden, (batch_size, num_context_points, -1))
        representation = tf.reduce_mean(hidden, axis=1)
        
        return representation