# decoder.py
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers

tfd = tfp.distributions

class Decoder(layers.Layer):
    def __init__(self, output_sizes):
        super(Decoder, self).__init__()
        self.output_sizes = output_sizes
        # Initialize dense layers for MLP, using ReLU for all but last layer
        self.dense_layers = [layers.Dense(size, activation='relu') for size in output_sizes[:-1]]
        # Last layer without activation function
        self.dense_layers.append(layers.Dense(output_sizes[-1]))

    def call(self, representation, target_x):
        # Infer num_total_points from target_x
        num_total_points = tf.shape(target_x)[1]

        # Expand dims of representation to match target_x, then concatenate
        representation = tf.tile(tf.expand_dims(representation, axis=1), [1, num_total_points, 1])
        decoder_input = tf.concat([representation, target_x], axis=-1)

        batch_size, _, filter_size = decoder_input.shape
        hidden = tf.reshape(decoder_input, (batch_size * num_total_points, -1))

        # Pass through MLP
        for layer in self.dense_layers:
            hidden = layer(hidden)

        # Reshape to get back to the original batch and num_total_points dimensions
        hidden = tf.reshape(hidden, (batch_size, num_total_points, -1))

        # Split the output into mean and log variance
        mu, log_sigma = tf.split(hidden, 2, axis=-1)

        # Ensure variance is positive and bounded
        sigma = 0.1 + 0.9 * tf.nn.softplus(log_sigma)

        # Return a tuple of distribution parameters (mu and sigma)
        return mu, sigma
