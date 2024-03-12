# reg_dataloader.py
import tensorflow_probability as tfp
import tensorflow as tf

tfd = tfp.distributions
psd_kernels = tfp.math.psd_kernels

class FunctionRegressionDataGenerator:
    def __init__(self, batch_size=64, max_num_context=10, testing=False):
        self.batch_size = batch_size
        self.max_num_context = max_num_context
        self.testing = testing

    def generate_curves(self):
        # Define the kernel
        kernel = psd_kernels.ExponentiatedQuadratic(amplitude=1.0, length_scale=0.4)
        num_context = tf.random.uniform(shape=[], minval=3, maxval=self.max_num_context, dtype=tf.int32)
        # If we are testing we want to have more targets and have them evenly
        # distributed in order to plot the function.
        if self.testing is True:
            num_target = 400
            num_total_points = num_target
            x_values = tf.expand_dims(
                tf.range(-2., 2., 1./100., dtype=tf.float32),
                axis=0)  # (1, 400)
            x_values = tf.tile(x_values, [self.batch_size, 1])  # (batch_size, 400)
            x_values = tf.expand_dims(x_values, axis=-1)  # (batch_size, 400, 1)
        else:
            num_target = tf.random.uniform(
                shape=[], minval=2, maxval=self.max_num_context, dtype=tf.int32)
            num_total_points = num_context + num_target
            x_values = tf.random.uniform(
                (self.batch_size, num_total_points, 1), minval=-2., maxval=2.)

        gp = tfd.GaussianProcess(
            kernel, index_points=x_values, jitter=1.0e-4)
        y_values = tf.expand_dims(gp.sample(), axis=-1)

        if self.testing is True:
            target_x = x_values
            target_y = y_values

            # Select the observations
            idx = tf.random.shuffle(tf.range(num_target))
            context_x = tf.gather(x_values, idx[:num_context], axis=1)
            context_y = tf.gather(y_values, idx[:num_context], axis=1)
        else:
            # Select the targets which will consist of the context points
            # as well as some new target points
            target_x = x_values[:, :num_target + num_context, :]
            target_y = y_values[:, :num_target + num_context, :]

            # Select the observations
            context_x = x_values[:, :num_context, :]
            context_y = y_values[:, :num_context, :]

        yield (context_x, context_y, target_x), target_y
