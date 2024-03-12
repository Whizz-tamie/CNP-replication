import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from src.models.decoder import Decoder  # Adjust the import path to match your file structure

class DecoderTest(tf.test.TestCase):

    def setUp(self):
        super(DecoderTest, self).setUp()
        self.layer_sizes = [128, 128, 2]  # Configuration including output layer for mean and variance
        self.decoder = Decoder(layer_sizes=self.layer_sizes)
        
    def create_synthetic_data(self, batch_size, target_points, x_dim, repr_dim):
        """
        Generates synthetic data to simulate the input to the Decoder.
        """
        # Simulating target inputs (target_x)
        target_x = np.random.randn(batch_size, target_points, x_dim).astype(np.float32)
        # Simulating aggregated representations (r)
        r = np.random.randn(batch_size, repr_dim).astype(np.float32)
        return target_x, r

    def test_decoder_output_distribution(self):
        """
        Tests that the Decoder outputs a tfd.Distribution with correct batch and event shapes.
        """
        batch_size = 5
        target_points = 4
        x_dim = 3
        repr_dim = 128  # Should match the second last layer size of the Decoder
        target_x, r = self.create_synthetic_data(batch_size, target_points, x_dim, repr_dim)
        
        # Convert synthetic data to tensors
        target_x_tensor = tf.convert_to_tensor(target_x)
        r_tensor = tf.convert_to_tensor(r)

        # Use the decoder to process the data
        output_dist = self.decoder(r_tensor, target_x_tensor)

        # Verify that the output is a tfd.Distribution
        self.assertIsInstance(output_dist, tfp.distributions.Distribution)

        # Check the shapes of the mean and variance in the output distribution
        self.assertAllEqual(output_dist.batch_shape, [batch_size, target_points])
        self.assertAllEqual(output_dist.event_shape, [1])
        
    def test_output_shapes_for_different_input_dimensions(self):
        """
        Verifies that the Decoder correctly handles inputs with different dimensions.
        """
        batch_size = 10
        target_points = 7
        x_dim = 2  # Testing with a different input dimension
        repr_dim = self.layer_sizes[-2]  # Assuming the representation dimension matches the second-to-last layer size

        target_x, r = self.create_synthetic_data(batch_size, target_points, x_dim, repr_dim)
        target_x_tensor = tf.convert_to_tensor(target_x)
        r_tensor = tf.convert_to_tensor(r)

        output_dist = self.decoder(r_tensor, target_x_tensor)

        # Perform shape checks as before
        self.assertAllEqual(output_dist.batch_shape, [batch_size, target_points])
        self.assertAllEqual(output_dist.event_shape, [1])

if __name__ == '__main__':
    tf.test.main()
