import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from src.models.cnp_model import CNPModel

class CNPModelTest(tf.test.TestCase):

    def setUp(self):
        super(CNPModelTest, self).setUp()
        # Define layer sizes for both encoder and decoder
        self.encoder_layers = [128, 128]
        self.decoder_layers = [128, 128, 2]  # Assuming output is mean and variance for 1D targets
        self.model = CNPModel(encoder_layers=self.encoder_layers, decoder_layers=self.decoder_layers)

    def create_synthetic_data(self, batch_size, context_points, target_points, x_dim, y_dim):
        """
        Generates synthetic data for testing, including context and target sets.
        """
        # Context set (features and labels)
        context_x = np.random.randn(batch_size, context_points, x_dim).astype(np.float32)
        context_y = np.random.randn(batch_size, context_points, y_dim).astype(np.float32)
        # Target set (only features, as we're predicting the labels)
        target_x = np.random.randn(batch_size, target_points, x_dim).astype(np.float32)
        return context_x, context_y, target_x

    def test_model_output_distribution(self):
        """
        Tests that the CNPModel produces a distribution with the correct shapes.
        """
        batch_size = 5
        context_points = 10
        target_points = 15
        x_dim = 3  # Dimensionality of the input features
        y_dim = 1  # Dimensionality of the output predictions (mean and variance)
        
        context_x, context_y, target_x = self.create_synthetic_data(
            batch_size, context_points, target_points, x_dim, y_dim)
        
        # Convert synthetic data to tensors
        context_x_tensor = tf.convert_to_tensor(context_x)
        context_y_tensor = tf.convert_to_tensor(context_y)
        target_x_tensor = tf.convert_to_tensor(target_x)
        
        # Process through the model
        output_dist = self.model((context_x_tensor, context_y_tensor, target_x_tensor))
        
        # Verify the output is a tfd.Distribution
        self.assertIsInstance(output_dist, tfp.distributions.Distribution)
        
        # Check the shapes of the mean and variance in the output distribution
        self.assertAllEqual(output_dist.batch_shape, [batch_size, target_points])
        self.assertAllEqual(output_dist.event_shape, (1,))

if __name__ == '__main__':
    tf.test.main()
