import tensorflow as tf
import numpy as np
from src.models.encoder import Encoder

class EncoderTest(tf.test.TestCase):

    def setUp(self):
        super(EncoderTest, self).setUp()
        self.layer_sizes = [128, 128]
        self.encoder = Encoder(layer_sizes=self.layer_sizes)
        
    def create_synthetic_data(self, batch_size, observation_points, x_dim, y_dim):
        context_x = np.random.randn(batch_size, observation_points, x_dim).astype(np.float32)
        context_y = np.random.randn(batch_size, observation_points, y_dim).astype(np.float32)
        return context_x, context_y

    def test_encoder_output_shape(self):
        batch_size = 10
        observation_points = 5
        x_dim = 3
        y_dim = 1
        context_x, context_y = self.create_synthetic_data(batch_size, observation_points, x_dim, y_dim)
        
        context_x_tensor = tf.convert_to_tensor(context_x)
        context_y_tensor = tf.convert_to_tensor(context_y)
        encoded_output = self.encoder(context_x_tensor, context_y_tensor)
        
        expected_shape = (batch_size, observation_points, self.layer_sizes[-1])
        self.assertAllEqual(encoded_output.shape, expected_shape)

if __name__ == "__main__":
    tf.test.main()
