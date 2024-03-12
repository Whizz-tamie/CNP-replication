# cnp_model.py
import tensorflow as tf
from tensorflow.keras import Model
from .encoder import Encoder
from .decoder import Decoder

class CNPModel(Model):
    def __init__(self, encoder_output_sizes, decoder_output_sizes):
        super(CNPModel, self).__init__()
        self.encoder = Encoder(encoder_output_sizes)
        self.decoder = Decoder(decoder_output_sizes)
    
    def call(self, inputs):
        context_x, context_y, target_x = inputs
        representations = self.encoder(context_x, context_y)
        mu, sigma = self.decoder(representations, target_x)

        return tf.concat([mu, sigma], axis=-1)
    