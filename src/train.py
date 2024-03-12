import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow_probability import distributions as tfd
import tensorflow_datasets as tfds
import numpy as np
from utils import plot_regression
import os
import argparse
from tqdm import tqdm

from src.models.cnp_model import CNPModel
# Assuming `encoder.py` and `decoder.py` are in the same directory
from src.models.encoder import Encoder
from src.models.decoder import Decoder
# Import dataset loaders (to be implemented)
from data_loader import reg_dataloader

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=15, help='Number of training epochs')
parser.add_argument('-b', '--batch', type=int, default=64, help='Batch size for training')
parser.add_argument('-t', '--task', type=str, default='function_regression', help='Task to perform : (function_regression | function_regression2 | omniglot)')
parser.add_argument('-c', '--max_num_context', type=int, default=10, help='MAximum number of context')


args = parser.parse_args()

# Training parameters
BATCH_SIZE = args.batch
EPOCHS = args.epochs
MAX_NUM_CONTEXT = args.max_num_context
TRAINING_ITERATIONS = int(2e5)
PLOT_AFTER = int(2e4)

if args.task == 'function_regression':
    # Instantiate the data generators
    train_data_generator = reg_dataloader.FunctionRegressionDataGenerator(batch_size=BATCH_SIZE,
                                                                          max_num_context=MAX_NUM_CONTEXT,
                                                                          testing=False)
    test_data_generator = reg_dataloader.FunctionRegressionDataGenerator(batch_size=1, testing=True)

    # Create TensorFlow datasets with specified output shapes
    train_dataset = tf.data.Dataset.from_generator(
        train_data_generator.generate_curves,
        output_signature=(
            (
                tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),  # context_x
                tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),  # context_y
                tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32)   # target_x
            ),
            tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32)       # target_y
        )
    )  # Ensure to batch your training dataset

    test_dataset = tf.data.Dataset.from_generator(
        test_data_generator.generate_curves,
        output_signature=(
            (
                tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),  # context_x
                tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),  # context_y
                tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32)   # target_x
            ),
            tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32)       # target_y
        )
    )  # No need to batch if evaluating one at a time, but you might want to add `.batch(1)` for consistency
        

    encoder_layers = [128, 128, 128]
    decoder_layers = [128, 128, 128, 128, 2]

    # Define the model
    model = CNPModel(encoder_layers, decoder_layers)

    def loss_fn(target_y, pred_y):
        # Get the distribution
        mu, sigma = tf.split(pred_y, num_or_size_splits=2, axis=-1)
        dist = tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        return -tf.math.reduce_mean(dist.log_prob(target_y))
    
elif args.task == 'function_regression':
        # Instantiate the data generators
    train_data_generator = reg_dataloader.FunctionRegressionDataGenerator(batch_size=BATCH_SIZE,
                                                                          max_num_context=MAX_NUM_CONTEXT,
                                                                          testing=False)
    test_data_generator = reg_dataloader.FunctionRegressionDataGenerator(batch_size=1, testing=True)

    # Create TensorFlow datasets with specified output shapes
    train_dataset = tf.data.Dataset.from_generator(
        train_data_generator.generate_curves,
        output_signature=(
            (
                tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),  # context_x
                tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),  # context_y
                tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32)   # target_x
            ),
            tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32)       # target_y
        )
    )  # Ensure to batch your training dataset

    test_dataset = tf.data.Dataset.from_generator(
        test_data_generator.generate_curves,
        output_signature=(
            (
                tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),  # context_x
                tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),  # context_y
                tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32)   # target_x
            ),
            tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32)       # target_y
        )
    )  # No need to batch if evaluating one at a time, but you might want to add `.batch(1)` for consistency
        

    encoder_layers = [128, 128, 128]
    decoder_layers = [128, 128, 128, 128, 2]

    # Define the model
    model = CNPModel(encoder_layers, decoder_layers)

    def loss_fn(target_y, pred_y):
        # Get the distribution
        mu, sigma = tf.split(pred_y, num_or_size_splits=2, axis=-1)
        dist = tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        return -tf.math.reduce_mean(dist.log_prob(target_y))

optimizer = tf.optimizers.Adam(learning_rate=1e-4)

@tf.function(reduce_retracing=True)
def train_step(model, context_x, context_y, target_x, target_y):
    with tf.GradientTape() as tape:
        # Pass context and target_x to the model to get the predicted distribution
        predicted_distribution = model((context_x, context_y, target_x))
        # Calculate the loss
        loss = loss_fn(target_y, predicted_distribution)

    # Compute gradients and apply them to update the model's parameters
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

# Path within Google Drive where you want to save checkpoints
drive_path = f'./experiments/{args.task}/training_checkpoints'

# Create the directory if it does not exist
if not os.path.exists(drive_path):
    os.makedirs(drive_path)

checkpoint_dir = drive_path  # Use the Google Drive path

checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=5)

for it in tqdm(range(TRAINING_ITERATIONS), desc='Training Progress'):
    for ((context_x, context_y, target_x), target_y) in train_dataset:
        loss = train_step(model, context_x, context_y, target_x, target_y)

    if it % PLOT_AFTER == 0:
        test_data_iter = iter(test_dataset.take(1))
        try:
            ((test_context_x, test_context_y, test_target_x), test_target_y) = next(test_data_iter)

            predicted_dist = model([test_context_x, test_context_y, test_target_x])
            mu, sigma = tf.split(predicted_dist, num_or_size_splits=2, axis=-1)

            print(f'Iteration: {it}, Loss: {loss.numpy()}')
            plot_regression(test_target_x.numpy(),
                            test_target_y.numpy(),
                            test_context_x.numpy(),
                            test_context_y.numpy(),
                            mu.numpy(),
                            sigma.numpy(),
                            filename=f"test{it}"
                            task=args.task)
        except StopIteration:
            print(f'Iteration: {it}, No more data to plot.')

        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{ith}")
        save_path = checkpoint_manager.save(checkpoint_prefix)
        print(f"Saved checkpoint for iteration {it} at {save_path}")