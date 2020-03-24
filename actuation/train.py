import os
import argparse
import logging
import tensorflow as tf
# Import losses:
from tensorflow.keras.losses import
# Import optimizers:
from tensorflow.keras.optimizers import
# Import metrics:
from tensorflow.keras.metrics import (
  Mean, SparseCategoricalAccuracy
)
# Import models:
from models.network import Network
# Import processing:
from preprocess.process import Process


class Train(object):
    def __init__(self, params):
        self.lr = params.lr
        self.epochs = params.epochs
        # Define loss:
        self.loss_object =
        # Define optimizer:
        self.optimizer =
        # Define metrics for loss:
        self.train_loss =
        self.train_accuracy =
        self.test_loss =
        self.test_accuracy =
        # Define model:
        self.model = Network()
        # Define pre processor (params):
        preprocessor = Process()
        self.train_ds, self.test_ds = preprocessor.get_datasets()
        # Define Checkpoints:
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer,
                net=self.model)
        # Define Checkpoint manager:
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, f'checkpoints{params.ckpt_dir}',
                max_to_keep=3)

    # Feed forward through and update model on train data:
    @tf.function
    def _update(self, inputs, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    # Feed forward through model on test data:
    @tf.function
    def _test(self, inputs, labels):
        predictions = self.model(inputs)
        loss = self.loss_object(labels, predictions)

        self.test_loss(loss)
        self.test_accuracy(labels, predictions)

    # Log status of each epoch:
    def _log(self, epoch):
        template = 'Epoch {}, Loss: {}, Acc: {}, Test Loss: {}, Test Acc: {}'
        print(template.format(epoch + 1,
            self.train_loss.result(),
            self.train_accuracy.result() * 100,
            self.test_loss.result(),
            self.test_accuracy.result() * 100))

    # Save model to checkpoint:
    def _save(self, verbose=False):
        save_path = self.ckpt_manager.save()]
        if verbose:
            ckptLog = f"Saved checkpoint for step {int(self.ckpt.step)}: {save_path}"
            print(ckptLog)

    # Restore model from checkpoint:
    def _restore(self):
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint).expect_partial()
        if self.ckpt_manager.latest_checkpoint:
            print(f"Restored from {self.ckpt_manager.latest_checkpoint}")
        else:
            print("Initializing from scratch.")

    # Reset network metrics:
    def _reset(self):
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()

    # Train loop for network:
    def train(self):
        self._restore()
        for epoch in range(self.epochs):
            for inputs, labels in self.train_ds:
                self._update(inputs, labels)
            for testInputs, testLabels in self.test_ds:
                self._test(testInputs, testLabels)
            self._log(epoch)
            self._save()
            self._reset()


if __name__ == '__main__':
    numCkpts = len([folder for folder in os.listdir('./checkpoints') if os.path.isdir(folder)])
    parser= argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--ckpt_dir', default=str(numCkpts), type=str)
    parser.add_argument('--data_dir', default='./data/train', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--pre_fetch', default=1, type=int)
    args = parser.parse_args()
    actuator = Train(args)
    actuator.train()
