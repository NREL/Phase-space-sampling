import os

import numpy as np
import tensorflow as tf
from prettyPlot.progressBar import print_progress_bar
from tensorflow.keras import backend as K
from tensorflow.keras import layers, losses, optimizers, regularizers
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.constraints import max_norm, unit_norm
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


@tf.function
def loss_fn(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)


class myNN(Model):
    def __init__(self, ndim, hidden_units):
        super(myNN, self).__init__()

        self.ndim = ndim
        self.logLossFolder = "Log"
        self.modelFolder = "Model"
        self.hidden_units = hidden_units

        reg = 1e-3
        inputs = Input(shape=(ndim,), name="input")

        tmp = inputs
        for unit in hidden_units:
            tmp = Dense(unit, kernel_regularizer=regularizers.l2(l2=reg))(tmp)
            tmp = LeakyReLU()(tmp)

        output = Dense(1, activation="linear")(tmp)

        self.model = Model(inputs, output)

    @tf.function
    def train_step(self, x_batch_train, y_batch_train, dsOptimizer):
        with tf.GradientTape() as tape:
            y_pred = self.model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, y_pred)
            # Add extra loss terms to the loss value.
            loss_value += sum(self.model.losses)

        grads = tape.gradient(loss_value, self.trainable_weights)
        dsOptimizer.apply_gradients(zip(grads, self.trainable_weights))

        return loss_value

    def call(self, xTrain):
        return self.model(xTrain)

    def train(
        self, batch_size, epochs, xTrain, yTrain, learningRate, lrScheduler
    ):
        self.checkDataShape(xTrain, yTrain)

        self.prepareLog()
        bestLoss = 0

        # Prepare data sets
        train_dataset = tf.data.Dataset.from_tensor_slices((xTrain, yTrain))
        train_dataset = train_dataset.shuffle(buffer_size=2000).batch(
            batch_size
        )
        lr = learningRate
        dsOptimizer = optimizers.Adam(learning_rate=lr)

        # Train
        print_progress_bar(
            0,
            epochs,
            prefix="Loss=%s  Epoch= %d / %d " % ("?", 0, epochs),
            suffix="Complete",
            length=50,
        )
        for epoch in range(epochs):
            epoch_mse_loss = 0
            train_mse_loss = 0
            nSample = 0
            # Iterate over the batches of the dataset.
            for (
                step,
                (x_batch_train, y_batch_train),
            ) in enumerate(train_dataset):
                mse = self.train_step(
                    x_batch_train, y_batch_train, dsOptimizer
                )

                lr = lrScheduler(epoch, lr)
                K.set_value(dsOptimizer.learning_rate, lr)

                train_mse_loss += tf.reduce_sum(mse)
                nSample += mse.shape[0]

            print_progress_bar(
                epoch + 1,
                epochs,
                prefix="Loss=%.2f  Epoch= %d / %d "
                % (train_mse_loss / nSample, epoch + 1, epochs),
                suffix="Complete",
                length=50,
            )

            bestLoss = self.logTraining(
                epoch, mse=train_mse_loss / nSample, bestLoss=bestLoss
            )

    def prepareLog(self):
        os.makedirs(self.modelFolder, exist_ok=True)
        os.makedirs(self.logLossFolder, exist_ok=True)
        try:
            os.remove(self.logLossFolder + "/log.csv")
        except:
            pass

        f = open(self.logLossFolder + "/log.csv", "a+")
        f.write("epoch;mseloss\n")
        f.close()

    def logTraining(self, epoch, mse, bestLoss):
        f = open(self.logLossFolder + "/log.csv", "a+")
        f.write(str(int(epoch)) + ";" + str(mse.numpy()) + "\n")
        f.close()

        epochLoss = mse

        if epochLoss < bestLoss or epoch == 0:
            bestLoss = epochLoss
            self.save_weights(self.modelFolder + "/best.h5")

        return bestLoss

    def checkDataShape(self, xTrain, yTrain):
        if not len(xTrain.shape) == 2:
            print("Expected tensor of rank 2 for xTrain")
            print("xTrain shape =", xTrain.shape)
            sys.exit()
        if not len(yTrain.shape) == 2:
            print("Expected tensor of rank 2 for yTrain")
            print("yTrain shape =", yTrain.shape)
            sys.exit()
        if not (xTrain.shape[1] == self.ndim):
            print("Expected xTrain.shape[1] =", ndim)
            print("xTrain shape =", xTrain.shape)
            sys.exit()
