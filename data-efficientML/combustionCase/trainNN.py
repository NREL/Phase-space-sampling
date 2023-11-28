import sys

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

sys.path.append("util")
# NN Stuff
import tensorflow as tf
from myNN_better import *
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from prettyPlot.progressBar import print_progress_bar
from parallel import irank, iroot

def partitionData(nData, nBatch):
    # ~~~~ Partition the data across batches
    # Simple parallelization across snapshots
    NDataGlob = nData
    tmp1 = 0
    tmp2 = 0
    nData_b = np.zeros(nBatch, dtype=int)
    startData_b = np.zeros(nBatch, dtype=int)
    for ibatch in range(nBatch):
        tmp2 = tmp2 + tmp1
        tmp1 = int(NDataGlob / (nBatch - ibatch))
        nData_b[ibatch] = tmp1
        startData_b[ibatch] = tmp2
        NDataGlob = NDataGlob - tmp1
    return nData_b, startData_b


def getPrediction(model, data):
    result = np.zeros(data.shape[0])
    # Batch
    nPoints = data.shape[0]
    ApproxBatchSize = 10000
    nBatch = max(int(round(nPoints / ApproxBatchSize)), 1)
    nData_b, startData_b = partitionData(nPoints, nBatch)
    print_progress_bar(
        0,
        nBatch,
        prefix="Eval " + str(0) + " / " + str(nBatch),
        suffix="Complete",
        length=50,
        extraCond=(irank==iroot),
    )
    for ibatch in range(nBatch):
        start_ = startData_b[ibatch]
        end_ = startData_b[ibatch] + nData_b[ibatch]
        result[start_:end_] = np.squeeze(model(data[start_:end_]))
        print_progress_bar(
            ibatch + 1,
            nBatch,
            prefix="Eval " + str(ibatch + 1) + " / " + str(nBatch),
            suffix="Complete",
            length=50,
            extraCond=(irank==iroot),
        )

    return result


nPoints = [1000, 10000]
downsSamplingModes = ["random", "kmeans", "phase"]

# Load full data
try:
    combustion_data = np.load("../data/fullData.npy")[:, :2].astype("float32")
    srcProg = np.load("../data/fullData.npy")[:, 11].astype("float32")
except FileNotFoundError:
    print(
        "File ../data/fullData.npy is needed and available upon request to malik!hassanaly!at!nrel!gov"
    )
    sys.exit()

# Rescale data
print("RESCALE DATA")
nDim = combustion_data.shape[1]
minVal = np.zeros(nDim)
maxVal = np.zeros(nDim)
freq = np.zeros(nDim)
for i in range(nDim):
    minVal[i] = np.amin(combustion_data[:, i])
    maxVal[i] = np.amax(combustion_data[:, i])
    if i == 0:
        freq[i] = (maxVal[i] - minVal[i]) / 2.5
    if i == 1:
        freq[i] = (maxVal[i] - minVal[i]) / 1.5

printedModel = False

for irep in range(5):
    for nPoint in nPoints:
        print("\tnPoint = ", nPoint)
        for mode in downsSamplingModes:
            if mode == "random":
                indices = np.load(
                    "../data/downSampledDataRandom_"
                    + str(nPoint)
                    + "_"
                    + str(irep + 1)
                    + ".npz"
                )["indices"]
            if mode == "kmeans":
                indices = np.load(
                    "../data/downSampledDataKmeans40_"
                    + str(nPoint)
                    + "_"
                    + str(irep + 1)
                    + ".npz"
                )["indices"]
            if mode == "phase":
                indices = np.load(
                    "../data/downSampledData_"
                    + str(nPoint)
                    + "_iter1_"
                    + str(irep + 1)
                    + ".npz"
                )["indices"]

            combustion_data_downSampled = combustion_data[indices]

            # Preprocess data
            InputNN = combustion_data_downSampled[:, :2].copy()
            OutputNN = srcProg[indices].copy()
            scalerMean = np.mean(InputNN, axis=0)
            scalerStd = np.std(InputNN, axis=0)
            scalerMeanOutput = np.mean(OutputNN)
            scalerStdOutput = np.std(OutputNN)
            InputNN = (InputNN - scalerMean) / scalerStd
            InputNN = InputNN.astype("float32")
            OutputNN = (OutputNN - scalerMeanOutput) / scalerStdOutput
            OutputNN = np.reshape(OutputNN, (-1, 1)).astype("float32")

            # Surrogate NN
            EPOCHS = int(400 * 10000 // nPoint)
            BATCH_SIZE = 100
            LEARNING_RATE = 1e-1

            def scheduler(epoch, lr):
                if epoch < EPOCHS // 4:
                    return lr
                else:
                    return max(lr * tf.math.exp(-0.1), 1e-3)

            hidden_units = [32, 32]
            nn = myNN(ndim=2, hidden_units=hidden_units)
            if not printedModel:
                nn.model.summary()
                printedModel = True

            # TRAIN NN
            nn.train(
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                xTrain=InputNN,
                yTrain=OutputNN,
                learningRate=LEARNING_RATE,
                lrScheduler=scheduler,
            )

            # TEST NN
            InputNN_all = (combustion_data[:, :2] - scalerMean) / scalerStd
            InputNN_all = InputNN_all.astype("float32")
            OutputNN_all = srcProg.astype("float32")
            Y_pred = getPrediction(nn.model, InputNN_all)
            Y_pred = Y_pred * scalerStdOutput + scalerMeanOutput

            # Compute Error
            Error = abs(Y_pred - OutputNN_all)

            # Log
            print(
                "\t\t"
                + mode
                + " max = %.4f , mean = %.4f , std = %.4f "
                % (np.amax(Error), np.mean(Error), np.std(Error))
            )

            f = open("MaxNN_" + mode + "_n" + str(nPoint), "a+")  # append mode
            f.write(str(np.amax(Error)) + "\n")
            f.close()
            f = open(
                "MeanNN_" + mode + "_n" + str(nPoint), "a+"
            )  # append mode
            f.write(str(np.mean(Error)) + "\n")
            f.close()
            f = open("StdNN_" + mode + "_n" + str(nPoint), "a+")  # append mode
            f.write(str(np.std(Error)) + "\n")
            f.close()
