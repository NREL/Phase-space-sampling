import sys

import numpy as np

import os

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from prettyPlot.plotting import pretty_labels
from prettyPlot.parser import parse_input_file


def plotScatterProjection(data, fullData, fieldNames, lims):
    nDim = data.shape[1]
    if nDim > 2:
        fig, axs = plt.subplots(
            nDim - 1, nDim - 1, figsize=(12, 12), sharex="col", sharey="row"
        )
        for idim in range(nDim - 1):
            for jdim in range(idim + 1, nDim):
                # plot contours of support of all data
                a = axs[jdim - 1, idim].scatter(
                    fullData[:, idim], fullData[:, jdim], color="gray", s=0.2
                )
                a = axs[jdim - 1, idim].scatter(
                    data[:, idim], data[:, jdim], color="blue", s=0.2
                )

        for idim in range(nDim - 1):
            axs[nDim - 2, idim].set_xlabel(fieldNames[idim])
            axs[nDim - 2, idim].set_xlim(lims[idim])
            for tick in axs[nDim - 2, idim].get_xticklabels():
                tick.set_rotation(33)
            axs[idim, 0].set_ylabel(fieldNames[idim + 1])
            axs[idim, 0].set_ylim(lims[idim + 1])

        for idim in range(nDim - 2):
            for jdim in range(idim + 1, nDim - 1):
                axs[idim, jdim].axis("off")
    if nDim == 2:
        fig = plt.figure()
        plt.scatter(fullData[:, 0], fullData[:, 1], color="gray", s=0.2)
        plt.scatter(data[:, 0], data[:, 1], color="blue", s=0.2)
        ax = plt.gca()
        pretty_labels(fieldNames[0], fieldNames[1], 14, ax=ax)
        ax.set_xlim(lims[0])
        for tick in ax.get_xticklabels():
            tick.set_rotation(33)
        ax.set_ylim(lims[1])

import argparse
parser = argparse.ArgumentParser(description="Visualize downsampled data")
parser.add_argument(
    "-i",
    "--input",
    type=str,
    metavar="",
    required=False,
    help="Input file",
    default="input",
)
args, unknown = parser.parse_known_args()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Parse input
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

inpt = parse_input_file(args.input)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Parameters to save
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# List of sample size
nSamples = [int(float(n)) for n in inpt["nSamples"].split()]
# Data size used to learn the data probability
nWorkingDatas = [int(float(n)) for n in inpt["nWorkingData"].split()]
if len(nWorkingDatas) == 1:
    nWorkingDatas = nWorkingDatas * int(inpt["num_pdf_iter"])
for nWorkingData in nWorkingDatas:
    if not nWorkingData in nSamples:
        nSamples += [nWorkingData]
# Data file name
fullDataFile = inpt["dataFile"]
# Scaler file name
scalerFile = inpt["scalerFile"]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Plot
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("LOAD DATA ... ", end="")
sys.stdout.flush()
fullData = np.load(fullDataFile)
print("DONE!")
mins = np.amin(fullData, axis=0)
maxs = np.amax(fullData, axis=0)
# mins = np.load(scalerFile)["minVal"]
# maxs = np.load(scalerFile)["maxVal"]
lims = [
    (xmin - 0.05 * (xmax - xmin), xmax + 0.05 * (xmax - xmin))
    for xmin, xmax in zip(mins, maxs)
]

fieldNames = ["feature" + str(i) for i in range(fullData.shape[1])]

# Folder where figures are saved
figureFolder = "Figures"
os.makedirs(figureFolder, exist_ok=True)

for pdf_iter in range(int(inpt["num_pdf_iter"])):
    print(f"Iter : {pdf_iter}")
    for nSample in nSamples:
        print(f"\tplot nSample : {nSample} ... ", end="")
        sys.stdout.flush()
        dataFile = (
            f"{inpt['prefixDownsampledData']}_{nSample}_it{pdf_iter}.npz"
        )
        downSampledData = fullData[np.load(dataFile)["indices"], :]
        plotScatterProjection(downSampledData, fullData, fieldNames, lims)
        plt.savefig(
            figureFolder
            + "/"
            + inpt["prefixDownsampledData"]
            + "_"
            + str(nSample)
            + ".png"
        )
        plt.close()
        print("DONE!")
        sys.stdout.flush()
