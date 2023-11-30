import numpy as np
from matplotlib import cm
from prettyPlot.plotting import plt, pretty_labels

from uips.utils.parallel import irank, iroot


def cornerPlotScatter(data, title=None):
    if not irank == iroot:
        return

    nDim = data.shape[1]
    if nDim > 2:
        fig, axs = plt.subplots(nDim - 1, nDim - 1, figsize=(8, 8))
        for idim in range(nDim - 1):
            for jdim in range(idim + 1, nDim):
                axs[idim, jdim - 1].plot(
                    data[:, idim], data[:, jdim], "o", markersize=0.5
                )
                pretty_labels(
                    "feat" + str(idim),
                    "feat" + str(jdim),
                    10,
                    ax=axs[idim, jdim - 1],
                )
        if not title == None:
            fig.suptitle(
                title,
                fontsize=16,
                fontweight="bold",
                fontname="Times New Roman",
            )
    elif nDim == 2:
        fig = plt.figure()
        plt.plot(data[:, 0], data[:, 1], "o", markersize=0.5)
        pretty_labels("feat" + str(0), "feat" + str(1), 10, title=title)


def cornerPlotScatterColor(data, colorData):
    if not irank == iroot:
        return

    nDim = data.shape[1]
    if nDim > 2:
        fig, axs = plt.subplots(nDim - 1, nDim - 1, figsize=(8, 8))
        for idim in range(nDim - 1):
            for jdim in range(idim + 1, nDim):
                a = axs[idim, jdim - 1].scatter(
                    data[:, idim],
                    data[:, jdim],
                    c=colorData,
                    s=0.5,
                    cmap=cm.gray_r,
                    alpha=0.9,
                )
                pretty_labels(
                    "feat" + str(idim),
                    "feat" + str(jdim),
                    10,
                    ax=axs[idim, jdim - 1],
                )
    elif nDim == 2:
        fig = plt.figure()
        plt.scatter(
            data[:, 0],
            data[:, 1],
            c=colorData,
            s=0.5,
            cmap=cm.gray_r,
            alpha=0.9,
        )
        pretty_labels("feat" + str(0), "feat" + str(1), 10)
        plt.colorbar()
