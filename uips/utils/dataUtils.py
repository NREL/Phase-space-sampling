import os
import sys

import numpy as np

import uips.utils.parallel as par
from uips.utils.fileFinder import find_data

# from memory_profiler import profile


# @profile
def checkData(shape, N, d, nWorkingData, nWorkingDataAdjustment, useNF):
    if not len(shape) == 2:
        par.printAll(
            "Expected 2 dimensions for the data, first is the number of samples, second is the dimension"
        )
        par.comm.Abort()
    else:
        par.printRoot(f"Dataset has {N} samples of dimension {d}")

    if useNF:
        # Check that sizes make sense
        if N < nWorkingData * 10:
            par.printRoot(f"WARNING: Only {N} samples, this may not work")
        if N < max(nWorkingData, nWorkingDataAdjustment):
            par.printAll(
                f"ERROR: At least {max(nWorkingData, nWorkingDataAdjustment)} samples required"
            )
            par.comm.Abort()

    return


# @profile
def prepareData(inpt):
    # Set parameters from input
    dataFile = find_data(inpt["dataFile"])
    preShuffled = inpt["preShuffled"] == "True"
    scalerFile = inpt["scalerFile"]
    nWorkingDatas = [int(float(n)) for n in inpt["nWorkingData"].split()]
    if len(nWorkingDatas) == 1:
        nWorkingDatas = nWorkingDatas * int(inpt["num_pdf_iter"])
    nWorkingDataAdjustment = int(float(inpt["nWorkingDataAdjustment"]))

    # Reduce Data
    try:
        dimList = [int(n) for n in inpt["dimList"].split()]
    except:
        dimList = None
    try:
        nDimReduced = int(float(inpt["nDimReduced"]))
    except:
        nDimReduced = -1
    try:
        nDatReduced = int(float(inpt["nDatReduced"]))
    except:
        nDatReduced = -1

    # Load the dataset but don't read it just yet
    dataset = np.load(dataFile, mmap_mode="r")

    # Check that dataset shape make sense
    if nDatReduced > 0:
        nFullData = min(dataset.shape[0], nDatReduced)
    else:
        nFullData = dataset.shape[0]
    if dimList is not None:
        nDim = min(dataset.shape[1], len(dimList))
    elif nDimReduced > 0:
        nDim = min(dataset.shape[1], nDimReduced)
    else:
        nDim = dataset.shape[1]
    if par.irank == par.iroot:
        useNF = inpt["pdf_method"].lower() == "normalizingflow"
        checkData(
            dataset.shape,
            nFullData,
            nDim,
            nWorkingDatas[-1],
            nWorkingDataAdjustment,
            useNF,
        )

    # Distribute dataset
    if par.irank == par.iroot:
        print("LOAD DATA ... ", end="")
        sys.stdout.flush()
    par.comm.Barrier()
    nSnap_, startSnap_ = par.partitionData(nFullData)
    if dimList is None:
        data_to_downsample_ = dataset[
            startSnap_ : startSnap_ + nSnap_, :nDim
        ].astype("float32")
    else:
        data_to_downsample_ = np.take(
            dataset[startSnap_ : startSnap_ + nSnap_],
            np.array(dimList),
            axis=1,
        ).astype("float32")
    par.printRoot("DONE!")

    # Rescale data
    if par.irank == par.iroot:
        print("RESCALE DATA ... ", end="")
        sys.stdout.flush()
    par.comm.Barrier()
    minVal = np.zeros(nDim)
    maxVal = np.zeros(nDim)
    for i in range(nDim):
        minVal_ = np.amin(data_to_downsample_[:, i])
        maxVal_ = np.amax(data_to_downsample_[:, i])
        minVal[i] = par.comm.allreduce(minVal_, op=par.MPI.MIN)
        maxVal[i] = par.comm.allreduce(maxVal_, op=par.MPI.MAX)
    # Root processor saves scaling
    if par.irank == par.iroot:
        np.savez(scalerFile, minVal=minVal, maxVal=maxVal)
    par.printRoot("DONE!")

    # Parallel shuffling
    dataInd_ = None
    if not preShuffled:
        if par.irank == par.iroot:
            print("SHUFFLE DATA ... ", end="")
            sys.stdout.flush()

        data_to_downsample_, dataInd_, tags_ = par.parallel_shuffle_np(
            data_to_downsample_, nFullData
        )
        if nFullData < int(1e7):
            tags_gathered = par.gather1DList(list(tags_), 0, nFullData)
            assert np.amin(np.diff(tags_gathered)) > -1e-12

        par.printRoot("DONE!")

    # Get subsampled dataset to work with
    working_data = par.gatherNelementsInArray(
        data_to_downsample_, nWorkingDatas[0]
    )

    return data_to_downsample_, dataInd_, working_data, nFullData
