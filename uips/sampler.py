import os
import sys
import time

import numpy as np
import torch
from prettyPlot.progressBar import print_progress_bar
from sklearn.neighbors import NearestNeighbors
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from torch.utils.data import DataLoader

import uips.nn as nn_
import uips.utils.parallel as par
from uips.nde import distributions, flows, transforms
from uips.utils.flowUtils import create_base_transform
from uips.utils.torchutils import tensor2numpy


def str2float(str):
    return float(str)


def str2int(str):
    return int(str)


def str2BigInt(str):
    return int(float(str))


def checkParamListLength(listToCheck, targetLength, name):
    if not len(listToCheck) == targetLength and (not len(listToCheck) == 1):
        par.printAll(
            f"Expected {name} of size {targetLength} got {len(listToCheck)}"
        )
        par.comm.Abort()


def makeParamList(strEntry, fun, inpt, pdf_iter):
    num_pdf_iter = int(inpt["num_pdf_iter"])
    param_list_inpt = [fun(n) for n in inpt[strEntry].split()]
    checkParamListLength(param_list_inpt, num_pdf_iter, strEntry)
    if len(param_list_inpt) == 1 and num_pdf_iter > 1:
        if pdf_iter == 0:
            par.printRoot(
                "WARNING: Assigned same " + strEntry + " for all pdf iter"
            )
        param_list = [param_list_inpt[0] for _ in range(num_pdf_iter)]
        return param_list
    else:
        return param_list_inpt


def computeDistanceToClosestNeighbor(data):
    if not par.irank == par.iroot:
        return

    neigh = NearestNeighbors(n_neighbors=2, n_jobs=-1)
    tree = neigh.fit(data)
    dist = tree.kneighbors(data)[0][:, 1]
    mean = np.mean(dist)
    std = np.std(dist)
    return mean, std


def rescaleData(np_data, inpt):
    scaler = np.load(inpt["scalerFile"])
    np_data_rescaled = np_data.copy()
    np_data_rescaled = (np_data_rescaled - scaler["minVal"]) / (
        0.125 * (scaler["maxVal"] - scaler["minVal"])
    ) - 4
    return np_data_rescaled.astype("float32")


def createFlow(dim, pdf_iter, inpt):
    distribution = distributions.StandardNormal((dim,))
    base_transform_type = "spline"
    grad_norm_clip_value = float(5)

    hidden_features_list = makeParamList(
        "hidden_features", str2int, inpt, pdf_iter
    )
    num_blocks_list = makeParamList("num_blocks", str2int, inpt, pdf_iter)
    num_bins_list = makeParamList("num_bins", str2int, inpt, pdf_iter)
    num_coupling_layer_list = makeParamList(
        "nCouplingLayer", str2int, inpt, pdf_iter
    )

    transform = transforms.CompositeTransform(
        [
            create_base_transform(
                i,
                base_transform_type,
                hidden_features=hidden_features_list[pdf_iter],
                num_blocks=num_blocks_list[pdf_iter],
                num_bins=num_bins_list[pdf_iter],
                tail_bound=5,
                dim=dim,
            )
            for i in range(num_coupling_layer_list[pdf_iter])
        ]
    )
    flow = flows.Flow(transform, distribution)

    return flow


def makePytorchData(
    np_data, BATCH_SIZE, inpt, shuffle=True, device=torch.device("cpu")
):
    np_data_rescaled = rescaleData(np_data, inpt)
    pytorch_data = torch.from_numpy(np_data_rescaled).to(device)
    n_data_points = pytorch_data.data.shape[0]
    data_loader = DataLoader(
        pytorch_data,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=torch.Generator(device=device),
    )
    return data_loader


def prepareLog(pdf_iter):
    os.makedirs("TrainingLog", exist_ok=True)
    filename = f"TrainingLog/log_iter{pdf_iter}.csv"
    try:
        os.remove(filename)
    except:
        pass
    f = open(filename, "a+")
    f.write("step;nll_loss\n")
    f.close()
    return


def logTraining(step, loss, pdf_iter):
    filename = f"TrainingLog/log_iter{pdf_iter}.csv"
    f = open(filename, "a+")
    f.write(f"{int(step)};{loss.item()}\n")
    f.close()
    return


def trainFlow(np_data, flow, pdf_iter, inpt):
    # Timer
    times = time.time()

    learning_rate_list = makeParamList(
        "learning_rate", str2float, inpt, pdf_iter
    )
    num_epochs_list = makeParamList("nEpochs", str2int, inpt, pdf_iter)
    batch_size_list = makeParamList(
        "batch_size_train", str2int, inpt, pdf_iter
    )

    BATCH_SIZE = batch_size_list[pdf_iter]
    EPOCHS = num_epochs_list[pdf_iter]

    grad_norm_clip_value = float(5)
    # create optimizer
    optimizer = optim.Adam(flow.parameters(), lr=learning_rate_list[pdf_iter])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, 0)
    last_loss = 100

    if not par.irank == par.iroot:
        return last_loss

    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    use_gpu = (inpt["use_gpu"] == "True") and (torch.cuda.is_available())

    # Perform training on GPU if possible
    if use_gpu:
        device = torch.device("cuda")
        torch.set_default_dtype(torch.cuda.float32)
    else:
        device = torch.device("cpu")
        torch.set_default_dtype(torch.float32)
    flow = flow.to(device)

    # Train
    data_loader = makePytorchData(
        np_data, BATCH_SIZE, inpt, shuffle=True, device=device
    )
    nBatch = len(data_loader)
    totalSteps = nBatch * EPOCHS
    # Init log
    prepareLog(pdf_iter)
    print_progress_bar(
        0,
        totalSteps,
        prefix="Loss = ? Step %d / %d " % (0, totalSteps),
        suffix="Complete",
        length=50,
        extraCond=(par.irank == par.iroot),
    )
    for epoch in range(EPOCHS):
        for step, batch in enumerate(data_loader):
            flow.train()
            optimizer.zero_grad()
            log_density = flow.log_prob(batch)
            loss = -torch.mean(log_density)
            loss.backward()
            if grad_norm_clip_value is not None:
                clip_grad_norm_(flow.parameters(), grad_norm_clip_value)
            optimizer.step()
            scheduler.step()

            print_progress_bar(
                epoch * nBatch + step + 1,
                totalSteps,
                prefix="Loss %.4f Step %d / %d "
                % (loss.item(), epoch * nBatch + step + 1, totalSteps),
                suffix="Complete",
                length=50,
                extraCond=(par.irank == par.iroot),
            )
            if ((epoch * nBatch + step + 1) % 10) == 0:
                logTraining(epoch * nBatch + step + 1, loss, pdf_iter)

    if use_gpu:
        # Send model back to cpu so that all cpu can read it
        device = torch.device("cpu")
        torch.set_default_dtype(torch.float32)
        flow = flow.to(device)

    torch.save(
        flow.state_dict(),
        f"TrainingLog/modelWeights_iter{pdf_iter}.pt",
    )

    # Timer
    timee = time.time()
    printTiming = inpt["printTiming"] == "True"
    if printTiming:
        par.printRoot(f"Time Train : {timee - times:.2f}s")

    last_loss = loss.item()

    return last_loss


def trainBinPDF(np_data, pdf_iter, inpt):
    # Timer
    times = time.time()
    if not par.irank == par.iroot:
        return None, None
    # Train
    np_data_rescaled = rescaleData(np_data, inpt)
    H, edges = np.histogramdd(np_data_rescaled, bins=int(inpt["num_pdf_bins"]))
    logProb = np.log(1e-16 + H / np.sum(H))

    os.makedirs("TrainingLog", exist_ok=True)
    np.savez(
        f"TrainingLog/modelWeights_iter{pdf_iter}.npz",
        edges=edges,
        logProb=logProb,
    )

    # Timer
    timee = time.time()
    printTiming = inpt["printTiming"] == "True"
    if printTiming:
        par.printRoot(f"Time Train : {timee - times:.2f}s")

    return logProb, edges


def as_list(x):
    if x.shape == (1, 1):
        return [x[0, 0]]
    elif x.shape == (1,):
        return [x[0]]
    else:
        return list(np.squeeze(x))


def computeExpectedNSamples(samplingProb):
    # return np.sum(np.clip(np.float128(samplingProb),0,1))
    return np.sum(np.clip(samplingProb, 0, 1))


def adjustLogSamplingProbMult(logSamplingProb, nDownSamples, nFullSample):
    reduced_logSamplingProb = logSamplingProb
    reduced_nDownSamples = nDownSamples * len(logSamplingProb) // nFullSample
    expectedNumber = computeExpectedNSamples(np.exp(reduced_logSamplingProb))
    factor = reduced_nDownSamples / expectedNumber
    expectedNumber = computeExpectedNSamples(
        np.exp(reduced_logSamplingProb + np.log(factor))
    )
    while (
        abs(expectedNumber - reduced_nDownSamples)
        > reduced_nDownSamples * 0.05
        or expectedNumber < reduced_nDownSamples
    ):
        if expectedNumber < reduced_nDownSamples:
            factor *= 1.1
        else:
            factor *= 0.9
        # print(factor)
        expectedNumber = computeExpectedNSamples(
            np.exp(reduced_logSamplingProb + np.log(factor))
        )
        # print(abs(expectedNumber - nDownSamples)/nDownSamples)

    # print('nSample expected = ',expectedNumber)

    return factor


def adjustLogSamplingProbMultPar(logSamplingProb_, nDownSamples, nFullSample):
    expectedNumber_ = computeExpectedNSamples(np.exp(logSamplingProb_))
    expectedNumber = par.allsumScalar(expectedNumber_)
    factor = nDownSamples / expectedNumber
    expectedNumber_ = computeExpectedNSamples(
        np.exp(logSamplingProb_ + np.log(factor))
    )
    expectedNumber = par.allsumScalar(expectedNumber_)
    while (
        abs(expectedNumber - nDownSamples) > nDownSamples * 0.05
        or expectedNumber < nDownSamples
    ):
        if expectedNumber < nDownSamples:
            factor *= 1.1
        else:
            factor *= 0.9
        # print(factor)
        expectedNumber_ = computeExpectedNSamples(
            np.exp(logSamplingProb_ + np.log(factor))
        )
        expectedNumber = par.allsumScalar(expectedNumber_)
        # print(abs(expectedNumber - nDownSamples)/nDownSamples)

    # print('nSample expected = ',expectedNumber)

    return factor


def evalLogProbNF(flow, np_data_to_downsample, pdf_iter, inpt):
    # Wait for root to be done with training
    par.comm.barrier()

    # Timer
    times = time.time()

    # Environment
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    device = torch.device("cpu")

    # Load trained flow
    flow.load_state_dict(
        torch.load(
            f"TrainingLog/modelWeights_iter{pdf_iter}.pt",
            map_location=device,
        )
    )

    # Evaluation
    flow.eval()
    log_density_np = []
    BATCH_SIZE = int(float(inpt["batch_size_eval"]))
    to_downsample_loader = makePytorchData(
        np_data_to_downsample, BATCH_SIZE, inpt, shuffle=False
    )
    nBatch = len(to_downsample_loader)
    print_progress_bar(
        0,
        nBatch,
        prefix="Eval Step %d / %d " % (0, nBatch),
        suffix="Complete",
        length=50,
        extraCond=(par.irank == par.iroot),
    )
    for step, batch in enumerate(to_downsample_loader):
        batch = batch.to(device)
        log_density = flow.log_prob(batch)
        log_density_np = np.concatenate(
            (log_density_np, tensor2numpy(log_density))
        )
        print_progress_bar(
            step + 1,
            nBatch,
            prefix="Eval Step %d / %d " % (step + 1, nBatch),
            suffix="Complete",
            length=50,
            extraCond=(par.irank == par.iroot),
        )

    # Timer
    timee = time.time()
    printTiming = inpt["printTiming"] == "True"
    if printTiming:
        par.printRoot(f"Time Eval : {par.allmaxScalar(timee - times):.2f}s")

    return log_density_np


def evalLogProbBIN(np_data_to_downsample, pdf_iter, inpt):
    # Wait for root to be done with training
    par.comm.barrier()

    # Timer
    times = time.time()

    # Load trained PDF estimate
    binPDF = np.load(f"TrainingLog/modelWeights_iter{pdf_iter}.npz")
    logProb = binPDF["logProb"]
    edges = binPDF["edges"]

    # Evaluation
    np_data_rescaled = rescaleData(np_data_to_downsample, inpt)
    digit = []
    digit_zeros = []
    for idim in range(np_data_rescaled.shape[1]):
        digit_zeros += list(
            np.argwhere(np_data_rescaled[:, idim] > edges[idim][-1])[:, 0]
        )
        digit_zeros += list(
            np.argwhere(np_data_rescaled[:, idim] < edges[idim][0])[:, 0]
        )
        tmp_dig = np.digitize(np_data_rescaled[:, idim], edges[idim]) - 1
        tmp_dig = np.clip(
            tmp_dig, a_min=0, a_max=logProb.shape[idim] - 1
        ).astype(int)
        digit.append(tmp_dig)
    log_density_np = logProb[tuple(digit)]
    log_density_np[list(set(digit_zeros))] = np.log(1e-16)

    # Timer
    timee = time.time()
    printTiming = inpt["printTiming"] == "True"
    if printTiming:
        par.printRoot(f"Time Eval : {par.allmaxScalar(timee - times):.2f}s")

    return log_density_np


def gatherDownsampledData(
    phaseSpaceSampledData_,
    dataInd_,
    indexSelected_,
    nSample,
    nFullData,
    nSamplesAssigned,
):
    nSnap_, startSnap_ = par.partitionData(nFullData)
    # Back to global indexing
    if dataInd_ is None:
        indexSelected_ = np.array(indexSelected_).astype("int32") + startSnap_
    else:
        indexSelected_ = dataInd_[indexSelected_]
    # Gather indices and data
    indexSelected = None
    indexSelected = par.gather1DArray(
        indexSelected_, rootId=par.iroot - 1, N=nSamplesAssigned
    ).astype("int32")
    phaseSpaceSampledData = None
    phaseSpaceSampledData = par.gather2DArray(
        phaseSpaceSampledData_,
        rootId=0,
        N1Loc=len(indexSelected_),
        N1Glob=nSamplesAssigned,
        N2=phaseSpaceSampledData_.shape[1],
    ).astype("float32")
    return phaseSpaceSampledData[:nSample], indexSelected[:nSample]


def downSample(
    data_to_downsample_,
    dataInd_,
    log_density_np_,
    log_density_np_adjustment,
    nSample,
    nFullData,
    inpt,
):
    # Mode of adjustment of the sampling probability
    nWorkingDataAdjustment = int(float(inpt["nWorkingDataAdjustment"]))
    if nWorkingDataAdjustment < 0:
        use_serial_adjustment = False
        try:
            data_freq_adjustment = int(inpt["data_freq_adjustment"])
        except KeyError:
            data_freq_adjustment = 1
    else:
        use_serial_adjustment = True

    # Adjust sampling probability to get the desired number of samples
    if use_serial_adjustment:
        if par.irank == par.iroot:
            factor = adjustLogSamplingProbMult(
                -log_density_np_adjustment, nSample, nFullData
            )
        else:
            factor = None
        # Broadcast multiplicative factor
        factor = par.bcast(factor)
    else:
        factor = adjustLogSamplingProbMultPar(
            -log_density_np_[::data_freq_adjustment],
            int(nSample / data_freq_adjustment),
            nFullData,
        )

    # Sample according to the acceptance probability
    log_samplingProb_ = np.clip(
        np.log(factor) - log_density_np_, a_min=None, a_max=0
    )
    samplingProb_ = np.exp(log_samplingProb_)
    nSamplesAssigned = 0
    nSamplesAssigned_ = 0
    niter = 0
    nData_ = data_to_downsample_.shape[0]
    indexSelected_ = []
    while nSamplesAssigned < nSample:
        randNum_ = np.random.uniform(0, 1, nData_).astype("float32")
        niter += 1
        Ind = np.argwhere(samplingProb_ > randNum_)
        set1 = set(as_list(Ind))
        set2 = set(indexSelected_)
        Ind = list(set1 - set2)
        for ind in Ind:
            indexSelected_.append(ind)
            nSamplesAssigned_ += 1
        nSamplesAssigned = par.allsumScalar(nSamplesAssigned_)
        # par.printRoot("nSample Assigned = ",nSamplesAssigned)

    phaseSpaceSampledData_ = data_to_downsample_[indexSelected_, :]
    # Gather
    phaseSpaceSampledData, indexDownsampledData = gatherDownsampledData(
        phaseSpaceSampledData_,
        dataInd_,
        indexSelected_,
        nSample,
        nFullData,
        nSamplesAssigned,
    )

    return (
        phaseSpaceSampledData,
        indexDownsampledData,
        samplingProb_,
        log_samplingProb_,
    )


def checkLoss(pdf_iter, flow_nll_loss):
    if (not pdf_iter == 1) or (not par.irank == par.iroot):
        return
    if flow_nll_loss[1] <= flow_nll_loss[0]:
        par.printRoot(
            "WARNING: Flow loss did not increase with iteration. Try increasing the number of epochs"
        )
    return


def checkProcedure(meanCriterion, nSample, randomCriterion):
    # Make sure improvement was observed compared to random sampling
    errorSignal = False
    if meanCriterion[-1] < randomCriterion:
        par.printRoot(
            f"ERROR: Failure of the sampling procedure for nSample : {nSample}"
        )
        errorSignal = True
    if errorSignal:
        return

    # Make sure that the sampling quality improved between first and second iteration
    num_pdf_iter = meanCriterion.shape[0]
    if num_pdf_iter < 2:
        return
    if meanCriterion[-1] < meanCriterion[0]:
        par.printRoot(
            f"WARNING: Possible failure of the sampling procedure for nSample : {nSample}"
        )
