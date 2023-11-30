import os
import sys
import time

import numpy as np
import torch
from prettyPlot.parser import parse_input_file

import uips.sampler as sampler
import uips.utils.parallel as par
from uips.utils.dataUtils import prepareData
from uips.utils.fileFinder import find_input
from uips.utils.plotFun import *
from uips.utils.torchutils import get_num_parameters


def downsample_dataset_from_input(inpt_file):
    inpt_file = find_input(inpt_file)
    inpt = parse_input_file(inpt_file)
    use_normalizing_flow = inpt["pdf_method"].lower() == "normalizingflow"
    use_bins = inpt["pdf_method"].lower() == "bins"

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~ Parameters to save
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # List of sample size
    nSamples = [int(float(n)) for n in inpt["nSamples"].split()]
    # Data size used to adjust the sampling probability
    nWorkingDataAdjustment = int(float(inpt["nWorkingDataAdjustment"]))
    if nWorkingDataAdjustment < 0:
        use_serial_adjustment = False
    else:
        use_serial_adjustment = True
    # Data size used to learn the data probability
    nWorkingDatas = [int(float(n)) for n in inpt["nWorkingData"].split()]
    if len(nWorkingDatas) == 1:
        nWorkingDatas = nWorkingDatas * int(inpt["num_pdf_iter"])
    for nWorkingData in nWorkingDatas:
        if not nWorkingData in nSamples:
            nSamples += [nWorkingData]
    # Do we compute the neighbor distance criterion
    computeCriterion = inpt["computeDistanceCriterion"] == "True"
    try:
        nSampleCriterionLimit = int(inpt["nSampleCriterionLimit"])
    except:
        nSampleCriterionLimit = int(1e5)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~ Environment
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    use_gpu = (
        (inpt["use_gpu"] == "True")
        and (torch.cuda.is_available())
        and (par.irank == par.iroot)
    )
    if use_gpu:
        # GPU SETTING
        device = torch.device("cuda")
        torch.set_default_dtype(torch.cuda.float32)
    else:
        # CPU SETTING
        device = torch.device("cpu")
        torch.set_default_dtype(torch.float32)
    # REPRODUCIBILITY
    torch.manual_seed(int(inpt["seed"]) + par.irank)
    np.random.seed(int(inpt["seed"]) + par.irank)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~ Prepare Data and scatter across processors
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    data_to_downsample_, dataInd_, working_data, nFullData = prepareData(inpt)

    dim = data_to_downsample_.shape[1]

    # Compute uniform sampling criterion of random data
    randomCriterion = np.zeros(len(nSamples))
    if par.irank == par.iroot and computeCriterion:
        par.printRoot("RANDOM: ")
        for inSample, nSample in enumerate(nSamples):
            if nSample <= nSampleCriterionLimit:
                random_sampled_data = working_data[:nSample, :]
                mean, std = sampler.computeDistanceToClosestNeighbor(
                    sampler.rescaleData(random_sampled_data, inpt)
                )
                randomCriterion[inSample] = mean
                par.printRoot(
                    f"\t nSample {nSample} mean dist = {mean:.4f}, std dist = {std:.4f}"
                )

    # Prepare arrays used for sanity checks
    meanCriterion = np.zeros((int(inpt["num_pdf_iter"]), len(nSamples)))
    stdCriterion = np.zeros((int(inpt["num_pdf_iter"]), len(nSamples)))
    flow_nll_loss = np.zeros(int(inpt["num_pdf_iter"]))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~ Downsample
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    data_for_pdf_est = working_data

    for pdf_iter in range(int(inpt["num_pdf_iter"])):
        if use_normalizing_flow:
            # Create the normalizing flow
            flow = sampler.createFlow(dim, pdf_iter, inpt)
            # flow = flow.to(device)
            n_params = get_num_parameters(flow)
            par.printRoot(
                "There are {} trainable parameters in this model.".format(
                    n_params
                )
            )

            # Train (happens on 1 proc)
            flow_nll_loss[pdf_iter] = sampler.trainFlow(
                data_for_pdf_est, flow, pdf_iter, inpt
            )
            sampler.checkLoss(pdf_iter, flow_nll_loss)

            # Evaluate probability: This is the expensive step (happens on multi processors)
            log_density_np_ = sampler.evalLogProbNF(
                flow, data_to_downsample_, pdf_iter, inpt
            )

        if use_bins:
            bin_pdfH, bin_pdfEdges = sampler.trainBinPDF(
                data_for_pdf_est, pdf_iter, inpt
            )
            # Evaluate probability: This is the expensive step (happens on multi processors)
            log_density_np_ = sampler.evalLogProbBIN(
                data_to_downsample_, pdf_iter, inpt
            )

        if use_serial_adjustment:
            log_density_np_for_adjust = par.gatherNelementsInArray(
                log_density_np_, nWorkingDataAdjustment
            )
        else:
            log_density_np_for_adjust = None

        # Correct probability estimate
        if pdf_iter > 0:
            log_density_np_ = log_density_np_ - log_samplingProb_
            if use_serial_adjustment:
                log_density_np_for_adjust = par.gatherNelementsInArray(
                    log_density_np_, nWorkingDataAdjustment
                )
            else:
                log_density_np_for_adjust = None

        par.printRoot(f"TRAIN ITER {pdf_iter}")

        for inSample, nSample in enumerate(nSamples):
            # Downsample
            (
                downSampledData,
                downSampledIndices,
                samplingProb_,
                log_samplingProb_,
            ) = sampler.downSample(
                data_to_downsample_,
                dataInd_,
                log_density_np_,
                log_density_np_for_adjust,
                nSample,
                nFullData,
                inpt,
            )

            # Plot
            # cornerPlotScatter(downSampledData,title='downSampled npts='+str(nSample)+', iter='+str(pdf_iter))
            # Get criterion
            if computeCriterion and par.irank == par.iroot:
                if nSample <= nSampleCriterionLimit:
                    mean, std = sampler.computeDistanceToClosestNeighbor(
                        sampler.rescaleData(downSampledData, inpt)
                    )
                    meanCriterion[pdf_iter, inSample] = mean
                    stdCriterion[pdf_iter, inSample] = std
                    par.printRoot(
                        f"\t nSample {nSample} mean dist = {mean:.4f}, std dist = {std:.4f}"
                    )

            if pdf_iter == int(inpt["num_pdf_iter"]) - 1:
                # Last pdf iter : Root proc saves downsampled data, and checks the outcome
                sampler.checkProcedure(
                    meanCriterion[:, inSample],
                    nSample,
                    randomCriterion[inSample],
                )
            if par.irank == par.iroot:
                np.savez(
                    f"{inpt['prefixDownsampledData']}_{nSample}_it{pdf_iter}.npz",
                    data=downSampledData,
                    indices=downSampledIndices,
                )

        if not (pdf_iter == int(inpt["num_pdf_iter"]) - 1):
            # Prepare data for the next training iteration
            (
                downSampledData,
                _,
                samplingProb_,
                log_samplingProb_,
            ) = sampler.downSample(
                data_to_downsample_,
                dataInd_,
                log_density_np_,
                log_density_np_for_adjust,
                nWorkingDatas[pdf_iter + 1],
                nFullData,
                inpt,
            )
            data_for_pdf_est = downSampledData

    # Advise for best sampling
    if par.irank == par.iroot:
        if np.amax(meanCriterion) > 0:
            print("\n")
            maxCrit = np.argmax(meanCriterion, axis=0)
            nSamples_asked = [int(float(n)) for n in inpt["nSamples"].split()]
            for iSample, nSample in enumerate(nSamples_asked):
                print(
                    f"For sample {nSample} use {inpt['prefixDownsampledData']}_{nSample}_it{maxCrit[iSample]}.npz"
                )
            print("\n")


# if par.irank==par.iroot:
#    plt.show()
