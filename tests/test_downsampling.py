import numpy
import os

from uips.options import UIPSOptions, stepOptions
from uips import UIPS_INPUT_DIR
from uips.wrapper import (
    downsample_dataset_from_input,
    downsample_dataset_from_input_file,
)


def test_nf_input_file():
    input_file = os.path.join(UIPS_INPUT_DIR, "input_test")
    downsample_dataset_from_input_file(input_file)


def test_bins_input():
    input_file = os.path.join(UIPS_INPUT_DIR, "input_test_bins")
    downsample_dataset_from_input_file(input_file)


def test_nf_input():

    step1_options = stepOptions(
        nEpochs = 1,
        batch_size_train = 2048,
        nWorkingData = 1e4,
        learning_rate = 2e-4,
        nCouplingLayer = 2,
        num_bins = 3,
        hidden_features = 12,
        num_blocks = 3
    )
    step2_options = step1_options
    downsampler_options = UIPSOptions(
        num_pdf_iter = 2,
        pdfMethod = "NormalizingFlow",
        use_gpu = False,
        printTiming = True,
        nDatReduced = 1e5,
        preShuffled = True,
        scalerFile = "scaler.npz",
        nSamples_list = [1e3],
        computeDistanceCriterion = True,
        prefixDownsampledData = "downSampledData",
        data_freq_adjustment = 1,
        nWorkingDataAdjustment = -1,
    )

    dataFile = "../data/combustion2DToDownsampleSmall.npy"
    dataset = np.load(dataFile)
    downsample_dataset_from_input(downsampler_options, dataset)


if __name__ == "__main__":
    test_nf_input()
    test_bins_input()
