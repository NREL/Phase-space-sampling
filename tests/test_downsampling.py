import os

from uips import UIPS_INPUT_DIR
from uips.wrapper import downsample_dataset_from_input_file


def test_nf_input():
    input_file = os.path.join(UIPS_INPUT_DIR, "input_test")
    downsample_dataset_from_input_file(input_file)


def test_bins_input():
    input_file = os.path.join(UIPS_INPUT_DIR, "input_test_bins")
    downsample_dataset_from_input_file(input_file)
