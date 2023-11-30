from phaseSpaceSampling.wrapper import downsample_dataset_from_input
from phaseSpaceSampling import PSS_INPUT_DIR
import os 

def test_nf_input():
    input_file = os.path.join(PSS_INPUT_DIR, "input_test")
    downsample_dataset_from_input(input_file)

def test_bins_input():
    input_file = os.path.join(PSS_INPUT_DIR, "input_test_bins")
    downsample_dataset_from_input(input_file)
