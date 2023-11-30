import argparse

from uips.wrapper import downsample_dataset_from_input

parser = argparse.ArgumentParser(description="Downsampler")
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

downsample_dataset_from_input(args.input)
