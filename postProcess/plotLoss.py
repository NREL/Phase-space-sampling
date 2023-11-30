import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from prettyPlot.parser import parse_input_file
from prettyPlot.plotting import pretty_labels

import uips.utils.parallel as par
from uips import UIPS_INPUT_DIR

parser = argparse.ArgumentParser(description="Loss plotting")
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
inpt_file = args.input
if not os.path.isfile(inpt_file):
    new_inpt_file = os.path.join(UIPS_INPUT_DIR, os.path.split(inpt_file)[-1])
    par.printRoot(f"WARNING: {inpt_file} not found trying {new_inpt_file} ...")
    if not os.path.isfile(new_inpt_file):
        par.printRoot(
            f"ERROR: could not open data {inpt_file} or {new_inpt_file}"
        )
        sys.exit()
    else:
        inpt_file = new_inpt_file

inpt = parse_input_file(inpt_file)


if inpt["pdf_method"].lower() == "normalizingflow":
    nIter = int(inpt["num_pdf_iter"])

    # Folder where figures are saved
    figureFolder = "Figures"
    os.makedirs(figureFolder, exist_ok=True)
    if nIter > 1:
        fig, axs = plt.subplots(1, nIter, figsize=(10, 3))
        for i in range(nIter):
            Loss = np.genfromtxt(
                f"TrainingLog/log_iter{i}.csv", delimiter=";", skip_header=1
            )
            axs[i].plot(Loss[:, 0], Loss[:, 1], color="k", linewidth=3)
            pretty_labels(
                "Step", "Loss", 14, title=f"iteration {i}", ax=axs[i]
            )
    else:
        fig = plt.figure()
        Loss = np.genfromtxt(
            f"TrainingLog/log_iter0.csv", delimiter=";", skip_header=1
        )
        plt.plot(Loss[:, 0], Loss[:, 1], color="k", linewidth=3)
        pretty_labels("Step", "Loss", 14, title=f"iteration 0")

    plt.savefig(figureFolder + "/loss.png")
    plt.close()
