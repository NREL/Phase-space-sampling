import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("utils")
import myparser
from plotsUtil import *

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Parse input
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

inpt = myparser.parseInputFile()

nIter = int(inpt["num_pdf_iter"])

# Folder where figures are saved
figureFolder = "Figures"
os.makedirs(figureFolder, exist_ok=True)

fig, axs = plt.subplots(1, nIter, figsize=(10, 3))
for i in range(nIter):
    Loss = np.genfromtxt(
        f"TrainingLog/log_iter{i}.csv", delimiter=";", skip_header=1
    )
    axs[i].plot(Loss[:, 0], Loss[:, 1], color="k", linewidth=3)
    axprettyLabels(axs[i], "Step", "Loss", 14, title=f"iteration {i}")

plt.savefig(figureFolder + "/loss.png")
plt.close()
