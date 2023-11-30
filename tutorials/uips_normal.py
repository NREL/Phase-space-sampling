import os

import numpy as np
from prettyPlot.parser import parse_input_file
from prettyPlot.plotting import plt, pretty_labels, pretty_legend

from uips import UIPS_INPUT_DIR
from uips.wrapper import downsample_dataset_from_input

ndat = int(1e5)
nSampl = 100
ds = np.random.multivariate_normal([0, 0], np.eye(2), size=ndat)
np.save("norm.npy", ds)

inpt = parse_input_file(os.path.join(UIPS_INPUT_DIR, "input2D"))
inpt["dataFile"] = "norm.npy"
inpt["nWorkingData"] = f"{ndat} {ndat}"
inpt["nEpochs"] = f"5 20"
inpt["nSamples"] = f"{nSampl}"
best_files = downsample_dataset_from_input(inpt)


downsampled_ds = {}
for nsamp in best_files:
    downsampled_ds[nsamp] = np.load(best_files[nsamp])["data"]

fig = plt.figure()
plt.plot(ds[:, 0], ds[:, 1], "o", color="k", label="full DS")
plt.plot(
    downsampled_ds[nSampl][:, 0],
    downsampled_ds[nSampl][:, 1],
    "o",
    color="r",
    label="downsampled",
)
pretty_labels("", "", 14)
pretty_legend()
plt.savefig("normal_downsample.png")
