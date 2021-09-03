# Scope
This folder is NOT necessary for using the phase-space sampling package. It only contains the code necessary to reproduce the results shown in the paper:

Uniform-in-phase-space data-selection with iterative normalizing flows, Under review.

# Data
The folder `data/` contains the phase space sampled data for five independent runs of each case described in the paper aforementionned. To run all the codes and reproduce the results, the file `fullData.npy` is necessary. Due to space constraints it is not in the repository but is available upon request.


# Usage
In the folders `artificialCase/` and `combustionCase/`, a script to train many Gaussian processes and Neural net is available. In addition, a script to transform the results in latex format is provided.
- `python trainGP.py`: trains Gaussian Processes
- `python trainNN.py`: trains a Neural net
- `python generateLatexTable.py`: generates the latex tables summarizing the results. By default, the raw results used for the paper are used. 


