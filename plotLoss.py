import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('utils')
from plotsUtil import *
import myparser

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Parse input
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

inpt = myparser.parseInputFile()

nIter = int(inpt['num_flow_iter'])

# Folder where figures are saved
figureFolder = 'Figures'
os.makedirs(figureFolder,exist_ok=True)

fig, axs = plt.subplots(1, nIter,figsize=(10,3))
for i in range(nIter):
    Loss=np.genfromtxt('TrainingLog/log_iter'+str(i)+'.csv',delimiter=';',skip_header=1)
    axs[i].plot(Loss[:,0],Loss[:,1],color='k',linewidth=3)
    axprettyLabels(axs[i],'Step','Loss',14,title='iteration '+str(i))

plt.savefig(figureFolder+'/loss.png')
plt.close()
