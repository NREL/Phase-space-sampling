import numpy as np
import torch
import os

#import matplotlib.pyplot as plt

import sys
sys.path.append('utils')
import utils

from myProgressBar import printProgressBar
from dataUtils import prepareData
from plotFun import *
import sampler
import time
import myparser

import parallel as par

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Parse input
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

inpt = myparser.parseInputFile()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Parameters to save
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# List of sample size
nSamples = [int(float(n)) for n in inpt['nSamples'].split()] 
# Data size used to adjust the sampling probability
nWorkingDataAdjustment = int(float(inpt['nWorkingDataAdjustment']))
if nWorkingDataAdjustment<0:
    use_serial_adjustment = False
else:
    use_serial_adjustment = True
# Data size used to learn the data probability
nWorkingData = int(float(inpt['nWorkingData']))
if not nWorkingData in nSamples:
    nSamples += [nWorkingData] 
# Do we compute the neighbor distance criterion
computeCriterion = (inpt['computeDistanceCriterion']=='True')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Environment
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

os.environ['KMP_DUPLICATE_LIB_OK']='True'
use_gpu = (inpt['use_gpu']=='True') and (torch.cuda.is_available()) and (par.irank==par.iroot)
if use_gpu:
    # GPU SETTING
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    # CPU SETTING
    device = torch.device('cpu')
    torch.set_default_tensor_type('torch.FloatTensor')
# REPRODUCIBILITY
torch.manual_seed(int(inpt['seed'])+par.irank)
np.random.seed(int(inpt['seed'])+par.irank)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Prepare Data and scatter across processors
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

data_to_downsample_, working_data, nFullData = prepareData(inpt)

dim = data_to_downsample_.shape[1]

# Compute uniform sampling criterion of random data
randomCriterion = np.zeros(len(nSamples))
if par.irank==par.iroot and computeCriterion:
    par.printRoot("RANDOM: ")
    for inSample, nSample in enumerate(nSamples):
        random_sampled_data = working_data[:nSample,:]
        mean, std = sampler.computeDistanceToClosestNeighbor(sampler.rescaleData(random_sampled_data,inpt))
        randomCriterion[inSample] = mean
        par.printRoot("\t nSample %d mean dist = %.4f, std dist = %.4f" %(nSample,mean,std))
       
# Prepare arrays used for sanity checks
meanCriterion = np.zeros((int(inpt['num_flow_iter']),len(nSamples)))
stdCriterion = np.zeros((int(inpt['num_flow_iter']),len(nSamples)))
flow_nll_loss = np.zeros(int(inpt['num_flow_iter']))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Downsample
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

data_flow = working_data

for flow_iter in range(int(inpt['num_flow_iter'])):

    # Create the normalizing flow 
    flow = sampler.createFlow(dim,flow_iter,inpt)
    #flow = flow.to(device) 
    n_params = utils.get_num_parameters(flow)
    par.printRoot('There are {} trainable parameters in this model.'.format(n_params))
    
    # Train (happens on 1 proc)
    flow_nll_loss[flow_iter] = sampler.trainFlow(data_flow,flow,flow_iter,inpt)
    sampler.checkLoss(flow_iter,flow_nll_loss)
   
    # Evaluate probability: This is the expensive step (happens on multi processors)
    log_density_np_ = sampler.evalLogProb(flow,data_to_downsample_, nFullData, flow_iter, inpt)
    if use_serial_adjustment:
        log_density_np_for_adjust = par.gatherNelementsInArray(log_density_np_,nWorkingDataAdjustment)
    else:
        log_density_np_for_adjust = None

    # Correct probability estimate
    if flow_iter>0:
        log_density_np_ = log_density_np_ - log_samplingProb_
        log_density_np_for_adjust = par.gatherNelementsInArray(log_density_np_,nWorkingDataAdjustment)
        

    par.printRoot('TRAIN ITER '+str(flow_iter))

    for inSample, nSample in enumerate(nSamples):
        # Downsample
        downSampledData, downSampledIndices, samplingProb_, log_samplingProb_ = sampler.downSample(data_to_downsample_,
                                                                                log_density_np_,
                                                                                log_density_np_for_adjust,
                                                                                nSample,nFullData,
                                                                                inpt) 

        # Plot
        #cornerPlotScatter(downSampledData,title='downSampled npts='+str(nSample)+', iter='+str(flow_iter))
        # Get criterion
        if computeCriterion and par.irank==par.iroot:
            mean, std = sampler.computeDistanceToClosestNeighbor(sampler.rescaleData(downSampledData,inpt))
            meanCriterion[flow_iter,inSample] = mean
            stdCriterion[flow_iter,inSample] = std
            par.printRoot("\t nSample %d mean dist = %.4f, std dist = %.4f" %(nSample,mean,std))

        if flow_iter == int(inpt['num_flow_iter'])-1:
            # Last flow iter : Root proc saves downsampled data, and checks the outcome
            if par.irank==par.iroot:
                 np.savez(inpt['prefixDownsampledData']+'_'+str(nSample)+'.npz',data=downSampledData,indices=downSampledIndices)
                 sampler.checkProcedure(meanCriterion[:,inSample],nSample,randomCriterion[inSample])
 
    if not (flow_iter == int(inpt['num_flow_iter'])-1):
        # Prepare data for the next training iteration
        downSampledData, _, samplingProb_, log_samplingProb_ = sampler.downSample(data_to_downsample_,
                                                                                  log_density_np_,
                                                                                  log_density_np_for_adjust,
                                                                                  nWorkingData,nFullData,
                                                                                  inpt)
        data_flow = downSampledData
   
 
#if par.irank==par.iroot:
#    plt.show()
