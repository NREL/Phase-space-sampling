import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from torch.utils.data import DataLoader
import numpy as np
import sys
sys.path.append('utils')
from nde import distributions, flows, transforms
from myProgressBar import printProgressBar
from sklearn.neighbors import NearestNeighbors
from flowUtils import *
import os

import nn as nn_
import utils

import parallel as par
import time

def str2float(str):
    return float(str)

def str2int(str):
    return int(str)

def str2BigInt(str):
    return int(float(str))

def checkParamListLength(listToCheck,targetLength,name):
    if not len(listToCheck) == targetLength and (not len(listToCheck)==1):
        par.printAll('Expected ' + name + ' of size ' + str(targetLength) + ' got ' + str(len(listToCheck)))
        par.comm.Abort()

def makeParamList(strEntry,fun,inpt,flow_iter):
    num_flow_iter = int(inpt['num_flow_iter'])
    param_list_inpt = [fun(n) for n in inpt[strEntry].split()]
    checkParamListLength(param_list_inpt,num_flow_iter,strEntry)
    if len(param_list_inpt)==1 and num_flow_iter>1:
        if flow_iter==0:
            par.printRoot('WARNING: Assigned same ' + strEntry + ' for all flow iter')
        param_list = [param_list_inpt[0] for _ in range(num_flow_iter)]
        return param_list
    else:
        return param_list_inpt

def computeDistanceToClosestNeighbor(data):

    if not par.irank==par.iroot: 
        return

    neigh = NearestNeighbors(n_neighbors=2,n_jobs=-1)
    tree = neigh.fit(data)
    dist = tree.kneighbors(data)[0][:,1]
    mean = np.mean(dist)
    std = np.std(dist)
    return mean, std

def rescaleData(np_data,inpt):
    scaler = np.load(inpt['scalerFile'])
    np_data_rescaled = np_data.copy()
    np_data_rescaled = (np_data_rescaled - scaler['minVal'])/(0.125*(scaler['maxVal']-scaler['minVal'])) - 4
    return np_data_rescaled.astype('float32')

def createFlow(dim,flow_iter,inpt):
    distribution = distributions.StandardNormal((dim,))
    base_transform_type = 'spline'
    grad_norm_clip_value = float(5)
   
    hidden_features_list    = makeParamList('hidden_features',str2int,inpt,flow_iter)
    num_blocks_list         = makeParamList('num_blocks',str2int,inpt,flow_iter)
    num_bins_list           = makeParamList('num_bins',str2int,inpt,flow_iter)
    num_coupling_layer_list = makeParamList('nCouplingLayer',str2int,inpt,flow_iter)
   
    transform = transforms.CompositeTransform(
    [
        create_base_transform(i,
                              base_transform_type,
                              hidden_features=hidden_features_list[flow_iter],
                              num_blocks=num_blocks_list[flow_iter],
                              num_bins=num_bins_list[flow_iter],
                              tail_bound=5, dim=dim) for i in range(num_coupling_layer_list[flow_iter])
    ]
    )
    flow = flows.Flow(transform, distribution)
    
    return flow

def makePytorchData(np_data, BATCH_SIZE, inpt, shuffle=True, device=torch.device('cpu')):
    np_data_rescaled = rescaleData(np_data,inpt)
    pytorch_data=torch.from_numpy(np_data_rescaled).to(device)
    n_data_points = pytorch_data.data.shape[0]
    data_loader = DataLoader(pytorch_data, batch_size=BATCH_SIZE, shuffle=shuffle)
    return data_loader

def prepareLog(flow_iter):
    os.makedirs('TrainingLog',exist_ok=True)
    filename = 'TrainingLog/log_iter'+str(flow_iter)+'.csv'
    try:
        os.remove(filename)
    except:
        pass
    f = open(filename,'a+')
    f.write('step;nll_loss\n')
    f.close()
    return

def logTraining(step,loss,flow_iter):
    filename = 'TrainingLog/log_iter'+str(flow_iter)+'.csv'
    f = open(filename,'a+')
    f.write(str(int(step))+';'+
            str(loss.item())+'\n')
    f.close()
    return 

def trainFlow(np_data,flow,flow_iter,inpt):
   
    # Timer
    times = time.time()    

    learning_rate_list    = makeParamList('learning_rate',str2float,inpt,flow_iter)
    num_epochs_list       = makeParamList('nEpochs',str2int,inpt,flow_iter)
    batch_size_list       = makeParamList('batch_size_train',str2int,inpt,flow_iter)

    BATCH_SIZE = batch_size_list[flow_iter]
    EPOCHS     = num_epochs_list[flow_iter]


    grad_norm_clip_value = float(5)
    # create optimizer
    optimizer = optim.Adam(flow.parameters(), lr=learning_rate_list[flow_iter])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, 0)
    last_loss = 100
    

    if not par.irank==par.iroot:
        return last_loss

    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    use_gpu = (inpt['use_gpu']=='True') and (torch.cuda.is_available())

    # Perform training on GPU if possible
    if use_gpu:
        device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')
    flow=flow.to(device)

    # Train
    data_loader = makePytorchData(np_data,BATCH_SIZE, inpt, shuffle=True,device=device)
    nBatch = len(data_loader)
    totalSteps = nBatch*EPOCHS
    # Init log
    prepareLog(flow_iter)
    printProgressBar(0, totalSteps, prefix = 'Loss = ? Step %d / %d ' % 
                                             ( 0, totalSteps),
                                    suffix = 'Complete', 
                                    length = 50)
    for epoch in range(EPOCHS):
        for step, batch in enumerate(data_loader):
            flow.train()
            optimizer.zero_grad()
            log_density = flow.log_prob(batch)
            loss = - torch.mean(log_density)
            loss.backward()
            if grad_norm_clip_value is not None:
                clip_grad_norm_(flow.parameters(), grad_norm_clip_value)
            optimizer.step()
            scheduler.step()
    
            printProgressBar(epoch*nBatch + step+1, totalSteps, prefix = 'Loss %.4f Step %d / %d ' % 
                                                                         ( loss.item(), epoch*nBatch + step+1, totalSteps),
                                                                suffix = 'Complete', 
                                                                length = 50)
            if ((epoch*nBatch + step+1) % 10) == 0:
                logTraining(epoch*nBatch + step+1,loss,flow_iter)
    

    if use_gpu:
        # Send model back to cpu so that all cpu can read it
        device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')
        flow = flow.to(device)
 
    torch.save(flow.state_dict(), 'TrainingLog/modelWeights_iter'+str(flow_iter)+'.pt')

    # Timer
    timee = time.time()
    printTiming = (inpt['printTiming'] == 'True')
    if printTiming:
        par.printRoot('Time Train : %.2f' % (timee-times))
    
    last_loss = loss.item()

    return last_loss

def as_list(x):
    if x.shape == (1,1):
        return [x[0,0]]
    elif x.shape == (1,):
        return [x[0]]
    else:
        return list(np.squeeze(x))

def computeExpectedNSamples(samplingProb):
    #return np.sum(np.clip(np.float128(samplingProb),0,1))
    return np.sum(np.clip(samplingProb,0,1))

def adjustLogSamplingProbMult(logSamplingProb,nDownSamples,nFullSample):
    reduced_logSamplingProb = logSamplingProb
    reduced_nDownSamples = nDownSamples * len(logSamplingProb) // nFullSample
    expectedNumber = computeExpectedNSamples(np.exp(reduced_logSamplingProb))
    factor = reduced_nDownSamples/expectedNumber
    expectedNumber = computeExpectedNSamples(np.exp(reduced_logSamplingProb+np.log(factor)))
    while abs(expectedNumber - reduced_nDownSamples)>reduced_nDownSamples*0.05 or expectedNumber<reduced_nDownSamples:
          if expectedNumber<reduced_nDownSamples:
              factor *= 1.1
          else:
              factor *= 0.9
          #print(factor)
          expectedNumber = computeExpectedNSamples(np.exp(reduced_logSamplingProb+np.log(factor)))
          #print(abs(expectedNumber - nDownSamples)/nDownSamples)

    #print('nSample expected = ',expectedNumber)

    return factor

def adjustLogSamplingProbMultPar(logSamplingProb_,nDownSamples,nFullSample):
    expectedNumber_ = computeExpectedNSamples(np.exp(logSamplingProb_))
    expectedNumber  = par.allsumScalar(expectedNumber_)
    factor = nDownSamples/expectedNumber
    expectedNumber_ = computeExpectedNSamples(np.exp(logSamplingProb_+np.log(factor)))
    expectedNumber  = par.allsumScalar(expectedNumber_)
    while abs(expectedNumber - nDownSamples)>nDownSamples*0.05 or expectedNumber<nDownSamples:
          if expectedNumber<nDownSamples:
              factor *= 1.1
          else:
              factor *= 0.9
          #print(factor)
          expectedNumber_ = computeExpectedNSamples(np.exp(logSamplingProb_+np.log(factor)))
          expectedNumber  = par.allsumScalar(expectedNumber_)
          #print(abs(expectedNumber - nDownSamples)/nDownSamples)

    #print('nSample expected = ',expectedNumber)

    return factor

def evalLogProb(flow,np_data_to_downsample, nFullData, flow_iter, inpt):

    # Wait for root to be done with training
    par.comm.barrier()

    # Timer
    times = time.time()

    # Environment
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    device = torch.device('cpu')

    # Load trained flow
    flow.load_state_dict(torch.load('TrainingLog/modelWeights_iter'+str(flow_iter)+'.pt',map_location=device))

    # Evaluation
    flow.eval()
    log_density_np = []
    BATCH_SIZE = int(float(inpt['batch_size_eval']))
    to_downsample_loader = makePytorchData(np_data_to_downsample,BATCH_SIZE, inpt, shuffle=False)
    nBatch = len(to_downsample_loader)
    printProgressBar(0, nBatch, prefix = 'Eval Step %d / %d ' % ( 0, nBatch),suffix = 'Complete', length = 50)
    for step, batch in enumerate(to_downsample_loader):
        batch = batch.to(device)
        log_density = flow.log_prob(batch)
        log_density_np = np.concatenate(
            (log_density_np, utils.tensor2numpy(log_density))
        )
        printProgressBar(step+1, nBatch, prefix = 'Eval Step %d / %d ' % ( step+1, nBatch),suffix = 'Complete', length = 50)

    # Timer
    timee = time.time()
    printTiming = (inpt['printTiming'] == 'True')
    if printTiming:
        par.printRoot('Time Eval : %.2f' % (par.allmaxScalar(timee-times)))

    return log_density_np

def gatherDownsampledData(phaseSpaceSampledData_,indexSelected_,nSample,nFullData,nSamplesAssigned):
    nSnap_, starSnap_ = par.partitionData(nFullData)
    # Back to global indexing
    indexSelected_ = np.array(indexSelected_).astype('int32') + starSnap_ 
    # Gather indices and data
    indexSelected = None
    indexSelected = par.gather1DArray(indexSelected_,rootId=par.iroot-1,N=nSamplesAssigned).astype('int32')
    phaseSpaceSampledData = None
    phaseSpaceSampledData = par.gather2DArray(phaseSpaceSampledData_,rootId=0,N1Loc=len(indexSelected_),
                                                                              N1Glob=nSamplesAssigned,
                                                                              N2=phaseSpaceSampledData_.shape[1]).astype('float32')
    return phaseSpaceSampledData[:nSample], indexSelected[:nSample]
 

def downSample(data_to_downsample_,log_density_np_,log_density_np_adjustment,nSample,nFullData,inpt):

    # Mode of adjustment of the sampling probability
    nWorkingDataAdjustment = int(float(inpt['nWorkingDataAdjustment']))
    if nWorkingDataAdjustment<0:
        use_serial_adjustment = False
        try:
            data_freq_adjustment = int(inpt['data_freq_adjustment'])
        except KeyError:
            data_freq_adjustment = 1 
    else:
        use_serial_adjustment = True

    # Adjust sampling probability to get the desired number of samples
    if use_serial_adjustment:
        if par.irank==par.iroot:
            factor = adjustLogSamplingProbMult(-log_density_np_adjustment,nSample, nFullData)
        else:
            factor = None
        # Broadcast multiplicative factor
        factor = par.bcast(factor)
    else:
        factor = adjustLogSamplingProbMultPar(-log_density_np_[::data_freq_adjustment],int(nSample/data_freq_adjustment), nFullData)

    # Sample according to the acceptance probability
    log_samplingProb_ = np.clip(np.log(factor) - log_density_np_,a_min=None,a_max=0)
    samplingProb_ = np.exp(log_samplingProb_)
    nSamplesAssigned = 0
    nSamplesAssigned_ = 0
    niter = 0
    nData_ = data_to_downsample_.shape[0]
    indexSelected_=[]
    while nSamplesAssigned < nSample:
          randNum_ = np.random.uniform(0,1,nData_).astype('float32')
          niter += 1
          Ind=np.argwhere(samplingProb_>randNum_)
          set1 = set(as_list(Ind))
          set2 = set(indexSelected_)
          Ind = list(set1-set2)
          for ind in Ind:
              indexSelected_.append(ind)
              nSamplesAssigned_ += 1
          nSamplesAssigned = par.allsumScalar(nSamplesAssigned_)
          #par.printRoot("nSample Assigned = ",nSamplesAssigned)

    phaseSpaceSampledData_ = data_to_downsample_[indexSelected_,:]
    # Gather
    phaseSpaceSampledData, indexDownsampledData =  gatherDownsampledData(phaseSpaceSampledData_,
                                                                         indexSelected_,
                                                                         nSample,
                                                                         nFullData,
                                                                         nSamplesAssigned)
    
    return phaseSpaceSampledData, indexDownsampledData, samplingProb_, log_samplingProb_

def checkLoss(flow_iter,flow_nll_loss):
    if (not flow_iter==1) or (not par.irank==par.iroot):
        return
    if flow_nll_loss[1]<=flow_nll_loss[0]:
        par.printRoot('WARNING: Flow loss did not increase with iteration. Try increasing the number of epochs')
    return

def checkProcedure(meanCriterion,nSample,randomCriterion):
    # Make sure improvement was observed compared to random sampling
    errorSignal = False
    if meanCriterion[-1]<randomCriterion:
        par.printRoot('ERROR: Failure of the sampling procedure for nSample : ' + str(nSample))
        errorSignal = True
    if errorSignal:
        return

    # Make sure that the sampling quality improved between first and second iteration
    num_flow_iter = meanCriterion.shape[0]
    if num_flow_iter<2:
        return
    if meanCriterion[-1]<meanCriterion[0]:
        par.printRoot('WARNING: Possible failure of the sampling procedure for nSample : ' + str(nSample))



