import numpy as np
import parallel as par
import sys
#from memory_profiler import profile

#@profile
def checkData(shape, nWorkingData,nWorkingDataAdjustment):
    if not len(shape)==2:
        par.printAll('Expected 2 dimensions for the data, first is the number of samples, second is the dimension')
        par.comm.Abort()
    else:
        par.printRoot('Dataset has ' + str(shape[0]) + ' samples of dimension ' + str(shape[1]) )

    # Check that sizes make sense
    if shape[0]<nWorkingData*10:
        par.printRoot('WARNING: Only ' + str(shape[0]) + ' samples, this may not work')
    if shape[0]<max(nWorkingData,nWorkingDataAdjustment):
        par.printAll('ERROR: At least ' + str(max(nWorkingData,nWorkingDataAdjustment)) +' samples required')
        par.comm.Abort()

    return

#@profile
def prepareData(inpt):

    # Set parameters from input 
    dataFile =			inpt['dataFile']
    preShuffled =		(inpt['preShuffled']=="True")
    scalerFile = 		inpt['scalerFile']
    nWorkingData = 		int(float(inpt['nWorkingData']))
    nWorkingDataAdjustment = 	int(float(inpt['nWorkingDataAdjustment']))

    # Load the dataset but don't read it just yet
    dataset = np.load(dataFile,mmap_mode='r')

    # Check that dataset shape make sense
    nFullData = dataset.shape[0]
    nDim = dataset.shape[1]
    if par.irank==par.iroot: 
        checkData(dataset.shape,nWorkingData,nWorkingDataAdjustment)
    
    # Distribute dataset
    if par.irank==par.iroot:
        print('LOAD DATA ... ', end='')
        sys.stdout.flush()
    par.comm.Barrier()
    nSnap_, startSnap_ = par.partitionData(nFullData) 
    data_to_downsample_ = dataset[startSnap_:startSnap_+nSnap_,:].astype('float32')
    par.printRoot('DONE!')  

    # Rescale data
    if par.irank==par.iroot:
        print('RESCALE DATA ... ', end='')
        sys.stdout.flush()
    par.comm.Barrier()
    minVal = np.zeros(nDim)
    maxVal = np.zeros(nDim)
    for i in range(nDim):
        minVal_ = np.amin(data_to_downsample_[:,i])
        maxVal_ = np.amax(data_to_downsample_[:,i])
        minVal[i] = par.comm.allreduce(minVal_, op=par.MPI.MIN)
        maxVal[i] = par.comm.allreduce(maxVal_, op=par.MPI.MAX)
    # Root processor saves scaling
    if par.irank==par.iroot:
        np.savez(scalerFile,minVal=minVal,maxVal=maxVal)  
    par.printRoot('DONE!')  

    # Parallel shuffling
    if not preShuffled:
        if par.irank==par.iroot:
            print('SHUFFLE DATA ... ', end='')
            sys.stdout.flush()
        par.comm.Barrier()
        par.parallel_shuffle(data_to_downsample_)
        par.printRoot('DONE!')  


    # Get subsampled dataset to work with
    working_data = par.gatherNelementsInArray(data_to_downsample_,nWorkingData)
 
    return data_to_downsample_, working_data, nFullData
