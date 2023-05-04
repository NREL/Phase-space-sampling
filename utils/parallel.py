import random
import sys

import numpy as np
from mpi4py import MPI

# MPI Init
comm = MPI.COMM_WORLD
irank = comm.Get_rank() + 1
iroot = 1
nProc = comm.Get_size()
status = MPI.Status()


# ~~~~ Print functions
def printProc(description, item=None, proc=iroot):
    if irank == proc:
        if type(item).__name__ == "NoneType":
            print(description)
        elif not type(item).__name__ == "ndarray":
            print(description + ": ", item)
        else:
            print(description + ": ", item.tolist())
        sys.stdout.flush()
    return


def printRoot(description, item=None, root=iroot):
    printProc(description, item, iroot)
    return


def printAll(description, item=None):
    if type(item).__name__ == "NoneType":
        print(f"[{irank}] {description}")
    elif not type(item).__name__ == "ndarray":
        print(f"[{irank}] {description}:", item)
    else:
        print(f"[{irank}] {description}:", item.tolist())
    sys.stdout.flush()
    return


def printAllOrdered(description, item=None):
    for rank in range(nProc):
        if irank == rank + 1:
            printProc(f"[{rank}] {description}", item=item, proc=irank)
        comm.Barrier()
    return


def partitionData(nSnap):
    # ~~~~ Partition data snapshots with MPI
    # Simple parallelization across snapshots
    NSnapGlob = nSnap
    tmp1 = 0
    tmp2 = 0
    for iproc in range(nProc):
        tmp2 = tmp2 + tmp1
        tmp1 = int(NSnapGlob / (nProc - iproc))
        if irank == (iproc + 1):
            nSnap_ = tmp1
            startSnap_ = tmp2
        NSnapGlob = NSnapGlob - tmp1
    return nSnap_, startSnap_


def gather1DList(list_, rootId, N):
    list_ = np.array(list_, dtype="double")
    sendbuf = list_
    recvbuf = np.empty(N, dtype="double")
    # Collect local array sizes:
    sendcounts = comm.gather(len(sendbuf), root=rootId)
    comm.Gatherv(sendbuf, recvbuf=(recvbuf, sendcounts), root=rootId)
    return recvbuf


def gather1DArray(array_, rootId, N):
    # Broadcast type
    dtype = None
    if irank == iroot:
        dtype = array_.dtype.name
    dtype = bcast(dtype)
    sendbuf = array_
    recvbuf = np.empty(N, dtype=dtype)
    # Collect local array sizes:
    sendcounts = comm.gather(len(sendbuf), root=rootId)
    comm.Gatherv(sendbuf, recvbuf=(recvbuf, sendcounts), root=rootId)
    return recvbuf


def gather2DList(list_, rootId, N1Loc, N1Glob, N2):
    # ~~~ The parallelization is across the axis 0
    # ~~~ This will not work if the parallelization is across axis 1
    list_ = np.array(list_, dtype="double")
    # Reshape the local data matrices:
    nElements_ = N1Loc * N2
    sendbuf = np.reshape(list_, nElements_, order="C")
    # Collect local array sizes:
    sendcounts = comm.gather(len(sendbuf), root=rootId)
    # Gather the data matrix:
    recvbuf = np.empty(N1Glob * N2, dtype="double")
    comm.Gatherv(sendbuf=sendbuf, recvbuf=(recvbuf, sendcounts), root=0)
    recvbuf = np.reshape(recvbuf, (N1Glob, N2), order="C")
    return recvbuf


def gather2DArray(array_, rootId, N1Loc, N1Glob, N2):
    # ~~~ The parallelization is across the axis 0
    # ~~~ This will not work if the parallelization is across axis 1
    # Broadcast type
    dtype = None
    if irank == iroot:
        dtype = array_.dtype.name
    dtype = bcast(dtype)
    # Reshape the local data matrices:
    nElements_ = N1Loc * N2
    sendbuf = np.reshape(array_, nElements_, order="C")
    # Collect local array sizes:
    sendcounts = comm.gather(len(sendbuf), root=rootId)
    # Gather the data matrix:
    recvbuf = np.empty(N1Glob * N2, dtype=dtype)
    comm.Gatherv(sendbuf=sendbuf, recvbuf=(recvbuf, sendcounts), root=0)
    recvbuf = np.reshape(recvbuf, (N1Glob, N2), order="C")
    return recvbuf


def allgather1DList(list_, N):
    list_ = np.array(list_, dtype="double")
    # Collect local array sizes:
    recvbuf = np.empty(N, dtype="double")
    comm.Allgatherv(list_, recvbuf)
    return recvbuf


def allsum1DArrays(A):
    buf = np.zeros(len(A), dtype=A.dtype.name)
    comm.Allreduce(A, buf, op=MPI.SUM)
    return buf


def allsumMultiDArrays(A):
    # Takes a 3D array as input
    # Returns 3D array
    shapeDim = A.shape
    nTotDim = int(np.prod(shapeDim))
    buf = np.zeros(nTotDim, dtype=A.dtype.name)
    comm.Allreduce(np.reshape(A, nTotDim), buf, op=MPI.SUM)
    return np.reshape(buf, shapeDim)


def allsumScalar(A):
    result = comm.allreduce(A, op=MPI.SUM)
    return result


def allmaxScalar(A):
    result = comm.allreduce(A, op=MPI.MAX)
    return result


def bcast(A):
    A = comm.bcast(A, root=0)
    return A


def distribute1DArray(A=None):
    """
    MPI-distribution of a 1D array by partitioning along the first dimension
    Input: Full 1D array
    Returns: Partitioned 1D array
    """

    # Broadcast dimensions and type
    nData = None
    dtype = None
    if irank == iroot:
        nData = A.shape[0]
        dtype = A.dtype.name
    nData = bcast(nData)
    dtype = bcast(dtype)

    # Partition
    nSnap_, startSnap_ = partitionData(nData)

    # Gather nSnap_ and startSnap_ on root processor
    nSnap_array = comm.gather(nSnap_, root=iroot - 1)
    startSnap_array = comm.gather(startSnap_, root=iroot - 1)

    # Root sends
    if irank == iroot:
        A_ = A[startSnap_ : startSnap_ + nSnap_]
        for iproc in range(nProc):
            if not iproc == iroot - 1:
                procStart = startSnap_array[iproc]
                procSize = nSnap_array[iproc]
                comm.Send(
                    A[procStart : procSize + procStart], dest=iproc, tag=iproc
                )
    else:
        A_ = np.empty(nSnap_, dtype=dtype)
        comm.Recv(A_, source=iroot - 1, tag=irank - 1)

    return A_


def distribute2DArray(A=None):
    """
    MPI-distribution of a 2D array by partitioning along the first dimension
    Input: Full 2D array
    Returns: Partitioned 2D array
    """

    # Broadcast dimensions and type
    nData = None
    nDim = None
    dtype = None
    if irank == iroot:
        nData = A.shape[0]
        nDim = A.shape[1]
        dtype = A.dtype.name
    nData = bcast(nData)
    nDim = bcast(nDim)
    dtype = bcast(dtype)

    # Partition
    nSnap_, startSnap_ = partitionData(nData)

    # Gather nSnap_ and startSnap_ on root processor
    nSnap_array = comm.gather(nSnap_, root=iroot - 1)
    startSnap_array = comm.gather(startSnap_, root=iroot - 1)

    # Root sends
    if irank == iroot:
        A_ = A[startSnap_ : startSnap_ + nSnap_, :]
        for iproc in range(nProc):
            if not iproc == iroot - 1:
                procStart = startSnap_array[iproc]
                procSize = nSnap_array[iproc]
                comm.Send(
                    A[procStart : procSize + procStart, :],
                    dest=iproc,
                    tag=iproc,
                )
    else:
        A_ = np.empty((nSnap_, nDim), dtype=dtype)
        comm.Recv(A_, source=iroot - 1, tag=irank - 1)

    return A_


def gatherNelementsInArray(A_, nSelect):
    """
    From a MPI-distributed dataset along the first dimension, gather nSelect datapoints
    on the root processor. The tensor dimensions is arbitrary.
    Input: distributed data
    Output : Gathered data for root processor, None for the other processor
    """

    A = None
    if irank == iroot:
        newShape = [A_.shape[i] for i in range(len(A_.shape))]
        newShape[0] = nSelect
        A = np.zeros(tuple(newShape), dtype=A_.dtype.name)

    # Gather size in each proc
    nSnap_array = comm.gather(A_.shape[0], root=iroot - 1)

    # Figure out the communications
    nSend = np.zeros(nProc, dtype="int")
    if irank == iroot:
        # Figure out who sends what
        nSend[0] = 0
        nLeftToSend = max(nSelect - nSnap_array[0], 0)
        for iproc in range(1, nProc):
            nSend[iproc] = min(nLeftToSend, nSnap_array[iproc])
            nLeftToSend -= nSend[iproc]
            if nLeftToSend == 0:
                break
    # Let everybody be aware of the communication
    nSend = allsum1DArrays(nSend)

    # Now, communicate
    # Root receives
    if irank == iroot:
        nToKeep = min(nSelect, nSnap_array[0])
        A[:nToKeep] = A_[:nToKeep]
        for iproc in range(nProc):
            if nSend[iproc] > 0:
                bufShape = [A_.shape[i] for i in range(len(A_.shape))]
                bufShape[0] = nSend[iproc]
                bufrecv = np.empty(bufShape, dtype=A_.dtype.name)
                comm.Recv(bufrecv, source=iproc, tag=iproc)
                start = np.sum(nSend[:iproc]) + nToKeep
                A[start : start + nSend[iproc]] = bufrecv[:]
    else:
        for iproc in range(nProc):
            if nSend[iproc] > 0:
                if irank - 1 == iproc:
                    bufShape = [A_.shape[i] for i in range(len(A_.shape))]
                    bufShape[0] = nSend[iproc]
                    bufsend = np.empty(bufShape, dtype=A_.dtype.name)
                    bufsend[:] = A_[: nSend[iproc]]
                    comm.Send(bufsend, dest=iroot - 1, tag=iproc)

    return A


def parallel_shuffle(A_):
    """
    Shuffle a MPI-distributed dataset along the first dimension
    Works for arbitrary tensor dimensions
    """

    # From https://stackoverflow.com/questions/36266968/parallel-computing-shuffle
    def exchange(localdata, sendrank, recvrank):
        """
        Perform a merge-exchange with a neighbour;
        sendrank sends local data to recvrank,
        which merge-sorts it, and then sends lower
        data back to the lower-ranked process and
        keeps upper data
        """
        assert irank - 1 == sendrank or irank - 1 == recvrank
        assert sendrank < recvrank
        if irank - 1 == sendrank:
            comm.send(localdata, dest=recvrank)
            newdata = comm.recv(source=recvrank)
        else:
            bothdata = list(localdata)
            otherdata = comm.recv(source=sendrank)
            bothdata = bothdata + otherdata
            bothdata.sort()
            comm.send(bothdata[: len(otherdata)], dest=sendrank)
            newdata = bothdata[len(otherdata) :]
        return newdata

    def odd_even_sort(data):
        data.sort()
        for step in range(1, nProc + 1):
            if ((irank - 1 + step) % 2) == 0:
                if irank - 1 < nProc - 1:
                    data = exchange(data, irank - 1, irank)
            elif irank - 1 > 0:
                data = exchange(data, irank - 2, irank - 1)
        return np.array([x for _, x in data])

    # Tag data with random numbers
    n_points = A_.shape[0]
    randomArray = np.random.uniform(size=n_points)
    Ashuffled_ = [(randomArray[i], A_[i]) for i in range(n_points)]
    # Sort by random num
    Ashuffled_ = odd_even_sort(Ashuffled_)

    return Ashuffled_


def parallel_shuffle_np(A_, nData):
    """
    Shuffle a MPI-distributed dataset along the first dimension
    Works for arbitrary tensor dimensions
    """

    def sortByTags(data, dataInd, tags):
        ind = np.argsort(tags)
        return data[ind], dataInd[ind], tags[ind]

    # From https://stackoverflow.com/questions/36266968/parallel-computing-shuffle
    def exchange(
        localdata, localinds, localtags, sendrank, recvrank, nSnap_array
    ):
        """
        Perform a merge-exchange with a neighbour;
        sendrank sends local data to recvrank,
        which merge-sorts it, and then sends lower
        data back to the lower-ranked process and
        keeps upper data
        """
        assert irank - 1 == sendrank or irank - 1 == recvrank
        assert sendrank < recvrank

        if irank - 1 == sendrank:
            comm.Send(localdata, dest=recvrank, tag=0)
            comm.Send(localtags, dest=recvrank, tag=1)
            comm.Send(localinds, dest=recvrank, tag=2)

            newdata = np.empty(
                (nSnap_array[sendrank], localdata.shape[1]), dtype=np.float32
            )
            newtags = np.empty(nSnap_array[sendrank], dtype=np.float32)
            newinds = np.empty(nSnap_array[sendrank], dtype=int)
            comm.Recv(newdata, source=recvrank, tag=0)
            comm.Recv(newtags, source=recvrank, tag=1)
            comm.Recv(newinds, source=recvrank, tag=2)

        else:
            otherdata = np.empty(
                (nSnap_array[sendrank], localdata.shape[1]), dtype=np.float32
            )
            othertags = np.empty(nSnap_array[sendrank], dtype=np.float32)
            otherinds = np.empty(nSnap_array[sendrank], dtype=int)
            comm.Recv(otherdata, source=sendrank, tag=0)
            comm.Recv(othertags, source=sendrank, tag=1)
            comm.Recv(otherinds, source=sendrank, tag=2)

            bothdata = np.concatenate((localdata, otherdata), axis=0)
            bothtags = np.concatenate((localtags, othertags))
            bothinds = np.concatenate((localinds, otherinds))

            sortedInd = np.argsort(bothtags)

            comm.Send(
                bothdata[sortedInd[: otherdata.shape[0]]], dest=sendrank, tag=0
            )
            comm.Send(
                bothtags[sortedInd[: otherdata.shape[0]]], dest=sendrank, tag=1
            )
            comm.Send(
                bothinds[sortedInd[: otherdata.shape[0]]], dest=sendrank, tag=2
            )
            newdata = bothdata[sortedInd[otherdata.shape[0] :]]
            newtags = bothtags[sortedInd[otherdata.shape[0] :]]
            newinds = bothinds[sortedInd[otherdata.shape[0] :]]

        return newdata, newinds, newtags

    def odd_even_sort(data, dataind, tags, nSnap_array):
        data, dataind, tags = sortByTags(data, dataind, tags)
        for step in range(1, nProc + 1):
            if ((irank - 1 + step) % 2) == 0:
                if irank - 1 < nProc - 1:
                    data, dataind, tags = exchange(
                        data, dataind, tags, irank - 1, irank, nSnap_array
                    )
            elif irank - 1 > 0:
                data, dataind, tags = exchange(
                    data, dataind, tags, irank - 2, irank - 1, nSnap_array
                )
        return data, dataind, tags

    # Tag data with random numbers
    n_points = A_.shape[0]
    tags_ = np.random.uniform(size=n_points).astype("float32")
    # Get data shape of each proc
    nSnap_, startSnap_ = partitionData(nData)
    nSnap_array = comm.allgather(nSnap_)
    dataInd_ = np.array(list(range(startSnap_, nSnap_ + startSnap_)))
    # Sort by random num
    Ashuffled_, dataInd_, tags_ = odd_even_sort(
        A_, dataInd_, tags_, nSnap_array
    )

    return Ashuffled_, dataInd_, tags_
