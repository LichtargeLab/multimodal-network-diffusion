#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Aug 26, 2016

@author: dmkonecki
'''
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix, csgraph, identity
from multiprocessing import cpu_count, Process, Queue, Manager
from scipy.sparse.linalg import lgmres
from scipy.stats import zscore
import numpy as np
import os
from AlgorithmParent import AlgorithmParent

class GraphBasedDiffusion(AlgorithmParent):
    '''
    Algorithm class of graph-based information diffusion
    '''

    def __init__(self, network, validation, selectedQueryMode=None, method='Diffusion', processors=None):
        '''
        Constructor:
        Calls the parent constructor
        '''
        AlgorithmParent.__init__(self, method=method, network=network,
                                 validation=validation, selectedQueryMode=selectedQueryMode)
        # adjacencyMatrix is the A matrix
        # Set up a problem
        # started with normed set to False
        adjMatrix = self.network.getAdjacencyMatrix()
        L = csgraph.laplacian(adjMatrix, normed=True)
        n = adjMatrix.shape[0]
        I = identity(n, dtype='int8', format='csr')
        # Parameter that weights smoothness versus loss.
        # Called "mu" in Shin et al 2007.
        #################################################################
        # Right now I am summing by zero dimension (column) but we may have to
        # sum and average or something if the matrix is not symmetric.
        #################################################################
        axisSum = coo_matrix.sum(np.abs(L), axis=0)
        sumMax = np.max(axisSum)
        self.alpha = (1 / float(sumMax))
        self.saveAlpha()
        # ps is the same for given adjacency matrix, so we precompute it for
        # every diffusion experiment (I+alpha*L)
        self.__ps = (I + (self.alpha * L))
        self.processors = processors

    def __del__(self):
        del(self.processors)
        AlgorithmParent.__del__(self)

    def perform(self, labeledNodes, labels=None, processors=None):
        os.chdir(self.resultDir)
        if(not(processors is None)):
            self.processors = processors
        if(self.method == 'Diffusion'):
            self.diffusionMethod(labeledNodes, labels)
        elif(self.method == 'Propagation'):
            raise ValueError('Propagation method not yet implemented!')
        else:
            raise ValueError('Unknown signal spreading method provided!')

    def diffusionMethod(self, labeledNodes, labels=None):
        if(os.path.exists(self.saveFile + '.npz')):
            self.loadMatrixResult(self.saveFile)
        else:
            os.chdir(self.networkDir)
            net = self.network
            nodePosition = net.getNodePosition()
            dim = self.__ps.shape[0]
            # adjMatrix = net.getAdjacencyMatrix()
            self.resultmat = lil_matrix((dim, dim), dtype=np.float64)
            if(labels is None):
                labels = [1] * len(labeledNodes)
            numLabels = len(labeledNodes)
            print('Performing diffusion for {} experiments.'.format(numLabels))
            inputs = [(dim, nodePosition[labeledNodes[i]],
                       labels[i]) for i in range(numLabels)]
            if((self.processors is None) or (self.processors > 1)):
                print('Performing multiprocessed diffusion.')
                # Determine the number of threads to create
                numInputs = len(inputs)
                if(self.processors is None):
                    numCPUs = cpu_count()
                    numCPUs = (3 * numCPUs / 4)
                else:
                    numCPUs = self.processors
                if numInputs < numCPUs:
                    numThreads = numInputs
                else:
                    numThreads = numCPUs
                print(
                    'Number of Threads/Queues created: {}'.format(numThreads))
                # Create a queue
                q = Queue()
                # Build a shared dictionary of adjMatrix by manager
                manager = Manager()
                d = manager.dict()
                d['input'] = self.__ps
                for i in range(numThreads):
                    argsInput = d, inputs[
                        int(i * numLabels / numThreads):int((i + 1) * numLabels / numThreads)], i
                    p1 = Process(
                        target=diffuse_multiprocess, args=(argsInput, q))
                    p1.start()
                results = []
                for i in range(numThreads):
                    # set block=True to block until we get a result
                    output = q.get(True)
                    results = results + output
            else:
                print('Performing diffusion with a single process')
                # For single threading run this instead of the previous
                # section.
                results = []
                for i in inputs:
                    # results.append(diffusionWorker(i,adjMatrix))
                    results.append(diffusionWorker(i, self.__ps))
            for res in results:
                self.resultmat[res[0], :] = res[1]
            self.resultmat = csr_matrix(self.resultmat)
            self.resultmat = ((self.resultmat + self.resultmat.transpose())
                              / 2)
            self.saveMatrixResult(self.saveFile)

    def getresult(self, problem='PredictionMatrix'):
        '''
        This returns the results in a specified format.

        problem=['PredictionMatrix', 'RelationshipList_Index', 'RelationshipList_Name']

        '''
        if problem == 'PredictionMatrix':
            return self.resultmat
        elif problem == 'ZScoreMatrix':
            try:
                return csr_matrix(zscore(self.resultmat, axis=1))
            except:
                return csr_matrix(zscore(self.resultmat.todense(), axis=1))
        elif problem == 'RelationshipList_Index':
            res = self.result
            if(res is None):
                res = {}
                resMat = self.resultmat
                indices = resMat.nonzero()
                for i in len(indices[0]):
                    row = indices[0][i]
                    col = indices[1][i]
                    res[(row, col)] = resMat[row, col]
            return res
        elif problem == 'RelationshipList_Name':
            currRes = self.getresult('RelationshipList_Index')
            res = {}
            positionNode = self.network.getPositionNode()
            for key in currRes:
                row = positionNode[key[0]]
                col = positionNode[key[1]]
                res[(row, col)] = currRes[key]
            return res
        else:
            raise ValueError('This return type is not supported')

    def saveAlpha(self):
        os.chdir(self.resultDir)
        fileName = self.saveFile + '.alpha'
        f = open(fileName, 'w')
        f.write("{}\n".format(self.alpha))
        f.close()
        return


def diffusionWorker(inTuple, ps):
    dim = inTuple[0]
    nodePos = inTuple[1]
    label = inTuple[2]
    print('Performing Diffusion  on node at pos {}'.format(nodePos))
    labelVector = np.zeros(dim)
    labelVector[nodePos] = label
    resultVector = diffuse(labelVector, ps)
    resultVector[nodePos] = 0
    return (nodePos, resultVector)


def diffuse_multiprocess(argsInput, q):
    d, subInputs, i = argsInput
    ps = d['input']
    output = []
    n = 0
    total = len(subInputs)
    for inTuple in subInputs:
        # n+=1
        # if n%100 ==0:
        #    print('Performing Diffusion on {}/{} nodes in process {}'.format(n,total,i))
        dim = inTuple[0]
        nodePos = inTuple[1]
        label = inTuple[2]
        labelVector = np.zeros(dim)
        labelVector[nodePos] = label
        resultVector = diffuse(labelVector, ps)
        resultVector[nodePos] = 0
        output.append((nodePos, resultVector))
    q.put(output)


def diffuse(labelVector, ps):
    svSum = labelVector.sum()
    if(svSum == 0):
        return lil_matrix(shape=(1, len(labelVector)), dtype=np.float64)
    y = labelVector
    f = lgmres(ps, y, tol=1e-10)[0]
    return f


if(__name__ == '__main__'):
    adjMatrix = csr_matrix(np.matrix([[0, 1, 0, 0, 1],
                                      [1, 0, 1, 0, 0],
                                      [0, 1, 0, 1, 1],
                                      [0, 0, 1, 0, 1],
                                      [1, 0, 0, 1, 0]]))
    label = np.zeros(5)
    label[0] = 1
    label[2] = 1
    label[1] = 1
    L = csgraph.laplacian(adjMatrix, normed=True)
    n = adjMatrix.shape[0]
    I = identity(n, dtype='int8', format='csr')
    # Parameter that weights smoothness versus loss.
    # Called "mu" in Shin et al 2007.
    #################################################################
    # Right now I am summing by one dimension (row) but we may have to sum and
    # average or something if the matrix is not symetric.
    #################################################################
    axisSum = coo_matrix.sum(np.abs(L), axis=0)
    sumMax = np.max(axisSum)
    diffusionParameter = (1 / float(sumMax))
    ps = (I + (diffusionParameter * L))
    print(sum((diffuse(label, ps))))
