#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Sep 6, 2016

@author: daniel
'''
from scipy.sparse import csr_matrix, issparse
from copy import deepcopy
import numpy as np
import os
import sys
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
PROJECT_DIR = os.getenv("PROJECT_DIR")
sys.path.append(PROJECT_DIR+'/src')
import paths

class AlgorithmParent(object):
    '''
    This class is intended to serve as parent class for all algorithm classes:
    It contains all the variables which should be required by an algorithm
    It contains all the methods which are common to the algorithms
    '''

    def __init__(self, method, network, validation, **kwargs):
        '''
        Constructor:
        All Algorithms should have a:
        Name
        Method
        Network
        Resultmat
        Result
        '''
        '''
        Store the name and method
        '''
        self.name = method
        self.method = method
        '''
        Store the entity type of query
        '''
        self.selectedQueryMode = None
        self.selectedPredictionMode = None
        '''
        Store the network
        '''
        self.network = network
        '''
        Create result place holders
        '''
        self.result = None
        self.resultmat = None
        '''
        Store the location to save algorithm results
        '''
        self.resultDir = paths.ALGORITHM_RESULT_TMP_DIR
        '''
        Store the location to save processed networks
        '''
        self.networkDir = paths.NETWORK_TMP_DIR
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self.__dict__.update(kwargs)
        if not os.path.exists(self.resultDir):
            os.mkdir(self.resultDir)
        if not os.path.exists(self.networkDir):
            os.mkdir(self.networkDir)
        '''
        Set name for result file to save to
        '''
        try:
            self.saveFile = '{}_Result_{}_{}_qMode_{}_pMode_{}'.format(self.name,
                                                                       self.network.getName(), validation, self.selectedQueryMode, self.selectedPredictionMode)
        except:
            try:
                self.saveFile = '{}_Result_{}_{}_qMode_{}'.format(self.name,
                                                                  self.network.getName(), validation, self.selectedQueryMode)
            except:
                try:
                    self.saveFile = '{}_Result_{}_{}'.format(self.name,
                                                             self.network.getName(), validation)
                except:
                    try:
                        self.saveFile = '{}_Result_{}'.format(self.name,
                                                              self.network.getName())
                    except:
                        self.saveFile = '{}_Result_{}'.format(
                            self.name, 'None')
        self.saveFile = self.saveFile.replace(
            "Mapping_Ccgdd16Sx91Extended_", "")

    def __del__(self):
        '''
        Destructor:
        All of the base variables should be destroyed
        '''
        '''
        Remove the name and method
        '''
        del(self.name)
        del(self.method)
        '''
        Remove the network
        '''
        del(self.network)
        '''
        Remove result variables
        '''
        del(self.result)
        del(self.resultmat)

    def __str__(self):
        '''
        str

        Overriding the default str method so that it returns relevant
        information about the instance of an algorithm.
        '''
        return '{} object to be performed over the {} network'.format(
            self.method, self.network.getName())

    def __repr__(self):
        '''
        repr

        Overriding the default repr method so that it returns relevant
        information about the instance of an algorithm.
        '''
        self.getName()

    def getName(self):
        '''
        getName

        Returns the name of the algorithm

        Returns:
        --------
        str
            The name of the algorithm.
        '''
        return deepcopy(self.name)

    def getMethod(self):
        '''
        getMethod

        Returns the name of the method used by this algorithm.

        Returns:
        str
            The name of the method.
        '''
        return deepcopy(self.method)

    def setMethod(self, method):
        '''
        setMethod

        Sets the name of the method employed by the current algorithm.

        Parameters:
        -----------
        method: str
            The name of the method employed by this algorithm.
        '''
        self.method = method

    def getNetwork(self):
        '''
        getNetwork

        Returns the network being used by this algorithm.

        Returns:
        --------
        GraphRepresentation5
            The network object being computed over by this algorithm.
        '''
        return deepcopy(self.network)

    def setNetwork(self, network):
        '''
        setNetwork

        Sets the network being used by this algorithm.

        Parameters:
        -----------
        network: GraphRepresentation5
            The network object to compute over when performing this algorithm.
        '''
        self.network = network

    def getResult(self):
        '''
        getResult

        Returns the result stored by this algorithm.

        Returns:
        list:
            A list of results in the format node position 1, node position 2,
            predicted value.
        '''
        return deepcopy(self.result)

    def getResultMat(self):
        '''
        getResultMat

        Returns the result matrix stored by this algorithm.

        Returns:
        numpy array like or scipy.sparse like
            Result matrix computed by this aglorithm.
        '''
        return deepcopy(self.resultmat)

    def getSaveFile(self):
        '''
        getSaveFile

        Returns the name of the file to which results from this algorithm will
        be saved.

        Returns:
        --------
        str
            The name of the file to which results from this algorithm will be
            saved.
        '''
        return deepcopy(self.saveFile)

    def perform(self):
        '''
        perform

        A method signature used to enforce implementation in algorithms
        inheriting from this class parent.
        '''
        raise NotImplementedError(
            'The perform method for this Algorithm has not been implemented yet!')

    def getresult(self, problem='PredictionMatrix'):
        '''
        getResult

        Returns the result of performing this algorithm.

        Parameters:
        -----------
        problem: str
            Currently 'PredictionMatrix' is the default value and will return
            the matrix stored at self.resultmat.  If any other option is return
            the self.result variable will be returned.
        Returns:
        --------
        numpy array like or scipy.sparse like
            The result matrix from performing this algorithm.
        '''
        if(problem == 'PredictionMatrix'):
            return deepcopy(self.resultmat)
        else:
            return deepcopy(self.result)

    def saveMatrixResult(self, fileName=None, matrix=None):
        '''
        saveMatrixResult

        A method for saving the result computed by this instance of an
        algorithm. If no matrix is provided, the saved result is stored
        in self.resultmat.

        Parameters:
        ----------
        fileName: str
            The file name under which to save the result data.  If no
            file path is provided the self.saveFile path is used.
        matrix: numpy.array like or scipy.sparse like
            If a matrix is provided the saved results will be those from the
            provided matrix, otherwise the self.resultmat matrix will be saved.
        --------
        '''
        os.chdir(self.resultDir)
        if(fileName is None):
            fileName = self.saveFile
        if(matrix is None):
            matrix = self.resultmat
        if(issparse(matrix)):
            np.savez(fileName, data=matrix.data,
                     indices=matrix.indices,
                     indptr=matrix.indptr,
                     shape=matrix.shape)
        else:
            np.savez(fileName, mat=matrix)

    def loadMatrixResult(self, fileName=None, matrix=None):
        '''
        loadMatrixResult

        A method for loading the result computed by this instance of an
        algorithm. If no matrix is provided, the loaded result is stored
        in self.resultmat.

        Parameters:
        ----------
        fileName: str
            The file name to search for when loading the result data.  If no
            file path is provided the self.saveFile path is loaded.
        matrix: numpy.array like or scipy.sparse like
            If a matrix is provided the loaded results will not be saved in the
            self.resultmat variable.
        --------
        '''
        if(fileName is None):
            fileName = self.saveFile
        loadedFiles = np.load(fileName + '.npz')
        print 'Loading pre-existing saved results ' + fileName
        if('mat' in loadedFiles):
            mat = loadedFiles['mat']
        else:
            mat = csr_matrix((loadedFiles['data'],
                              loadedFiles['indices'],
                              loadedFiles['indptr']),
                             shape=loadedFiles['shape'])
        if(matrix is None):
            self.resultmat = mat
