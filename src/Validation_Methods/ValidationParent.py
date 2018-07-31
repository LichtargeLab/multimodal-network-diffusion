#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Aug 25, 2016

@author: Chih-Hsu Lin
'''
import os
import sys
import paths
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             average_precision_score)
from scipy.stats import rankdata
#from scipy.sparse import csr_matrix, triu
from scipy.sparse import lil_matrix
# from copy import deepcopy
import cPickle as pickle
import bisect
from IPython import embed
import pandas as pd
from GraphRepresentation5 import GraphRepresentation5

class ValidationParent(object):
    '''
    This class is intended to serve as parent class for all validation method classes:
    It contains all the variables which should be required by an validation method
    It contains all the methods which are common to the validation methods
    '''

    def __init__(self, method, algorithm, network, split,
                 selectedEntityList=None, selectedQueryMode=None,
                 selectedPredictionMode=None, bridgeMode=None,
                 answerNetwork=None,
                 splitPerformanceByEntity=False, 
                 param=None):
        '''
        Constructor

        All validation methods should have a the variables built in this class
        and be able to perform the included functions.

        Parameters:
        method: str
            The type of validation experiment being performed
        algorithm: AlgorithmParent child
            An instance of a class which inherits from the AlgorithmParent 
            class.
        network: GraphRepresentation5
            The network which will be used as the input for performing the
            experiment.
        split: str
            The parameter on which the network will be split for validation.
        selectedEntityList: list
            Entity types for which to create separate results.
        seletedQueryMode: str
            Entity type for which to look for new interactions.
        selectedPredictionMode: str
            Entity type for which to look for new interactions with query type.
        answerNetwork: GraphRepresentation5
            If a custom answer network should be used in validations this is
            where it should be provided.
        splitPerformanceByEntity: bool
            Whether or not to present results on an entity by entity basis.

        Returns
        -------
        ValidationParent
            An instance of this class.
        '''

        # Store the validation method name
        self.name = method
        self.method = method
        # Store the algorithm
        self.algorithm = algorithm
        # Store the parameters for AptRank
        self.param = param
        # Store the network of initial input
        self.network = network
        # Store the parameter to split network E.g., in timeStamped experiment,
        # this is the year to split.
        self.split = split
        # Store the entity list for generating predictions
        self.selectedEntityList = selectedEntityList
        # Store the node object of the entity list for generating predictions
        self.selectedNodes = None
        # Store the entity type of query
        self.selectedQueryMode = selectedQueryMode
        # Store the entity type of predictions
        self.selectedPredictionMode = selectedPredictionMode
        # Store the edge type used for Proximity method
        self.bridgeMode = bridgeMode
        # Indicate whether AUC and PR are computed separately by each given
        # entity
        self.splitPerformanceByEntity = splitPerformanceByEntity
        # Store input network for running algorithm
        self.inputNetwork = None
        # Store answer network for evaluating performance
        self.answerNetwork = answerNetwork
        # A list storing input and answer network file names
        # Should be specified in each validation class
        self.validationNetworkNames = None
        # Store the location to save performance results
        if 'prospective' in method:
            self.dirResult = paths.VALIDATION_RESULT_DIR+'/prospective/'
        elif 'timeStamped' in method:
            self.dirResult = paths.VALIDATION_RESULT_DIR+'/time_stamped/'
        elif 'leaveAModeOut' in method:
            self.dirResult = paths.VALIDATION_RESULT_DIR+'/leave_one_mode_out/'
        elif 'kFoldCV' in method:
            self.dirResult = paths.VALIDATION_RESULT_DIR+'/k_fold_cv/'
        else:
            self.dirResult = paths.VALIDATION_RESULT_DIR+'/results/'
        if not os.path.exists(self.dirResult):
            os.mkdir(self.dirResult)
        # Store the location to save processed networks
        self.dirNetworks = paths.NETWORK_TMP_DIR
        if not os.path.exists(self.dirNetworks):
            os.mkdir(self.dirNetworks)
        # Store the output file name prefix
        self.outputPrefix = None
        # Store the prediction matrix
        self.predictions = None
        # Store the row entity names
        self.y_node1 = None
        # Store the column entity names
        self.y_node2 = None
        # Store the prediction 1-d array for computing performance
        self.y_score = None
        # Store the prediction 1-d array for computing performance for
        # Diffusion
        self.y_score_scaled = None
        # Store the answer 1-d array for computing performance
        self.y_true = None
        # Store the total number of predictions
        self.predictionsN = None
        # Store the total number of true positives
        self.truePositiveN = None
        # Store the % interval of top predictions in which the total number of
        # true positives will be counted
        self.topPercentList = None
        # Store the list of total number of true positives in top % interval
        # predictions
        self.truePositiveNtopPercent = None
        # True positive rate; Recall; Sensitivity; used for computing AUC
        self.tpr = None
        # False positive rate
        self.fpr = None
        # Store the performance metrics
        self.AUC = None
        self.AUC_bootstrap = []
        self.AUC_bootstrapN = 100
        self.AUC_bootstrapMedian = None
        self.AUC_bootstrapStd = None
        self.pAUC = None
        self.pAUC_bootstrap = []
        self.pAUC_bootstrapMedian = None
        self.pAUC_bootstrapStd = None
        self.pAUC_thrshold = 0.01
        # Positive predictive value; Precision
        self.precision = None
        # True positive rate; Recall; Sensitivity; used for computing AUC of PR
        # curve
        self.recall = None
        # Average precision; Area under the precision-recall curve
        self.AUC_PR = None
        self.AUC_PR_bootstrap = []
        self.AUC_PR_bootstrapMedian = None
        self.AUC_PR_bootstrapStd = None
        self.pAUC_PR = None
        self.pAUC_PR_bootstrap = []
        self.pAUC_PR_bootstrapMedian = None
        self.pAUC_PR_bootstrapStd = None
        # F1 score
        self.F1 = np.nan
        # Mean reciprocal rank
        self.MRR = np.nan
        # Mean average precision at K
        self.MAPat10 = None
        self.MAPat20 = None
        self.MAPat30 = None
        self.MAPat40 = None
        self.MAPat50 = None
        self.savedVars = ["fpr", "tpr", "AUC", "pAUC", "AUC_bootstrap", "pAUC_bootstrap", 
        "AUC_bootstrapMedian", "pAUC_bootstrapMedian","AUC_bootstrapStd", "pAUC_bootstrapStd", "recall", "precision", "AUC_PR", 
        "pAUC_PR", "AUC_PR_bootstrap", "pAUC_PR_bootstrap", "AUC_PR_bootstrapMedian", 
        "pAUC_PR_bootstrapMedian","AUC_PR_bootstrapStd", "pAUC_PR_bootstrapStd","F1", "MRR", "predictionsN",
        "selectedEntityList", "selectedQueryMode", "selectedPredictionMode", 
        "outputPrefix", "y_node1", "y_node2", "splitPerformanceByEntity", 
        "pAUC_thrshold", "truePositiveN", "AUC_bootstrapN", "truePositiveNtopPercent", 
        "topPercentList", "MAPat10", "MAPat20", "MAPat30", "MAPat40", "MAPat50"]

    def __del__(self):
        '''
        Destructor

        Deletes class data before deleting instance of class.
        '''
        del(self.name)
        del(self.method)
        del(self.algorithm)
        del(self.network)
        del(self.split)
        del(self.selectedEntityList)
        del(self.selectedQueryMode)
        del(self.selectedPredictionMode)
        del(self.splitPerformanceByEntity)
        del(self.inputNetwork)
        del(self.answerNetwork)
        del(self.dirResult)
        del(self.dirNetworks)
        del(self.outputPrefix)
        del(self.predictions)
        del(self.y_node1)
        del(self.y_node2)
        del(self.y_score)
        del(self.y_score_scaled)
        del(self.y_true)
        del(self.predictionsN)
        del(self.truePositiveN)
        del(self.tpr)
        del(self.fpr)
        del(self.AUC)
        del(self.AUC_bootstrap)
        del(self.AUC_bootstrapMedian)
        del(self.pAUC)
        del(self.pAUC_bootstrap)
        del(self.pAUC_bootstrapMedian)
        del(self.pAUC_thrshold)
        del(self.precision)
        del(self.recall)
        del(self.AUC_PR)
        del(self.AUC_PR_bootstrap)
        del(self.AUC_PR_bootstrapMedian)
        del(self.pAUC_PR)
        del(self.pAUC_PR_bootstrap)
        del(self.pAUC_PR_bootstrapMedian)
        del(self.F1)
        del(self.MRR)
        del(self.validationNetworkNames)

    def __str__(self):
        '''
        To String

        This method generates an informative string to describe the current
        ValidationParent or inheriting class instance.

        Returns
        -------
        str
            A string detailing the method, network name, and split of the
            current validation experiment.
        '''
        objStr = 'Validation name: {}\n'.format(self.getName())
        # objStr += 'Algorithm name: {}\n'.format(self.algorithm.getName())
        objStr += 'Network name: {}'.format(self.network.getName())
        objStr += 'Split factor: {}'.format(self.split)
        return objStr

    def getName(self):
        '''
        Get Name

        Standard Getter

        This method returns the name state. The method is not safe as it passes
        a reference to the actual instance name.
        '''
        return self.name

    def setName(self, newName):
        '''
        Set Name

        Standard Setter

        This method alters the name state.
        '''
        self.name = newName

    def getMethod(self):
        '''
        Get method

        Standard Getter

        This method returns the method state. The method is not safe as it
        passes a reference to the actual instance method.
        '''
        return self.method

    def setMethod(self, newMethod):
        '''
        Set Method

        Standard Setter

        This method alters the method state.
        '''
        self.method = newMethod

    def getAlgorithm(self):
        '''
        Get algorithm

        Standard Getter

        This method returns the algorithm state. The method is not safe as it
        passes a reference to the actual instance algorithm.
        '''
        return self.algorithm

    def setAlgorithm(self, newAlgorithm):
        '''
        Set algorithm

        Standard Setter

        This method alters the algorithm state.
        '''
        self.algorithm = newAlgorithm

    def getNetwork(self):
        '''
        Get network

        Standard Getter

        This method returns the network state. The method is not safe as it
        passes a reference to the actual instance network.
        '''
        return self.network

    def setNetwork(self, newNetwork):
        '''
        Set network

        Standard Setter

        This method alters the network state.
        '''
        self.network = newNetwork

    def getSplit(self):
        '''
        Get split

        Standard Getter

        This method returns the split state. The method is not safe as it passes
        a reference to the actual instance split.
        '''
        return self.split

    def setSplit(self, newSplit):
        '''
        Set split

        Standard Setter

        This method alters the split state.
        '''
        self.split = newSplit

    def getSelectedEntityList(self):
        '''
        Get selected entity list

        Standard Getter

        This method returns the selected entity list state. The method is not
        safe as it passes a reference to the actual instance selected entity
        list.
        '''
        return self.selectedEntityList

    def setSelectedEntityList(self, selectedEntityList):
        '''
        Set selected entity list

        Standard Setter

        This method alters the selected entity list state.
        '''
        self.selectedEntityList = selectedEntityList

    def getSelectedQueryMode(self):
        '''
        Get selected query mode

        Standard Getter

        This method returns the selected query mode state. The method is not
        safe as it passes a reference to the actual instance selected query
        mode.
        '''
        return self.selectedQueryMode

    def setSelectedQueryMode(self, selectedQueryMode):
        '''
        Set selected query mode

        Standard Setter

        This method alters the selected query mode state.
        '''
        self.selectedQueryMode = selectedQueryMode

    def getSelectedPredictionMode(self):
        '''
        Get selected prediction mode

        Standard Getter

        This method returns the selected prediction mode state. The method is
        not safe as it passes a reference to the actual instance selected
        prediction mode.
        '''
        return self.selectedPredictionMode

    def setSelectedPredictionMode(self, selectedPredictionMode):
        '''
        Set selected prediction mode

        Standard Setter

        This method alters the selected prediction mode state.
        '''
        self.selectedPredictionMode = selectedPredictionMode

    def getSplitPerformanceByEntity(self):
        '''
        Get split performance by entity

        Standard Getter

        This method returns the split performance by entity state. The method is
        not safe as it passes a reference to the actual instance split
        performance by entity.
        '''
        return self.splitPerformanceByEntity

    def setSplitPerformanceByEntity(self, splitPerformanceByEntity):
        '''
        Set split performance by entity

        Standard Setter

        This method alters the split performance by entity state.
        '''
        self.splitPerformanceByEntity = splitPerformanceByEntity

    def getInputNetwork(self):
        '''
        Get input network

        Standard Getter

        This method returns the input network state. The method is not safe as
        it passes a reference to the actual instance input network.
        '''
        return self.inputNetwork

    def setInputNetwork(self, newInputNetwork):
        '''
        Set input network

        Standard Setter

        This method alters the input network state.
        '''
        self.inputNetwork = newInputNetwork

    def getAnswerNetwork(self):
        '''
        Get answer network

        Standard Getter

        This method returns the answer network state. The method is not safe as
        it passes a reference to the actual instance answer network.
        '''
        return self.answerNetwork

    def setAnswerNetwork(self, answerNetwork):
        '''
        Set answer network

        Standard Setter

        This method alters the answer network state.
        '''
        self.answerNetwork = answerNetwork

    def getDirResult(self):
        '''
        Get result directory

        Standard Getter

        This method returns the result directory state. The method is not safe
        as it passes a reference to the actual instance result directory.
        '''
        return self.dirResult

    def setDirResult(self, dirResult):
        '''
        Set result directory

        Standard Setter

        This method alters the result directory state.
        '''
        self.dirResult = dirResult

    def getDirNetworks(self):
        '''
        Get dir networks

        Standard Getter

        This method returns the dir networks state. The method is not safe as it passes
        a reference to the actual instance dir networks.
        '''
        return self.dirNetworks

    def setDirNetworks(self, dirNetworks):
        '''
        Set network directory

        Standard Setter

        This method alters the network directory state.
        '''
        self.dirNetworks = dirNetworks

    def getOutputPrefix(self):
        '''
        Get output prefix

        Standard Getter

        This method returns the output prefix state. The method is not safe as
        it passes a reference to the actual instance output prefix.
        '''
        return self.outputPrefix

    def getPredictions(self):
        '''
        Get predictions

        Standard Getter

        This method returns the predictions state. The method is not safe as it passes
        a reference to the actual instance predictions.
        '''
        return self.predictions

    def getYNode1(self):
        '''
        Get y node 1

        Standard Getter

        This method returns the y node 1 state. The method is not safe as it
        passes a reference to the actual instance y node 1.
        '''
        return self.y_node1

    def getYNode2(self):
        '''
        Get y node 2

        Standard Getter

        This method returns the y node 2 state. The method is not safe as it
        passes a reference to the actual instance y node 2.
        '''
        return self.y_node2

    def getYScore(self):
        '''
        Get y score

        Standard Getter

        This method returns the y score state. The method is not safe as it
        passes a reference to the actual instance y score.
        '''
        return self.y_score

    def getYScoreScaled(self):
        '''
        Get y score scaled

        Standard Getter

        This method returns the y score scaled state. The method is not safe as
        it passes a reference to the actual instance y score scaled.
        '''
        return self.y_score_scaled

    def getYTrue(self):
        '''
        Get y true

        Standard Getter

        This method returns the y true state. The method is not safe as it
        passes a reference to the actual instance y true.
        '''
        return self.y_true

    def getPredictionsN(self):
        '''
        Get number of predictions

        Standard Getter

        This method returns the number of predictions. The method is not safe as
        it passes a reference to the actual instance number of predictions.
        '''
        return self.predictionsN

    def getTruePositiveN(self):
        '''
        Get number of true positives

        Standard Getter

        This method returns the number of true positives state. The method is
        not safe as it passes a reference to the actual instance number of true
        positives.
        '''
        return self.truePositiveN

    def getTPR(self):
        '''
        Get TPR

        Standard Getter

        This method returns the TPR state. The method is not safe as it passes
        a reference to the actual instance TPR.
        '''
        return self.tpr

    def getFPR(self):
        '''
        Get FPR

        Standard Getter

        This method returns the FPR state. The method is not safe as it passes
        a reference to the actual instance FPR.
        '''
        return self.fpr

    def getAUC(self):
        '''
        Get AUC

        Standard Getter

        This method returns the AUC state. The method is not safe as it passes
        a reference to the actual instance AUC.
        '''
        return self.AUC

    def getAUCBootstrap(self):
        '''
        Get AUC Bootstrap

        Standard Getter

        This method returns the AUC Bootstrap state. The method is not safe as
        it passes a reference to the actual instance AUC Bootstrap.
        '''
        return self.AUC_bootstrap

    def getAUCBootstrapMedian(self):
        '''
        Get AUC Bootstrapping Median

        Standard Getter

        This method returns the AUC Bootstrapping Median state. The method is
        not safe as it passes a reference to the actual instance AUC
        Bootstrapping Median.
        '''
        return self.AUC_bootstrapMedian

    def getPAUC(self):
        '''
        Get Partial AUC

        Standard Getter

        This method returns the Partial AUC state. The method is not safe as it
        passes a reference to the actual instance Partial AUC.
        '''
        return self.pAUC

    def getPAUCBootstrap(self):
        '''
        Get Partial AUC Bootstrap

        Standard Getter

        This method returns the Partial AUC Bootstrap state. The method is not
        safe as it passes a reference to the actual instance Partial AUC
        Bootstrap.
        '''
        return self.pAUC_bootstrap

    def getPAUCBootstrapMedian(self):
        '''
        Get Partial AUC Bootstrapping Median

        Standard Getter

        This method returns the Partial AUC Bootstrapping Median state. The
        method is not safe as it passes a reference to the actual instance
        Partial AUC Bootstrapping Median.
        '''
        return self.pAUC_bootstrapMedian

    def getPAUCThreshold(self):
        '''
        Get Partial AUC Threshold

        Standard Getter

        This method returns the Partial AUC Threshold state. The method is not
        safe as it passes a reference to the actual instance Partial AUC
        Threshold.
        '''
        return self.pAUC_thrshold

    def getPrecision(self):
        '''
        Get precision

        Standard Getter

        This method returns the precision state. The method is not safe as it
        passes a reference to the actual instance precision.
        '''
        return self.precision

    def getRecall(self):
        '''
        Get recall

        Standard Getter

        This method returns the recall state. The method is not safe as it
        passes a reference to the actual instance recall.
        '''
        return self.recall

    def getAUCPR(self):
        '''
        Get AUC PR

        Standard Getter

        This method returns the AUC PR state. The method is not safe as it
        passes a reference to the actual instance AUC PR.
        '''
        return self.AUC_PR

    def getAUCPRBootstrap(self):
        '''
        Get AUC PR Bootstrap

        Standard Getter

        This method returns the AUC PR Bootstrap state. The method is not safe
        as it passes a reference to the actual instance AUC PR Bootstrap.
        '''
        return self.AUC_PR_bootstrap

    def getAUCPRBootstrapMedian(self):
        '''
        Get AUC PR Bootstrapping Median

        Standard Getter

        This method returns the AUC PR Bootstrapping Median state. The method is
        not safe as it passes a reference to the actual instance AUC PR
        Bootstrapping Median.
        '''
        return self.AUC_PR_bootstrapMedian

    def getPAUCPR(self):
        '''
        Get Partial AUC PR

        Standard Getter

        This method returns the Partial AUC PR state. The method is not safe as
        it passes a reference to the actual instance Partial AUC PR.
        '''
        return self.pAUC_PR

    def getPAUCPRBootstrap(self):
        '''
        Get Partial AUC PR Bootstrap

        Standard Getter

        This method returns the Partial AUC PR Bootstrap state. The method is
        not safe as it passes a reference to the actual instance Partial AUC PR
        Bootstrap.
        '''
        return self.pAUC_PR_bootstrap

    def getPAUCPRBootstrapMedian(self):
        '''
        Get Partial AUC PR Bootstrap Median

        Standard Getter

        This method returns the Partial AUC PR Bootstrap Median state.
        The method is not safe as it passes a reference to the actual instance
        Partial AUC PR Bootstrap Median.
        '''
        return self.pAUC_PR_bootstrapMedian

    def getF1(self):
        '''
        Get F1

        Standard Getter

        This method returns the F1 state. The method is not safe as it passes
        a reference to the actual instance F1.
        '''
        return self.F1

    def getMRR(self):
        '''
        Get MRR

        Standard Getter

        This method returns the MRR state. The method is not safe as it passes
        a reference to the actual instance MRR.
        '''
        return self.MRR

    def getResults(self):
        '''
        Get results from validation experiment

        Standard Getter

        '''
        return {x: self.__dict__[x] for x in self.savedVars}
        # if self.selectedEntityList is None and self.splitPerformanceByEntity is False:
        #     return (self.fpr, self.tpr, self.AUC, self.pAUC, self.AUC_bootstrap,
        #             self.pAUC_bootstrap, self.AUC_bootstrapMedian,
        #             self.pAUC_bootstrapMedian, self.recall, self.precision,
        #             self.AUC_PR, self.pAUC_PR, self.AUC_PR_bootstrap,
        #             self.pAUC_PR_bootstrap, self.AUC_PR_bootstrapMedian,
        #             self.pAUC_PR_bootstrapMedian, self.F1, self.predictionsN,
        #             self.truePositiveN, self.truePositiveNtopPercent)
        # else:
        #     return (self.fpr, self.tpr, self.AUC, self.pAUC, self.AUC_bootstrap,
        #             self.pAUC_bootstrap, self.AUC_bootstrapMedian,
        #             self.pAUC_bootstrapMedian, self.recall, self.precision,
        #             self.AUC_PR, self.pAUC_PR, self.AUC_PR_bootstrap,
        #             self.pAUC_PR_bootstrap, self.AUC_PR_bootstrapMedian,
        #             self.pAUC_PR_bootstrapMedian, self.F1, self.predictionsN,
        #             self.truePositiveN, self.truePositiveNtopPercent,
        #             self.y_node1, self.y_node2)

    def getDegreeByMode(self, node, mode):
        n = 0
        for x in ['InDegree','OutDegree']:
            try:
                n += node.getDegreeByMode()[x][mode]
            except:
                pass
        return n

    def perform(self):
        raise NotImplementedError(
            'The perform method for this Validation method has not been implemented yet!')

    def generateOutputPrefix(self, Alg, networkName=None):
        '''Generate output prefix'''
        if networkName is None:
            networkName = self.network.getName()
        outputPrefixList = [Alg.getName(), networkName,
                            self.getName()]
        self.outputPrefix = '_'.join(outputPrefixList)
        if self.selectedEntityList is not None:
            self.outputPrefix = self.outputPrefix + '_selectedEntity'
        if self.selectedQueryMode is not None:
            if(hasattr(self.selectedQueryMode, '__iter__')):
                self.outputPrefix = self.outputPrefix + \
                    '_qMode_' + '_'.join(self.selectedQueryMode)
            else:
                self.outputPrefix = self.outputPrefix + \
                    '_qMode_' + self.selectedQueryMode
        if self.selectedPredictionMode is not None:
            if(hasattr(self.selectedPredictionMode, '__iter__')):
                self.outputPrefix = self.outputPrefix + '_pMode_' + \
                    '_'.join(self.selectedPredictionMode)
            else:
                self.outputPrefix = self.outputPrefix + \
                    '_pMode_' + self.selectedPredictionMode
        if self.splitPerformanceByEntity:
            self.outputPrefix = self.outputPrefix + '_byEntity'
        self.outputPrefix = self.outputPrefix.replace(
            "Mapping_Ccgdd16Sx91Extended_101217_", "")
        return self.outputPrefix

    def generateValidationNetwork(self):
        '''Template to generate validation network'''
        raise NotImplementedError(
            'The generateValidationNetwork method for this Validation method has not been implemented yet!')

    def loadValidationNetwork(self, validationNetworkNames=None):
        '''
        Load validation network

        Reload the old network and gained edges from saved files
        '''
        if validationNetworkNames is None:
            validationNetworkNames = self.validationNetworkNames
        fileExists = map(lambda x: os.path.exists(self.dirNetworks+ '/' + x + '.pkl'),
                         validationNetworkNames)
        if sum(fileExists) == len(fileExists):
            # print 'Reloading saved validation networks'
            if 'prospective' in self.getName():
                self.answerNetwork = GraphRepresentation5()
                self.answerNetwork.loadGraph(validationNetworkNames[0] +
                                             '.pkl')
                # print('{} was loaded.'.format(
                #     validationNetworkNames[0]))
            else:
                self.inputNetwork = GraphRepresentation5()
                self.inputNetwork.loadGraph(validationNetworkNames[0] +
                                            '.pkl')
                self.answerNetwork = GraphRepresentation5()
                self.answerNetwork.loadGraph(validationNetworkNames[1] +
                                             '.pkl')
                # print('{} and {} were loaded.'.format(
                #     validationNetworkNames[0], validationNetworkNames[1]))
            return True
        else:
            return False

    def saveValidationNetwork(self, validationNetworkNames=None):
        '''
        Save validation network

        Save the input network and answer network
        '''
        if validationNetworkNames is None:
            validationNetworkNames = self.validationNetworkNames
        if not os.path.exists(self.dirNetworks):
            os.mkdir(self.dirNetworks)
        os.chdir(self.dirNetworks)
        if 'prospective' in self.getName():
            for i, net in enumerate([self.answerNetwork]):
                pFN = validationNetworkNames[i] + '.pkl'
                net.saveGraph(pFN)
                # print '{} was saved.'.format(pFN)            
        else:
            for i, net in enumerate([self.inputNetwork, self.answerNetwork]):
                pFN = validationNetworkNames[i] + '.pkl'
                net.saveGraph(pFN)
                # print '{} was saved.'.format(pFN)
        return

    def savePerformance(self):
        '''
        Save Performance

        This method saves the test results for the current validation
        experiment.
        '''
        if self.selectedEntityList is None and self.splitPerformanceByEntity == False:
            fout = open(self.outputPrefix + '_result.tsv', 'w')
            header = ['AUC', 'AUC median', 'pAUC', 'pAUC median',
                      'AUC_PR', 'AUC_PR median', 'pAUC_PR', 'pAUC_PR median']
            fout.write('\t'.join(header) + '\n')
            fout.write('{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5:.4f}\t{6:.4f}\t{7:.4f}\n'.format(
                self.AUC, self.AUC_bootstrapMedian, self.pAUC,
                self.pAUC_bootstrapMedian, self.AUC_PR,
                self.AUC_PR_bootstrapMedian, self.pAUC_PR,
                self.pAUC_PR_bootstrapMedian))
            fout.close()
        elif self.selectedEntityList is not None or self.splitPerformanceByEntity:
            fout = open(self.outputPrefix + '_result_entity.tsv', 'w')
            fout.write('Entity\tName\tDegree_D\tDegree_G\tDegree_C\tDegree_all\tTruePositiveN\tAUC\tAUC_bootstrapMedian\tAUC_bootstrapStd\tpAUC\tpAUC_bootstrapMedian\tpAUC_bootstrapStd\tAUC_PR\tAUC_PR_bootstrapMedian\tAUC_PR_bootstrapStd\tpAUC_PR\tpAUC_PR_bootstrapMedian\tpAUC_PR_bootstrapStd\n')
            for i, x in enumerate(self.AUC):
                # if not os.path.exists(self.selectedEntityList[i]):
                #     os.mkdir(self.selectedEntityList[i])
                # os.chdir(self.selectedEntityList[i])
                # fout = open((self.outputPrefix + '_' +
                # self.selectedEntityList[i] + '_result.tsv'), 'w')
                degree_d = self.getDegreeByMode(self.y_node1[i],'Disease')
                degree_g = self.getDegreeByMode(self.y_node1[i],'Gene')
                degree_c = self.getDegreeByMode(self.y_node1[i],'Chemical')
                degree_all = degree_d+degree_g+degree_c
                try:
                    fout.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7:.4f}\t{8:.4f}\t{9:.4f}\t{10:.4e}\t{11:.4e}\t{12:.4e}\t{13:.4f}\t{14:.4f}\t{15:.4f}\t{16:.4e}\t{17:.4e}\t{18:.4e}\n'.format(self.selectedEntityInitName[i], 
                        self.y_node1[i].getRelSynonyms()[0], 
                        degree_d, degree_g, degree_c, degree_all,
                        self.truePositiveN[i],
                        self.AUC[i], self.AUC_bootstrapMedian[i], self.AUC_bootstrapStd[i],
                        self.pAUC[i], self.pAUC_bootstrapMedian[i],self.pAUC_bootstrapStd[i],
                        self.AUC_PR[i], self.AUC_PR_bootstrapMedian[i], self.AUC_PR_bootstrapStd[i],
                        self.pAUC_PR[i], self.pAUC_PR_bootstrapMedian[i], self.pAUC_PR_bootstrapStd[i]))
                except:
                    fout.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7:.4f}\t{8:.4f}\t{9:.4f}\t{10:.4e}\t{11:.4e}\t{12:.4e}\t{13:.4f}\t{14:.4f}\t{15:.4f}\t{16:.4e}\t{17:.4e}\t{18:.4e}\n'.format(self.y_node1[i].getName(), 
                        self.y_node1[i].getRelSynonyms()[0], 
                        degree_d, degree_g, degree_c, degree_all,
                        self.truePositiveN[i],
                        self.AUC[i], self.AUC_bootstrapMedian[i], self.AUC_bootstrapStd[i],
                        self.pAUC[i], self.pAUC_bootstrapMedian[i],self.pAUC_bootstrapStd[i],
                        self.AUC_PR[i], self.AUC_PR_bootstrapMedian[i], self.AUC_PR_bootstrapStd[i],
                        self.pAUC_PR[i], self.pAUC_PR_bootstrapMedian[i], self.pAUC_PR_bootstrapStd[i]))                    
            fout.close()
        elif self.splitPerformanceByEntity:
            fout = open(self.outputPrefix + '_result.tsv', 'w')
            header = ['Entity ID', 'Entity name', 'True positive #', 'AUC',
                      'AUC_PR']
            fout.write('\t'.join(header) + '\n')
            for i in range(len(self.y_node1)):
                fout.write('{0}\t{1}\t{2}\t{3:.4f}\t{4:.4f}\n'.format(self.y_node1[i].getName(
                ), self.y_node1[i].getCanonical(), self.truePositiveN[i], self.AUC[i], self.AUC_PR[i]))
            fout.close()

        pFile = open(self.outputPrefix + '_result.pkl', 'wb')
        data = {x:self.__dict__[x] for x in self.savedVars}
        try:
            pickle.dump(data, pFile, protocol=pickle.HIGHEST_PROTOCOL)
        except:
            self.downsizeFprTprPrecisionRecall()
            data = {x:self.__dict__[x] for x in self.savedVars}
            pickle.dump(data, pFile, protocol=pickle.HIGHEST_PROTOCOL)
        pFile.close()
            # os.chdir('..')
        return

    def loadPerformance(self):
        '''
        Load performance

        Loads the results of all tests performed in this validation experiment.
        '''
        pFN = self.outputPrefix + '_result.pkl'
        if(os.path.exists(pFN)):
            # print(
            #     'Reloading pre-existing saved validation performance metrics')
            pFile = open(pFN, 'rb')
            loadedData = pickle.load(pFile)
            self.__dict__.update(loadedData)
            pFile.close()
            os.chdir('..')
            return True
        else:
            os.chdir('..')
            return False

    def printPerformance(self):
        return

    def savePredictions(self, n=100):
        return

    def getReindexedAnswerPrediciton(self, selectedRowMode=None, selectedColMode=None, selectedNodes=None):
        '''
        Get re-inidexed answer predictions

        Re-indexes the results generated by the prediction algorithm.
        '''
        if selectedNodes is None:
            selectedRowEntityList = self.selectedNodes
        else:
            selectedRowEntityList = selectedNodes
        if selectedRowMode is None:
            selectedRowMode = self.selectedQueryMode
        if selectedColMode is None:
            selectedColMode = self.selectedPredictionMode
        answerNodes = self.answerNetwork.getNodePosition()
        # print "answerNodes length {}".format(len(answerNodes.keys()))
        # Get nodes of prediction matrix, which should be the same as the input
        # network
        predictionNodes = self.inputNetwork.getNodePosition()
        # print "predictionNodes length {}".format(len(predictionNodes.keys()))
        if ('AptRank' in self.outputPrefix) and ('FoldCV' not in self.name):
            predict_rows = self.inputNetwork.getModePosition(selectedRowMode)
            predict_cols = self.inputNetwork.getModePosition(selectedColMode)
            sortedPos = sorted([predict_rows, predict_cols])
            predictionNodesNew = {}
            for v, x in sorted(zip(predictionNodes.values(), predictionNodes.keys())):
                if sortedPos[0][0] <= v <= sortedPos[0][1]:
                    predictionNodesNew[x] = v - sortedPos[0][0]
                elif sortedPos[1][0] <= v <= sortedPos[1][1]:
                    predictionNodesNew[x] = v - sortedPos[1][0]
            predictionNodes = predictionNodesNew

        # Get overlapping nodes in answer matrix and prediction matrix
        overlappingNodes = self.answerNetwork.determineOverlappingNodes(
            self.inputNetwork)[1]
        # print "overlappingNodes length {}".format(len(overlappingNodes))
        if((selectedRowEntityList is None) and (selectedRowMode is None)):
            overlappingNodesSelectedRow = overlappingNodes
        elif((selectedRowEntityList is not None) and (selectedRowMode is not None)):
            overlappingNodesSelectedRow = [
                x for x in overlappingNodes if x.getMode() == selectedRowMode]
            overlappingNodesSelectedRow = set(
                selectedRowEntityList).intersection(overlappingNodesSelectedRow)
        elif(selectedRowEntityList is not None) and (selectedRowMode is None):
            overlappingNodesSelectedRow = set(
                selectedRowEntityList).intersection(overlappingNodes)
        elif(selectedRowEntityList is None) and (selectedRowMode is not None):
            overlappingNodesSelectedRow = [
                x for x in overlappingNodes if x.getMode() == selectedRowMode]

        overlappingNodeIndexAnswerSelectedRow = [answerNodes[x] for x in
                                                 overlappingNodesSelectedRow]
        overlappingNodeIndexPredictionSelectedRow = [predictionNodes[x] for x in
                                                     overlappingNodesSelectedRow]
        if(selectedColMode is not None):
            overlappingNodesSelectedCol = [
                x for x in overlappingNodes if x.getMode() == selectedColMode]
        else:
            overlappingNodesSelectedCol = overlappingNodes
        overlappingNodeIndexAnswerSelectedCol = [answerNodes[x] for x in
                                                 overlappingNodesSelectedCol]
        overlappingNodeIndexPredictionSelectedCol = [predictionNodes[x] for x in
                                                     overlappingNodesSelectedCol]
        # Get predictions
        predictions = self.getPredictions()
        # Convert to lil matrix
        # predictions = predictions.tolil()
        os.chdir(self.dirNetworks)
        # Reindex prediction matrix
        predictions = predictions[overlappingNodeIndexPredictionSelectedRow, :][
            :, overlappingNodeIndexPredictionSelectedCol]
        # Use answer network as true answer
        answerMatrix = self.answerNetwork.getAdjacencyMatrix()
        # Reindex answer matrix
        try:
            answerMatrix = answerMatrix[overlappingNodeIndexAnswerSelectedRow, :][
            :, overlappingNodeIndexAnswerSelectedCol]
        except:
            embed()
            sys.exit()

        # # To remove rows of all zeros in answer matrix, which are the
        # # cases we don't have answers to validate the performance
        # nonzeroRow = answerMatrix.sum(axis=1).nonzero()[0]
        # # nonzeroCol = answerMatrix.sum(axis=0).nonzero()[1]
        # answerMatrix = answerMatrix[nonzeroRow, :]
        # # answerMatrix = answerMatrix[:,nonzeroCol]
        # predictions = predictions[nonzeroRow, :]
        # # predictions = predictions[:,nonzeroCol]
        # overlappingNodesSelectedRow = [
        #     overlappingNodesSelectedRow[x] for x in nonzeroRow]
        # # overlappingNodesSelectedCol = [overlappingNodesSelectedCol[x] for x in nonzeroRow]

        if 'FoldCV' in self.name:
            # Do not remove known edges in overall k-fold CV
            knownEdgeIdx = None
        else:
            inputNetwork = self.inputNetwork.getAdjacencyMatrix()
            predictionNodes = self.inputNetwork.getNodePosition()
            overlappingNodeIndexPredictionSelectedRow = [predictionNodes[x] for x in
                                                         overlappingNodesSelectedRow]
            overlappingNodeIndexPredictionSelectedCol = [predictionNodes[x] for x in
                                                         overlappingNodesSelectedCol]
            inputNetwork2 = inputNetwork[overlappingNodeIndexPredictionSelectedRow, :][
                :, overlappingNodeIndexPredictionSelectedCol]
            try:
                inputNetwork2 = inputNetwork2[nonzeroRow, :]
            except:
                pass
            knownEdgeIdx = inputNetwork2.nonzero()
            # Remove predictions which is known in old network
            predictions[knownEdgeIdx] = 0
        return (answerMatrix, predictions, overlappingNodesSelectedRow,
                overlappingNodesSelectedCol, knownEdgeIdx)

    def generatePredicitonAnswerArray(self):
        '''
        Generate Prediction Answer Array

        This method sets the y_node1, y_node2, y_score, y_true, predictionsN,
        and truePositiveN variables by processing the existing predictions.
        '''
        # Synchronize indices of prediction and answer matrix
        # ['Parkinson Disease', 'Parkinsonian Disorders', 'Parkinson Disease, Secondary']
        # selectedEntityList = ['Parkinson Disease', 'Parkinsonian Disorders', 'Parkinson Disease, Secondary']
        # selectedPredictionMode = 'Gene'
        if ((self.selectedQueryMode is not None) and (self.selectedPredictionMode is not None)):
            if((hasattr(self.selectedQueryMode, '__iter__')) and (hasattr(self.selectedPredictionMode, '__iter__'))):
                self.y_score = np.array([])
                self.y_true = np.array([])
                self.predictionsN = 0
                self.truePositiveN = 0
                for selectedRowMode, selectedColMode in zip(self.selectedQueryMode, self.selectedPredictionMode):
                    res = self.getReindexedAnswerPrediciton(
                        selectedRowMode, selectedColMode)
                    y_score, y_true, predictionsN, truePositiveN = self.selectScoredSamples(
                        res)
                    self.y_score = np.append(self.y_score, y_score)
                    self.y_true = np.append(self.y_true, y_true)
                    self.predictionsN += predictionsN
                    self.truePositiveN += truePositiveN
            elif((hasattr(self.selectedQueryMode, '__iter__')) or (hasattr(self.selectedPredictionMode, '__iter__'))):
                raise NotImplementedError(
                    "The length of selectedRowMode and selectedColMode should be the same!!!")
            else:
                if 'prospective' in self.getName() or self.splitPerformanceByEntity:
                    self.y_score = []
                    self.y_true = []
                    self.predictionsN = []
                    self.truePositiveN = []
                    if self.selectedNodes is None:
                        if self.splitPerformanceByEntity and self.selectedEntityList is None:
                            query_cols = self.inputNetwork.getModePosition(
                                self.selectedQueryMode)
                            self.positionNode = self.inputNetwork.getPositionNode()
                            self.selectedNodes = [self.positionNode[x] for x in range(query_cols[0], query_cols[1]+1)]
                            self.nodePosition = self.inputNetwork.getNodePosition()
                            selectedRowPos = [self.nodePosition[x] for x in self.selectedNodes]
                            predict_cols = self.inputNetwork.getModePosition(
                                self.selectedPredictionMode)
                            if 'AptRank' in self.outputPrefix:
                                predict_rows = self.inputNetwork.getModePosition(
                                    self.selectedQueryMode)
                                sortedPos = sorted([predict_rows, predict_cols])
                                predictionNodesNew = {}
                                for v, x in sorted(zip(self.nodePosition.values(), self.nodePosition.keys())):
                                    if sortedPos[0][0] <= v <= sortedPos[0][1]:
                                        predictionNodesNew[x] = v - sortedPos[0][0]
                                    elif sortedPos[1][0] <= v <= sortedPos[1][1]:
                                        predictionNodesNew[x] = v - sortedPos[1][0]
                                selectedRowPosNew = [predictionNodesNew[x] for x in self.selectedNodes]
                                predictions = self.predictions[selectedRowPosNew, :]
                            else:
                                predictions = self.predictions[selectedRowPos, :][
                                    :, predict_cols[0]:predict_cols[1] + 1]
                            self.y_node1 = self.selectedNodes
                            self.y_node2 = [self.positionNode[x]
                                            for x in range(predict_cols[0], predict_cols[1] + 1)]
                        else:
                            raise ValueError("self.selectedNodes has to be assigned")
                    for selectedNode in self.selectedNodes:
                        if selectedNode in self.answerNetwork.getNodes():
                            res = self.getReindexedAnswerPrediciton(selectedNodes=[selectedNode])
                            y_score, y_true, predictionsN, truePositiveN = self.selectScoredSamples(res)
                        else:
                            y_score, y_true, predictionsN, truePositiveN = [np.nan],[np.nan],np.nan,np.nan
                        self.y_score.append(y_score)
                        self.y_true.append(y_true)
                        self.predictionsN.append(predictionsN)
                        self.truePositiveN.append(truePositiveN)
                else:
                    res = self.getReindexedAnswerPrediciton()
                    self.y_score, self.y_true, self.predictionsN, self.truePositiveN = self.selectScoredSamples(
                        res)
        else:
            res = self.getReindexedAnswerPrediciton()
            self.y_score, self.y_true, self.predictionsN, self.truePositiveN = self.selectScoredSamples(
                res)
        if type(self.y_score) == list:
            pass
        else:
            if np.inf in self.y_score:
                maxNonInf = np.max(self.y_score[self.y_score != np.inf])
                self.y_score[self.y_score == np.inf] = maxNonInf + 10
            self.computeTopPercentTruePositives()
        return

    def selectScoredSamples(self, res):
        '''
        Convert Prediction and Answer Matrix to 1-D Array and Delete Known Edges

        This method return the y_node1, y_node2, y_score, y_true, predictionsN,
        and truePositiveN variables by processing the existing predictions.
        '''
        answerMatrix = res[0]
        if answerMatrix.shape[0] == 0:
            return np.nan,np.nan,np.nan,np.nan
        predictions = res[1]
        overlappingNodesSelectedRow = res[2]
        overlappingNodesSelectedCol = res[3]
        knownEdgeIdx = res[4]
        # print "answerMatrix size: {}".format(answerMatrix.shape)
        # print "predictions size: {}".format(predictions.shape)
        # try:
        #     print "knownEdgeIdx size:{}".format(len(knownEdgeIdx[0]))
        # except:
        #     print "knownEdgeIdx size:0"
        if(answerMatrix.shape[0] == answerMatrix.shape[1]):
            # Only take upper triangle of matrix to reduce computing time
            upperTriIdx = np.triu_indices(answerMatrix.shape[0])
            upperTriIdxSet = set(zip(*upperTriIdx))
            if knownEdgeIdx is None:
                knownEdgeIdxSet = set()
            else:
                knownEdgeIdxSet = set(zip(*knownEdgeIdx))
            # Only take elements which are unknown in input network
            scoredIdxSet = upperTriIdxSet.difference(knownEdgeIdxSet)
            scoredIdx = zip(*scoredIdxSet)
            scoredIdx = (np.array(scoredIdx[0]), np.array(scoredIdx[1]))
            upperAnswerMatrix = answerMatrix[scoredIdx]
            # Reshape answer matrix to a 1-d vector
            y_true_array = upperAnswerMatrix.A1
            # Convert answer matrix to binary values
            y_true_array = y_true_array > 0
            # Get upper prediction matrix
            upperPredictions = predictions[scoredIdx]
            # Reshape upper prediction matrix to a 1-d vector
            if(type(upperPredictions) == np.matrixlib.defmatrix.matrix):
                upperPredictions = upperPredictions.A1
            elif((type(upperPredictions) == np.ndarray) and
                 (len(upperPredictions.shape) == 1)):
                pass
            elif((type(upperPredictions) == lil_matrix)):
                upperPredictions = upperPredictions.toarray()[0]
            else:
                raise TypeError(
                    "PredictionMatrix: Prediction type cannot be recognized!")
            # Remove zeros from predictions and corresponding element in y_true
            # This step was taken out because it should be consistent size of
            # matrix across every algorithm

            # nonZeroIdx = upperPredictions.nonzero()
            # y_score = upperPredictions[nonZeroIdx]
            # y_true = y_true_array[nonZeroIdx]
            y_score = upperPredictions
            y_true = y_true_array
            predictionsN = len(y_score)
            truePositiveN = np.sum(y_true)
        else:
            if type(predictions) == np.matrixlib.defmatrix.matrix:
                y_score = predictions.A1
                y_true = answerMatrix.toarray().flatten() > 0
                raise NotImplementedError(
                    'type(predictions) == np.matrixlib.defmatrix.matrix has not been implemented.')
            else:
                if 'FoldCV' in self.name:
                    # Do not remove known edges in overall k-fold CV
                    y_score = predictions.toarray().flatten()
                    y_true = answerMatrix.toarray().flatten() > 0
                else:
                    predMat = predictions.toarray()
                    ncol = predMat.shape[1]
                    OneDknownEdgeIdx = [
                        x * ncol + y for x, y in zip(knownEdgeIdx[0], knownEdgeIdx[1])]
                    y_score = np.delete(predMat, OneDknownEdgeIdx, None)
                    y_true = np.delete(
                        answerMatrix.toarray() > 0, OneDknownEdgeIdx)
                    # print '{} predictions were deleted because they are known in the current networks'.format(len(OneDknownEdgeIdx))
            predictionsN = len(y_score)
            truePositiveN = np.sum(y_true)
        return (y_score, y_true, predictionsN, truePositiveN)

    def computeTopPercentTruePositives(self, topPercentList=[0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1]):
        if self.topPercentList is None:
            # self.topPercentList is not predefined
            self.topPercentList = topPercentList
        else:
            topPercentList = self.topPercentList
        sortedTruePositives = [
            x for y, x in sorted(zip(self.y_score, self.y_true), reverse=True)]
        # sortedScores = sorted(self.y_score, reverse=True)
        self.truePositiveNtopPercent = []
        for i in range(len(topPercentList) - 1):
            start = int(topPercentList[i] * self.predictionsN)
            end = int(topPercentList[i + 1] * self.predictionsN)
            self.truePositiveNtopPercent.append(
                np.sum(sortedTruePositives[start:end]))
            # print topPercentList[i]*100,topPercentList[i+1]*100,float(np.sum(sortedTruePositives[start:end]))/self.truePositiveN*100
            # print topPercentList[i]*100,topPercentList[i+1]*100,float(np.sum(sortedTruePositives[start:end]))/(end-start)*100
            # print start,end,np.sum(sortedTruePositives[start:end])
        return

    def computePartialAUCfromFprTpr(self, fprGiven=None, tprGiven=None,
                                    trapezoid=False):
        '''
        Compute partial auc

        Use the FPR and TPR to compute the partial AUC.  This code has been
        scribbed from the following link:
        http://stackoverflow.com/questions/39537443/how-to-calculate-a-partial-area-under-the-curve-auc

        Parameters
        ----------
        fprGiven: list
            A list of fpr values for a given experiment for which to calculate
            a partial AUC.
        tprGiven list
            A list of tpr values for a given experiment for which to calculate
            a partial AUC.
        trapezoid: bool
            Whether or not to use the trapezoid rule when calculating the
            partial AUC.  Set to False by default.

        Returns
        -------
        float
            The partial AUC given for the first section of the AUC curve as
            defined by the pAUC_thrshold variable.
        '''
        if fprGiven is None:
            p = bisect.bisect_left(self.fpr, self.pAUC_thrshold)
            fpr = self.fpr.copy()
            fpr[p] = self.pAUC_thrshold
            pFpr = fpr[: p + 1]
            pTpr = self.tpr[: p + 1]
        else:
            p = bisect.bisect_left(fprGiven, self.pAUC_thrshold)
            fprGiven[p] = self.pAUC_thrshold
            pFpr = fprGiven[: p + 1]
            pTpr = tprGiven[: p + 1]
        area = 0
        ft = zip(pFpr, pTpr)
        for p0, p1 in zip(ft[: -1], ft[1:]):
            area += (p1[0] - p0[0]) * \
                ((p1[1] + p0[1]) / 2 if trapezoid else p0[1])
        return area

    def computePartialAUCfromFprTpr2(self):
        '''
        Compute partial auc

        Use the FPR and TPR to compute the partial AUC.  This code has been
        scribbed from the following link:
        http://stackoverflow.com/questions/39537443/how-to-calculate-a-partial-area-under-the-curve-auc

        Returns
        -------
        float
            The partial AUC given for the first section of the AUC curve as
            defined by the pAUC_thrshold variable.
        '''
        p = bisect.bisect_left(self.fpr, self.pAUC_thrshold)
        fpr = self.fpr.copy()
        fpr[p] = self.pAUC_thrshold
        pFpr = fpr[: p + 1]
        pTpr = self.tpr[: p + 1]
        area = np.trapz(pFpr, pTpr)
        return area

    def computeAUCbootstrap(self, y_true=None, y_score=None, bootstrapRatio=0.2, min_positive_n=3):
        '''
        Compute AUC Bootstrap

        Computes the data to fill in AUC_bootstrapMedian,
        AUC_PR_bootstrapMedian, pAUC_bootstrapMedian,
        and pAUC_PR_bootstrapMedian.
        '''
        if y_true is None:
            y_true = self.y_true
        if y_score is None:
            y_score = self.y_score
        truePositiveN = sum(y_true)
        if truePositiveN < min_positive_n/bootstrapRatio:
            return [np.nan]*self.AUC_bootstrapN,[np.nan]*self.AUC_bootstrapN,[np.nan]*self.AUC_bootstrapN,[np.nan]*self.AUC_bootstrapN,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
        idx_TP = y_true.nonzero()[0]
        idx_TN = np.array(list(set(range(len(
            y_true))).difference(set(idx_TP))))
        sampleN = int(truePositiveN * bootstrapRatio)
        # print('Computing bootstraped AUC for {} samples per run and {} iterations'.format(
        #     sampleN * 2, self.AUC_bootstrapN))
        AUC_bootstrap = []
        AUC_PR_bootstrap = []
        pAUC_bootstrap = []
        pAUC_PR_bootstrap = []
        for j in range(self.AUC_bootstrapN):
            idx_TP_selected = np.random.choice(idx_TP, sampleN, replace=False)
            idx_TN_selected = np.random.choice(idx_TN, sampleN, replace=False)
            idx = np.concatenate([idx_TP_selected, idx_TN_selected])
            fpr, tpr, _threshold = roc_curve(y_true[idx],y_score[idx])
            precision, recall, _ = precision_recall_curve(y_true[idx], y_score[idx])
            AUC_PR_bootstrap.append(average_precision_score(y_true[idx], y_score[idx]))
            AUC_bootstrap.append(auc(fpr, tpr))
            pAUC_bootstrap.append(
                self.computePartialAUCfromFprTpr(fpr, tpr))
            # recall_sorted = sorted(recall)
            # precision_sorted = [x for y,x in sorted(zip(recall,precision))]
            # The reversed array is the same as the sorted array from small to
            # large
            recall_sorted = recall[::-1]
            precision_sorted = precision[::-1]
            pAUC_PR_bootstrap.append(
                self.computePartialAUCfromFprTpr(recall_sorted,
                                                 precision_sorted))
        AUC_bootstrapMedian = np.median(AUC_bootstrap)
        AUC_PR_bootstrapMedian = np.median(AUC_PR_bootstrap)
        pAUC_bootstrapMedian = np.median(pAUC_bootstrap)
        pAUC_PR_bootstrapMedian = np.median(pAUC_PR_bootstrap)
        AUC_bootstrapStd = np.std(AUC_bootstrap)
        AUC_PR_bootstrapStd = np.std(AUC_PR_bootstrap)
        pAUC_bootstrapStd = np.std(pAUC_bootstrap)
        pAUC_PR_bootstrapStd = np.std(pAUC_PR_bootstrap)
        return AUC_bootstrap,AUC_PR_bootstrap,pAUC_bootstrap,pAUC_PR_bootstrap,AUC_bootstrapMedian,AUC_PR_bootstrapMedian,pAUC_bootstrapMedian,pAUC_PR_bootstrapMedian,AUC_bootstrapStd,AUC_PR_bootstrapStd,pAUC_bootstrapStd,pAUC_PR_bootstrapStd

    def downsizeFprTprPrecisionRecall(self, size=10000):
        '''
        Down size FPR, TPR, and Precision recall

        This method reduces the size of the FPR, TPR, and precision recall
        results stored by this class.

        Parameters
        ----------
        size: int
            The number of values to reduce the TPR, FPR, and precision recall
            to.
        '''
        idx = np.linspace(0, len(self.fpr), num=size, endpoint=False)
        idx = [int(x) for x in idx]
        self.fpr = self.fpr[idx]
        self.fpr = np.append(self.fpr, [1])
        self.tpr = self.tpr[idx]
        self.tpr = np.append(self.tpr, [1])
        idx = np.linspace(0, len(self.recall), num=size, endpoint=False)
        idx = [int(x) for x in idx]
        self.recall = self.recall[idx]
        self.recall = np.append(self.recall, [1])
        self.precision = self.precision[idx]
        self.precision = np.append(self.precision, [0])
        return

    def computeFprTprPrecisionMRR(self):
        '''
        Compute FPR, TPR, Precision, and MRR

        This function computes the false positive rate, true positive rate,
        precision, and mean reciprocal rank for the experiments performed by
        this instance of the ValidationParent or inheriting child class.
        '''
        if((self.selectedEntityList is None) and
           (self.splitPerformanceByEntity is False)):
            self.fpr, self.tpr, self.AUC, self.pAUC, self.MRR, self.precision, self.recall, self.AUC_PR, self.pAUC_PR,self.AUC_bootstrap,self.AUC_PR_bootstrap,self.pAUC_bootstrap,self.pAUC_PR_bootstrap,self.AUC_bootstrapMedian,self.AUC_PR_bootstrapMedian,self.pAUC_bootstrapMedian,self.pAUC_PR_bootstrapMedian,self.AUC_bootstrapStd,self.AUC_PR_bootstrapStd,self.pAUC_bootstrapStd,self.pAUC_PR_bootstrapStd = self._computeMetric(self.y_score, self.y_true)
        else:
            self.tpr = []
            self.fpr = []
            self.AUC = []
            self.pAUC = []
            self.precision = []
            self.recall = []
            self.AUC_PR = []
            self.pAUC_PR = []
            self.F1 = []
            self.MRR = []
            self.AUC_bootstrap = []
            self.AUC_PR_bootstrap = []
            self.pAUC_bootstrap = []
            self.pAUC_PR_bootstrap = []
            self.AUC_bootstrapMedian = []
            self.AUC_PR_bootstrapMedian = []
            self.pAUC_bootstrapMedian = []
            self.pAUC_PR_bootstrapMedian = []
            self.AUC_bootstrapStd = []
            self.AUC_PR_bootstrapStd = []
            self.pAUC_bootstrapStd = []
            self.pAUC_PR_bootstrapStd = []            
            for i in range(len(self.y_score)):
                fpr, tpr, AUC, pAUC, MRR, precision, recall, AUC_PR, pAUC_PR, AUC_bootstrap,AUC_PR_bootstrap,pAUC_bootstrap,pAUC_PR_bootstrap,AUC_bootstrapMedian,AUC_PR_bootstrapMedian,pAUC_bootstrapMedian,pAUC_PR_bootstrapMedian,AUC_bootstrapStd,AUC_PR_bootstrapStd,pAUC_bootstrapStd,pAUC_PR_bootstrapStd = self._computeMetric(self.y_score[i], self.y_true[i], bootstrapRatio=0.5, min_positive_n=3)
                self.fpr.append(fpr)
                self.tpr.append(tpr)
                self.AUC.append(AUC)
                self.precision.append(precision)
                self.recall.append(recall)
                self.AUC_PR.append(AUC_PR)
                self.MRR.append(MRR)
                self.pAUC.append(pAUC)
                self.pAUC_PR.append(pAUC_PR)

                self.AUC_bootstrap.append(AUC_bootstrap)
                self.AUC_PR_bootstrap.append(AUC_PR_bootstrap)
                self.pAUC_bootstrap.append(pAUC_bootstrap)
                self.pAUC_PR_bootstrap.append(pAUC_PR_bootstrap)
                self.AUC_bootstrapMedian.append(AUC_bootstrapMedian)
                self.AUC_PR_bootstrapMedian.append(AUC_PR_bootstrapMedian)
                self.pAUC_bootstrapMedian.append(pAUC_bootstrapMedian)
                self.pAUC_PR_bootstrapMedian.append(pAUC_PR_bootstrapMedian)
                self.AUC_bootstrapStd.append(AUC_bootstrapStd)
                self.AUC_PR_bootstrapStd.append(AUC_PR_bootstrapStd)
                self.pAUC_bootstrapStd.append(pAUC_bootstrapStd)
                self.pAUC_PR_bootstrapStd.append(pAUC_PR_bootstrapStd)
        return

    def _computeMetric(self,y_score,y_true,bootstrapRatio=0.2, min_positive_n=3):
        if np.sum(y_score) == 0 or str(np.sum(y_score)) == str(np.nan) or np.sum(y_true)<5:
            # return fpr, tpr, AUC, pAUC, MRR, precision, recall, AUC_PR,pAUC_PR, AUC_bootstrap,AUC_PR_bootstrap,pAUC_bootstrap,pAUC_PR_bootstrap,AUC_bootstrapMedian,AUC_PR_bootstrapMedian,pAUC_bootstrapMedian,pAUC_PR_bootstrapMedian,AUC_bootstrapStd,AUC_PR_bootstrapStd,pAUC_bootstrapStd,pAUC_PR_bootstrapStd
            return [np.nan], [np.nan], np.nan, np.nan, np.nan, [np.nan], [np.nan], np.nan,np.nan, [np.nan]*self.AUC_bootstrapN,[np.nan]*self.AUC_bootstrapN,[np.nan]*self.AUC_bootstrapN,[np.nan]*self.AUC_bootstrapN,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
        # Compute fpr tpr
        # print('Computing fpr and tpr')
        if(('Diffusion' in self.outputPrefix) or
           ('RndWalk' in self.outputPrefix)):
            minNonZero = np.min(y_score[y_score.nonzero()])
            # print('The minimum of non-zero scores: {}'.format(minNonZero))
            minNonZero = 1.0 / minNonZero
            y_score_scaled = y_score * minNonZero
            fpr, tpr, _threshold = roc_curve(y_true, y_score_scaled)
        else:
            fpr, tpr, _threshold = roc_curve(y_true, y_score)
        # print('# of unique scores: {}'.format(len(fpr)))
        # print('# of non-zero scores: {}'.format(len(y_score.nonzero()[0])))
        # print('# of predictions: {}'.format(len(y_score)))
        # print('Ratio of non-zero scores: {}'.format(float(len(y_score.nonzero()[0])) / (len(y_score))))
        # print('Computing AUC')
        # Compute AUC
        # self.downsizeFprTprPrecisionRecall()
        AUC = auc(fpr, tpr)
        pAUC = self.computePartialAUCfromFprTpr(fpr, tpr)
        # print('Computing precision and recall')
        # Compute precision
        if(('Diffusion' in self.outputPrefix) or
           ('RndWalk' in self.outputPrefix)):
            precision, recall, _ = precision_recall_curve(
                y_true, y_score_scaled)
            AUC_PR = average_precision_score(y_true,
                                                  y_score_scaled)
            y_score_rank = len(y_score_scaled) + 1 - rankdata(y_score_scaled)
            AUC_bootstrap,AUC_PR_bootstrap,pAUC_bootstrap,pAUC_PR_bootstrap,AUC_bootstrapMedian,AUC_PR_bootstrapMedian,pAUC_bootstrapMedian,pAUC_PR_bootstrapMedian,AUC_bootstrapStd,AUC_PR_bootstrapStd,pAUC_bootstrapStd,pAUC_PR_bootstrapStd = self.computeAUCbootstrap(y_true,y_score_scaled,bootstrapRatio, min_positive_n)
        else:
            precision, recall, _ = precision_recall_curve(
                y_true, y_score)
            AUC_PR = average_precision_score(y_true,
                                                  y_score)
            y_score_rank = len(y_score) + 1 - rankdata(y_score)
            AUC_bootstrap,AUC_PR_bootstrap,pAUC_bootstrap,pAUC_PR_bootstrap,AUC_bootstrapMedian,AUC_PR_bootstrapMedian,pAUC_bootstrapMedian,pAUC_PR_bootstrapMedian,AUC_bootstrapStd,AUC_PR_bootstrapStd,pAUC_bootstrapStd,pAUC_PR_bootstrapStd = self.computeAUCbootstrap(y_true,y_score,bootstrapRatio, min_positive_n)
        pAUC_PR = self.computePartialAUCfromFprTpr(recall[::-1],
                                                        precision[::-1])
        '''Compute MRR'''
        y_score_rank_true = y_score_rank[np.nonzero(y_true)[0]]
        if len(y_score_rank_true) == 0:
            MRR = 0
        else:
            MRR = np.mean(1 / y_score_rank_true)
        return fpr, tpr, AUC, pAUC, MRR, precision, recall, AUC_PR,pAUC_PR, AUC_bootstrap,AUC_PR_bootstrap,pAUC_bootstrap,pAUC_PR_bootstrap,AUC_bootstrapMedian,AUC_PR_bootstrapMedian,pAUC_bootstrapMedian,pAUC_PR_bootstrapMedian,AUC_bootstrapStd,AUC_PR_bootstrapStd,pAUC_bootstrapStd,pAUC_PR_bootstrapStd


    @staticmethod
    def precision_at_k(r, k):
        """Score is precision @ k
        Relevance is binary (nonzero is relevant).
        >>> r = [0, 0, 1]
        >>> precision_at_k(r, 1)
        0.0
        >>> precision_at_k(r, 2)
        0.0
        >>> precision_at_k(r, 3)
        0.33333333333333331
        >>> precision_at_k(r, 4)
        Traceback (most recent call last):
            File "<stdin>", line 1, in ?
        ValueError: Relevance score length < k
        Args:
            r: Relevance scores (list or numpy) in rank order
                (first element is the first item)
        Returns:
            Precision @ k
        Raises:
            ValueError: len(r) must be >= k
        """
        assert k >= 1
        r = np.asarray(r)[:k] != 0
        if r.size != k:
            raise ValueError('Relevance score length < k')
        return np.mean(r)

    def ComputeAveragePrecision(self, r, k):
        """Score is average precision (area under PR curve)
        Relevance is binary (nonzero is relevant).
        >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
        >>> delta_r = 1. / sum(r)
        >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
        0.7833333333333333
        >>> average_precision(r)
        0.78333333333333333
        Parameters
        ----------
        r: Relevance scores (list or numpy) in rank order
                (first element is the first item)
        Returns:
            Average precision
        """
        assert k >= 1
        if len(r) < k:
            raise ValueError('Relevance score length < k')
        r = r[:k]
        r = np.asarray(r) != 0
        out = [self.precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
        if not out:
            return 0.
        return np.mean(out)

    def ComputeMAPatK(self, rs=None, k=50):
        """Score is mean average precision
        Relevance is binary (nonzero is relevant).
        >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
        >>> mean_average_precision(rs)
        0.78333333333333333
        >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
        >>> mean_average_precision(rs)
        0.39166666666666666
        Args:
            rs: Iterator of relevance scores (list or numpy) in rank order
                (first element is the first item)
        Returns:
            Mean average precision
        """
        if rs is None:
            try:
                rs = []
                for i in range(len(self.y_score)):
                    rs.append([x for y, x in sorted(zip(self.y_score[i], self.y_true[
                              i]), reverse=True)])
            except:
                rs = [[x for y, x in sorted(
                    zip(self.y_score, self.y_true), reverse=True)]]
        return np.mean([self.ComputeAveragePrecision(r, k=k) for r in rs])

    @staticmethod
    def ComputeAveragePrecisionAtK(actual, predicted, k=10):
        """
        Computes the average precision at k.
        This function computes the average prescision at k between two lists of
        items.
        Parameters
        ----------
        actual : list
                 A list of elements that are to be predicted (order doesn't matter)
        predicted : list
                    A list of predicted elements (order does matter)
        k : int, optional
            The maximum number of predicted elements
        Returns
        -------
        score : double
                The average precision at k over the input lists
        """
        if len(predicted) > k:
            predicted = predicted[:k]

        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        if not actual:
            return 0.0

        return score / min(len(actual), k)

    def ComputeMAPatK2(self, actual=None, predicted=None, k=10):
        """
        Computes the mean average precision at k.
        This function computes the mean average prescision at k between two lists
        of lists of items.
        Parameters
        ----------
        actual : list
                 A list of lists of elements that are to be predicted 
                 (order doesn't matter in the lists)
        predicted : list
                    A list of lists of predicted elements
                    (order matters in the lists)
        k : int, optional
            The maximum number of predicted elements
        Returns
        -------
        score : double
                The mean average precision at k over the input lists
        """
        if actual is None:
            try:
                predicted = []
                actual = []
                for i in range(len(self.y_score)):
                    predicted.append([x for y, x in sorted(
                        zip(self.y_score[i], range(len(self.y_score[i]))), reverse=True)])
                    actual.append(list(self.y_true[i].nonzero()[0]))
            except:
                pass
                rs = [[x for y, x in sorted(
                    zip(self.y_score, self.y_true), reverse=True)]]
        return np.mean([self.ComputeAveragePrecisionAtK(a, p, k) for a, p in zip(actual, predicted)])

    def plotScoreHist(self):
        '''
        Plots a histogram of the prediction scores.

        Plots a histogram of the prediction scores made while performing this
        validation experiment.
        '''
        plt.figure(figsize=(8, 8), dpi=100)
        plt.xlabel("Prediction scores", fontsize=30)
        plt.ylabel("Frequency", fontsize=30)
        # plt.title("ROC Curve", fontsize=30)
        plt.hist(self.y_score, bins=np.logspace(-10, 2, 30))
        plt.gca().set_xscale("log")
        plt.tight_layout()
        plt.savefig(self.outputPrefix + "_scoreHist.png")
        plt.close()
        return

    def plot_violinplot(self, metric='AUC'):
        '''
        Plot violin plots

        This method uses the Seaborn package to plot violin plots of 
        AUC per entity data.

        Parameters
        ----------
        array: list or array
            Data to plot.
        keyword: str
            A string specifier which helps identify the kind of plot to generate.
        labels: list
            A set of strings describing the data to be plotted.
        outputPrefix: str
            A string used as part of the filename to save the plot.
        '''
        import seaborn as sns
        if metric == 'AUC':
            data = [x for x in self.AUC if str(x) != 'nan']
        elif metric == 'AUC_PR':
            data = [x for x in self.AUC_PR if str(x) != 'nan']
        array = np.array(data)
        sns.set(font_scale=3)
        sns.set_style("white")
        sns.set_style("ticks")
        # plt.figure(figsize=(20, 10))
        plt.figure(figsize=(7, 10))
        # # dark blue
        # df = pd.DataFrame()
        # df['AUC'] = AUC + AUC + AUC + AUC + AUC + AUC
        # df['Network'] = ['CTD2014_DG'] * len(AUC) + ['CTD2014'] * len(AUC) + ['CTD_MeSH_STRING'] * len(AUC) +['CTD2014_DG'] * len(AUC) + ['CTD2014'] * len(AUC) + ['CTD_MeSH_STRING'] * len(AUC)
        # df['Algorithm'] = ['GID']* len(AUC) + ['GID']* len(AUC) + ['GID']* len(AUC) + ['RW']* len(AUC) + ['RW']* len(AUC) + ['RW']* len(AUC)
        # color_ordered = [u'#4477AA'] * len(labels)
        labels = [metric]
        # green
        color_ordered = [u'#4daf4a'] * len(labels)
        g = sns.violinplot(data=array, palette=color_ordered, inner=None)
        g = sns.swarmplot(data=array, color='white')
        # g = sns.violinplot(x="Algorithm",y='AUC',hue="Network", data=df, palette='muted', inner="stick" )
        # g = sns.boxplot(x="Algorithm",y='AUC',hue="Network", data=df, palette='muted')
        # g = sns.violinplot(x="Algorithm",y='AUC',hue="Network", data=df, palette='muted',inner=None)
        # g = sns.swarmplot(x="Algorithm",y='AUC',hue="Network", data=df, color='white', edgecolor="gray")
        # g = sns.violinplot(x="Algorithm",y='AUC',hue="Network", data=df, palette='muted')
        # x = range(0, len(labels))
        # plt.xticks(x, labels, rotation=90, fontsize=30)
        # plt.yticks(fontsize=30)
        # plt.xticks(fontsize=30)
        g.set_xlabel('')
        # if 'pAUC' in keyword:
        #     g.set_ylabel('Partial AUC (0.1)', fontsize=30)
        #     g.set_ylim([0, 0.1])
        # else:
        # g.set_ylabel('AUC', fontsize=30)
        g.set_ylabel(metric)
        g.set_ylim([0, 1])
        # sns.despine(offset=10, trim=True)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig('{}_{}_violinplot_swarm.pdf'.format(self.outputPrefix, metric),
                    bbox_inches='tight')
        # plt.savefig('{}_{}_violinplot.pdf'.format(outputPrefix, keyword),
        #             bbox_inches='tight')
        plt.close()
        return

    def plotAUChist(self):
        '''
        Plots a histogram of AUCs

        Plots a histogram of the area under the receiver operating curves
        generated during this experiment.
        '''
        plt.figure(figsize=(8, 8), dpi=100)
        plt.xlabel("AUC", fontsize=30)
        plt.ylabel("Frequency", fontsize=30)
        # plt.title("ROC Curve", fontsize=30)
        AUC = [x for x in self.AUC if str(x) != 'nan']
        plt.hist(AUC)
        # plt.tight_layout()
        plt.savefig(self.outputPrefix + "_AUChist.png", bbox_inches='tight')
        plt.close()
        return

    def plotBoxplot(self, metric, xticklabels):
        '''
        Plot a box and whisker plot of metrics.
        '''
        plt.switch_backend('Agg')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        data = self.__dict__[metric]
        ax.boxplot(data)
        ax.set_xticklabels(xticklabels, fontsize=20)
        # ax.set_xlabel('# of callers', fontsize=20)
        ax.set_ylabel(metric, fontsize=20)
        # ax.set_ylim([0,0.04])
        if 'bootstrap' in metric:
            plt.savefig('{}_{}_i{}_boxplot.png'.format(
                self.outputPrefix, metric, self.AUC_bootstrapN), bbox_inches='tight')
        else:
            plt.savefig('{}_{}_boxplot.png'.format(
                self.outputPrefix, metric), bbox_inches='tight')            
        plt.close()
        return

    def plotAUCbootstrap(self):
        '''
        Plot a box and whisker plot of the AUCs from bootstrapping.

        Plots a box and whisker plot of the area under the receiver operating
        curves generated while performing bootstrapping over the prediction
        results.
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.boxplot(self.AUC_bootstrap)
        ax.set_xticklabels(self.network.getName(), fontsize=20)
        # ax.set_xlabel('# of callers', fontsize=20)
        ax.set_ylabel('AUC', fontsize=20)
        # ax.set_ylim([0,0.04])
        plt.savefig('{}_AUC_i{}_boxplot.png'.format(
            self.outputPrefix, self.AUC_bootstrapN), bbox_inches='tight')
        plt.close()
        return

    def plotpAUCbootstrap(self):
        '''
        Plot box and whisker plot of pAUCs from bootstrapping

        Plots a box and whisker plot of partial area under the receiver
        operating curves from bootstrapping experiment performed by this
        instance of the ValidationParent or child class.
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.boxplot(self.pAUC_bootstrap)
        ax.set_xticklabels(self.network.getName(), fontsize=20)
        # ax.set_xlabel('# of callers', fontsize=20)
        ax.set_ylabel(
            'Partial AUC ({})'.format(self.pAUC_thrshold), fontsize=20)
        # ax.set_ylim([0,0.04])
        plt.savefig('{}_pAUC_i{}_boxplot.png'.format(self.outputPrefix,
                                                     self.AUC_bootstrapN),
                    bbox_inches='tight')
        plt.close()
        return

    def plotAUCPRhist(self):
        '''
        Plot AUC Histogram

        Plots a histogram of the area under the receiver operating curve for the
        experiment performed by this instance of the ValidationParent or child 
        class.
        '''
        plt.figure(figsize=(8, 8), dpi=100)
        plt.xlabel("AUPRC", fontsize=30)
        plt.ylabel("Frequency", fontsize=30)
        # plt.title("ROC Curve", fontsize=30)
        plt.hist(self.AUC_PR)
        plt.tight_layout()
        plt.savefig(self.outputPrefix + "_AUCPRhist.png")
        plt.close()
        return

    def plotROC(self, randomline=True):
        '''
        Plot Receiver Operating Curve

        Plots the receiver operating curve for the experiment performed by this
        instance of the ValidationParent or child class.
        '''
        if self.selectedEntityList is None:
            # Plot ROC curve
            plt.figure(figsize=(8, 8), dpi=100)
            plt.xlabel("FPR", fontsize=30)
            plt.ylabel("TPR", fontsize=30)
            # plt.title("ROC Curve", fontsize=30)
            plt.plot(self.fpr, self.tpr, linewidth=2,
                     label='Prediction #= {0}, AUC = {1:.3f}'.format(
                         self.predictionsN, self.AUC))
            if randomline:
                x = [0.0, 1.0]
                # , label='random')
                plt.plot(x, x, linestyle='dashed', color='red', linewidth=2)
            plt.xlim(0.0, 1.0)
            plt.ylim(0.0, 1.0)
            plt.legend(fontsize=16, loc='best')
            plt.tight_layout()
            plt.savefig(self.outputPrefix + "_ROC.png")
        else:
            for i in range(len(self.y_node1)):
                # Plot ROC curve
                plt.figure(figsize=(8, 8), dpi=100)
                plt.xlabel("FPR", fontsize=30)
                plt.ylabel("TPR", fontsize=30)
                # plt.title("ROC Curve", fontsize=30)
                plt.plot(self.fpr[i], self.tpr[i], linewidth=2,
                         label='Prediction #= {0}, AUC = {1:.3f}'.format(
                    self.predictionsN[i], self.AUC[i]))
                if randomline:
                    x = [0.0, 1.0]
                    # , label='random')
                    plt.plot(
                        x, x, linestyle='dashed', color='red', linewidth=2)
                plt.xlim(0.0, 1.0)
                plt.ylim(0.0, 1.0)
                plt.legend(fontsize=16, loc='best')
                plt.tight_layout()
                if not os.path.exists(self.y_node1[i]):
                    os.mkdir(self.y_node1[i])
                plt.savefig(self.y_node1[i] + '/' + self.outputPrefix + "_" +
                            self.y_node1[i] + "_ROC.png")
        plt.close()
        return

    def plotPRcurve(self):
        '''
        Plot precision recall curve

        Plots the precision recall curve for the experiment performed by this
        instance of the ValidationParent or child class.
        '''
        if self.selectedEntityList is None:
            # Plot precision recall curve
            plt.figure(figsize=(8, 8), dpi=100)
            plt.xlabel("Recall (TPR)", fontsize=30)
            plt.ylabel("Precision", fontsize=30)
            # plt.title("ROC Curve", fontsize=30)
            plt.plot(self.recall, self.precision, linewidth=2,
                     label='Prediction #= {0}, AUC = {1:.3f}'.format(
                         self.predictionsN, self.AUC_PR))
            plt.xlim(0.0, 1.0)
            plt.ylim(0.0, 1.0)
            plt.legend(fontsize=16, loc='best')
            plt.tight_layout()
            plt.savefig(self.outputPrefix + "_PRcurve.png")
        else:
            for i in range(len(self.y_node1)):
                # Plot precision recall curve
                plt.figure(figsize=(8, 8), dpi=100)
                plt.xlabel("Recall (TPR)", fontsize=30)
                plt.ylabel("Precision", fontsize=30)
                # plt.title("ROC Curve", fontsize=30)
                plt.plot(self.recall[i], self.precision[i], linewidth=2,
                         label='Prediction #= {0}, AUC = {1:.3f}'.format(
                    self.predictionsN[i], self.AUC_PR[i]))
                plt.xlim(0.0, 1.0)
                plt.ylim(0.0, 1.0)
                plt.legend(fontsize=16, loc='best')
                plt.tight_layout()
                if not os.path.exists(self.y_node1[i]):
                    os.mkdir(self.y_node1[i])
                plt.savefig(self.y_node1[i] + '/' + self.outputPrefix + "_" +
                            self.y_node1[i] + "_PRcurve.png")
        plt.close()
        return

    def computeF1(self, thresholdRecall=0.1):
        '''
        Compute F1 score

        This method computes the F1 score for the validation experiment given
        the results stored in its state variables.

        Parameters
        thresholdRecall: float
            The threshold at which to stop when computing the F1 score, set by
            default to 0.1.

        F1 = 2*(precision*recall)/(precision+recall)
        '''
        if self.selectedEntityList is None:
            p = self.precision
            r = self.recall
            for i in range(len(p)):
                if r[i] <= thresholdRecall:
                    if(abs(r[i] - thresholdRecall) <
                       abs(thresholdRecall - r[i - 1])):
                        self.F1 = 2 * (p[i] * r[i]) / (p[i] + r[i])
                    else:
                        self.F1 = 2 * \
                            (p[i - 1] * r[i - 1]) / (p[i - 1] + r[i - 1])
                    break
        else:
            self.F1 = []
            for j in range(len(self.selectedEntityList)):
                p = self.precision[j]
                r = self.recall[j]
                for i in range(len(p)):
                    if r[i] <= thresholdRecall:
                        if(abs(r[i] - thresholdRecall) <
                           abs(thresholdRecall - r[i - 1])):
                            self.F1.append(2 * (p[i] * r[i]) / (p[i] + r[i]))
                        else:
                            self.F1.append(2 * (p[i - 1] * r[i - 1]) /
                                           (p[i - 1] + r[i - 1]))
                        break
        return
