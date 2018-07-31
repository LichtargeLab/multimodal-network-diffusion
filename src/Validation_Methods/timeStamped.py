#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Aug 25, 2016

@author: Chih-Hsu Lin
'''
import os
import sys
from time import time
import paths
from ValidationParent import ValidationParent
from IPython import embed


class timeStamped(ValidationParent):
    '''
    Take the year to split the network into old and new networks and conduct time-stamped experiment
    '''

    def __init__(self, algorithm, network, split, selectedEntityList=None,
                 selectedQueryMode=None, selectedPredictionMode=None,
                 splitPerformanceByEntity=False, answerNetwork=None, bridgeMode=None,
                 **kwargs):
        '''
        Constructor

        This function generates an instance of the timeStamped class.

        Parameters
        ----------
        algorithm: child class of AlgorithmParent
            The algorithm to be performed during the leave a mode out experiment
            currently expects classes: GraphBasedDiffusion, MF,
            and networkLassDiff.
        network: GraphRepresentation5
            An object representing the network to start with when performing
            this experiment.
        split: str
            The year by which to split the networks.
        selectedEntityList: list
            A set of entities of interest to compute results for.
        selectedQueryMode: str
            The node type from which to diffuse labels in this experiment
            (i.e. Chemical, Disease, Gene). The default value for this parameter
            is None which means predictions will be made starting from all node
            types.
        selectedPredictionMode: str
            The node type to which to diffuse labels in this experiment
            (i.e. Chemical, Disease, Gene). The default value for this parameter
            is None which means predictions will be made for all node types.
        splitPerformanceByEntity: str
            An entity type to split results on.
        answerNetwork: GraphRepresentation5
            A graph object containing the edges (and supporting nodes) which are
            in the mode left out in the experiment.  That is those edges you
            wish to predict back by performing the algorithm. Default value is
            None, in which case the correct result graph will be generated using
            the starting network.

        Returns
        -------
        timeStamped
            An instance of this class.
        '''
        method = 'timeStamped'
        ValidationParent.__init__(
            self, method=method, algorithm=algorithm, network=network,
            split=split, selectedEntityList=selectedEntityList,
            selectedQueryMode=selectedQueryMode,
            selectedPredictionMode=selectedPredictionMode,
            bridgeMode=bridgeMode,
            splitPerformanceByEntity=splitPerformanceByEntity,
            answerNetwork=answerNetwork, param=kwargs)
        try:
            self.validationNetworkNames = [
                '{}_Year_lt_{}_Deleted_Edges_qMode{}_pMode{}'.format(
                    self.network.getName(),
                    split,
                    selectedQueryMode, selectedPredictionMode),
                '{}_Edge_Year_gte_{}_qMode{}_pMode{}'.format(
                    self.network.getName(),
                    split,
                    selectedQueryMode, selectedPredictionMode)]
        except:
            self.validationNetworkNames = [
                '{}_Year_lt_{}_Deleted_Edges'.format(
                    self.network.getName(), split),
                '{}_Edge_Year_gte_{}'.format(
                    self.network.getName(), split)]
        self.name = method + '_Year_' + str(split)

    def __str__(self):
        '''
        To String

        This method generates an informative string to describe the current
        timeStamped class experiment.

        Returns
        -------
        str
            A string detailing the name, node count, and edge count of the
            current instance.
        '''
        objStr = 'Validation name: {}\n'.format(self.getName())
        objStr += 'Algorithm name: {}\n'.format(self.algorithm.getName())
        objStr += 'Network name: {}'.format(self.network.getName())
        objStr += 'Year to split: {}'.format(self.split)
        return objStr

    def perform(self):
        '''
        Perform a leave a mode out experiment.

        This method performs the specified algorithm, trying to predict a
        specific type of edge from all other information in the network except
        for that edge type.

        Returns
        -------
        float
            The fpr of the experiment.
        float
            The tpr of the epxeriment.
        float
            The AUC of the experiment.
        float
            The partial AUC of the experiment (i.e. the AUC of the first 10% of
            the curve).
        list
            A list of AUCs created by performing 100 bootstrapped experiments.
        list
            A list of pAUCs created by performing 100 bootstrapped experiments.
        float
            The median AUC value the performed bootstrapped experiments.
        float
            The median pAUC value the performed bootstrapped experiments.
        float
            The recall score of the experiment.
        float
            The precision score of the experiment.
        float
            The AUC PR of this experiment.
        float
            The partial AUC PR of this experiment.
        list
            A list of AUC PRs created by performing 100 bootstrapped
            experiments.
        list
            A list of pAUC PRs created by performing 100 bootstrapped
            experiments.
        float
            The median value observed when creating AUC PR curves over a
            bootstrapped set of experiments.
        float
            The median value observed when creating partial AUC PR curves over
            a bootstrapped set of experiments.
        float
            The F1 score of the experiment.
        int
            The overall number of predictions made.
        int
            The number of predictions in the true positive set.
        '''
        # Perform validation method
        # print('Generating old edges till year ' + str(self.split - 1) +
        #       ' and gained edges since year ' + str(self.split))
        self.generateValidationNetwork()
        # Perform algorithm using old network
        if 'Diffusion' in str(self.algorithm):
            Alg = self.algorithm(self.inputNetwork, self.name, self.selectedQueryMode)
        elif 'AptRank' in str(self.algorithm):
            Alg = self.algorithm(self.inputNetwork, self.name,
                                 self.selectedQueryMode, self.selectedPredictionMode, param=self.param)
        else:
            raise NotImplementedError('{} has not been implemented yet'.format(self.algorithm))
        self.generateOutputPrefix(Alg)
        valid = time()
        if not os.path.exists(self.dirResult):
            os.mkdir(self.dirResult)
        self.dirResult = self.dirResult + self.outputPrefix + '/'
        if not os.path.exists(self.dirResult):
            os.mkdir(self.dirResult)
        os.chdir(self.dirResult)
        if not self.loadPerformance():
            # print('Running ' + Alg.getName() + ' on ' + self.network.getName() +
            #       ' year ' + str(self.split))
            start = time()
            if((Alg.getName() == 'Diffusion') or (Alg.getName() == 'RndWalk')):
                nodes = self.inputNetwork.getNodes().keys()
                if(hasattr(self.selectedQueryMode, '__iter__')):
                    selectedLabels = set()
                    for x in self.selectedQueryMode:
                        mappedNodes, unmappedNodes = self.answerNetwork.getNodesOfType(
                            x)
                        selectedLabels |= set(mappedNodes + unmappedNodes)
                    labeledNodes = list(
                        selectedLabels.intersection(set(nodes)))
                    Alg.perform(labeledNodes=labeledNodes)
                else:
                    mappedNodes, unmappedNodes = self.answerNetwork.getNodesOfType(
                        self.selectedQueryMode)
                    labeledNodes = list(
                        set(mappedNodes + unmappedNodes).intersection(set(nodes)))
                    Alg.perform(labeledNodes=labeledNodes)
            elif Alg.getName() == 'Proximity':
                try:
                    Alg.perform(bridgeMode=self.bridgeMode, startMode=self.selectedQueryMode,
                                endMode=self.selectedPredictionMode)
                except:
                    raise ValueError(
                        'selectedQueryMode and selectedPredictionMode are required for {} method'.format(Alg.getName()))
            elif 'AptRank' in Alg.getName():
                predict_rows = self.inputNetwork.getModePosition(
                    self.selectedQueryMode)
                predict_cols = self.inputNetwork.getModePosition(
                    self.selectedPredictionMode)
                trainIdx = [(predict_rows, predict_cols)]
                predIdx = [(predict_rows, predict_cols)]
                Alg.perform(trainIdx, predIdx)
            else:
                Alg.perform()
            pred = time()
            # print('Predictions took {} min to make'.format(
            #     (pred - start) / 60))
            self.predictions = Alg.getresult()
            valid = time()
            # print('Predictions took {} min to retrieve'.format(
            #     (valid - pred) / 60))
            # Compute performance metrics
            # print('Validating predictions')
            self.generatePredicitonAnswerArray()
            self.computeFprTprPrecisionMRR()
            # self.computeF1()
        # Switch to the result directory
        os.chdir(self.dirResult)
        self.savePerformance()
        self.savePredictions()
        self.printPerformance()
        # Plot and save ROC and PR curve
        if self.splitPerformanceByEntity is True:
            # self.plotAUChist()
            # self.plotAUCPRhist()
            self.plot_violinplot('AUC')
            self.plot_violinplot('AUC_PR')
        else:
            self.plotROC()
            self.plotPRcurve()
            self.plotAUCbootstrap()
            self.plotpAUCbootstrap()
        os.chdir('..')
        end = time()
        return

    def generateValidationNetwork(self):
        '''
        Creates input and answer network for validation.

        This method creates the starting graph and the target graph from the
        provided network.
        '''
        # Use answer matrix when it is given
        os.chdir(self.dirNetworks)
        if ((self.answerNetwork is None) and (self.inputNetwork is None)):
            # Generate the network without a mode and the mode by itself
            # Try to reload saved oldNetwork and gainedEdges first
            if not self.loadValidationNetwork():
                # Get all edges greater and equal to specified year (new edges)
                # self.network would be replaced by the all edges less than
                # specified year (old network) as input network
                excludedMode = self.network.getSubNetWork(parameterName='Year',
                                                          parameterValue=self.split,
                                                          mode='gte')
                self.inputNetwork = self.network
                self.inputNetwork.deleteNodes('Degree', 0, save=False)
                # Get the intersected nodes for experiment
                _nodeOld, nodeInterection, _nodeNew = self.inputNetwork.determineOverlappingNodes(
                    excludedMode)
                # Get the gained edges for experiment
                _edgeOld, _edgeInterection, edgeNew = self.inputNetwork.determineOverlappingEdges(
                    excludedMode)
                # Get edges only found in new edges of intersected nodes as
                # answer network
                self.answerNetwork = excludedMode.generateSubNetworkFromEdgesAndNodes(
                    self.validationNetworkNames[1], edgeNew, nodeInterection)
                self.saveValidationNetwork()
        elif self.inputNetwork is None:
            # Answer network was specified. Only need to generate input network
            if not self.loadValidationNetwork():
                # print 'Use answer network given in input arguments without generating again.'
                # print 'Use input network as old network'
                # Get edges before certain year (including self.split) as input
                self.inputNetwork = self.network
                self.inputNetwork.deleteNodes('Degree', 0, save=False)
                self.saveValidationNetwork()
        elif self.answerNetwork is None:
            raise NotImplementedError(
                'generateValidationNetwork in timeStamped.py should be implemented!')
        # else:
        #     print 'Both input network and answer network are given.'
        return
