#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Aug 25, 2016

@author: Chih-Hsu Lin and Daniel Konecki
'''
import os
import sys
from time import time
from IPython import embed
from ValidationParent import ValidationParent
import itertools

class leaveAModeOut(ValidationParent):
    '''
    Take the network modes to attempt leaving out.
    '''

    def __init__(self, algorithm, network, mode, answerNetwork=None,
                 selectedQueryMode=None, selectedPredictionMode=None, bridgeMode=None,
                 **kwargs):
        '''
        Constructor

        This function generates an instance of the leaveAModeOut class.

        Parameters
        ----------
        algorithm: child class of AlgorithmParent
            The algorithm to be performed during the leave a mode out experiment
            currently expects classes: GraphBasedDiffusion, MF,
            and networkLassDiff.
        network: GraphRepresentation5
            An object representing the network to start with when performing
            this experiment.
        mode: str
            A string with the format '{}-{}' where {} is a node type e.g.
            (Chemical, Disease, Gene).  This parameter specifies which edges
            will be dropped from the network and predicted back during the
            experiment.
        answerNetwork: GraphRepresentation5
            A graph object containing the edges (and supporting nodes) which are
            in the mode left out in the experiment.  That is those edges you
            wish to predict back by performing the algorithm. Default value is
            None, in which case the correct result graph will be generated using
            the starting network.
        selectedQueryMode: str
            The node type from which to diffuse labels in this experiment
            (i.e. Chemical, Disease, Gene). The default value for this parameter
            is None which means predictions will be made starting from all node
            types.
        selectedPredictionMode: str
            The node type to which to diffuse labels in this experiment
            (i.e. Chemical, Disease, Gene). The default value for this parameter
            is None which means predictions will be made for all node types.
        Returns
        -------
        leaveAModeOut
            An instance of this class.
        '''
        ValidationParent.__init__(self, method='leaveAModeOut',
                                  algorithm=algorithm, selectedQueryMode=selectedQueryMode,
                                  selectedPredictionMode=selectedPredictionMode,
                                  bridgeMode=bridgeMode,
                                  network=network, split=mode,
                                  answerNetwork=answerNetwork,
                                  param=kwargs)
        # print self.network.getName()
        if '_Mode_{}_Retained_Edges'.format(mode) in self.network.getName():
            self.network.name = self.network.name.replace(
                '_Mode_{}_Retained_Edges'.format(mode), '')
        # print self.network.getName()
        self.validationNetworkNames = [self.network.getName() + '_Mode_{}_Retained_Edges'.format(mode),
                                       self.network.getName() + '_Mode_{}_Deleted_Edges'.format(mode)]
        # print self.validationNetworkNames
        self.name = 'LOMO_' + mode

    def __del__(self):
        '''
        Destructor

        This method deletes all data associated with an instance of the
        leaveAModeOut class.
        '''
        ValidationParent.__del__(self)

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
                modeTypes = self.split.split('-')
                try:
                    type1 = modeTypes[0]
                    type2 = modeTypes[1]
                except:
                    print modeTypes
                excludedMode = self.network.excludeEdgeType(type1=type1,
                                                            type2=type2)
                self.inputNetwork = self.network
                self.inputNetwork.deleteNodes('Degree', 0, save=False)
                # Get all edges not in the specified mode
                _nodeOld, nodeInterection, _nodeNew = self.inputNetwork.determineOverlappingNodes(
                    excludedMode)
                _edgeOld, _edgeInterection, edgeNew = self.inputNetwork.determineOverlappingEdges(
                    excludedMode)
                # Get edges only found in new network of intersected nodes as
                # answer network
                self.answerNetwork = excludedMode.generateSubNetworkFromEdgesAndNodes(
                    self.validationNetworkNames[1], edgeNew, nodeInterection)
                self.saveValidationNetwork()
        elif self.inputNetwork is None:
            # Answer network was specified. Only need to generate input network
            if not self.loadValidationNetwork():
                print 'Use answer network given in input arguments without generating again.'
                print 'Generating input network'
                modeTypes = self.split.split('-')
                type1 = modeTypes[0]
                type2 = modeTypes[1]
                self.network.excludeEdgeType(type1=type1,
                                             type2=type2)
                # Get edges excluding specified mode as input
                self.inputNetwork = self.network
                self.inputNetwork.deleteNodes('Degree', 0, save=False)
                self.saveValidationNetwork()
        elif self.answerNetwork is None:
            raise NotImplementedError(
                'generateValidationNetwork in timeStamped.py should be implemented!')
        else:
            print 'Both input network and answer network are given.'
        return

    def perform(self):
        '''
        Perform a leave a mode out experiment.

        This method performs the specified algorithm, trying to predict a
        specific type of edge from all other information in the network except
        for that edge type.
        '''
        # Perform validation method
        # print('Generating network without edges from ' + self.split + ' mode.')
        self.generateValidationNetwork()
        if 'Diffusion' in str(self.algorithm):
            Alg = self.algorithm(self.inputNetwork, self.method, self.selectedQueryMode)
        elif 'AptRank' in str(self.algorithm):
            Alg = self.algorithm(self.inputNetwork, self.method,
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
            #       ' mode ' + self.split)
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
            elif 'AptRank' in Alg.getName():
                predict_rows = self.inputNetwork.getModePosition(
                    self.selectedQueryMode)
                predict_cols = self.inputNetwork.getModePosition(
                    self.selectedPredictionMode)
                allModes = sorted(set(self.inputNetwork.mappedNodes.keys()).union(
                    set(self.inputNetwork.unmappedNodes.keys())))
                trainIdx = []
                for x, y in itertools.combinations(allModes, 2):
                    if (x == self.selectedQueryMode and y == self.selectedPredictionMode) or (y == self.selectedQueryMode and x == self.selectedPredictionMode):
                        pass
                    else:
                        trainIdx.append((self.inputNetwork.getModePosition(
                            x), self.inputNetwork.getModePosition(y)))
                predIdx = [(predict_rows,predict_cols)]
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
        self.printPerformance()
        os.chdir(self.dirResult)
        # Plot and save ROC and PR curve
        if self.splitPerformanceByEntity is True:
            self.plotAUChist()
            self.plotAUCPRhist()
        else:
            self.plotROC()
            self.plotPRcurve()
            self.plotAUCbootstrap()
            self.plotpAUCbootstrap()
        self.savePerformance()
        self.savePredictions()
        os.chdir('..')
        end = time()
        # print('Predictions took {} min to validate'.format((end - valid) / 60))
