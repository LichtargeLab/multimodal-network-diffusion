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
from GraphRepresentation5 import GraphRepresentation5
from ValidationParent import ValidationParent
import cPickle as pickle
from scipy.sparse import csr_matrix
import numpy as np

class kFoldCV(ValidationParent):
    '''
    Take the network, split into k folds and validate the prediction based on k-1 fold by the one fold leaving out.
    '''

    def __init__(self, algorithm, network, k, answerNetwork=None, selectedQueryMode=None, selectedPredictionMode=None, **kwargs):
        '''
        Constructor

        Parameters
        ----------
        k: integer
            A number specifying the fold number to split

        Returns
        -------
        kFoldCV
            An instance of this class.
        '''
        ValidationParent.__init__(self, method='kFoldCV',
                                  algorithm=algorithm, selectedQueryMode=selectedQueryMode,
                                  selectedPredictionMode=selectedPredictionMode,
                                  network=network, split=k,
                                  answerNetwork=answerNetwork, param=kwargs)
        self.validationNetworkNames = [
            [self.network.getName() + '_{}fold_{}kept'.format(self.split, i + 1),
             self.network.getName() + '_{}fold_{}removed'.format(self.split, i + 1)]
            for i in range(self.split)]
        self.validationNetworkNames += [
            [self.network.getName() + '_{}fold_input'.format(self.split),
             self.network.getName() + '_{}fold_answer'.format(self.split)]
        ]
        self.name = '{}FoldCV'.format(k)
        # Store a dictionary mapping a split number to a matrix mask where all edges
        # to consider are marked with a 1 and all other edges are 0. This
        # tracks not only existing edges, but all possible edges.
        self.splitDict = None
        # Save the initial network name to reload the network later
        self.initialNetworkName = self.network.getName()

    def __del__(self):
        ValidationParent.__del__(self)

    def generateValidationNetwork(self, validationNetworkNames):
        # Use answer matrix when it is given
        os.chdir(self.dirNetworks)
        if ((self.answerNetwork is None) and (self.inputNetwork is None)):
            # Generate the input network and answer network
            # Try to reload saved networkds first
            if not self.loadValidationNetwork(validationNetworkNames):
                # Start to generate k sets of input and answer networks
                # Get k sets of masks based on the graph
                # splitDict = self.network.generateKFoldSplits(self.split)
                # Start from the end so that the final set would be the first
                # fold to be removed
                for i in range(self.split)[::-1]:
                    # Deleted edges in mask as input network
                    # Returned edges are edges in the mask which is the answer
                    # network
                    self.network.loadGraph(self.initialNetworkName)
                    self.answerNetwork = self.network.deleteEdges('Mask',
                                                                  self.splitDict[
                                                                      i],
                                                                  name1=self.validationNetworkNames[
                                                                      i][1],
                                                                  name2=self.validationNetworkNames[i][0])
                    self.answerNetwork.deleteNodes(
                        attr='Degree', value=0, save=False)
                    self.answerNetwork.saveGraph(
                        self.validationNetworkNames[i][1] + '.pkl')
                    self.inputNetwork = self.network
                    self.inputNetwork.deleteNodes(
                        attr='Degree', value=0, save=False)
                    self.inputNetwork.saveGraph(
                        self.validationNetworkNames[i][0] + '.pkl')
        else:
            raise NotImplementedError(
                'K-fold CV generateValidationNetwork has not been implemented completely!')
        return

    def saveSplitDict(self):
        '''
        Save k fold split 

        This method saves the the split dictionary made in this validation experiment.
        '''
        # Save performance metrics
        pFile = open(self.outputPrefix + '_split.pkl', 'wb')
        pickle.dump(self.splitDict, pFile, protocol=pickle.HIGHEST_PROTOCOL)
        pFile.close()
        return

    def loadSplitDict(self):
        '''
        Load k fold split 

        Loads the split dictionary made in this validation experiment.
        '''
        # Loading split dict requires running experiment for GraphBasedDiffusion first.
        # Running other algorithms would try to load split dict from
        # GraphBasedDiffusion directory
        if('GraphBasedDiffusion' in str(self.algorithm)):
            os.chdir(self.dirResult2)
            pFN = self.outputPrefix + '_split.pkl'
        else:
            correspondingDiffusionFolder = self.dirResult2
            correspondingDiffusionFile = self.outputPrefix
            otherAlgorithms = ['RndWalk', 'NMF', 'AptRank1',
                               'AptRank2', 'AptRank3', 'AptRank4']
            for x in otherAlgorithms:
                correspondingDiffusionFolder = correspondingDiffusionFolder.replace(
                    x, 'Diffusion')
                correspondingDiffusionFile = correspondingDiffusionFile.replace(
                    x, 'Diffusion')
            try:
                os.chdir(correspondingDiffusionFolder)
            except:
                raise ValueError(
                    '{} should load the corresponding Diffusion split\nNo corresponding Diffusion experiment was found'.format(str(self.algorithm)))
            pFN = correspondingDiffusionFile + '_split.pkl'

        if(os.path.exists(pFN)):
            # print('Reloading pre-existing saved split dictionary')
            pFile = open(pFN, 'rb')
            self.splitDict = pickle.load(pFile)
            pFile.close()
            os.chdir('..')
            return True
        else:
            if('GraphBasedDiffusion' in str(self.algorithm)):
                return False
            else:
                raise ValueError(
                    '{} should load the corresponding Diffusion split'.format(str(self.algorithm)))
                sys.exit(0)

    def generateSplits(self):
        if not self.loadSplitDict():
            self.splitDict = self.network.generateKFoldSplits(self.split)
            self.saveSplitDict()
            os.chdir('..')
        return

    def generateMappedIdx(self, initialPositionNode, maskedIdx):
        workingNodePosition = self.inputNetwork.getNodePosition()
        row = list(maskedIdx[0])
        col = list(maskedIdx[1])
        tmpListCmp = [[x, y, workingNodePosition[initialPositionNode[x]], workingNodePosition[initialPositionNode[y]]] for x, y in zip(
            row, col) if initialPositionNode[x] in workingNodePosition and initialPositionNode[y] in workingNodePosition]
        tmpListCmp = np.array(tmpListCmp).T
        selectedMaskedIdx = (np.append(tmpListCmp[0], tmpListCmp[1]), np.append(
            tmpListCmp[1], tmpListCmp[0]))
        mappedSelectedMaskedIdx = (np.append(
            tmpListCmp[2], tmpListCmp[3]), np.append(tmpListCmp[3], tmpListCmp[2]))
        return selectedMaskedIdx, mappedSelectedMaskedIdx

    def perform(self):
        # Perform validation method
        # print ('Generating network split by {} fold').format(self.split)
        os.chdir(self.dirNetworks)
        if 'Diffusion' in str(self.algorithm):
            Alg = self.algorithm(self.network, self.method, self.selectedQueryMode)
        elif 'AptRank' in str(self.algorithm):
            Alg = self.algorithm(self.network, self.method,
                                 self.selectedQueryMode, self.selectedPredictionMode, param=self.param)
        else:
            raise NotImplementedError('{} has not been implemented yet'.format(self.algorithm))
        self.generateOutputPrefix(Alg, networkName=self.initialNetworkName)
        initialOutputPrefix = self.outputPrefix
        self.dirResult2 = self.dirResult + self.outputPrefix + '/'
        if not os.path.exists(self.dirResult2):
            os.mkdir(self.dirResult2)
        os.chdir(self.dirResult2)
        valid = time()
        if not self.loadPerformance():
            self.generateSplits()
            self.predictions = csr_matrix(np.zeros([self.network.getNodeCount(),
                                                    self.network.getNodeCount()]))
            os.chdir(self.dirNetworks)
            initialNetwork = GraphRepresentation5()
            initialNetwork.loadGraph(self.initialNetworkName)
            initialNodePosition = initialNetwork.getNodePosition()
            initialPositionNode = initialNetwork.getPositionNode()
            for i in range(self.split):
                k = i + 1
                self.answerNetwork = None
                self.inputNetwork = None
                self.generateValidationNetwork(self.validationNetworkNames[i])
                # Perform algorithm using k-1 fold network
                if 'Diffusion' in str(self.algorithm):
                    Alg = self.algorithm(self.inputNetwork, self.name + '_{}'.format(k), self.selectedQueryMode)
                elif 'AptRank' in str(self.algorithm):
                    Alg = self.algorithm(self.inputNetwork, self.name + '_{}'.format(k),
                                         self.selectedQueryMode, self.selectedPredictionMode, param=self.param)
                else:
                    raise NotImplementedError('{} has not been implemented yet'.format(self.algorithm))
                self.generateOutputPrefix(
                    Alg, networkName=self.initialNetworkName)
                valid = time()
                # print 'Running ' + Alg.getName() + ' on ' +
                # self.initialNetworkName + ' of ' + str(k) + 'th fold'
                start = time()
                if((Alg.getName() == 'Diffusion') or (Alg.getName() == 'RndWalk')):
                    if(hasattr(self.selectedQueryMode, '__iter__')):
                        selectedLabels = set()
                        for x in self.selectedQueryMode:
                            mappedNodes, unmappedNodes = self.inputNetwork.getNodesOfType(
                                x)
                            selectedLabels |= set(mappedNodes + unmappedNodes)
                        Alg.perform(
                            labeledNodes=list(selectedLabels))
                    elif self.selectedQueryMode is not None:
                        mappedNodes, unmappedNodes = self.inputNetwork.getNodesOfType(
                            self.selectedQueryMode)
                        Alg.perform(
                            labeledNodes=mappedNodes + unmappedNodes)
                    else:
                        inputOnly, interactedNodes, answerOnly = self.inputNetwork.determineOverlappingNodes(
                            self.inputNetwork)
                        Alg.perform(
                            labeledNodes=interactedNodes)
                elif 'AptRank' in Alg.getName():
                    if(hasattr(self.selectedQueryMode, '__iter__') and hasattr(self.selectedPredictionMode, '__iter__')):
                        trainIdx = []
                        for x, y in zip(self.selectedQueryMode, self.selectedPredictionMode):
                            predict_rows = self.inputNetwork.getModePosition(x)
                            predict_cols = self.inputNetwork.getModePosition(y)
                            trainIdx.append((predict_rows, predict_cols))
                    elif(hasattr(self.selectedQueryMode, '__iter__') or hasattr(self.selectedPredictionMode, '__iter__')):
                        raise ValueError(
                            'Both selectedQueryMode and selectedPredictionMode should be iterable.')
                    else:
                        predict_rows = self.inputNetwork.getModePosition(
                            self.selectedQueryMode)
                        predict_cols = self.inputNetwork.getModePosition(
                            self.selectedPredictionMode)
                        trainIdx = [(predict_rows, predict_cols)]
                    predIdx = [((0, self.inputNetwork.nodeCount - 1),
                                (0, self.inputNetwork.nodeCount - 1))]
                    Alg.perform(trainIdx, predIdx)
                else:
                    Alg.perform()
                pred = time()
                # print(
                #     'Predictions took {} min to make'.format((pred - start) / 60))
                mask = self.splitDict[i]
                maskedIdx = mask.nonzero()
                # Ignore those edges which nodes do not exist in working
                # predictions
                # Create mapped masked idx between final prediction and working
                # prediction of each fold
                selectedMaskedIdx, mappedSelectedMaskedIdx = self.generateMappedIdx(
                    initialPositionNode, maskedIdx)
                self.predictions[selectedMaskedIdx] = Alg.getresult()[
                    mappedSelectedMaskedIdx]
                valid = time()
                # print(
                #     'Predictions took {} min to retrieve'.format((valid - pred) / 60))
            # Compute performance metrics
            # print 'Validating predictions'
            os.chdir(self.dirNetworks)
            self.outputPrefix = initialOutputPrefix
            self.answerNetwork.loadGraph(self.initialNetworkName)
            self.inputNetwork.loadGraph(self.initialNetworkName)
            self.generatePredicitonAnswerArray()
            self.computeFprTprPrecisionMRR()
            # self.computeF1()
        os.chdir(self.dirResult2)
        # Plot and save ROC and PR curve
        self.plotROC()
        self.plotPRcurve()
        self.printPerformance()
        self.savePerformance()
        os.chdir('..')
        end = time()
        # print('Predictions took {} min to validate'.format((end - valid) / 60))
