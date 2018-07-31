#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Aug 26, 2016

@author: Chih-Hsu Lin

This script is for running leave-one-mode-out (LOMO) experiments across different networks

Example:
    This example runs a prediction of disease-gene associations.
        $ python LeaveAModeOut.py --e DG

Arguments:
    --e: Could be one of {'DG','DC','GC'}; DG: disease-gene; DC: disease-chemical; GC: gene-chemical

"""
import argparse
import sys
import os
import paths
from ValidationParent import ValidationParent
from GraphRepresentation5 import GraphRepresentation5
from AptRank import AptRank
from GraphBasedDiffusion import GraphBasedDiffusion
from RandomWalkDiffusion import RandomWalkDiffusion
from leaveAModeOut import leaveAModeOut
from IPython import embed
from time import time
import argparse
from utility import setNetworks, generateDegree1net, parseArguments


if __name__ == '__main__':
    start = time()
    args = parseArguments()
    if(type(args['mapping']) == str):
        mappingFile = args['mapping']
    else:
        mappingFile = args['mapping'][0]
    if(type(args['data']) == str):
        dataDir = args['data']
    else:
        dataDir = args['data'][0]
    if(type(args['experiment']) == str):
        experiment = args['experiment']
    else:
        experiment = args['experiment'][0]

    print 'Running leave-one-mode-out (LOMO) experiment for {}'.format(experiment)
    networkNames, conns, expModes, selectedQueryModes, selectedPredictionModes = setNetworks(
        2016, 'LOMO', experiment, mappingFile, dataDir)

    # Defining the algorithm
    algorithms = [GraphBasedDiffusion, RandomWalkDiffusion, AptRank]
    algorithmNames = ['GID', 'RW', 'AptRank1']

    # Trim iteratively the nodes which have degree = 1 in the last input network
    # (Assuming the last input network is the largest one)
    _net, removedNodes = generateDegree1net(conns[-1], mappingFile)

    for algorithm, algorithmName in zip(algorithms, algorithmNames):
        for i in range(len(expModes)):
            currMode = expModes[i]
            currQueryMode = selectedQueryModes[i]
            currPredictionMode = selectedPredictionModes[i]
            currNet, _removedNodes = generateDegree1net(
                conns[i], mappingFile, removedNodes)
            currName = networkNames[i]
            # Use the first CTD2016 1-mode network to generate the answer network for
            # all networks
            if i == 0:
                validation = leaveAModeOut(algorithm=algorithm, network=currNet,
                                           mode=currMode,
                                           selectedQueryMode=currQueryMode,
                                           selectedPredictionMode=currPredictionMode)
                validation.generateValidationNetwork()
                ansNet = validation.answerNetwork
            validation = leaveAModeOut(algorithm=algorithm, network=currNet,
                                       answerNetwork=ansNet, mode=currMode,
                                       selectedPredictionMode=currPredictionMode,
                                       selectedQueryMode=currQueryMode)
            validation.perform()
    end = time()
    print('This program ran {} min'.format((end - start) / 60))
