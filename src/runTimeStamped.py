#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Aug 26, 2016

@author: Chih-Hsu Lin

This script is for running time-stamped experiments across different networks.

Example:
    This example runs a prediction of disease-gene associations.
        $ python runTimeStamped.py --e DG

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
from timeStamped import timeStamped
from copy import deepcopy
from IPython import embed
from time import time
import pandas as pd
from utility import generateAnsNet, setNetworks, generateDegree1net, parseArguments

if __name__ == '__main__':
    start = time()
    previous = time()
    args = parseArguments()
    year = 2015
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

    print 'Running time-stamped experiment for {}'.format(experiment)
    networkNames, newNetworkNames, oldConns, newConns, selectedQueryModes, selectedPredictionModes = setNetworks(
        2014, 'TS', experiment, mappingFile, dataDir)

    # Defining the algorithm
    algorithms = [GraphBasedDiffusion, RandomWalkDiffusion, AptRank]
    algorithmNames = ['GID', 'RW', 'AptRank1']

    # Trim iteratively the nodes which have degree = 1 in the last input network
    # (Assuming the last input network is the largest one)
    _net, removedNodes = generateDegree1net(oldConns[-1], mappingFile)

    for algorithm, algorithmName in zip(algorithms, algorithmNames):
        for i in range(len(networkNames)):
            oldNetwork, _removedNodes = generateDegree1net(
                oldConns[i], mappingFile, removedNodes)
            currQueryMode = selectedQueryModes[i]
            currPredictionMode = selectedPredictionModes[i]
            currName = networkNames[i]
            currNewNetName = newNetworkNames[i]
            if i == 0:
                answerNetwork = generateAnsNet(
                    oldNetwork, newConns[i], mappingFile, currNewNetName, experiment)
            validation = timeStamped(algorithm, oldNetwork, year, answerNetwork=answerNetwork,
                                     ## Option to compute performance for each entity; Default is False
                                     # splitPerformanceByEntity=True, 
                                     selectedQueryMode=currQueryMode,
                                     selectedPredictionMode=currPredictionMode)
            validation.perform()
    end = time()
    print('This program ran {} min'.format((end - start) / 60))
