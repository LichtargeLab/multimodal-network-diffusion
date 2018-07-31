#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Aug 26, 2016

@author: Chih-Hsu Lin

This script is for running k-fold cross-validation experiments

Example:
    This example runs a 10-fold cross-validation of 1-mode networks
        $ python runKfoldCV.py --e 1mode

Arguments:
    --e: Could be one of {'1mode','3mode','6mode'}; 1mode: 1-mode networks; 3mode: 3-mode networks; 6mode: 6-mode networks

"""
import argparse
import sys
import os
import paths
from GraphRepresentation5 import GraphRepresentation5
from AptRank import AptRank
from GraphBasedDiffusion import GraphBasedDiffusion
from RandomWalkDiffusion import RandomWalkDiffusion
from kFoldCV import kFoldCV
from IPython import embed
from time import time
import pandas as pd
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

    k = 10
    print 'Running {}-fold cross-validation for {}'.format(k, experiment)
    networkNames, conns, selectedQueryModes, selectedPredictionModes = setNetworks(
        2016, 'kFold', experiment, mappingFile, dataDir)

    os.chdir(paths.NETWORK_TMP_DIR)
    networks = [GraphRepresentation5(mappingFile=mappingFile, connectivity=conn,
                                     checkForNet=True) for conn in conns]

    # Defining the algorithm
    algorithms = [GraphBasedDiffusion, RandomWalkDiffusion, AptRank]
    algorithmNames = ['GID', 'RW', 'AptRank1']

    for algorithm, algorithmName in zip(algorithms, algorithmNames):
        for i in range(len(networkNames)):
            network, _removedNodes = generateDegree1net(conns[i], mappingFile)
            currQueryMode = selectedQueryModes[i]
            currPredictionMode = selectedPredictionModes[i]
            currName = networkNames[i]
            validation = kFoldCV(algorithm, network, k,
                                 selectedQueryMode=currQueryMode,
                                 selectedPredictionMode=currPredictionMode)
            validation.perform()
    end = time()
    print('This program ran {} min'.format((end - start) / 60))
