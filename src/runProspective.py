#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Aug 26, 2017

@author: Chih-Hsu Lin

This script is for running prospective experiments across different networks

Example:
    This example runs a prospective experiment of predicting genes for given diseases
        $ python runProspective.py --e DG

Arguments:
    --e: Could be one of {'DG','DC','GC'}; DG: disease-gene; DC: disease-chemical; GC: gene-chemical

"""
import sys
import os
import paths
from GraphRepresentation5 import GraphRepresentation5
from AptRank import AptRank
from GraphBasedDiffusion import GraphBasedDiffusion
from RandomWalkDiffusion import RandomWalkDiffusion
from prospective import prospective
from IPython import embed
from time import time
import argparse
from utility import setNetworks, generateDegree1net, parseArguments
import pandas as pd

if __name__ == '__main__':
    start = time()
    previous = time()
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

    print 'Running Prospective for {}'.format(experiment)
    networkNames, conns, selectedQueryModes, selectedPredictionModes = setNetworks(
        2016, 'Prospective', experiment, mappingFile, dataDir)
    # Defining the algorithm
    algorithms = [GraphBasedDiffusion, RandomWalkDiffusion, AptRank]
    algorithmNames = ['GID', 'RW', 'AptRank1']

    # Trim iteratively the nodes which have degree = 1 in the last input network
    # (Assuming the last input network is the largest one)
    _net, removedNodes = generateDegree1net(conns[-1], mappingFile)

    if experiment.startswith('D') or experiment=='test':
        inputListFile = paths.PROJECT_DIR + '/data/prospective_input_list/top10DieasesFormatedmappedFiltered.txt'
    elif experiment.startswith('G'):
        inputListFile = paths.PROJECT_DIR + '/data/prospective_input_list/interestingGenes_ratioFormatedmappedFiltered.txt'
    inputList = pd.read_csv(inputListFile, sep="\t", header=None)[0].values
    inputListInitialNames = pd.read_csv(
        inputListFile, sep="\t", header=None)[4].values
    inputPrefix = inputListFile.split('/')[-1].split('.')[0]

    FDAdrugsFile = paths.PROJECT_DIR + '/data/other/full database_12132016_FDAdrugsFormatedmappedFiltered.txt'
    MeSH_name_file1 = paths.PROJECT_DIR + '/data/other/c2016.bin_dict'
    MeSH_name_file2 = paths.PROJECT_DIR + '/data/other/d2016.bin_dict'

    for algorithm, algorithmName in zip(algorithms, algorithmNames):
        for i in range(len(networkNames)):
            currQueryMode = selectedQueryModes[i]
            currPredictionMode = selectedPredictionModes[i]
            currNet, _removedNodes = generateDegree1net(
                conns[i], mappingFile, removedNodes)
            currName = networkNames[i]
            print(currNet)
            validation = prospective(name=inputPrefix, algorithm=algorithm, network=currNet,
                                     selectedEntityList=inputList, selectedEntityInitName=inputListInitialNames,
                                     selectedPredictionMode=currPredictionMode,
                                     selectedQueryMode=currQueryMode,
                                     mappingFile=mappingFile, FDAdrugsFile=FDAdrugsFile,
                                     MeSH_name_file1=MeSH_name_file1, MeSH_name_file2=MeSH_name_file2,
                                     splitPerformanceByEntity=True)
            validation.perform()
    end = time()
    print('This program ran {} min'.format((end - start) / 60))
