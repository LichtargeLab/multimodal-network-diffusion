#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on June 9, 2017

@author: Chih-Hsu Lin

This script is for functions shared by running experiments
'''

import sys
import os
import paths
from GraphRepresentation5 import GraphRepresentation5
from copy import deepcopy
from IPython import embed
from time import time
import pandas as pd
import argparse
import numpy as np
from pandas import DataFrame
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict as ddict

def parseArguments():
    '''
    parse arguments

    This method provides a nice interface for parsing command line arguments
    and includes help functionality.

    Returns:
    --------
    dict:
        A namespace containing the arguments parsed from the command line and
        their arguments.
    '''
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--experiment', metavar='e', type=str, nargs=1,
                        default='test',
                        help='The experiment option to perform when running this code.')
    parser.add_argument('--data', metavar='d', type=str, nargs=1,
                        default=paths.NETWORK_DATA_DIR,
                        help='The file path to the directory where networks are stored.')
    parser.add_argument('--mapping', metavar='m', type=str, nargs=1,
                        default=paths.NETWORK_MAPPING_FILE,
                        help='The file path to the mapping file used for nodes in this graph.')
    parser.add_argument('--results', metavar='R', type=str, nargs=1,
                        help='The file path to the directory where results should be stored.')
    args = parser.parse_args()
    return vars(args)


def loadNet(mappingFile, conn):
    '''
    Load network

    This method is a shortcut to load GraphRepresentation object

    Parameters
    ----------
    mappingFile: str
        A string providing the file path to a mapping file.  
    connectivity: list
        This should be a list of three-tuples of (source, filepath, directed)

    Returns:
    --------
    GraphRepresentation5
        An instance of GraphRepresentation5.
    '''
    return GraphRepresentation5(mappingFile=mappingFile, connectivity=conn, checkForNet=True)


def generateAnsNet(oldNetwork, newConn, mappingFile, newNetworkName, mode):
    '''
    Generate answer network for time-stamped experiment

    Parameters
    ----------
    oldNetwork: GraphRepresentation5
        GraphRepresentation5 object of the old network
    newConn: list
        Connectivity tuples of the new network
    mappingFile: str
        A string providing the file path to a mapping file.  
    newNetworkName: str
        A string providing the name of the answer network.
    mode: str
        A string indicating the mode type e.g. 'CD', 'DG' or 'CD'

    Returns:
    --------
    GraphRepresentation5
        A network used as answer for time-stamped experiment.
    '''
    os.chdir(paths.NETWORK_TMP_DIR)
    if os.path.exists(newNetworkName + '.pkl'):
        answerNetwork = GraphRepresentation5()
        answerNetwork.loadGraph(newNetworkName + '.pkl')
    else:
        # Get overlapping nodes in input and answer matrices of different networks
        # These nodes are able to be compared across networks
        newNetwork = loadNet(mappingFile, newConn)
        # oldNetwork.deleteNodes('Degree', 0, save=False)
        _nodeOld, nodeInterection, _nodeNew = oldNetwork.determineOverlappingNodes(
            newNetwork)
        _edgeOld, _edgeInterection, edgeNew = oldNetwork.determineOverlappingEdges(
            newNetwork)
        answerNetwork = newNetwork.generateSubNetworkFromEdgesAndNodes(
            newNetworkName, edgeNew, nodeInterection)
        answerNetwork.saveGraph(
            pFN='{}.pkl'.format(answerNetwork.getName()))
    return answerNetwork


def generateDegree1net(conn, mappingFile, removedNodes=None):
    '''
    Generate network of nodes which degree > 1

    This function iteratively delete nodes of degree <=1 in the network.
    If the removedNodes is specified, it would remove those nodes only.

    Parameters
    ----------
    conn: list
        Connectivity tuples of the network
    mappingFile: str
        A string providing the file path to a mapping file.  
    removedNodes: list
        A list of Node objects, which would be removed from network

    Returns:
    --------
    GraphRepresentation5
        A network of nodes which degree > 1
    '''
    os.chdir(paths.NETWORK_TMP_DIR)
    net = loadNet(mappingFile, conn)
    # Trim iteratively the nodes which have degree = 1 in the last input network
    # (Assuming the last input network is the largest one)
    newName = net.name + '_Degree_1_Retained_Nodes'
    if os.path.exists(newName + '.pkl'):
        net.loadGraph(newName + '.pkl')
        try:
            otherGraph = GraphRepresentation5()
            otherGraph.loadGraph(net.name + '_Degree_1_Deleted_Nodes.pkl')
        except:
            pass
    else:
        if removedNodes is None:
            otherGraph, _lostEdges = net.deleteNodes('Degree', 1)
            removedNodes = otherGraph.getNodes().keys()
        else:
            net.deleteNodes('Node', removedNodes, save=False)
            net.setName(newName)
            net.saveGraph(pFN='{}.pkl'.format(net.getName()))
    return net, removedNodes


def setNetworks(version, experiment, mode, mappingFile='../../Networks/Mapping_Ccgdd16Sx91Extended_101217.tsv', dataDir='../../Networks/NetworkData', cancer=False):
    '''
    This function set networks used for experiments

    Parameters
    ----------
    version: int
        A integer specifying the version/year of the loaded networks.
        Currently it only supports '2014' and '2016'
    experiment: str
        A string representing the experiment
        Currently it is either '10fold', 'LOMO' (leave-one-mode-out), or 'TS' (time-stamped)
    mode: str
        A string indicating the mode type e.g. 'CD': Chemical-Disease, 'DG': Disease-Gene,
        'CG': Chemical-Gene.
        Currently it is either 'CD'/'DC', 'DG'/'GD' or 'CD'/'DC'
    cancer: boolean
        A boolean indicating whether to conduct cancer-specific experiments
        Default = False
    mappingFile: str
        A string indicating the mappingFile including path
    dataDir: str
        A string indicating the path to network data

    Returns
    -------
    list
        A list of strings, which are network names
    list
        A list of strings, which are newer network names (This is only return when experiment=='TS')
    list
        A list of Connectivity tuples of the network
    list
        A list of Connectivity tuples of the newer network (This is only return when experiment=='TS')
    list
        A list of strings, which are the modes of query
    list
        A list of strings, which are the modes of prediction

    '''
    mappingFile = mappingFile
    # All network files for comparison
    CTD2014CD_file = dataDir + '/CTD_2014/Chemical-Diseasemapped.txt'
    CTD2014CG_file = dataDir + '/CTD_2014/Chemical-Genemapped.txt'
    CTD2014DG_file = dataDir + '/CTD_2014/Disease-Genemapped.txt'
    CTD2016CD_file = dataDir + '/CTD_2016/Chemical-Diseasemapped.txt'
    CTD2016CG_file = dataDir + '/CTD_2016/Chemical-Genemapped.txt'
    CTD2016DG_file = dataDir + '/CTD_2016/Disease-Genemapped.txt'
    STRING905_file = dataDir + '/STRING_9.05/experimental_9.05_year_labelsmapped.txt'
    STRING905_900_file = dataDir + \
        '/STRING_9.05/experimental_9.05_year_labels_threshold_900mapped.txt'
    STRING10_file = dataDir + '/STRING_10/experimental_10_year_labelsmapped.txt'
    STRING10_900_file = dataDir + \
        '/STRING_10/experimental_10_year_labels_threshold_900mapped.txt'
    MeSH2013CC_file = dataDir + \
        '/MeSH/mtrees2013Chemical-Chemical_network_parentChildmapped.txt'
    MeSH2013DD_file = dataDir + \
        '/MeSH/mtrees2013Disease-Disease_network_parentChildmapped.txt'
    MeSH2016CC_file = dataDir + \
        '/MeSH/mtrees2016Chemical-Chemical_network_parentChildmapped.txt'
    MeSH2016DD_file = dataDir + \
        '/MeSH/mtrees2016Disease-Disease_network_parentChildmapped.txt'
    # Connectivity tuples
    CTD2014CD = ('CTD2014CD', CTD2014CD_file, False)
    CTD2014CG = ('CTD2014CG', CTD2014CG_file, False)
    CTD2014DG = ('CTD2014DG', CTD2014DG_file, False)
    CTD2016CD = ('CTD2016CD', CTD2016CD_file, False)
    CTD2016CG = ('CTD2016CG', CTD2016CG_file, False)
    CTD2016DG = ('CTD2016DG', CTD2016DG_file, False)
    STRING905 = ('STRING905', STRING905_file, False)
    STRING905_900 = ('STRING905_900', STRING905_900_file, False)
    MeSH2013CC = ('MeSH2013CC', MeSH2013CC_file, False)
    MeSH2013DD = ('MeSH2013DD', MeSH2013DD_file, False)
    STRING10 = ('STRING10', STRING10_file, False)
    STRING10_900 = ('STRING10_900', STRING10_900_file, False)
    MeSH2016CC = ('MeSH2016CC', MeSH2016CC_file, False)
    MeSH2016DD = ('MeSH2016DD', MeSH2016DD_file, False)

    modeDict = {'C': 'Chemical', 'D': 'Disease', 'G': 'Gene',
                'CG': ['Gene', 'Chemical'],'GC': ['Gene', 'Chemical'],
                 'CD': ['Disease', 'Chemical'], 'DC': ['Disease', 'Chemical'],
                'DG': ['Disease', 'Gene'],'GD': ['Disease', 'Gene']}

    if mode == 'test' and experiment == 'kFold':
        raise ValueError('Please specify parameter by adding \'--e {prediction mode}\' \n {prediction mode}={\'1mode\',\'3mode\',\'6mode\'}')
    elif mode == 'test':
        raise ValueError('Please specify parameter by adding \'--e {prediction mode}\' \n {prediction mode}={\'DG\',\'DC\',\'GC\'}')
    if experiment == 'TS' or experiment == 'timeStamped':
        if mode == 'test':
            selectedQueryMode = 'Disease'
            selectedPredictionMode = 'Gene'
            newNetworkName = "CTD-DG_gainedEdges_2014trimmed"
            networkNames = ['CTD-DG', 'CTD-DG_MeSH-DD']
            oldConns = [[CTD2014DG], [CTD2014DG, MeSH2013DD]]
            newConns = [[CTD2016DG]]
        elif mode in modeDict:
            if mode == 'DC':
                mode = 'CD'
            if mode == 'GD':
                mode = 'DG'
            if mode == 'GC':
                mode = 'CG'            
            selectedQueryMode = modeDict[mode][0]
            selectedPredictionMode = modeDict[mode][1]
            newNetworkName = "CTD-{}_gainedEdges_2014trimmed".format(mode)
            if cancer:
                newNetworkName = newNetworkName + '_cancer'
            networkNames = [
                'CTD-{}'.format(mode), 'CTD-All', 'CTD+STRING+MeSH']
            exec("""oldConns = [
                [CTD2014{}],
                [CTD2014CG, CTD2014CD, CTD2014DG],
                [CTD2014CG, CTD2014CD, CTD2014DG, STRING905_900, MeSH2013DD, MeSH2013CC]]""".format(mode))
            exec("newConns = [[CTD2016{}]]".format(mode))
        newNetworkNames = [newNetworkName] * len(networkNames)
        selectedQueryModes = [selectedQueryMode] * len(networkNames)
        selectedPredictionModes = [selectedPredictionMode] * len(networkNames)
        return networkNames, newNetworkNames, oldConns, newConns, selectedQueryModes, selectedPredictionModes
    elif experiment == 'LOMO':
        if mode == 'test':
            selectedQueryMode = 'Disease'
            selectedPredictionMode = 'Gene'
            networkNames = ['CTD', 'CTD+STRING+MeSH']
            if version == 2016:
                conns = [[CTD2016CG, CTD2016CD, CTD2016DG],
                         [CTD2016CG, CTD2016CD, CTD2016DG, STRING10_900, MeSH2016DD, MeSH2016CC]]
            elif version == 2014:
                conns = [[CTD2014CG, CTD2014CD, CTD2014DG],
                         [CTD2014CG, CTD2014CD, CTD2014DG, STRING905_900, MeSH2013DD, MeSH2013CC]]
        elif mode in modeDict:
            if mode == 'DC':
                mode = 'CD'
            if mode == 'GD':
                mode = 'DG'
            if mode == 'GC':
                mode = 'CG'
            selectedQueryMode = modeDict[mode][0]
            selectedPredictionMode = modeDict[mode][1]
            expMode = "{}-{}".format(modeDict[mode[0]], modeDict[mode[1]])
            networkNames = ['CTD', 'CTD+STRING+MeSH']
            if version == 2016:
                conns = [[CTD2016CG, CTD2016CD, CTD2016DG],
                         [CTD2016CG, CTD2016CD, CTD2016DG, STRING10_900, MeSH2016DD, MeSH2016CC]]
            elif version == 2014:
                conns = [[CTD2014CG, CTD2014CD, CTD2014DG],
                         [CTD2014CG, CTD2014CD, CTD2014DG, STRING905_900, MeSH2013DD, MeSH2013CC]]
        expModes = [expMode] * len(networkNames)
        selectedQueryModes = [selectedQueryMode] * len(networkNames)
        selectedPredictionModes = [selectedPredictionMode] * len(networkNames)
        return networkNames, conns, expModes, selectedQueryModes, selectedPredictionModes
    elif experiment == 'kFold':
        if mode == 'test':
            selectedQueryModes = ['Disease']
            selectedPredictionModes = ['Gene']
            networkNames = ['CTD-DG']
            if version == 2016:
                conns = [[CTD2016DG]]
            elif version == 2014:
                conns = [[CTD2014DG]]
        elif mode == '1mode':
            selectedQueryModes = ['Disease', 'Gene', 'Disease']
            selectedPredictionModes = ['Gene', 'Chemical', 'Chemical']
            networkNames = ['CTD-DG', 'CTD-CG', 'CTD-CD']
            if version == 2016:
                conns = [[CTD2016DG], [CTD2016CG], [CTD2016CD]]
            elif version == 2014:
                conns = [[CTD2014DG], [CTD2014CG], [CTD2014CD]]
        elif mode == '3mode':
            selectedQueryModes = [['Disease', 'Chemical', 'Disease']]
            selectedPredictionModes = [['Gene', 'Gene', 'Chemical']]
            networkNames = ['CTD-All']
            if version == 2016:
                conns = [[CTD2016DG, CTD2016CG, CTD2016CD]]
            elif version == 2014:
                conns = [[CTD2014DG, CTD2014CG, CTD2014CD]]
        elif mode == '6mode':
            networkNames = ['CTD+STRING+MeSH']
            selectedQueryModes = [['Disease', 'Chemical',
                                   'Disease', 'Gene', 'Chemical', 'Disease']]
            selectedPredictionModes = [['Gene', 'Gene', 'Chemical', 'Gene', 'Chemical', 'Disease']]
            if version == 2016:
                conns = [[CTD2016CG, CTD2016CD, CTD2016DG, STRING10_900, MeSH2016DD, MeSH2016CC]]
            elif version == 2014:
                conns = [[CTD2014CG, CTD2014CD, CTD2014DG, STRING905_900, MeSH2013DD, MeSH2013CC]]
        elif mode == 'Multimode':
            networkNames = ['CTD-All', 'CTD+STRING+MeSH']
            selectedQueryModes = [['Disease', 'Chemical', 'Disease'],
                                  ['Disease', 'Chemical', 'Disease', 'Gene', 'Chemical', 'Disease']]
            selectedPredictionModes = [['Gene', 'Gene', 'Chemical'],
                                       ['Gene', 'Gene', 'Chemical', 'Gene', 'Chemical', 'Disease']]
            if version == 2016:
                conns = [[CTD2016DG, CTD2016CG, CTD2016CD],
                     [CTD2016CG, CTD2016CD, CTD2016DG, STRING10_900, MeSH2016DDpc, MeSH2016CCpc]]
            elif version == 2014:
                conns = [[CTD2014DG, CTD2014CG, CTD2014CD],
                     [CTD2014CG, CTD2014CD, CTD2014DG, STRING905_900, MeSH2013DD, MeSH2013CC]]                
        return networkNames, conns, selectedQueryModes, selectedPredictionModes
    elif experiment == 'Prospective':
        if mode == 'test':
            selectedQueryMode = 'Disease'
            selectedPredictionMode = 'Gene'
            networkNames = ['CTD-DG']
            if version == 2016:
                conns = [[CTD2016DG]]
            elif version == 2014:
                conns = [[CTD2014DG]]
        elif mode[0] in modeDict and mode[1] in modeDict:
            selectedQueryMode = modeDict[mode[0]]
            selectedPredictionMode = modeDict[mode[1]]
            networkNames = ['CTD-{}'.format(mode), 'CTD', 'CTD+STRING+MeSH']
            if mode == 'DC':
                mode = 'CD'
            if mode == 'GD':
                mode = 'DG'
            if mode == 'GC':
                mode = 'CG'
            if version == 2016:
                exec("""conns = [
                    [CTD2016{}],
                    [CTD2016CG, CTD2016CD, CTD2016DG],
                    [CTD2016CG, CTD2016CD, CTD2016DG, STRING10_900, MeSH2016DD, MeSH2016CC]]""".format(mode))
            elif version == 2014:
                exec("""conns = [
                    [CTD2014{}],
                    [CTD2014CG, CTD2014CD, CTD2014DG],
                    [CTD2014CG, CTD2014CD, CTD2014DG, STRING905_900, MeSH2013DD, MeSH2013CC]]""".format(mode))
        selectedQueryModes = [selectedQueryMode] * len(networkNames)
        selectedPredictionModes = [selectedPredictionMode] * len(networkNames)
        return networkNames, conns, selectedQueryModes, selectedPredictionModes


def printPerformance(results, keywords, labels, outputPrefix):
    '''
    Print performance.

    Print the performance of an experiment.

    Parameters
    ----------
    keywords: list
        Set of keywords to define column names.
    labels: list
        Set of keywords to define row names.
    outputPrefix: str
        String defining first part of the name to save data under.
    '''
    df = DataFrame(columns=keywords)
    for x in keywords:
        df[x] = results[x]
    df.index = labels
    df.to_csv(outputPrefix + '_metrics.tsv', sep="\t")

    selectedKeywords = ['AUC', 'pAUC', 'truePositiveN']
    df = DataFrame(columns=selectedKeywords)
    for x in selectedKeywords:
        df[x] = results[x]
    topPercentListName = ['0-1%', '1-10%', '10-20%',
                          '20-30%', '30-40%', '40-50%', '50-100%']
    truePositiveNtopPercent = np.array(results['truePositiveNtopPercent']).T
    for i, x in enumerate(topPercentListName):
        try:
            df[x] = truePositiveNtopPercent[i]
        except:
            df[x] = np.nan
    df.index = labels
    df.to_csv(outputPrefix + '_metrics2.tsv', sep="\t")

    selectedKeywords = ['AUC_bootstrap', 'AUC_PR_bootstrap']
    for keyword in selectedKeywords:
        df_dict = ddict(pd.DataFrame)
        for y, l in zip(results[keyword], labels):
            data = y
            algorithm = l.split(' (')[0]
            network = l.split(' (')[1]
            # df_tmp = pd.DataFrame({network: data, 'algorithm':algorithm})
            # df_dict[network] = pd.concat([df_dict[network],df_tmp])
            df_tmp = pd.DataFrame({algorithm: data, 'network': network})
            df_dict[algorithm] = pd.concat([df_dict[algorithm], df_tmp])
        df = df_dict[df_dict.keys()[0]]
        for k in df_dict.keys()[1:]:
            df = pd.concat([df, df_dict[k]], axis=1)
        df.T.to_csv(outputPrefix + '_' + keyword + '.tsv', sep="\t")

    return


def plotMultipleAUCperEntity_xyplot(results, keyword, labels, outputPrefix):
    df_dict = ddict(pd.DataFrame)
    for y, l, node in zip(results[keyword], labels, results['y_node1']):
        data = y
        nodeName = [n.getCanonical() for n in node]
        algorithm = l.split(' (')[0]
        network = l.split(' (')[1]
        df_tmp = pd.DataFrame(
            {algorithm + '_' + keyword: data, 'Entity': nodeName, 'Network': network})
        df_dict[algorithm] = pd.concat([df_dict[algorithm], df_tmp])

    df = df_dict[df_dict.keys()[0]]
    for alg in df_dict.keys()[1:]:
        df = df.merge(df_dict[alg], on=['Entity', 'Network'])
    df.dropna().to_csv('{}_{}_xyplot_data.tsv'.format(outputPrefix, keyword), sep="\t")
    import seaborn as sns
    sns.set(font_scale=1.5)
    sns.set_style("white")
    sns.set_style("ticks")
    plt.figure(figsize=(10, 10))
    cases = list(set([k.replace('_' + keyword, '')
                      for k in df.columns if keyword in k]))
    for x, y in itertools.combinations(cases, 2):
        g = sns.lmplot(x + '_' + keyword, y + '_' + keyword,
                       data=df.dropna(),
                       fit_reg=False,
                       hue='Network')
        # scatter_kws={"marker": ['o','x'],
        #              "s": 100}
        g.set(ylim=(0, 1.1))
        g.set(xlim=(0, 1.1))
        g.ax.plot([0, 1], [0, 1], ':k')
        plt.savefig('{}_{}_{}_{}_xyplot.pdf'.format(outputPrefix, x, keyword, y),
                    bbox_inches='tight')
        plt.close()
    return


def plotMultipleAUCperEntity_violinplot(results, keyword, labels, outputPrefix):
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
    df = pd.DataFrame()
    for y, l in zip(results[keyword], labels):
        data = [x for x in y if str(x) != 'nan']
        algorithm = l.split(' (')[0]
        network = l.split(' (')[1]
        df_tmp = pd.DataFrame(
            {keyword: data, 'Algorithm': algorithm, 'Network': network})
        df = pd.concat([df, df_tmp], axis=0)

    import seaborn as sns
    sns.set(font_scale=3)
    sns.set_style("white")
    sns.set_style("ticks")
    # plt.figure(figsize=(20, 10))
    plt.figure()
    # # dark blue
    # color_ordered = [u'#4477AA'] * len(labels)
    # green
    # color_ordered = [u'#4daf4a'] * len(labels)
    # g = sns.violinplot(data=array, palette=color_ordered)
    # x = range(0, len(labels))
    # plt.xticks(x, labels, rotation=90, fontsize=30)
    g = sns.violinplot(x="Algorithm", y=keyword, hue="Network",
                       data=df, palette='muted', inner=None)
    g = sns.swarmplot(x="Algorithm", y=keyword, hue="Network",
                      data=df, color='white', edgecolor="gray", split=True)
    # g = sns.swarmplot(x="Algorithm", y=keyword, hue="Network",
    #                   data=df)
    # plt.yticks(fontsize=30)
    g.set_xlabel('')
    if 'p' in keyword:
        # g.set_ylabel('Partial {} (0.1)'.format(keyword), fontsize=30)
        g.set_ylabel('Partial {} (0.1)'.format(keyword))
        g.set_ylim([0, 0.1])
    else:
        # g.set_ylabel(keyword, fontsize=30)
        g.set_ylabel(keyword)
        # g.set_ylim([0.8, 1])
        g.set_ylim([0, 1])
    # sns.despine(offset=10, trim=True)

    # Get the handles and labels.
    # https://stackoverflow.com/questions/35538882/seaborn-boxplot-stripplot-duplicate-legend
    handles, labels2 = g.get_legend_handles_labels()
    l = plt.legend(handles[0:len(set(labels2))], labels[0:len(
        set(labels2))], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.savefig('{}_{}_violinSwarmplot2.pdf'.format(outputPrefix, keyword),
                bbox_inches='tight')
    plt.close()
    return


def plotMultipleAUCbootstrap_violinplot(array, keyword, labels, outputPrefix):
    '''
    Plot violin plots

    This method uses the Seaborn package to plot violin plots of bootstrapped
    AUC data.

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
    array = np.array(array).T
    sns.set_style("white")
    sns.set_style("ticks")
    plt.figure(figsize=(20, 10))
    # # dark blue
    # color_ordered = [u'#4477AA'] * len(labels)
    # green
    color_ordered = [u'#4daf4a'] * len(labels)
    g = sns.violinplot(data=array, palette=color_ordered)
    x = range(0, len(labels))
    plt.xticks(x, labels, rotation=90, fontsize=30)
    plt.yticks(fontsize=30)
    g.set_xlabel('')
    if 'pAUC' in keyword:
        g.set_ylabel('Partial AUC (0.1)', fontsize=30)
        g.set_ylim([0, 0.1])
    else:
        g.set_ylabel('AUC', fontsize=30)
        g.set_ylim([0.8, 1])
    # sns.despine(offset=10, trim=True)
    plt.savefig('{}_{}_violinplot.pdf'.format(outputPrefix, keyword),
                bbox_inches='tight')
    plt.close()
    return


def plotMultipleAUCbootstrap_boxplot(AUC_bootstrap, labels, outputPrefix):
    '''
    Plot box plots

    This method uses matplotlib to plot box plots of bootstrapped AUC data.

    Parameters
    ----------
    AUC_bootstrap: list or array
        Data to plot.
    labels: list
        A set of strings describing the data to be plotted.
    outputPrefix: str
        A string used as part of the filename to save the plot.
    '''
    plt.figure(figsize=(30, 10))
    ax = plt.subplot(111)
    # ax.set_xlabel("FPR", fontsize=30)
    ax.set_ylabel("AUC", fontsize=30)
    # plt.title("ROC Curve", fontsize=30)
    ax.boxplot(AUC_bootstrap, labels=labels)
    x = range(1, len(AUC_bootstrap) + 1)
    plt.xticks(x, labels, rotation=90)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylim(0.8, 1.0)
    # ax.legend(fontsize=16, loc='best')
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(30)
    # ax.tight_layout()
    plt.savefig('{}_AUC_boxplot.pdf'.format(outputPrefix), bbox_inches='tight')
    plt.close()
    return


def plotMultipleROC(fpr, tpr, predictionsN, AUC, pAUC, labels, outputPrefix,
                    randomline=False):
    '''
    Plot ROC Curves

    Plot multiple ROC curves on a single plot.

    Parameters
    ----------
    fpr: list or array
        Set of points describing the fpr over an AUC curve.
    tpr: list or array
        Set of points descrbing the tpr over an AUC curve.
    predictionsN: int
        Number of predictions made overall.
    AUC: float
        The AUC score for a plot.
    pAUC: float
        The pAUC score for a plot.
    labels: list
        A set of labels for the data in the ROC curves.
    outputPrefix: str
        A string to be used for part of the filename to save the plot under.
    randomline: bool
        Whether or not to plot the random line on the plot.
    '''
    # colors = ("blue", "green", "orange", "darkviolet", "pink", "black", "red",
    #           "yellowgreen", "darkred", "skyblue", "magenta", "steelblue",
    #           "yellow", "cyan")
    # linestyles = ['solid', 'solid', 'solid', 'dashed']
    if len(fpr) == 6:
        # colors = ("blue","blue","blue", "green","green","green")
        colors = ("red", "red", "red", "blue", "blue", "blue")
        linestyles = [':', '--', '-']
    elif len(fpr) == 4:
        # colors = ("blue","blue", "green","green")
        colors = ("red", "red", "blue", "blue")
        linestyles = ['--', '-']
    else:
        colors = ("blue", "green", "orange", "darkviolet", "pink", "black", "red",
                  "yellowgreen", "darkred", "skyblue", "magenta", "steelblue",
                  "yellow", "cyan")
        linestyles = ['solid', 'solid', 'solid', 'dashed']

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)
    ax.set_xlabel("FPR", fontsize=30)
    ax.set_ylabel("TPR", fontsize=30)
    # plt.title("ROC Curve", fontsize=30)
    for i in range(len(fpr)):
        # plt.plot(fpr[i], tpr[i], linewidth=2, label=labels[i] + ", #= %i, AUC = %.3f" % (
            # predictionsN[i], AUC[i]))
        if len(fpr) == 6:
            ax.plot(fpr[i], tpr[i], linewidth=2, linestyle=linestyles[i % 3],
                    color=colors[i], label=('{0}, AUC={1:.3f})'.format(
                        labels[i], AUC[i])))
        elif len(fpr) == 4:
            ax.plot(fpr[i], tpr[i], linewidth=2, linestyle=linestyles[i % 2],
                    color=colors[i], label=('{0}, AUC={1:.3f})'.format(
                        labels[i], AUC[i])))
        else:
            ax.plot(fpr[i], tpr[i], linewidth=2, linestyle=linestyles[i % 2],
                    color=colors[i % len(colors)], label=('{0}, AUC={1:.3f})'.format(
                        labels[i], AUC[i])))
            # color=colors[i], label=('{0}, AUC={1:.4f}, pAUC={2:.4f}'.format(
            #     labels[i],AUC[i],pAUC[i])))
            # color=colors[i], label=(labels[i] +
            #                         ', AUC={0:.4f}'.format(AUC[i]) +
            #                         ', pAUC={0:.4f}' % (pAUC[i])))
    if randomline:
        x = [0.0, 1.0]
        # , label='random')
        ax.plot(x, x, linestyle='dashed', color='red', linewidth=2)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    # ax.legend(fontsize=20, loc='best')
    if 'cancer' in outputPrefix:
        ax.legend(fontsize=16, loc=4)
    elif 'Proximity' in outputPrefix:
        ax.legend(fontsize=14, loc=4)
    else:
        ax.legend(fontsize=17, loc=4)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(24)
    # ax.tight_layout()
    plt.savefig(outputPrefix + "_ROC.pdf", bbox_inches='tight')
    plt.close()
    return


def plotMultiplePRcurve(recall, precision, predictionsN, AUC_PR, pAUC_PR,
                        labels, outputPrefix):
    '''
    Plot PR Curves

    Plot multiple precision recall curves.

    Parameters
    ----------
    recall: list or array
        Set of recall values for a group of experiments to plot.
    precision: list or array
        Set of precision values for a group of experiments to plot.
    predictionsN: int
        Number of predictions made.
    AUC_PR: float
        AUC score for this PR curve.
    pAUC_PR: float
        Partial AUC score for this PR curve.
    labels: list
        Set of labels for the data being plotted.
    outputPrefix: str
        String to be used in the filename for saving this plot.
    '''
    colors = ("blue", "green", "orange", "darkviolet", "pink", "black", "red",
              "yellowgreen", "darkred", "skyblue", "magenta", "steelblue",
              "yellow", "cyan")
    linestyles = ['solid', 'solid', 'solid', 'dashed']
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)
    ax.set_xlabel("Recall (TPR)", fontsize=30)
    ax.set_ylabel("Precision", fontsize=30)
    # plt.title("ROC Curve", fontsize=30)
    for i in range(len(recall)):
        # plt.plot(recall[i], precision[i], linewidth=2,
        # label=labels[i] + ", # = %i, AUC = %.3f" % (predictionsN[i],
        # AUC_PR[i]))
        ax.plot(recall[i], precision[i], linewidth=2, linestyle=linestyles[i % 2],
                color=colors[i % len(colors)], label=('{0}, AUC={1:.4f}, pAUC={2:.3E}'.format(
                    labels[i], AUC_PR[i], pAUC_PR[i])))
        # color=colors[i], label=(labels[i] +
        #                         ', AUC={0:.4f}'.format(AUC[i]) +
        #                         ', pAUC={0:.4f}' % (pAUC[i])))
        # plt.plot(recall[i], precision[i], linewidth=2, color=colors[i],
        #          linestyle=linestyles[i % 2],
        #          label=(labels[i] + ', AUC = {0:.4f}'.format(AUC_PR[i]) +
        #                 ', pAUC = {0:.3E}'.format(pAUC_PR[i])))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=16, loc='best')
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    plt.savefig(outputPrefix + "_PRcurve.pdf", bbox_inches='tight')
    plt.close()
    return


# def getOverlappingNodes(networkList):
#     '''
#     Get Overlapping Nodes
#
#     Get overlapping nodes in input and answer matrices of different networks
#
#     Parameters
#     ----------
#     networkList: list
#         Set of GraphReprsentation5 objects to check for overlaps between.
#
#     Returns
#     -------
#     set
#         A set of nodes which overlap between two networks.
#     '''
#     overlappingNodes = set(networkList[0].getNodePosition().keys())
#     for net in networkList[1:]:
#         overlappingNodes = overlappingNodes.intersection(
#             set(net.getNodePosition().keys()))
#     return overlappingNodes
#
#
# def getOverlappingNodes2(networkList):
#     '''
#     Get Overlapping Nodes
#
#     Get overlapping nodes in input and answer matrices of different networks
#
#     Parameters
#     ----------
#     networkList: list
#         Set of GraphReprsentation5 objects to check for overlaps between.
#
#     Returns
#     -------
#     set
#         A set of nodes which overlap between two networks.
#     '''
#     overlappingNodes = set(networkList[0].getNodes().keys())
#     print(networkList[0])
#     for net in networkList[1:]:
#         print(net)
#         print('At breakpoint')
#         overlappingNodes = overlappingNodes.intersection(
#             set(net.getNodes().keys()))
#     return overlappingNodes


def getIntersectedNetworkByOverlappingNodes(network1, network2,
                                            newNetworkName=None):
    '''
    Get matrix of edges from networks based on intersecting nodes.

    Get overlapping nodes in input and answer matrices of different networks.

    Parameters
    ----------
    network1: GraphRepresentation5
        First network to use for checking for overlap
    network2: GraphRepresentation5
        Second network to use for checking for overlap
    newNetworkName: str
        Optional string to use for saving the resulting overlapping network.

    Returns
    -------
    GraphReprsentation5
        A network object holding the result of the overlap between two networks.
    '''
#     overlappingNodes = getOverlappingNodes2([network1, network2])
#     overlappingNodes = {x: x for x in overlappingNodes}
#     onKeys = overlappingNodes.keys()
#     for ok in onKeys:
#         print ok
#     # exit()
#     gainEdges = set(network2.getEdgeList().keys()).difference(
#         set(network1.getEdgeList().keys()))
#     gainEdges2 = {}
#     for edge in gainEdges:
#         node1 = edge.getNode1()
#         check1 = node1 in onKeys
#         node2 = edge.getNode2()
#         check2 = node2 in onKeys
#         check = (check1 and check2)
#         if(check):
#             gainEdges2[edge] = edge
#     # gainEdges2 = {x: x for x in gainEdges
#     #               if ((x.getNode1() in overlappingNodes.keys()) and
#     #                   (x.getNode2() in overlappingNodes.keys()))}
#     answerMatrix = network2.generateSubNetworkFromEdgesAndNodes(
#         gainEdges2, overlappingNodes)
#     if newNetworkName is not None:
#         answerMatrix.saveGraph(newNetworkName + '.pkl')
    answerMatrix = network1.determineOverlap(network2)
    if(newNetworkName):
        answerMatrix.setName(newNetworkName)
    return answerMatrix


def outputConnectivityFile(adjMatrix, nodeNameList, networkNames,
                           nodeAttributes, year):
    '''
    Output Connectivity File

    Create a graph connectivity file based on an adjacency matrix and the data
    describing the nodes from that matrix.

    Parameters
    ----------
    adjMatrix: csr_matrix
        A matrix representing edges in a graph.
    nodeNameList: list
        A set of node names ordered in the same way as the adjacency matrix
        and with the same number of elements as the shape of the adjacency
        matrix.
    networkNames: list
        List of network names to use in saving the file produced.
    nodeAttributes: dictionary
        Attributes for each node.
    year: int
        Year in which data of this graph is created.

    Returns
    -------
    fileName
    '''
    nonZeroIdx = adjMatrix.nonzero()
    fileName = "_".join(networkNames) + "_timeStamped_" + str(year) + ".txt"
    f = open(fileName, 'w')
    for i in range(len(nonZeroIdx[0])):
        node1 = nodeNameList[nonZeroIdx[0][i]]
        node2 = nodeNameList[nonZeroIdx[1][i]]
        weight = adjMatrix[nonZeroIdx[0][i], nonZeroIdx[1][i]]
        attr1 = nodeAttributes[node1]
        attr2 = nodeAttributes[node2]
        metaList = ['Entity1Label::' + attr1['Type'],
                    'Entity2Label::' + attr2['Type']]
        meta = ";;".join(metaList)
        f.write('{}\t{}\t{}\t{}\n'.format(node1, node2, weight, meta))
    f.close()
    return fileName


def getUnionEdges(networkList, selectedNodes, networkNames, year):
    '''
    Get Union Edges.

    Used to find the union across different answer matrices as the answer across
    networks/algorithms based on the selected overlapping nodes.

    Parameters
    ----------
    networkList: list
        Set of networks to join.
    selectedNodes: list
        List of nodes to use when generating union.
    networkNames: list
        Set of names for networks joined to be used in generating the name of
        the final network.
    year: int
        Year in which data of this graph is created.

    Returns
    -------
    GraphRepresentation5
        New graph created from networks passed in.
    '''
    unionEdges = networkList[0].subnetworkByNodes('Node', selectedNodes,
                                                  'Equal')
    unionNodes = unionEdges.getNodePosition()
    nodeAttributes = networkList[0].getNodeAttributes()
    # Sorted node name list for all matrices to align
    unionSortedNodes = sorted(unionNodes.keys(), key=unionNodes.get)
    unionMatrix = unionEdges.getAdjacencyMatrix()
    for net in networkList[1:]:
        subnet = net.subnetworkByNodes('Node', selectedNodes, 'Equal')
        subNodes = subnet.getNodePosition()
        subMatrix = subnet.getAdjacencyMatrix()
        newIndex = [subNodes[x] for x in unionSortedNodes]
        reindexedSubMatrix = subMatrix[newIndex, :][:, newIndex]
        unionMatrix += reindexedSubMatrix
    fileName = outputConnectivityFile(
        unionMatrix, unionSortedNodes, networkNames, nodeAttributes, year)
    newNetworkName = "_".join(networkNames) + "_timeStamped_" + str(year)
    finalNetwork = GraphRepresentation5(connectivity=[
        (newNetworkName, fileName, False)])
    return finalNetwork


def getUnionEdges2(networkList, selectedNodes, networkNames, year):
    '''
    Get Union Edges.

    Used to find the union across different answer matrices as the answer across
    networks/algorithms based on the selected overlapping nodes.

    Parameters
    ----------
    networkList: list
        Set of networks to join.
    selectedNodes: list
        List of nodes to use when generating union.
    networkNames: list
        Set of names for networks joined to be used in generating the name of
        the final network.
    year: int
        Year in which data of this graph is created.

    Returns
    -------
    GraphRepresentation5
        New graph created from networks passed in.
    '''
    # selectedNodeNames = [x.getName() for x in selectedNodes]
    finalNetwork, _ = networkList[0].subnetworkByNodes(
        #'Node', list(selectedNodeNames), 'Equal')
        'Node', selectedNodes, 'Equal')
    unionEdges = set(finalNetwork.getEdgeList().keys())
    # unionNodes = unionEdges.getNodePosition()
    # nodeAttributes = networkList[0].getNodeAttributes()
    # '''Sorted node name list for all matrices to align'''
    # unionSortedNodes = sorted(unionNodes.keys(), key=unionNodes.get)
    # unionMatrix = unionEdges.getAdjacencyMatrix()
    for net in networkList[1:]:
        subnet, _ = net.subnetworkByNodes(
            # 'Name', list(selectedNodeNames), 'equal')
            'Node', selectedNodes, 'Equal')
        edges = subnet.getEdgeList().keys()
        unionEdges |= set(edges)
    finalNodes = {}
    for x in selectedNodes:
        finalNodes[x] = x
    finalEdges = {}
    for x in unionEdges:
        finalEdges[x] = x
    finalNetwork.generateSubNetworkFromEdgesAndNodes(finalEdges, finalNodes)
    # fileName = outputConnectivityFile(unionMatrix, unionSortedNodes, networkNames, nodeAttributes, year)
    # newNetworkName = "_".join(networkNames) + "_timeStamped_" + str(year) + '_Node'+str(finalNetwork.getSize()['Nodes'])+ '_Edge'+str(finalNetwork.getSize()['Nodes'])
    newNetworkName = "_".join(networkNames) + "_timeStamped_" + str(year)
    finalNetwork.setName(newNetworkName)
    finalNetwork.saveGraph(finalNetwork.getName() + '.pkl')
    return finalNetwork


def loadNetworks(networkFiles, networkNames, year):
    '''
    Load networks

    Load a set of networks and generate a set of time stamped networks.

    Parameters
    ----------
    networkFiles: list
        A list of file paths to network pickle files.
    networkNames: list
        A list of network names to correspond with the network files passed in.
    year: int
        An integer value giving the year on which to split the loaded network.

    Returns
    -------
    list
        A set of networks containing nodes and edges known before the specified
        year.
    list
        A set of networks containing nodes and edges identified in or after the
        specified year.
    '''
    oldNetworks = []
    gainedNetworks = []
    for f, x in zip(networkFiles, networkNames):
        print 'Generating graph object ' + x
        network = GraphRepresentation5()
        network.load(f)
        network.setName(x)
        gainedEdges = network.getSubNetwork(
            parameterName='Year', parameterValue=year, mode='gte')
        oldNetworks.append(network)
        gainedNetworks.append(gainedEdges)
    return oldNetworks, gainedNetworks


def loadNetworks2(networkFiles, networkNames, split, parameterName='Year'):
    '''
    Load networks

    Load a set of networks and generate a set of time stamped networks.

    Parameters
    ----------
    networkFiles: list
        A list of file paths to network pickle files.
    networkNames: list
        A list of network names to correspond with the network files passed in.
    split: int or str
        An integer value giving the year on which to split the loaded network,
        or the value which should be used for the corresponding parameter name.
    parameterName: str
        The value on which to split the network, set to 'Year' by default.

    Returns
    -------
    list
        A set of networks containing nodes and edges known before the specified
        year.
    list
        A set of networks containing nodes and edges identified in or after the
        specified year.
    '''
    oldNetworks = []
    gainedNetworks = []
    for f, x in zip(networkFiles, networkNames):
        print 'Generating graph object ' + x
        validationNetworkNames = [x + '_Year_{}_Retained_Edges'.format(split),
                                  x + '_Year_{}_Deleted_Edges'.format(split)]
        fileExists = map(lambda x: os.path.exists(x + '.pkl'),
                         validationNetworkNames)
        if sum(fileExists) == len(fileExists):
            oldNetwork = GraphRepresentation5()
            oldNetwork.loadGraph(validationNetworkNames[0])
            gainedEdges = GraphRepresentation5()
            gainedEdges.loadGraph(validationNetworkNames[1])
        else:
            if f.endswith('.pkl'):
                network = GraphRepresentation5()
                network.loadGraph(f)
            else:
                mappingFile = '../../Networks/Mapping_Ccgdd16Sx91Extended.tsv'
                network = GraphRepresentation5(mapping=mappingFile,
                                               connectivityInfo=[(x, f, False)])
            gainedEdges = network.getSubNetwork(
                parameterName=parameterName, parameterValue=split, mode='lte')
            pFN = '{}.pkl'.format(validationNetworkNames[0])
            oldNetwork = network
            oldNetwork.saveGraph(pFN)
            pFN = '{}.pkl'.format(validationNetworkNames[1])
            gainedEdges.saveGraph(pFN)
        oldNetworks.append(oldNetwork)
        gainedNetworks.append(gainedEdges)
    return oldNetworks, gainedNetworks

