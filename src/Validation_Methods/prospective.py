#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Aug 8, 2017

@author: Chih-Hsu Lin
'''
import os
import sys
from time import time
from dotenv import load_dotenv, find_dotenv
import paths
from ValidationParent import ValidationParent
from IPython import embed
import pandas as pd
import numpy as np
from pubmed import multithreadAssociation
from scipy.stats import chi2_contingency, fisher_exact
from multiprocessing import cpu_count, Process, Queue
from  statsmodels.stats.multitest import multipletests
from GraphRepresentation5 import GraphRepresentation5,Edge
import logging
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%Y-%m-%d- %I:%M:%S', level=logging.INFO)
from prediction_to_html import output_html

class prospective(ValidationParent):
    '''
    Take the year to split the network into old and new networks and conduct time-stamped experiment
    '''

    def __init__(self, name, algorithm, network, selectedEntityList, selectedEntityInitName=None,
                 selectedQueryMode=None, selectedPredictionMode=None,
                 bridgeMode=None, **kwargs):
        '''
        Constructor

        This function generates an instance of the prospective class.

        Parameters
        ----------
        name: str
            The short name for this prospective prediction input entities
        algorithm: child class of AlgorithmParent
            The algorithm to be performed during the leave a mode out experiment
            currently expects classes: GraphBasedDiffusion, MF,
            and networkLassDiff.
        network: GraphRepresentation5
            An object representing the network to start with when performing
            this experiment.
        selectedEntityList: list
            A set of entities of interest to compute results for.
        selectedCanonicalList: list
            The canonical names of interested entities to compute results for.
        selectedQueryMode: str
            The node type from which to diffuse labels in this experiment
            (i.e. Chemical, Disease, Gene). The default value for this parameter
            is None which means predictions will be made starting from all node
            types.
        selectedPredictionMode: str
            The node type to which to diffuse labels in this experiment
            (i.e. Chemical, Disease, Gene). The default value for this parameter
            is None which means predictions will be made for all node types.
        answerNetwork: GraphRepresentation5
            A graph object containing the edges (and supporting nodes) which are
            in the mode left out in the experiment.  That is those edges you
            wish to predict back by performing the algorithm. Default value is
            None, in which case the correct result graph will be generated using
            the starting network.

        Returns
        -------
        prospective
            An instance of this class.
        '''
        method = 'prospective'
        ValidationParent.__init__(
            self, method=method, algorithm=algorithm, network=network,
            split=None,
            selectedEntityList=selectedEntityList,
            selectedQueryMode=selectedQueryMode,
            selectedPredictionMode=selectedPredictionMode,
            bridgeMode=bridgeMode,
            param=kwargs)
        self.inputNetwork = network
        self.selectedEntityInitName = selectedEntityInitName
        self.name = method + "_" + name
        self.outputDf = pd.DataFrame()
        self.mappingFile = '../../Networks/Mapping_Ccgdd16Sx91Extended.tsv'
        self.FDAdrugsFile = '../../LabResources/mapped/DrugBank/full database_12132016_FDAdrugsFormatedmappedFiltered.txt'
        self.MeSH_name_file1 = '/lab/cedar/home/LabData/Data/c2016.bin_dict'
        self.MeSH_name_file2 = '/lab/cedar/home/LabData/Data/d2016.bin_dict'
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self.validationNetworkNames = ['{}_{}_{}_PubMed'.format(self.name, self.selectedQueryMode, self.selectedPredictionMode)]
        self.FDAonly = True
        self.y_true = []
        self.y_score = []
        self.truePositiveN = []
        self.predictionsN = []
        self.resultDictPubMed = {}
        self.__dict__.update(kwargs)
        self.canonicalToSynonyms = pd.read_csv(self.mappingFile, sep="\t").set_index(
            'CanonicalID')['Synonyms'].to_dict()
        self.FDAdrugs = pd.read_csv(
            self.FDAdrugsFile, sep="\t", header=None)[0].values
        self.MeSH_name = pd.read_csv(
            self.MeSH_name_file1, sep="\t", header=None).set_index(0).to_dict()[1]
        MeSH_name2 = pd.read_csv(
            self.MeSH_name_file2, sep="\t", header=None).set_index(0).to_dict()[1]
        self.MeSH_name.update(MeSH_name2)
        self.nodePosition = network.getNodePosition()
        self.positionNode = network.getPositionNode()
        os.chdir(paths.NETWORK_TMP_DIR)
        self.inputAdjacencyMatrix = network.getAdjacencyMatrix()
        self.nodeNames = [self.positionNode[x].getCanonical() if '.' not in self.positionNode[x].getCanonical(
        ) else ";".join(self.positionNode[x].getRelSynonyms()) for x in range(self.inputAdjacencyMatrix.shape[1])]
        self.nodeDegree = [self.positionNode[x].getDegree()
                           for x in range(self.inputAdjacencyMatrix.shape[1])]
        self.nodeSynonyms = [self.canonicalToSynonyms[
            x] if x in self.canonicalToSynonyms else 'NA' for x in self.nodeNames]
        self.nodeSimpleName = []
        self.nodeFDA = [self.positionNode[x].getName() in self.FDAdrugs
                           for x in range(self.inputAdjacencyMatrix.shape[1])]
        for x in self.nodeNames:
            if x.replace("MESH:","") in self.MeSH_name:
                self.nodeSimpleName.append(self.MeSH_name[x.replace("MESH:","")].replace(' protein, human',''))
            else:
                tmp = x.split(';')
                if len(tmp) > 1:
                    # Filter out gene names with Ensembl ID e.g., 9606ENSP00000225245
                    tmp = [y.replace(' protein, human','') for y in tmp if (('GENE' not in y) and ('9606EN' not in y))]
                self.nodeSimpleName.append(";".join(tmp))

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
        return objStr

    def perform(self):
        '''
        Perform a leave a mode out experiment.

        This method performs the specified algorithm, trying to predict a
        specific type of edge from all other information in the network except
        for that edge type.

        '''
        if 'Diffusion' in str(self.algorithm):
            Alg = self.algorithm(self.inputNetwork, self.name, self.selectedQueryMode)
        elif 'AptRank' in str(self.algorithm):
            Alg = self.algorithm(self.inputNetwork, self.name,
                                 self.selectedQueryMode, self.selectedPredictionMode, param=self.param)
        else:
            raise NotImplementedError('{} has not been implemented yet'.format(self.algorithm))
        self.generateOutputPrefix(Alg)
        self.mapEntities()
        valid = time()
        self.dirResult = self.dirResult + self.outputPrefix + '/'
        if not os.path.exists(self.dirResult):
            os.mkdir(self.dirResult)
        os.chdir(self.dirResult)
        if not self.loadPerformance():
            # print 'Predicting {} entities for {} {} based on {}'.format(self.selectedPredictionMode, len(self.selectedEntityList), self.selectedQueryMode, self.network.getName())
            start = time()
            if((Alg.getName() == 'Diffusion') or (Alg.getName() == 'RndWalk')):
                nodes = self.inputNetwork.getNodes().keys()
                if(hasattr(self.selectedQueryMode, '__iter__')):
                    selectedLabels = set()
                    for x in self.selectedQueryMode:
                        mappedNodes, unmappedNodes = self.inputNetwork.getNodesOfType(
                            x)
                        selectedLabels |= set(mappedNodes + unmappedNodes)
                    labeledNodes = list(
                        selectedLabels.intersection(set(nodes)))
                    Alg.perform(labeledNodes=labeledNodes)
                else:
                    mappedNodes, unmappedNodes = self.inputNetwork.getNodesOfType(
                        self.selectedQueryMode)
                    labeledNodes = list(
                        set(mappedNodes + unmappedNodes).intersection(set(nodes)))
                    Alg.perform(labeledNodes=labeledNodes)
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
                # (pred - start) / 60))
            self.predictions = Alg.getresult()
            valid = time()
            # print('Predictions took {} min to retrieve'.format(
                # (valid - pred) / 60))
            # Compute performance metrics
            # print('Outputing predictions')
            self.outputPredictions()
            # print('Validating predictions')
            # self.validateAll()
            # print('Validation took {} min to retrieve'.format(
            #     (valid - pred) / 60))
            self.validate()
        # Switch to the result directory
        os.chdir(self.dirResult)
        # self.savePredictions()
        # self.savePerformance()
        # metrics = ['AUC_bootstrap','AUC_PR_bootstrap']
        # for m in metrics:
        #     self.plotBoxplot(m, self.selectedEntityInitName)
        os.chdir('..')
        end = time()
        # print('Predictions took {} min to validate'.format((end - valid) / 60))

    def mapEntities(self):
        fout = open('{}_missingEntity.txt'.format(self.outputPrefix), 'w')
        fout.write('Total #\tMissing #\tMissing entity\n')
        self.selectedNodeToInital = {}
        self.selectedNodes = []
        missingEntity = []
        missingEntityN = 0
        for x, y in zip(self.selectedEntityList, self.selectedEntityInitName):
            checkCanonical = [
                node for node in self.nodePosition.keys() if node.getCanonical() == y]
            if len(checkCanonical) == 1:
                if checkCanonical[0] in self.selectedNodes:
                    print node
                    print checkCanonical[0]
                    print x, y
                self.selectedNodes += checkCanonical
                self.selectedNodeToInital[checkCanonical[0]] = y
            else:
                checkNodeID = [
                    node for node in self.nodePosition.keys() if node.getName() == x]
                if len(checkNodeID) == 1:
                    if checkNodeID[0] in self.selectedNodes:
                        print node
                        print checkNodeID[0]
                        print x, y
                    self.selectedNodes += checkNodeID
                    self.selectedNodeToInital[checkNodeID[0]] = y
                else:
                    print x,y
                    missingEntityN += 1
                    missingEntity.append(y)
        fout.write('{}\t{}\t{}\n'.format(
            len(self.selectedEntityInitName), missingEntityN, ";".join(missingEntity)))
        fout.close()
        print '{} known entities are missing.'.format(missingEntityN)
        print '{} entities were mapped and labeled'.format(len(self.selectedNodes))
        return

    def outputPredictions(self):
        os.chdir(self.dirResult)
        predictions, known = self.selectPredicitons()
        predict_cols = self.inputNetwork.getModePosition(
            self.selectedPredictionMode)
        self.outputDf['ID'] = self.nodeNames[
            predict_cols[0]:predict_cols[1] + 1]
        # self.outputDf['Synonyms'] = self.nodeSynonyms[predict_cols[0]:predict_cols[1]+1]
        self.outputDf['Synonyms'] = self.nodeSimpleName[predict_cols[0]:predict_cols[1]+1]
        self.outputDf['Degree'] = self.nodeDegree[
            predict_cols[0]:predict_cols[1] + 1]
        nodeFDAorNot = self.nodeFDA[predict_cols[0]:predict_cols[1] + 1]
        for i in range(len(self.selectedNodes)):
            try:
                name = self.MeSH_name[self.selectedNodes[
                i].getCanonical().replace("MESH:", "")]
            except:
                names = self.selectedNodes[i].getRelSynonyms()
                try:
                    name = [x for x in names if (('GENE' not in x) and ('9606EN' not in x))][0]
                except:
                    name = names[0]
            name = name.replace(' protein, human','')
            self.outputDf['score_{}'.format(name)] = predictions[
                i, :].toarray()[0]
            self.outputDf['known_{}'.format(name)] = known[
                i, :].toarray()[0]
        if self.selectedPredictionMode == 'Chemical' and self.FDAonly:
            self.outputDf = self.outputDf.loc[nodeFDAorNot]
        self.outputDf.to_csv(self.outputPrefix +
                             '_pred.tsv', sep="\t", index=None)
        alg = self.outputPrefix.split('_')[0]
        if 'CTD2016CG_CTD2016CD_CTD2016DG_STRING10_900_MeSH2016DD_MeSH2016CC' in self.outputPrefix:
            net = 'CTD2016CG_CTD2016CD_CTD2016DG_STRING10_900_MeSH2016DD_MeSH2016CC'
            mode ='6'
        elif 'CTD2016CG_CTD2016CD_CTD2016DG' in self.outputPrefix:
            net = 'CTD2016CG_CTD2016CD_CTD2016DG'
            mode = '3'
        elif 'CTD2016CG' in self.outputPrefix:
            net = 'CTD2016CG'
            mode = '1'
        elif 'CTD2016CD' in self.outputPrefix:
            net = 'CTD2016CD'
            mode = '1'
        elif 'CTD2016DG' in self.outputPrefix:
            net = 'CTD2016DG'
            mode = '1'
        output_html(self.outputDf,alg,mode,net,self.selectedQueryMode,self.selectedPredictionMode)
        os.chdir('..')
        return

    def validate(self, topN=20, randomFold = 10, pThreshold=0.001, numThreads=50):
        os.chdir(self.dirResult)
        fout = open(self.outputPrefix + "_predTop{}_PM_random{}.tsv".format(topN,randomFold), 'w')
        fout.write("ID\tPred hit\tPred hit #\tPred %\tRandom hit #\tRandom %\tp_chisq\tIntersected hit #\tIntersection\tPred non-hit\n")
        obsAll = [[0, 0], [0, 0]]
        for i in range(1, (self.outputDf.shape[1]) / 2):
            col = self.outputDf.columns[i * 2+1]
            # Sort entities based on the score
            sortedDf = self.outputDf.sort_values(col, ascending=False)
            # Remove entities were known related
            sortedDf = sortedDf.loc[
                sortedDf[col.replace('score_', 'known_')] == 0]
            # Remove mapped entities without proper canonical names
            # print [x for x in sortedDf['ID'] if '.' in x]
            sortedDf = sortedDf.loc[['.' not in x for x in sortedDf['Synonyms']]]
            query = col.replace('score_', '').replace('Neoplasms', 'cancer').replace('Colonic','Colon')
            fout2 = open(self.outputPrefix + "_predTop{}_PM_random{}_{}.tsv".format(topN,randomFold,query.replace(' ','_').replace('Pulmonary_Disease,_Chronic_Obstructive','COPD')), 'w')
            term = [(query, x) for x in sortedDf.head(topN)['Synonyms']]
            results, self.resultDictPubMed = multithreadAssociation(term, numThreads=numThreads, load_path='../', save_path='../', precomputed_dict=self.resultDictPubMed)
            p = [x[0] for x in results]
            hit = np.array(p) < pThreshold
            randomTerm = []
            for i in range(randomFold):
                randomTerm += [(query, x) for x in sortedDf.sample(topN)['Synonyms']]
            intersectedPrediction = []
            for x in randomTerm:
                fout2.write("{}\n".format(x[1]))
                if x in term:
                    if hit[term.index(x)]:
                        intersectedPrediction.append(x[1])
            fout2.close()
            try:
                results_random, self.resultDictPubMed = multithreadAssociation(randomTerm, numThreads=numThreads, load_path='../', save_path='../', precomputed_dict=self.resultDictPubMed)
                p = [x[0] for x in results_random]
                randomHit = np.array(p) < pThreshold
            except:
                embed()
                sys.exit(0)
            obs = [[sum(hit), topN - sum(hit)],
                   [sum(randomHit), topN * randomFold - sum(randomHit)]]
            obsAll[0][0] += obs[0][0]
            obsAll[0][1] += obs[0][1]
            obsAll[1][0] += obs[1][0]
            obsAll[1][1] += obs[1][1]
            try:
                p = chi2_contingency(obs, correction=False)[1]
            except:
                p = 'NA'
            fout.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(term[0][0], ";".join([term[i][1] for i,x in enumerate(hit) if x]), sum(hit),
                                                         100.0 *
                                                         sum(hit) / len(hit),
                                                         sum(randomHit),
                                                         100.0 * sum(randomHit) / len(randomHit), p, len(intersectedPrediction),";".join(intersectedPrediction),
                                                         ";".join([term[i][1] for i,x in enumerate(hit) if not x])))
        p = chi2_contingency(obsAll, correction=False)[1]
        fout.write("ALL\t{}\t{}\t{}\t{}\t{}\t{}\n".format(obsAll[0][0],'NA',
                                                      100.0 *
                                                      obsAll[0][0] /
                                                      sum(obsAll[0]),
                                                      obsAll[1][0],
                                                      100.0 * obsAll[1][0] / sum(obsAll[1]), p))

        fout.close()
        os.chdir('..')
        return

    def validateAll(self):
        self.generateValidationNetwork()
        self.generatePredicitonAnswerArray()
        self.computeFprTprPrecisionMRR()
        return

    def generateValidationNetwork(self, q_threshold=0.1, numThreads=50):
        '''
        Creates input and answer network for validation.

        This method creates the starting graph and the target graph from the
        provided network.
        '''
        # Use answer matrix when it is given
        os.chdir(self.dirNetworks)
        source = 'PubMed'
        year = 2018
        weight = 1
        mdDict = {}
        weights = []
        edges = []
        if ((self.answerNetwork is None) and (self.inputNetwork is not None)):
            # Generate the network without a mode and the mode by itself
            # Try to reload saved oldNetwork and gainedEdges first
            if not self.loadValidationNetwork():
                # mapped, unmapped = self.inputNetwork.getNodesOfType(self.selectedPredictionMode)
                predict_cols = self.inputNetwork.getModePosition(
                    self.selectedPredictionMode)
                predictionNodes = [self.positionNode[x] for x in range(predict_cols[0],predict_cols[1]+1)]
                self.answerNetwork = GraphRepresentation5()
                self.answerNetwork.setNodes(predictionNodes+self.selectedNodes)
                pred = self.outputDf['Synonyms'].tolist()
                if self.selectedPredictionMode == 'Chemical' and self.FDAonly:
                    nodeFDAorNot = self.nodeFDA[predict_cols[0]:predict_cols[1] + 1]
                    predictionNodes = np.array(predictionNodes)[nodeFDAorNot]
                validatedQuery = set()
                for i,queryNode in enumerate(self.selectedNodes):
                    query = self.selectedEntityInitName[i].replace('Neoplasms', 'cancer').replace('Colonic','Colon')
                    if query in validatedQuery:
                        continue
                    else:
                        validatedQuery |= set([query])
                    # selectedRowPos = [self.nodePosition[x] for x in self.selectedNodes]
                    known = self.inputAdjacencyMatrix[self.nodePosition[queryNode], predict_cols[0]:predict_cols[1] + 1]
                    terms = [(query, x) for x in pred]
                    results, self.resultDictPubMed = multithreadAssociation(terms, numThreads=numThreads, load_path=self.dirResult+'/../', save_path=self.dirResult+'/../', precomputed_dict=self.resultDictPubMed)
                    p = [x[0] for x in results]
                    unknown = (known.toarray() == 0)[0]
                    if self.selectedPredictionMode == 'Chemical' and self.FDAonly:
                        unknown = unknown[nodeFDAorNot]
                    _reject, q, _alphacSidak, _alphacBonf = multipletests(p, method='fdr_bh')
                    p_unknown = np.array(p)[unknown]
                    predictionNodes_unknown = np.array(predictionNodes)[unknown]
                    _reject, q_unknown, _alphacSidak, _alphacBonf = multipletests(p_unknown, method='fdr_bh')
                    truePredNode = np.array(predictionNodes)[q< q_threshold]
                    truePredNode_unknown1 = np.array(predictionNodes)[(q< q_threshold) & (unknown)]
                    truePredNode_unknown2 = np.array(predictionNodes_unknown)[(q_unknown< q_threshold)]
                    # set(truePredNode_unknown1).difference(truePredNode_unknown2)
                    # for predNode in truePredNode:
                    for predNode in truePredNode_unknown2:
                        edge = queryNode.hasNeighbor(predNode)

                        edge = Edge(queryNode, predNode, directed=False)
                        edge.updateSource(source, year, weight, mdDict)
                        self.answerNetwork.addEdge(edge)
                        weights.append(weight)
                        edges.append(edge)
                    logging.info('Finish {} validation with {}'.format(query, self.selectedPredictionMode))
                self.answerNetwork.setName(self.validationNetworkNames[0])
                self.answerNetwork.normalizeEdgeWeights(weights, edges, source)
                self.answerNetwork.combineEdgeWeights()
                self.saveValidationNetwork()
        elif self.inputNetwork is None:
            raise ValueError('inputNetwork is required in the input!')
        else:
            print 'Both input network and answer network are given.'
        return

    def selectPredicitons(self):
        predict_cols = self.inputNetwork.getModePosition(
            self.selectedPredictionMode)
        selectedRowPos = [self.nodePosition[x] for x in self.selectedNodes]
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
            selectedRowPosNew = [predictionNodesNew[x]
                              for x in self.selectedNodes]
            predictions = self.predictions[selectedRowPosNew, :]
        else:
            predictions = self.predictions[selectedRowPos, :][
                :, predict_cols[0]:predict_cols[1] + 1]
        self.y_node1 = self.selectedNodes
        self.y_node2 = [self.positionNode[x]
                        for x in range(predict_cols[0], predict_cols[1] + 1)]
        known = self.inputAdjacencyMatrix[selectedRowPos, :][
            :, predict_cols[0]:predict_cols[1] + 1]
        return predictions, known

