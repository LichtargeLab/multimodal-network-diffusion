#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Nov 22, 2016

@author: daniel
'''
from GeneralParser import GeneralParser
from NodeV2 import Node
from EdgeV2 import Edge
from Queue import Queue, PriorityQueue
from scipy.sparse import csr_matrix
from itertools import combinations
from scipy.stats import rankdata
from csv import writer, reader
import cPickle as pickle
from time import time
import numpy as np
import operator
import os
import re
from IPython import embed


class GraphRepresentation5(object):
    ##########################################################################
    # Over writing class functions
    ##########################################################################
    '''
    classdocs
    '''

    def __init__(self, mappingFile=None, connectivity=[], normMethod=None,
                 filterNodes=True, checkForNet=False):
        '''
        Constructor

        This function generates an instance of a GraphRepresentation5 class.
        All variables can be instantiated with None and the class can then
        load it's data from a pickle file. Otherwise the data provided to this
        function will be processed to generate a graph.

        Parameters
        ----------
        mappingFile: str
            A string providing the file path to a mapping file.  This file
            should have the following format:
                -A header specifying the columns and starting with #
                -A column for IndexedID which should be a unique id for the
                entity and be preceded by the first letter of the entity type.
                -A column for CanonicalID which is the most recognized name for
                the entity described.
                -A column for Synonyms where each synonym is separated by ";;".
                -A column for MappingSource where each source is separated
                by "|".
                -A column for MetaData which starts with "(" and ends with ")".
                Each entry should be an attribute-value pair which is separated
                by "::" and each entry should be separated by ";;".
        connectivity: list
            This should be a list of three-tuples each of which specifies a
            single edge source and should have the following format:
                -source: A string specifying the name of the source being added.
                -filepath: A string specifying the path to the file where the
                data can be found.
                -directed: A boolean indicating if the edges found in this
                source are directed or not.
        normMethod: str
            A specification for the method by which edge weights should be
            normalized.  See normalizedEdgeWeights for a description of the
            current options.
        filter: bool
            Specifies whether or not to remove all nodes with a degree of 0.
            The default value of this parameter is True.

        Returns
        -------
        GraphRepresentation5
            An instance of this class.
        '''
        start = time()
        self.mappingFile = mappingFile
        self.uids = None
        self.canonicalIDs = None
        self.synonyms = None
        self.connectivitySources = connectivity
        self.nodeCount = 0
        self.edgeCount = 0
        self.mappedNodes = {}
        self.unmappedNodes = {}
        self.name = []
        if(checkForNet):
            mfName = '.'.join(self.mappingFile.split('/')[-1].split('.')[0:-1])
            connNames = [src[0] for src in connectivity]
            name = '_'.join([mfName] + connNames)
            name += '.pkl'
            if(os.path.exists(name)):
                self.loadGraph(name)
                return
        if(not(self.mappingFile is None)):
            self.parseMappingFile()
        if(not(self.connectivitySources == [])):
            for conn in self.connectivitySources:
                self.parseConnectivityData(conn[0], conn[1], conn[2],
                                           normMethod)
            self.combineEdgeWeights()
        self.name = '_'.join(self.name)
        if(filterNodes and
           ((not(self.mappingFile is None)) or
                (not(self.connectivitySources == [])))):
            self.deleteNodes(attr='Degree', value=0, save=False)
        self.saveGraph(pFN='{}.pkl'.format(self.name))
        self.writeOutGraphProperties()
        end = time()
        # print('Graph initialization took: {} min'.format((end - start) / 60.0))

    def __del__(self):
        '''
        Destructor

        This method deletes all data associated with an instance of the
        GraphRepresentation5 class.
        '''
        del(self.mappingFile)
        del(self.connectivitySources)
        del(self.nodeCount)
        del(self.edgeCount)
        del(self.mappedNodes)
        del(self.unmappedNodes)
        del(self.name)
        del(self)

    def __str__(self):
        '''
        To String

        This method generates an informative string to describe the current
        GraphRepresentation5 instance.

        Returns
        -------
        str
            A string detailing the name, node count, and edge count of the
            current instance.
        '''
        objStr = 'Graph object with name: {}'.format(self.name)
        objStr += '\nTotal Node Count: {}'.format(self.nodeCount)
        nodeCounts = self.getNodeCountByMode()
        for nType in nodeCounts:
            objStr += '\n\t{} Nodes: {}'.format(nType, nodeCounts[nType])
        objStr += '\nEdges: {}'.format(self.edgeCount)
        edgeCounts = self.getEdgeCountByMode()
        for eType in edgeCounts:
            objStr += '\n\t{} Edges: {}'.format(eType, edgeCounts[eType])
        return objStr

    def __repr__(self):
        '''
        To Representation

        This method generates an informative string to describe the current
        GraphRepresentation5 instance.

        Returns
        -------
        str
            A string detailing the name, node count, and edge count of the
            current instance.
        '''
        objStr = str(self)
        return objStr

    def __cmp__(self, other):
        '''
        Comparator

        This method compares to instances of the GraphRepresentation5 class.

        Parameters
        ----------
        other: GraphRepresentation5
            The other graph instance to compare to.

        Returns
        -------
        int
            If the first graph is considered smaller than the second graph then
            -1 is returned.  If the other graph passed in is considered smaller
            then 1 is returned.  If both graphs are equivalent then 0 is
            returned. Graph similarity is judged based on name, node count,
            and edge count.
        '''
        if(self.name < other.name):
            return -1
        elif(self.name > other.name):
            return 1
        else:
            if(self.nodeCount < other.nodeCount):
                return 1
            elif(self.nodeCount > other.nodeCount):
                return -1
            else:
                if(self.edgeCount < other.edgeCount):
                    return 1
                elif(self.edgeCount < other.edgeCount):
                    return -1
                else:
                    return 0

    ##########################################################################
    # Helper functions for initialization
    ##########################################################################

    def parseMappingFile(self):
        '''
        Parse mapping file

        Iterate over entries in the mapping file to create nodes.  The mapping
        file should have the format specified in the constructor.  If this
        function encounters two entries in the mapping file where the mode and
        the indexed ID are the same it will raise a value error and describe the
        collision. The count for the number of nodes created from this mapping
        file is used to set the node count. The mappedNodes dictionary which
        maps mode types to another dictionary mapping indexed IDs to the nodes
        representing them is also generated by this method.
        '''
        start = time()
        name = self.mappingFile.split('/')[-1]
        self.name.append('.'.join(name.split('.')[0:-1]))
        parser = GeneralParser(fileName=self.mappingFile, delimiter='\t',
                               numHeaderLines=1, skipChar='#',
                               columnNames=['IndexedID', 'CanonicalID',
                                            'Synonyms', 'MappingSource',
                                            'MetaData'])
        count, data = parser.parse()
        for i in range(count):
            indexedID = data['IndexedID'][i]
            mode = indexedID.split('.')[0]
            canonicalID = data['CanonicalID'][i]
            synonyms = list(set(data['Synonyms'][i].split(';;')))
            try:
                synonyms.remove('')
            except ValueError:
                pass
            sources = list(set(data['MappingSource'][i].split('|')))
            try:
                sources.remove('')
            except ValueError:
                pass
            metaData = data['MetaData'][i].lstrip('(').rstrip(')').split(';;')
            mdDict = {}
            for entry in metaData:
                if(entry == ''):
                    continue
                attVal = entry.split('::')
                try:
                    attVal.remove('')
                except ValueError:
                    pass
                if(len(attVal) != 2):
                    continue
                mdDict[attVal[0]] = attVal[1]
            node = Node(uID=indexedID, canonical=canonicalID, synonyms=synonyms,
                        relSynonyms=[],
                        mode=mode, sources=sources, metaData=mdDict)
            mode = node.getMode()
            if(mode not in self.mappedNodes):
                self.mappedNodes[mode] = {}
            if(indexedID not in self.mappedNodes[mode]):
                self.mappedNodes[mode][indexedID] = node
            else:
                errorString = 'Node already exists for mode:{} and id:{}!\n'
                errorString += 'Old Node: {}\nNew Node: {}'
                raise ValueError(errorString.format(mode, indexedID,
                                                    str(self.mappedNodes[mode]
                                                        [indexedID]),
                                                    str(node)))
        self.nodeCount = count
        end = time()
        # print('Parsing mapping took: {} min'.format((end - start) / 60.0))

    def parseConnectivityData(self, source, filePath, directed, normMethod):
        '''
        Parse in data from a connectivity file.

        Use the data in a connectivity file to generate new edges and if
        necessary new nodes.  New nodes are created if an unmapped node is
        encountered in the adjacency data provided in the connectivity file.
        Data from a new source for an existing edge is just added to the edge.
        This function also provides the components for the graph name.

        Parameters
        ----------
        source: str
            A string specifying the name of the source being added.
        filePath: str
            A string specifying the path to the connectivity file to be parsed.
            This file should have the following format:
            -A header
            -A column for Entity1, which specifies an entity's indexed ID
            -A column for Entity2, which specifies an entity's indexed ID
            -A column for Confidence, which specifies the edge weight as a float
            -A cloumn for MetaData, which specifies attribute-value pairs which
            are joined on "::" and which are separated by ";;".
        directed: bool
            Specify True if the edges from this source should be considered
            directed, specify False if they are undirected.
        '''
        start = time()
        weights = []
        edges = []
        if(source not in self.name):
            self.name.append(source)
        parser = GeneralParser(fileName=filePath, delimiter='\t',
                               numHeaderLines=1, skipChar='#',
                               columnNames=['Entity1', 'Entity2', 'Confidence',
                                            'MetaData'])
        count, data = parser.parse()
        for i in range(count):
            # Gather edge information
            entity1 = data['Entity1'][i]
            entity2 = data['Entity2'][i]
            weight = float(data['Confidence'][i])
            # print('Parsed weight: {}'.format(weight))
            metaData = data['MetaData'][i].replace(';;;;', ';;')
            metaData = metaData.replace('<undefined>', '')
            metaData = metaData.strip()
            metaData = list(set(metaData.split(';;')))
            mdDict = {}
            # Parse MetaData
            try:
                metaData.remove('')
            except ValueError:
                pass
            for entry in metaData:
                attVal = entry.split('::')
                if(entry == ''):
                    continue
                attVal = entry.split('::')
                try:
                    attVal.remove('')
                except ValueError:
                    pass
                if(len(attVal) != 2):
                    continue
                mdDict[attVal[0]] = attVal[1]
            # Retrieve required MetaData
            try:
                year = int(mdDict['Year'])
            except:
                # print entity1
                # print entity2
                # print weight
                # print metaData
                year = mdDict['Year']
            del(mdDict['Year'])
            type1 = mdDict['Entity1Label']
            del(mdDict['Entity1Label'])
            type2 = mdDict['Entity2Label']
            del(mdDict['Entity2Label'])
            # Get original name
            relSyn1 = mdDict['OriginalEntity1/conf'].split('/')[0]
            relSyn2 = mdDict['OriginalEntity2/conf'].split('/')[0]
            # Check for or create nodes
            node1 = self.findOrCreateUnmappedNode(entity1, type1, source)
            node2 = self.findOrCreateUnmappedNode(entity2, type2, source)
            # Set original name as relavent synonyms
            node1.setRelSynonyms(list(set(node1.getRelSynonyms() + [relSyn1])))
            node2.setRelSynonyms(list(set(node2.getRelSynonyms() + [relSyn2])))
            # Check for or create edge
            edge = node1.hasNeighbor(node2)
            if(edge is None):
                edge = Edge(node1, node2, directed)
                try:
                    self.addEdge(edge)
                except:
                    if(node1 == node2):
                        pass
                    else:
                        print node1, node2
                        raise ValueError('Edge already exists!')
            # Add edge source information
            edge.updateSource(source, year, weight, mdDict)
            # print('Edge weights:')
            # print edge.getEdgeWeights()
            weights.append(weight)
            edges.append(edge)
        self.normalizeEdgeWeights(weights, edges, source, normMethod)
        end = time()
        # print('Edge parsing took: {} min'.format((end - start) / 60.0))

    def normalizeEdgeWeights(self, weights, edges, source, method=None):
        '''
        Normalize edge weights

        This method iterates over all edges form a single source and then
        normalizes them so they fall between 0 and 1.  This normalization can
        be performed in different ways. Once normalized, the edge weights are
        assigned to the corresponding edges.

        Parameters
        ----------
        weights: list
            The list of edge weights to be normalized, these should all come
            from a single source.  The number of edge weights should be the
            same as the number of edges passed in with the next parameter.
        edges: list
            A list of EdgeV2 objects.  This should have the same length as
            the list of edge weights given in the previous parameter.
        source: str
            A string identifying which source the edge weights come from so that
            the normalized edge weights can be stored correctly in their
            corresponding edges.
        method: str
            This string specifies how to normalize the data.  If 'Percentile' is
            specified then all edges are ranked an each one is converted to a
            percentile based on its rank which is then stored in the
            corresponding edge. If None or any other value is specified all
            edge weights are divided by the largest edge weight. The result is
            then stored in the corresponding edge.
        '''
        start = time()
        if(method == 'Percentile'):
            norm = (rankdata(weights, 'max') /
                    float(len(weights)))
        else:
            maxVal = max(weights)
            norm = [weight / maxVal for weight in weights]
        for i in range(len(edges)):
            currNormalized = edges[i].getNormalizedEdgeWeights()
            currNormalized[source] = norm[i]
            edges[i].setNormalizedEdgeWeights(currNormalized)
        end = time()
        # print('Edge weight normalization took: {} min'.format(
        #     (end - start) / 60.0))

    def combineEdgeWeights(self):
        '''
        Combine edge weights.

        This method iterates over all edges represented by the graph object
        and combines the according to the function described in the EdgeV2 class
        named combineEdgeWeights.
        '''
        start = time()
        for status in ['Mapped', 'Unmapped']:
            if status == 'Mapped':
                nDict = self.mappedNodes
            else:
                nDict = self.unmappedNodes
            for mode in nDict:
                for uID in nDict[mode]:
                    for edge in nDict[mode][uID].getOutEdges():
                        edge.combineEdgeWeights()
        end = time()
        # print('Edge weight combination took: {} min'.format(
        #     (end - start) / 60.0))

    ##########################################################################
    # Serialization methods
    ##########################################################################

    def saveGraph(self, pFN):
        '''
        Save the current graph.

        This method uses the cPickle module with the HIGHEST_PROTOCOL to save
        the state of all of the variables in the GraphRepresentation.

        Parameters
        ----------
        pFN: str
            This parameter stands for pickle file name and should be given the
            path at which the file should be saved.
        '''
        start = time()
        if(not pFN.endswith('.pkl')):
            pFN += '.pkl'
        pFile = open(pFN, 'wb')
        pickle.dump((self.name, self.mappingFile, self.connectivitySources,
                     self.mappedNodes, self.unmappedNodes, self.nodeCount,
                     self.edgeCount),
                    pFile, protocol=pickle.HIGHEST_PROTOCOL)
        end = time()
        # print('Saving graph took: {} min'.format((end - start) / 60.0))

    def loadGraph(self, pFN):
        '''
        Load a previous graph.

        This method uses the cPickle module to load the state of a previously
        generated graph from a pickle file.  This is intended for use with
        empty graph instances as it will over-write all data currently in the
        graph.

        Parameters
        ----------
        pFN: str
            This parameter stands for pickle file name and should be given the
            path at which the file can be located and loaded from.
        '''
        start = time()
        if(not pFN.endswith('.pkl')):
            pFN += '.pkl'
        pFile = open(pFN, 'rb')
        storedData = pickle.load(pFile)
        self.name = storedData[0]
        self.mappingFile = storedData[1]
        self.connectivitySources = storedData[2]
        self.mappedNodes = storedData[3]
        self.unmappedNodes = storedData[4]
        self.nodeCount = storedData[5]
        self.edgeCount = storedData[6]
        end = time()
        # print('Loading graph took: {} min'.format((end - start) / 60.0))

    def writeOutGraphProperties(self, fn=None):
        '''
        Write Out Graph Properties

        This method writes the properties of the graph out to a file so that
        there is an easily accessible record of the features of each graph.
        Currently the properties provided in the file include: node count,
        mapped node count, unmapped node count, and edge count. The file created
        is a tab separated file where the property is listed in the left handed
        column and the value is listed in the right hand column.

        Parameters
        ----------
        fn: string
            A string providing the file name, or full file path where the
            created file should be stored.  This parameter is optional, if not
            specified the file will be written to the working directory and the
            name will be the name of the graph with the suffix '_properties.tsv'
        '''
        if(fn is None):
            fn = self.name + '_properties.tsv'
        handle = open(fn, 'wb')
        propWriter = writer(handle, delimiter='\t')
        # Write Name
        propWriter.writerow(['Name:', self.name])
        propWriter.writerow([])
        # Write Mapping File
        propWriter.writerow(['Mapping Source:', self.mappingFile])
        propWriter.writerow([])
        # Write Connectivity Sources
        numSources = len(self.connectivitySources)
        if(numSources == 0):
            propWriter.writerow(['Connectivity Sources', ''])
        else:
            for i in range(numSources):
                if(i == 0):
                    label = 'Connectivity Sources'
                else:
                    label = ''
                propWriter.writerow([label, self.connectivitySources[i][0],
                                     self.connectivitySources[i][1],
                                     self.connectivitySources[i][2]])
        propWriter.writerow([])
        # Write Node Counts
        propWriter.writerow(['Total Node Count:', self.nodeCount])
        propWriter.writerow(['Mapped Node Count:', self.getMappedNodeCount()])
        for mode in self.mappedNodes:
            propWriter.writerow(['Mapped {} Nodes:'.format(mode),
                                 self.getSpecificNodeCount('Mapped', mode)])
        propWriter.writerow(['Unmapped Node Count:',
                             self.getUnmappedNodeCount()])
        for mode in self.unmappedNodes:
            propWriter.writerow(['Unmapped {} Nodes:'.format(mode),
                                 self.getSpecificNodeCount('Unmapped', mode)])
        # Write Edge Counts
        propWriter.writerow(['Total Edge Count:', self.edgeCount])
        edgeTypeCounts = self.getEdgeCountByMode()
        for mode in edgeTypeCounts:
            propWriter.writerow(['{} Edges:'.format(mode),
                                 edgeTypeCounts[mode]])
        handle.close()

    ##########################################################################
    # Add and remove entity functions
    ##########################################################################

    def addMappedNode(self, node):
        '''
        Add a single mapped node.

        This function adds a single mapped node to the graph.  This assumes
        there are no edges attached to the node upon insertion.

        Parameters
        ----------
        node: NodeV2
        '''
        newNode = Node(uID=node.getUID(), canonical=node.getCanonical(),
                       synonyms=node.getSynonyms(), relSynonyms=node.getRelSynonyms(),
                       mode=node.getMode(),
                       sources=node.getSources(), metaData=node.getMetaData())
        mode = newNode.getMode()
        uID = newNode.getUID()
        if(self.hasUnmappedNode(newNode)):
            raise ValueError('The specified node already exists!')
        if(self.hasMappedNode(newNode)):
            pass
        else:
            if(not(mode in self.mappedNodes)):
                self.mappedNodes[mode] = {}
            self.mappedNodes[mode][uID] = newNode
            self.nodeCount += 1

    def addUnmappedNode(self, node):
        '''
        Add a single unmapped node.

        This function adds a single unmapped node to the graph.  This assumes
        there are no edges attached to the node upon insertion.

        Parameters
        ----------
        node: NodeV2
            The node to be added to the graph.
        '''
        newNode = Node(uID=node.getUID(), canonical=node.getCanonical(),
                       synonyms=node.getSynonyms(), relSynonyms=node.getRelSynonyms(),
                       mode=node.getMode(),
                       sources=node.getSources(), metaData=node.getMetaData())
        mode = newNode.getMode()
        uID = newNode.getUID()
        if(self.hasMappedNode(newNode)):
            raise ValueError('The specified node already exists!')
        if(self.hasUnmappedNode(newNode)):
            pass
        else:
            if(not(mode in self.unmappedNodes)):
                self.unmappedNodes[mode] = {}
            self.unmappedNodes[mode][uID] = newNode
            self.nodeCount += 1

    def addEdge(self, edge):
        '''
        Add a single edge.

        This function adds a single edge to the graph.  This requires that both
        nodes which are connected by this edge already exist in the graph.

        Parameters
        ----------
        edge: EdgeV2
            The edge to be added to the graph.
        '''
        key1 = edge.getNode1()
        node1 = self.findNode(key1[1], key1[0])
        if((node1[0] is None) or (node1[1] is None)):
            raise ValueError('A required node is missing! {}:{}'.format(
                key1[0], key1[1]))
        key2 = edge.getNode2()
        node2 = self.findNode(key2[1], key2[0])
        if((node2[0] is None) or (node2[1] is None)):
            raise ValueError('A required node is missing! {}:{}'.format(
                key2[0], key2[1]))
        node1 = node1[0]
        node2 = node2[0]
        check1 = node1.addEdge(edge)
        check2 = node2.addEdge(edge)
        if(check1 or check2):
            self.edgeCount += 1
        else:
            raise ValueError('The specified edge already exists!')

    def deleteNode(self, node):
        '''
        Remove a single node.

        This function remove a single node from the graph.  This includes
        removing all edges attached to the node.

        Parameters
        ----------
        node: NodeV2
            The node to be deleted.

        Returns
        -------
        list
            A list of the edges which had been attached to this node.
        '''
        mode = node.getMode()
        uID = node.getUID()
        edges = node.getInEdges() + node.getOutEdges()
        for edge in edges:
            try:
                self.deleteEdge(edge)
            except:
                if(edge.getNode1() == edge.getNode2()):
                    continue
                else:
                    raise ValueError(
                        'The specified edge was already removed from')
            #     embed()
        if(self.hasMappedNode(node)):
            del(self.mappedNodes[mode][uID])
        elif(self.hasUnmappedNode(node)):
            del(self.unmappedNodes[mode][uID])
        else:
            raise ValueError('Node is not present in graph!')
        self.nodeCount -= 1
        return edges

    def deleteNodes(self, attr, value, op=None, save=True):
        '''
        Remove nodes from the graph.

        This method removes nodes from the graph based on a specific parameter.
        All edges belonging to a removed node are also removed.

        Parameters
        ----------
        attr: str
            The node attribute on which to base deletion.  The currently
            supported attributes are: 'In-Degree', 'Out-Degree', 'Degree',
            'Mode', and 'Node'.
        value: str or int
            The value for which to check on the given attribute in a node.  For
            the degree based options the expected input is an integer.  Any
            node with a degree less than or equal to that specified will be
            removed.  For any other attribute option a string value is expected.
            If 'Mode' is specified any node which has the specified mode will be
            deleted. If 'Node' is specified the value is expected to be a list
            of NodeV2 objects to keep in the graph.

        Returns
        -------
        GraphRepresentation5
            An instance of the graph class containing removed nodes and any
            edges supported by those nodes which were also removed.
        list
            A list of edges which are no longer represented in the current graph
            and which could not be represented in the graph containing deleted
            nodes.  These edges constitute those which have nodes in both
            the graphs.
        '''
        start = time()
        if op is None:
            if(attr in ['In-Degree', 'Out-Degree', 'Degree']):
                op = getattr(operator, 'le')
            elif(attr == 'Node'):
                op = getattr(operator, 'contains')
            else:
                raise NotImplementedError(
                    'Node removal on this attribute has not been implemented!')
        else:
            if not hasattr(operator, op):
                raise ValueError('Invalid operator passed in: {}'.format(op))
            else:
                op = getattr(operator, op)
        # Intiailize graph to accept deleted nodes and edges
        otherGraph = GraphRepresentation5()
        otherGraph.setName(self.name + '_{}_{}_Deleted_Nodes'.format(attr,
                                                                     value))
        otherGraph.setMappingFile(self.mappingFile)
        otherGraph.setConnectivitySources(self.connectivitySources)
        # Create set to track all edges deleted during the node removal process
        tmpEdges = set()
        # Initialize a Queue with all nodes so that each node is checked at
        # least once for the desired condition.
        nodeQueue = Queue()
        deleteQueue = Queue()
        map(nodeQueue.put, self.getNodes().keys())
        # As long as there are nodes in the queue check whether the next node
        # meets the condition and should be removed.
        while(not nodeQueue.empty()):
            node = nodeQueue.get_nowait()
            # Skip any node not in a mode of interest.
            if((attr == 'Mode') and (not node.hasMode(value))):
                continue
            # Perform check for deletion
            check = False
            if(node.getDegree() == 0):
                check = True
            elif(attr in ['In-Degree', 'Out-Degree', 'Degree']):
                if(attr == 'In-Degree'):
                    check = node.getInDegree()
                elif(attr == 'Out-Degree'):
                    check = node.getOutDegree()
                else:
                    check = node.getDegree()
#                 if(check <= value):
                if(op(check, value)):
                    check = True
                else:
                    check = False
            elif(attr == 'Mode'):
                check = True
            elif(attr == 'Node'):
                #                 check = (node in value)
                check = (op(value, node))
            else:
                raise NotImplementedError(
                    'Node removal on this attribute has not been implemented!')
            # If check comes back true
            if(check):
                deleteQueue.put_nowait(node)

            if(nodeQueue.empty()):
                while(not deleteQueue.empty()):
                    dNode = deleteQueue.get_nowait()
                    # Identify node status, or skip any node which has already
                    # been deleted.
                    if(self.hasMappedNode(dNode)):
                        status = 'Mapped'
                    elif(self.hasUnmappedNode(dNode)):
                        status = 'Unmapped'
                    else:
                        continue
                    # Add the node to the other graph
                    if(status == 'Mapped'):
                        otherGraph.addMappedNode(dNode)
                    else:
                        otherGraph.addUnmappedNode(dNode)
                    # Add all edges to the set of edges for removal
                    edges = self.deleteNode(dNode)
                    tmpEdges |= set(edges)
                    # Add all neighbors which are still in the graph, back to the
                    # queue so they can be checked again for removal.
                    for edge in edges:
                        keys1 = edge.getNode1()
                        node1 = self.findNode(keys1[1], keys1[0])[0]
                        if((node1 is not None) and (node1 != node)):
                            nodeQueue.put(node1)
                        else:
                            keys2 = edge.getNode2()
                            node2 = self.findNode(keys2[1], keys2[0])[0]
                            if((node2 is not None) and (node2 != node)):
                                nodeQueue.put(node2)
        # Attempt to add removed edges to the other graph. If not possible,
        # store in a list to be returned.
        lostEdges = []
        for edge in tmpEdges:
            try:
                otherGraph.addEdge(edge)
            except ValueError:
                lostEdges.append(edge)
        # Save only if asked to do so.
        if(save):
            otherGraph.saveGraph(otherGraph.getName())
            otherGraph.writeOutGraphProperties()
            newName = self.name + '_{}_{}_Retained_Nodes'.format(attr, value)
            self.setName(newName)
            self.saveGraph(newName)
            self.writeOutGraphProperties()
        end = time()
        # print('Deleting nodes took: {} min'.format((end - start) / 60.0))
        return (otherGraph, lostEdges)

    def deleteEdge(self, edge):
        '''
        Remove a single edge.

        This function removes a single edge by deleting it from both nodes edge
        lists to which it is attached.

        Parameters
        ----------
        edge: EdgeV2
            The edge which should be removed.

        Returns
        -------
        list
            A list containing the two nodes which were connected by this edge.
        '''
        keys1 = edge.getNode1()
        node1 = self.findNode(keys1[1], keys1[0])
        if((node1[0] is None) or (node1[1] is None)):
            # print edge
            # print keys1
            raise ValueError('Node already missing for edge!')
        check1 = node1[0].removeEdge(edge)
        keys2 = edge.getNode2()
        node2 = self.findNode(keys2[1], keys2[0])
        if((node2[0] is None) or (node2[1] is None)):
            # print edge
            # print keys2
            raise ValueError('Node already missing for edge!')
        check2 = node2[0].removeEdge(edge)
        if(check1 or check2):
            self.edgeCount -= 1
        else:
            raise ValueError(
                'The specified edge was already removed from: {}!'.format(edge))
        return [node1[0], node2[0]]

    def deleteEdges(self, attr, value, name1=None, name2=None):
        '''
        Remove edges from the graph.

        This method removes edges from the graph based on a specific parameter.
        The edges are removed from the nodes to which they are attached.  Any
        node which reaches a degree of 0 through this operation is also removed.

        Parameters
        ----------
        attr: str
            The edge attribute on which to base deletion.  The currently
            supported attributes are: 'Mask', 'Weight', 'Mode', 'Source',
            and 'Year'.
        value: str or int
            The value for which to check on the given attribute in an edge.  For
            the Mask option, the value is expected to be a csr_matrix
            (scipy.sparse) containing non-zero values at every edge position
            which should be deleted.  This matrix should have the same
            dimensions as the adjacency matrix of the graph. For
            the weight based option the expected input is an integer.  Any
            edge with a weight less than or equal to that specified will be
            removed.  for the year option the expected input is also an integer.
            In this case edges first seen at or after (greater than or equal to)
            the specified year will be removed. For any other attribute option a
            string value is expected. If 'Mode' is specified any edge which has
            the specified mode will be deleted. A mode is expected to be in
            format '{Type}-{Type}' where Type is an entity type.
        name1: str
            The name of the graph containing only the deleted edges. If not
            provided a name will be generated for the graph.
        name2: str
            The name of the graph containing all remaining edges. If not
            provided a name will be generated for the graph.
        Returns
        -------
        GraphRepresentation5
            An instance of the graph class containing removed edges and their
            corresponding nodes.
        '''
        start = time()
        nodePos = self.getNodePosition()
        edgesToRemove = []
        otherGraph = GraphRepresentation5()
        if(attr == 'Mask'):
            saveVal = repr(value) + '_' + str(len(value.nonzero()[0]))
        else:
            saveVal = value
        if(name1 is None):
            reName1 = self.name + '_{}_{}_Deleted_Edges'.format(attr,
                                                                saveVal)
        else:
            reName1 = name1
        otherGraph.setName(reName1)
        otherGraph.setMappingFile(self.mappingFile)
        otherGraph.setConnectivitySources(self.connectivitySources)
        for status in ['Mapped', 'Unmapped']:
            if(status == 'Mapped'):
                nodes = self.mappedNodes
            else:
                nodes = self.unmappedNodes
            for mode in nodes:
                if((attr == 'Mode') and (not (mode in value))):
                    continue
                for uID in nodes[mode]:
                    node1 = nodes[mode][uID]
                    outEdges = node1.getOutEdges()
                    for edge in outEdges:
                        key2 = edge.getNode2()
                        node2 = self.findNode(key2[1], key2[0])
                        if(attr == 'Mask'):
                            pos1 = nodePos[node1]
                            pos2 = nodePos[node2[0]]
                            lower = (value[pos1, pos2] != 0)
                            upper = (value[pos2, pos1] != 0)
                            check = (lower or upper)
                        elif(attr == 'Weight'):
                            combinedWeight = edge.getCombinedEdgeWeight()
                            check = (combinedWeight
                                     < value)
                        elif(attr == 'Mode'):
                            types = value.split('-')
                            check = edge.hasModes(type1=types[0],
                                                  type2=types[1])
                        elif(attr == 'Source'):
                            includesSource = edge.inSource(value)
                            numSources = edge.getNumSources()
                            check = (
                                includesSource and (numSources == 1))
                        elif(attr == 'Year'):
                            year = int(value)
                            earliest = edge.getEarliestYear()
                            check = (earliest >= year)
                        else:
                            raise NotImplementedError(
                                'Edge removal on this attribute has not been implemented!')
                        if(check):
                            edgesToRemove.append(edge)
                            # key2 = edge.getNode2()
                            # node2 = self.findNode(key2[1], key2[0])
                            if(status == 'Mapped'):
                                otherGraph.addMappedNode(node1)
                            else:
                                otherGraph.addUnmappedNode(node1)
                            if(node2[1] == 'Mapped'):
                                otherGraph.addMappedNode(node2[0])
                            else:
                                otherGraph.addUnmappedNode(node2[0])
        for edge in edgesToRemove:
            try:
                self.deleteEdge(edge)
            except ValueError:
                print('Edge no longer attached to node:\n{}'.format(str(edge)))
            try:
                otherGraph.addEdge(edge)
            except ValueError:
                print('Edge already attached to node in new graph:\n{}'.format(
                    str(edge)))
        otherGraph.saveGraph(otherGraph.getName())
        otherGraph.writeOutGraphProperties()
        # self.deleteNodes(attr='Degree', value=0, save=False)
        if(name1 is None):
            reName2 = self.name + '_{}_{}_Retained_Edges'.format(attr, saveVal)
        else:
            reName2 = name2
        self.setName(reName2)
        self.saveGraph(reName2)
        self.writeOutGraphProperties()
        end = time()
        # print('Deleting edges took: {} min'.format((end - start) / 60.0))
        return otherGraph

    ###########################################################################
    # Check for entity methods
    ###########################################################################

    def hasMappedNode(self, node):
        '''
        Check for a mapped nodes.

        Checks whether a node is present in the mapped nodes dictionary.

        Parameters
        ----------
        node: NodeV2
            The node to check for in the mappedNodes dictionary.

        Returns
        -------
        bool
            Returns True if the node was found in the mappedNodes dictionary,
            otherwise returns False.
        '''
        mode = node.getMode()
        uID = node.getUID()
        if(mode in self.mappedNodes):
            if(uID in self.mappedNodes[mode]):
                return True
        return False

    def hasUnmappedNode(self, node):
        '''
        Check for an unmapped nodes.

        Checks whether a node is present in the unmapped nodes dictionary.

        Parameters
        ----------
        node: NodeV2
            The node to check for in the unmappedNodes dictionary.

        Returns
        -------
        bool
            Returns True if the node was found in the unmappedNodes dictionary,
            otherwise returns False.
        '''
        mode = node.getMode()
        uID = node.getUID()
        if(mode in self.unmappedNodes):
            if(uID in self.unmappedNodes[mode]):
                return True
        return False

    def findNode(self, mode, uID):
        '''
        Find node in graph

        This method can be used to find a node in the graph.

        Parameters
        ----------
        mode: str
            The entity mode to look in for the node.
        uID: str
            The entity uid to search for.

        Returns
        -------
        NodeV2:
            The node object if it could be found.  Otherwise None is returned.
        str
            The status of the node object.  Either 'Mapped', 'Unmapped', or
            None can be returned.
        '''
        if((mode in self.mappedNodes) and
           (uID in self.mappedNodes[mode])):
            return (self.mappedNodes[mode][uID], 'Mapped')
        elif((mode in self.unmappedNodes) and
             (uID in self.unmappedNodes[mode])):
            return (self.unmappedNodes[mode][uID], 'Unmapped')
        else:
            return (None, None)

    def findOrCreateUnmappedNode(self, entity, mode, source):
        '''
        Find a node in the mapped or unmapped dictionary.

        This method checks for a node in the mapped and unmapped node
        dictionaries based on its mode and its uID.  If no node can be found
        based on that data a new node is created and an attempt is made to
        add it to the unmapped dictionary.  If this fails an error is raised
        though it should not be possible for this to occur given the first
        check for the nodes presence in that dictionary.

        Parameters
        ----------
        entity: str
            The unique identifier by which a node can be found.
        mode: str
            The mode which describes the node's entity type.
        source: str
            The source from which the entity has been identified.  This is only
            used in the case that a new node must be created.
        '''
        node, status = self.findNode(mode, entity)
        if((node is None) and (status is None)):
            modes = self.mappedNodes.keys()
            modes.remove(mode)
            originalMode = None
            for m in modes:
                tmpNode = Node(uID=entity, canonical=None, relSynonyms=[],
                               synonyms=[], mode=m, sources=[], metaData={})
                if(self.hasMappedNode(tmpNode)):
                    originalMode = m
            if(originalMode is not None):
                originalNode = self.mappedNodes[originalMode][entity]
                node = Node(uID=originalNode.uID,
                            canonical=originalNode.canonical,
                            synonyms=originalNode.synonyms,
                            relSynonyms=originalNode.relSynonyms,
                            mode=mode, sources=originalNode.sources + [source],
                            metaData=originalNode.metaData)
            else:
                node = Node(uID=entity, canonical=entity, synonyms=[],
                            relSynonyms=[], mode=mode, sources=[source],
                            metaData={})
            if(not(mode in self.unmappedNodes)):
                self.unmappedNodes[mode] = {}
            if(not(entity in self.unmappedNodes[mode])):
                self.unmappedNodes[mode][entity] = node
                self.nodeCount += 1
            else:
                errorString = 'Node already exists for mode:{} and id:{}!\n'
                errorString += 'Old Node: {}\nNew Node: {}'
                raise ValueError(errorString.format(mode, entity,
                                                    str(self.unmappedNodes[mode]
                                                        [entity]),
                                                    str(node)))
        return node

    ###########################################################################
    # Subnetwork generation
    ###########################################################################

    def generateSubNetworkFromEdgesAndNodes(self, name, edgeList, nodeList):
        '''
        Generate a subnetwork.

        This method creates a subnetwork based on a list of nodes and edges.

        Parameters
        ----------
        name: name of new network
        edgeList: iterable of EdgeV2 objects.
        nodeList: iterable of NodeV2 objects.

        Returns
        -------
        GraphRepresentation5
            A graph object with nodes and edges specified in the parameters.
        '''
        newGraph = GraphRepresentation5()
        newGraph.setName(name)
        newGraph.setMappingFile(self.mappingFile)
        newGraph.setConnectivitySources(self.connectivitySources)
        for node in nodeList:
            uID = node.getUID()
            check = re.match(r'^[a-zA-Z]\.\d+$', uID)
            newNode = Node(uID=node.getUID(), canonical=node.getCanonical(),
                           synonyms=node.getSynonyms(), relSynonyms=node.getRelSynonyms(),
                           mode=node.getMode(),
                           sources=node.getSources(), metaData=node.getMetaData())
            if(check):
                newGraph.addMappedNode(newNode)
            else:
                newGraph.addUnmappedNode(newNode)
        for edge in edgeList:
            try:
                newGraph.addEdge(edge)
            except:
                pass
                # print('Nodes missing not adding edge: {}'.format(edge))
        # newGraph.deleteNodes(attr='Degree', value=0, save=False)
        newGraph.saveGraph(name)
        newGraph.writeOutGraphProperties()
        return newGraph

    def generateSubNetworkFromEdges(self, name1, name2, edgeList):
        '''
        Generate Subnetwork From Edges

        This method uses an edge list to generate two new networks from the
        current network.  One network contains only edges in the provided edge
        list while the second network contains all other edges.

        Parameters
        ----------
        name1: str
            Name which should be used for the network containing only edges from
            the provided edge list.
        name2: str
            Name which should be used for the network containing edges not in
            the provided edge list.
        edgeList: list
            Edges on which to split the current network.

        Returns
        -------
        GraphRepresentation5
            The graph consisting only of edges from the edge list and their
            supporting nodes.
        GraphRepresentation5
            The graph consisting of all edges not in the edge list and their
            supporting nodes.
        '''
        start = time()
        nodes = self.getNodes().keys()
        otherEdges = set(self.getEdges().keys()) - set(edgeList)
        graph1 = self.generateSubNetworkFromEdgesAndNodes(name=name1,
                                                          edgeList=edgeList,
                                                          nodeList=nodes)
        graph2 = self.generateSubNetworkFromEdgesAndNodes(name=name2,
                                                          edgeList=otherEdges,
                                                          nodeList=nodes)
        end = time()
        # print('Generating graphs from edge list took {} min.'.format(
        #     (end - start) / 60))
        return graph1, graph2

    def determineOverlappingNodes(self, other):
        '''
        Determine Overlapping Nodes

        This function takes the graph on which the function was called and
        another graph and forms a venn diagram out of the nodes in each graph.
        This allows it to return nodes which are in the first, both, or the
        second graph.

        Parameters
        ----------
        other: GraphRepresentation5
            Another instance of the GraphRepresentation5 class against which to
            check for overlaps.

        Returns
        -------
        list
            A set of nodes which is only in the graph on which this function was
            called.
        list
            A set of nodes which is in both the starting graph and the provided
            graph.
        list
            A set of nodes which is only in the graph which was passed to this
            function.
        '''
        sNodes = set(self.getNodes().keys())
        oNodes = set(other.getNodes().keys())
        intersection = sNodes.intersection(oNodes)
        sNodes = list(sNodes - intersection)
        oNodes = list(oNodes - intersection)
        intersection = list(intersection)
        return sNodes, intersection, oNodes

    def determineOverlappingEdges(self, other):
        '''
        Determine Overlapping Edges

        This function takes the graph on which the function was called and
        another graph and forms a venn diagram out of the edges in each graph.
        This allows it to return edges which are in the first, both, or the
        second graph.

        Parameters
        ----------
        other: GraphRepresentation5
            Another instance of the GraphRepresentation5 class against which to
            check for overlaps.

        Returns
        -------
        list
            A set of edges which is only in the graph on which this function was
            called.
        list
            A set of edges which is in both the starting graph and the provided
            graph.
        list
            A set of edges which is only in the graph which was passed to this
            function.
        '''
        sEdges = set(self.getEdges().keys())
        oEdges = set(other.getEdges().keys())
        intersection = sEdges.intersection(oEdges)
        sEdges = list(sEdges - intersection)
        oEdges = list(oEdges - intersection)
        intersection = list(intersection)
        return sEdges, intersection, oEdges

    def determineOverlap(self, other):
        '''
        Determine Overlap

        This method takes another graph object and creates a new graph which
        contains both the overlapping nodes and edges from this graph and the
        provided graph.  In addition the name is formed from the names of the
        two graphs to keep track of the origin of the new graph.  The mapping
        file and connectivity sources from each graph are also joined to
        provide the background of the new graph.

        Parameters
        ----------
        other:GraphRepresentation5
            The graph to join with this graph.

        Returns
        -------
        GraphRepresentation5
            A new graph containing the nodes and edges which exist in both this
            graph and the provided graph. In addition the mapping files,
            connectivity sources, and names of the two graphs are joined in
            the new graph.
        '''
        # Create empty graph to fill in
        newGraph = GraphRepresentation5()
        # Create name for new graph
        sNameComp = self.name.split('_')
        oNameComp = other.name.split('_')
        name = ('_'.join(list(set(sNameComp).union(oNameComp))) +
                '_intersection')
        newGraph.setName(name)
        # Set mapping file
        if(self.mappingFile == other.mappingFile):
            newGraph.setMappingFile(self.mappingFile)
        else:
            newGraph.setMappingFile(self.mappingFile + '_AND_' +
                                    other.mappingFile)
        # Set connectivity sources
        newGraph.setConnectivitySources(
            list(set(self.connectivitySources).union(
                set(other.connectivitySources))))
        # Find overlapping nodes
        # embed()
        oNodes = other.getNodes()
        for node in oNodes:
            if(self.hasMappedNode(node)):
                newGraph.addMappedNode(node)
                sNode = self.mappedNodes[node.getMode()][node.getUID()]
            elif(self.hasUnmappedNode(node)):
                newGraph.addUnmappedNode(node)
                sNode = self.unmappedNodes[node.getMode()][node.getUID()]
            else:
                continue
            # Find overlapping edges from an overlapping node
            edges = node.getOutEdges()
            for edge in edges:
                keys2 = edge.getNode2()
                node2 = other.findNode(keys2[1], keys2[0])[0]
                if(sNode.hasNeighbor(node2)):
                    if((not newGraph.hasMappedNode(node2)) and
                       self.hasMappedNode(node2)):
                        newGraph.addMappedNode(node2)
                    elif((not newGraph.hasUnmappedNode(node2)) and
                         self.hasUnmappedNode(node2)):
                        newGraph.addUnmappedNode(node2)
                    else:
                        pass
                    newGraph.addEdge(edge)
        # newGraph.deleteNodes(attr='Degree', value=0, save=False)
        newGraph.saveGraph(name)
        newGraph.writeOutGraphProperties()
        return newGraph

    def generateKFoldSplitsOnExistingEdges(self, k):
        '''
        Generate K Fold Splits

        Parameters
        ----------
        k: int
            The number of partitions to separate the network into.

        Returns
        -------
        dict
            A dictionary mapping a split number to a matrix mask where all edges
            to consider are marked with a 1 and all other edges are 0. This
            tracks not only existing edges, but all possible edges.
        '''
        splitDict = {x: ([], []) for x in range(k)}
        # Store node position for recomputing indices
        nodePos = self.getNodePosition()
        if(k == self.edgeCount):
            # print('Generating leave an edge out')
            edges = self.getEdges().keys()
            for i in range(k):
                edge = edges[i]
                keys1 = edge.getNode1()
                node1 = self.findNode(keys1[1], keys1[0])[0]
                pos1 = nodePos[node1]
                keys2 = edge.getNode2()
                node2 = self.findNode(keys2[1], keys2[0])[0]
                pos2 = nodePos[node2]
                # Enforcing upper triangle
                if(pos1 < pos2):
                    temp = pos2
                    pos2 = pos1
                    pos1 = temp
                splitDict[i][0].append(pos1)
                splitDict[i][1].append(pos2)
        else:
            # print('Generating k-fold cross validation')
            edgeCounts = self.getEdgeCountByMode()
            testing = 0
            for mode in edgeCounts:
                if(edgeCounts[mode] == 0):
                    continue
                edges = self.getEdgesOfType(mode)
                currCounts = {i: (edgeCounts[mode] / k) for i in range(k)}
                remainder = edgeCounts[mode] % k
                for i in range(k):
                    if(i < remainder):
                        currCounts[i] += 1
                    toPick = np.random.choice(a=range(len(edges)),
                                              size=currCounts[i], replace=False)
                    currEdges = set([edges[x] for x in toPick])
                    for edge in currEdges:
                        keys1 = edge.getNode1()
                        node1 = self.findNode(keys1[1], keys1[0])[0]
                        pos1 = nodePos[node1]
                        keys2 = edge.getNode2()
                        node2 = self.findNode(keys2[1], keys2[0])[0]
                        pos2 = nodePos[node2]
                        # Enforcing upper triangle
                        if(pos1 < pos2):
                            temp = pos2
                            pos2 = pos1
                            pos1 = temp
                        splitDict[i][0].append(pos1)
                        splitDict[i][1].append(pos2)
                    testing += len(currEdges)
                    edges = list(set(edges) - currEdges)
            # print(
            #     'Number of edges seen in k fold cross validation = {}'.format(testing))
        for k in splitDict:
            mat = csr_matrix(([1] * len(splitDict[k][0]),
                              (splitDict[k][0], splitDict[k][1])),
                             shape=(self.nodeCount, self.nodeCount))
            splitDict[k] = mat
        return splitDict

    def generateKFoldSplits(self, k):
        '''
        Generate K Fold Splits

        Parameters
        ----------
        k: int
            The number of partitions to separate the network into.

        Returns
        -------
        dict
            A dictionary mapping a split number to a matrix mask where all edges
            to consider are marked with a 1 and all other edges are 0. This
            tracks not only existing edges, but all possible edges.
        '''
        start = time()
        # Define results structure
        splitDict = {x: ([], []) for x in range(k)}
        # print splitDict
        # Identify all possible edge types
        edgeCounts = self.getEdgeCountByMode()
        # Store node position for recomputing indices
        nodePos = self.getNodePosition()
        # For each edge type
        for edgeType in edgeCounts:
            if(edgeCounts[edgeType] == 0):
                continue
            # Get all nodes that are relevant to the edge type
            types = edgeType.split('-')
            t1 = types[0]
            t1Nodes = self.getNodesOfType(t1)
            t1Nodes = t1Nodes[0] + t1Nodes[1]
            t1Size = len(t1Nodes)
            t2 = types[1]
            if(t2 != t1):
                t2Nodes = self.getNodesOfType(t2)
                t2Nodes = t2Nodes[0] + t2Nodes[1]
                t2Size = len(t2Nodes)
            else:
                t2Nodes = t1Nodes
                t2Size = t1Size
            # Identify their real indices
            t1Indices = [nodePos[x] for x in t1Nodes]
            t2Indices = [nodePos[x] for x in t2Nodes]
            # Index mapping from original indices to current matrix
            t1Map = {i: t1Indices[i] for i in range(t1Size)}
            t2Map = {i: t2Indices[i] for i in range(t2Size)}
            # A matrix with a all possible edges randomly assigned.
            assignments = np.random.randint(low=0, high=k,
                                            size=(t1Size, t2Size))
            # Find all edges to put into a specific split
            for s in splitDict:
                toAdd = np.where(assignments == s)
                for i in range(len(toAdd[0])):
                    splitDict[s][0].append(t1Map[toAdd[0][i]])
                    splitDict[s][1].append(t2Map[toAdd[1][i]])
        for m in splitDict:
            mat = csr_matrix(([1] * len(splitDict[m][0]),
                              (splitDict[m][0], splitDict[m][1])),
                             shape=(self.nodeCount, self.nodeCount))
            splitDict[m] = mat
        end = time()
        # print('Generating K Splits took {} min'.format((end - start) / 60.0))
        return splitDict

    def identifyNodesInSplit(self, split):
        '''
        Identify the Nodes in a K-Fold split

        This method takes a k-fold split and returns the nodes involved in that
        split.  This method is intended for use in k-fold cross validation
        experiments using the diffusion and random walk algorithms so that the
        nodes for which to perform the algorithm are provided.

        Parameters
        ----------
        split: csr_matrix
            A matrix with the dimensions of the adjacency matrix, containing
            edges of interest with non-zero values.

        Returns
        -------
        list
            A list of node objects from this graph which map to the positions
            in which there
        '''
        edges = split.nonzero()
        indices = list(set(list(edges[0])).union(set(list(edges[1]))))
        posNode = self.getPositionNode()
        nodes = [posNode[x] for x in indices]
        return nodes

    ###########################################################################
    # Network characterization
    ###########################################################################

    def degreeBasedDistributions(self):
        '''
        Degree Based Distributions

        Iterates over the graph to determine the distribution of nodes and edges
        accounted for by each degree.

        Returns:
        --------
        dict
            A dictionary where the keys are node degrees and the values are
            counts of how many nodes have that degree.
        dict
            A dictionary where the keys are the node degrees and the values are
            the number of edges in which nodes of that type are involved
            (degree x count).
        '''
        degreeNC = {}
        degreeEC = {}
        for status in ['Mapped', 'Unmapped']:
            if(status == 'Mapped'):
                nDict = self.mappedNodes
            else:
                nDict = self.unmappedNodes
            for mode in sorted(nDict.keys()):
                for uID in sorted(nDict[mode].keys()):
                    node = nDict[mode][uID]
                    degree = node.getDegree()
                    if degree not in degreeNC:
                        degreeNC[degree] = 1
                        degreeEC[degree] = degree
                    else:
                        degreeNC[degree] += 1
                        degreeEC[degree] += degree
        return degreeNC, degreeEC

    ###########################################################################
    # Getters
    ###########################################################################

    def getName(self):
        '''
        Get name

        Standard getter

        Returns
        -------
        str
            This method returns the name of the current graph instance.
            This method is not safe it returns a reference to the actual object.
        '''
        return self.name

    def getMappingFile(self):
        '''
        Get mapping file

        Standard getter

        Returns
        -------
        str
            This method returns the mapping file of the current graph instance.
            This method is not safe it returns a reference to the actual object.
        '''
        return self.mappingFile

    def getConnectivitySources(self):
        '''
        Get connectivity sources

        Standard getter

        Returns
        -------
        list
            This method returns the connectivity sources of the current graph
            instance. This method is not safe it returns a reference to the
            actual object.
        '''
        return self.connectivitySources

    def getMappedNodes(self):
        '''
        Get mapped nodes

        Standard getter

        Returns
        -------
        dict
            This method returns the mapped nodes of the current graph instance.
            This method is not safe it returns a reference to the actual object.
        '''
        return self.mappedNodes

    def getUnmappedNodes(self):
        '''
        Get unmapped nodes

        Standard getter

        Returns
        -------
        dict
            This method returns the unmapped nodes of the current graph
            instance. This method is not safe it returns a reference to the
            actual object.
        '''
        return self.unmappedNodes

    def getNodeCount(self):
        '''
        Get node count

        Standard getter

        Returns
        -------
        int
            This method returns the node count of the current graph instance.
            This method is not safe it returns a reference to the actual object.
        '''
        return self.nodeCount

    def getSpecificNodeCount(self, status, mode):
        '''
        Get Specific Node Count

        Get the node count of a specific subset of nodes which can be specified
        by their mapping status and mode.

        Parameters
        ----------
        status: str
            The mapping status of the subset of nodes to count.  Expected values
            are 'Mapped' or 'Unmapped'.
        mode: str
            The mode type of the nodes to count, e.g. 'Gene'.

        Returns
        -------
        int
            The count of nodes in the specified mapping and mode.
        '''
        if(status == 'Mapped'):
            nodes = self.mappedNodes
        elif(status == 'Unmapped'):
            nodes = self.unmappedNodes
        else:
            raise ValueError('Unknown node mapping status: {}'.format(status))
        if(mode in nodes):
            return len(nodes[mode])
        else:
            raise ValueError('Mode type {} not present in {} nodes'.format())

    def getMappedNodeCount(self):
        '''
        Get mapped node count

        Iterates over all mapped node modes and counts the nodes there.

        Returns
        -------
        int
            This method returns the mapped node count of the current graph
            instance.
        '''
        count = 0
        for mode in self.mappedNodes:
            count += self.getSpecificNodeCount('Mapped', mode)
        return count

    def getUnmappedNodeCount(self):
        '''
        Get mapped node count

        Iterates over all mapped node modes and counts the nodes there.

        Returns
        -------
        int
            This method returns the mapped node count of the current graph
            instance.
        '''
        count = 0
        for mode in self.unmappedNodes:
            count += self.getSpecificNodeCount('Unmapped', mode)
        return count

    def getNodeCountByMode(self):
        '''
        Get node counts by mode

        Iterates over mapped and unmapped nodes to determine how many nodes are
        present from each mode.

        Returns
        -------
        dict
            A dictionary mapping mode to node count.
        '''
        countDict = {}
        modes = list(set(self.mappedNodes.keys()).union(
                     set(self.unmappedNodes.keys())))
        for mode in modes:
            for status in ['Mapped', 'Unmapped']:
                if(not(mode in countDict)):
                    countDict[mode] = 0
                try:
                    countDict[mode] += self.getSpecificNodeCount(status, mode)
                except:
                    pass
        return countDict

    def getEdgeCountByMode(self):
        '''
        Get Edge Count By Mode

        This method determines the edge count for all types of edges in the
        graph.

        Returns
        -------
        dict
            A dictionary containing all edge types and their node counts.
        '''
        countDict = {}
        modes = list(set(self.mappedNodes.keys()).union(
                     set(self.unmappedNodes.keys())))
        edgeTypes = ['{}-{}'.format(x, x) for x in modes]
        combos = combinations(modes, 2)
        for combo in combos:
            edgeTypes.append('-'.join(combo))
        for eT in edgeTypes:
            countDict[eT] = len(self.getEdgesOfType(eT))
        return countDict

    def getEdgeCount(self):
        '''
        Get edge count

        Standard getter

        Returns
        -------
        int
            This method returns the edge count of the current graph instance.
            This method is not safe it returns a reference to the actual object.
        '''
        return self.edgeCount

    def getModePosition(self, mode):
        '''
        Get node position of the given mode

        A getter which provides a tuple of the start and end position of a mode
        in the adjacency matrix of the graph.

        Parameters
        ----------
        mode: iterable of string or string
            An iterable containing node modes or a string specifying the node mode

        Returns
        -------
        list
            A list of two-element tuple of start and end position of the input mode.
        '''
        nodeCount = self.getNodeCountByMode()
        if type(mode) == str:
            start = 0
            end = -1
            for m in sorted(set(self.mappedNodes.keys()).union(set(self.unmappedNodes.keys()))):
                start = end + 1
                end = start + nodeCount[m] - 1
                if m == mode:
                    return (start, end)
        else:
            result = [(np.nan, np.nan)] * len(mode)
            start = 0
            end = -1
            for m in sorted(set(self.mappedNodes.keys()).union(set(self.unmappedNodes.keys()))):
                start = end + 1
                end = start + nodeCount[m] - 1
                for i, x in enumerate(mode):
                    if x == m:
                        result[i] = (start, end)
        return result

    def getNodePosition(self):
        '''
        Get node position

        A getter which provides a mapping between a node and its position
        in the adjacency matrix of the graph.

        Returns
        -------
        dict
            A dictionary mapping nodes to their position in the adjacency
            matrix.
        '''
        npDict = {}
        counter = 0
        for mode in sorted(set(self.mappedNodes.keys()).union(set(self.unmappedNodes.keys()))):
            for status in ['Mapped', 'Unmapped']:
                if(status == 'Mapped'):
                    nDict = self.mappedNodes
                else:
                    nDict = self.unmappedNodes
                try:
                    for uID in sorted(nDict[mode].keys()):
                        node = nDict[mode][uID]
                        npDict[node] = counter
                        counter += 1
                except:
                    pass
        return npDict

    def getPositionNode(self):
        '''
        Get position node

        A getter which provides a mapping between a the position of a node
        in the adjacency matrix of the graph and the node itself.

        Returns
        -------
        dict
            A dictionary mapping node position in the adjacency matrix of the
            graph to the node itself.
        '''
        pnDict = {}
        counter = 0
        for mode in sorted(set(self.mappedNodes.keys()).union(set(self.unmappedNodes.keys()))):
            for status in ['Mapped', 'Unmapped']:
                if(status == 'Mapped'):
                    nDict = self.mappedNodes
                else:
                    nDict = self.unmappedNodes
                try:
                    for uID in sorted(nDict[mode].keys()):
                        node = nDict[mode][uID]
                        pnDict[counter] = node
                        counter += 1
                except:
                    pass
        return pnDict

    def getAdjacencyMatrix(self):
        '''
        Get adjacency matrix

        This method creates a sparse matrix representation of the adjacency
        matrix of the current graph instance.  The values in each cell will be
        the combined edge weights of each edge.  The nodes will be arranged in
        the following hierarchy:
            All Mapped Nodes first
            Nodes sorted by mode then by indexed ID
            All Unmapped Nodes second
            Nodes sorted by mode then by indexed ID
        This mapping can be found by calling getNodePosition()
        '''
        start = time()
        nFileName = '{}_ConfAdjacencyMatrix'.format(self.name)
        if(os.path.exists(nFileName + '.npz')):
            loadedFiles = np.load(nFileName + '.npz')
            adjacencyMatrix = csr_matrix((loadedFiles['data'],
                                          loadedFiles['indices'],
                                          loadedFiles['indptr']),
                                         shape=loadedFiles['shape'])
        else:
            # Build adjacency matrix
            nodePos = self.getNodePosition()
            rows = []
            cols = []
            contents = []
            nodes = self.getNodes()
            for node in nodes:
                pos1 = nodePos[node]
                edges = node.getOutEdges()
                for edge in edges:
                    keys2 = edge.getNode2()
                    node2 = self.findNode(keys2[1], keys2[0])
                    try:
                        pos2 = nodePos[node2[0]]
                    except:
                        print('In adjacency matrix')
                        embed()
                    weight = edge.getCombinedEdgeWeight()
                    rows.append(pos1)
                    cols.append(pos2)
                    contents.append(weight)
                    if(edge.getNode1() != edge.getNode2()):
                        if(not edge.isDirected()):
                            rows.append(pos2)
                            cols.append(pos1)
                            contents.append(weight)
            adjacencyMatrix = csr_matrix((contents, (rows, cols)),
                                         shape=(self.nodeCount,
                                                self.nodeCount))
            np.savez(nFileName, data=adjacencyMatrix.data,
                     indices=adjacencyMatrix.indices,
                     indptr=adjacencyMatrix.indptr,
                     shape=adjacencyMatrix.shape)
        end = time()
        # print('Took {} min to return adjacency matrix'.format(
        #     (end - start) / 60))
        return adjacencyMatrix

    def getTextBasedEdgeList(self, entityTypes=False):
        '''
        Get Text Based Edge List

        This function creates a list of Strings representing edges in the
        network this can be used to write the network to file in a format
        recognized by other graph packages like networkx.

        Parameters
        ----------
        entityTypes: boolean
            If set to true the edge list returend has two additional fields,
            one for each node in the edge, specifying entity types.  The default
            value is False, for compatibility with networkx.
        Returns
        -------
        list
            A list of strings with the format node1\tnode2\tweight describing
            the edges present in the graph.
        '''
        start = time()
        edgeList = []
        for status in ['Mapped', 'Unmapped']:
            if(status == 'Mapped'):
                nDict = self.mappedNodes
            else:
                nDict = self.unmappedNodes
            for mode in nDict:
                for uID in nDict[mode]:
                    node1 = nDict[mode][uID]
                    edges = node1.getOutEdges()
                    for edge in edges:
                        node2 = edge.getNode2()
                        weight = edge.getCombinedEdgeWeight()
                        if(entityTypes):
                            rowList = [node1.getName(), mode, node2[0],
                                       node2[1], str(weight)]
                        else:
                            rowList = [node1.getName(), node2[0], str(weight)]
                        eStr = '\t'.join(rowList)
                        # eStr = '{}\t{}\t{}'.format(node1.getName(),
                        #                           node2.getName(), weight)
                        edgeList.append(eStr)
        end = time()
        # print('Took {} min to return text edge list'.format(
        #     (end - start) / 60))
        return edgeList

    def getNodesOfType(self, mode):
        '''
        Return nodes of a specific type.

        This method identifies nodes of a specific type within the graph and
        returns them.

        Parameters
        ----------
        mode: str
            A string describing the mode of the nodes to be returned.

        Returns
        -------
        list
            All mapped nodes of the mode provided in the parameters.
        list
            All unmapped nodes of the mode provided in the parameters.
        '''
        mappedNodes = []
        unmappedNodes = []
        for status in ['Mapped', 'Unmapped']:
            if(status == 'Mapped'):
                nDict = self.mappedNodes
                res = mappedNodes
            else:
                nDict = self.unmappedNodes
                res = unmappedNodes
            if mode in nDict.keys():
                res.extend(nDict[mode].values())
        return (mappedNodes, unmappedNodes)

    def getEdgesOfType(self, edgeType):
        '''
        Return edges of a specific type

        This method identifies edges of a specific type within the graph and
        returns them.

        Parameters
        ----------
        edgeType: str
            Description of an edge mode specified by '{type}-{type}'.

        Returns
        -------
        list
            A list of EdgeV2 objects from this graph instance which have the
            specified mode.
        '''
        types = edgeType.split('-')
        nodes1 = self.getNodesOfType(types[0])
        nodes1 = nodes1[0] + nodes1[1]
        nodes2 = self.getNodesOfType(types[1])
        nodes2 = nodes2[0] + nodes2[1]
        if(types[0] != types[1]):
            nodes = nodes1 + nodes2
        else:
            nodes = nodes1
        edges = []
        for node1 in nodes:
            outEdges = node1.getOutEdges()
            for edge in outEdges:
                if(edge.hasModes(type1=types[0], type2=types[1])):
                    edges.append(edge)
        return edges

    def findShortestPath(self, start, end=None, invert=False):
        '''
        Find Shortest Path

        This method implements Dijkstra's shortest path algorithm, using a
        priority queue in order to find the shortest path.  The method can find
        the shortest path to all nodes in the starting nodes connected component
        (nodes outside the connected component get a distance of infinity), or
        to a specified node.

        Code written based on the psuedocode provided for the priority queue
        implementation of Dijkstra's algorithm at:
        https://en.wikipedia.org/wiki/Dijkstra's_algorithm

        Parameters
        ----------
        start: NodeV2
            The node at which to start the shortest path calculation
        end: NodeV2
            (Optional) The node to which to find the shortest path from
            the starting node.
        invert: bool
            (Optional) If the weights on the nodes are edges are confidences and
            you wish to maximize the confidence instead of minimizing the
            distance set this parameter to true.

        Returns
        -------
        dict
            A dictionary mapping node to distance. The distance to the starting
            node will be 0.  The distance to any node which could not be reached
            will be infinity as defined by float('inf').
        dict
            A dictionary mapping node to the node which preceded it in the
            shortest path.
        '''
        # Get the node in the graph corresponding to the passed in node
        start = self.findNode(start.getMode(), start.getUID())[0]
        # Create a distance dictionary to hold the distances to the nodes
        # Build a node set
        nodes = set(self.getNodes().keys())
        # Initialize all nodes and positions with distance infinity and no
        # previous node.
        dist = {}
        prev = {}
        for node in nodes:
            dist[node] = float('inf')
            prev[node] = None
        # Initialize it with the start node and a distance of 0.
        dist[start] = 0
        # Create a previous dictionary to hold the node leading to this node
        prev[start] = None
        # Create priority queue to hold nodes to traverse
        pq = PriorityQueue()
        # Add the starting node to the priority queue with distance 0
        pq.put_nowait((0, start))
        # Find distances to tall nodes, or untill the target node distance is
        # found.
        while(not pq.empty()):
            # Get the optimal node
            pick = pq.get_nowait()
            currNode = pick[1]
            # If a single target has been identified and we have reached it,
            # terminate.
            if((end is not None) and (currNode == end)):
                break
            else:
                pass
            # Since there is no dequeue or update priority setting for the built
            # in priority queue this throws away any old references to nodes
            # which have already been reached.
            if(currNode not in nodes):
                continue
            else:
                nodes.discard(currNode)
            # Get out edges to find neighbors
            outEdges = currNode.getOutEdges()
            outNeighbors = [(oE.getNode2(), oE.getCombinedEdgeWeight())
                            for oE in outEdges]
            # Get undirected in edges to find neighbors
            inEdges = [x for x in currNode.getInEdges() if not x.isDirected()]
            inNeighbors = [(iE.getNode1(), iE.getCombinedEdgeWeight())
                           for iE in inEdges]
            neighbors = outNeighbors + inNeighbors
            for nTuple in neighbors:
                neighbor = self.findNode(nTuple[0][1], nTuple[0][0])[0]
                # If the weights are to be inverted do that here. I subtract
                # from 2 instead of from 1 because there are graphs where we
                # have binary edge weights, and I did not want the inversion to
                # lead to a weight of 0.
                length = nTuple[1]
                if(invert):
                    length = (1.0 / length)
                # Determine the distance to all neighbors from the source
                # through the current node.
                alt = dist[currNode] + length
                # If the distance is less update it and update the priority in
                # the queue.
                if(alt < dist[neighbor]):
                    dist[neighbor] = alt
                    prev[neighbor] = currNode
                    pq.put_nowait((alt, neighbor))
        return dist, prev

    ###########################################################################
    # Setters
    ###########################################################################

    def setName(self, name):
        '''
        Set name

        Standard setter

        Parameters
        ----------
        str
            This method alters the name of the current graph instance.
        '''
        self.name = name

    def setMappingFile(self, mappingFile):
        '''
        Set mapping file

        Standard setter

        Parameters
        -------
        str
            This method alters the mapping file of the current graph instance.
        '''
        self.mappingFile = mappingFile

    def setConnectivitySources(self, connectivitySources):
        '''
        Set connectivity sources

        Standard setter

        Parameters
        ----------
        list
            This method alters the connectivity sources of the current graph
            instance. 
        '''
        self.connectivitySources = connectivitySources

    def setMappedNodes(self, mappedNodes):
        '''
        Set mapped nodes

        Standard setter

        Parameters
        ----------
        dict
            This method alters the mapped nodes of the current graph instance.
        '''
        self.mappedNodes = mappedNodes

    def setUnmappedNodes(self, unmappedNodes):
        '''
        Set unmapped nodes

        Standard setter

        Parameters
        ----------
        dict
            This method alters the unmapped nodes of the current graph
            instance.
        '''
        self.unmappedNodes = unmappedNodes

    def setNodeCount(self, nodeCount):
        '''
         Set node count

        Standard setter

        Parameters
        ----------
        int
            This method alters the node count of the current graph instance.
        '''
        self.nodeCount = nodeCount

    def setEdgeCount(self, edgeCount):
        '''
        Set edge count

        Standard setter

        Parameters
        ----------
        int
            This method alters the edge count of the current graph instance.
        '''
        return self.edgeCount

    ##########################################################################
    # Functions after this point are deprecated functions intended for use
    # only to facilitate the transition from previous iterations of the
    # GraphRepresentation class.
    ##########################################################################

    def getSize(self):
        '''
        Get size

        Deprecated getter

        Returns
        -------
        dict
            The dictionary containing the node and edge counts as specified by
            the keys 'Nodes' and 'Edges' respectively.
        '''
        size = {}
        size['Nodes'] = self.nodeCount
        size['Edges'] = self.edgeCount
        return size

    def setSize(self, size):
        '''
        Set size

        Deprecated setter

        Parameters
        ----------
        size: dict
            A dictionary with keys 'Nodes' and 'Edges' that map to values
            indicating the count of each type of entity.
        '''
        self.nodeCount = size['Nodes']
        self.edgeCount = size['Edges']

    def getConnectivityFiles(self):
        '''
        Get connectivity file

        Deprecated getter

        Returns
        -------
        list
            A list of source information in the form of three-tuples which
            contain a source name, file path, and directedness boolean.
        '''
        return self.connectivitySources

    def setConnectivityFiles(self, connectivityFiles):
        '''
        Set connectivity file

        Deprecated setter

        Parameters
        ----------
        connectivityFiles: list
            A list of connectivity source information in the format of
            three-tuples with a name, file path, and directednes boolean.
        '''
        self.connectivitySources = connectivityFiles

    def getNodes(self):
        '''
        Get nodes

        Deprecated getter

        Returns
        -------
        dict
            A dictionary containing all nodes from the graph with a node mapping
            to itself.
        '''
        nodes = {}
        for mode in set(self.mappedNodes.keys()).union(set(self.unmappedNodes.keys())):
            for status in ['Mapped', 'Unmapped']:
                if(status == 'Mapped'):
                    nDict = self.mappedNodes
                else:
                    nDict = self.unmappedNodes
                try:
                    for uID in nDict[mode]:
                        node = nDict[mode][uID]
                        nodes[node] = node
                except:
                    pass
        return nodes

    def getEdges(self):
        '''
        Get edges

        Deprecated getter

        Returns
        -------
        dict
            A dictionary containing all edges from the graph with a edge mapping
            to itself.
        '''
        results = {}
        for mode in set(self.mappedNodes.keys()).union(set(self.unmappedNodes.keys())):
            for status in ['Mapped', 'Unmapped']:
                if(status == 'Mapped'):
                    nDict = self.mappedNodes
                else:
                    nDict = self.unmappedNodes
                try:
                    for uID in nDict[mode]:
                        node = nDict[mode][uID]
                        edges = node.getOutEdges()
                        for edge in edges:
                            results[edge] = edge
                except:
                    pass
        return results

    def setNodes(self, nodes):
        '''
        Set nodes

        Deprecated setter

        Parameters
        ----------
        nodes: dict
            A dictionary mapping a node to itself, which will be mapped into
            the mappedNodes and unmappedNodes structures of this graph.
        '''
        for node in nodes:
            uID = node.getUID()
            check = re.match(r'^[a-zA-Z]\.\d+$', uID)
            if(check):
                self.addMappedNode(node)
            else:
                self.addUnmappedNode(node)

    def getEdgeList(self):
        '''
        Get edge list

        Deprecated getter

        Returns
        -------
        dict
            A dictionary mapping edges to themselves.
        '''
        edges = {}
        for status in ['Mapped', 'Unmapped']:
            if(status == 'Mapped'):
                nDict = self.mappedNodes
            else:
                nDict = self.unmappedNodes
            for mode in nDict:
                for uID in nDict[mode]:
                    node = nDict[mode][uID]
                    outEdges = node.getOutEdges()
                    for edge in outEdges:
                        edges[edge] = edge
        return edges

    def setEdgeList(self, edges):
        '''
        Set edge list

        Deprecated setter. This function will attempt to add all edges from the
        edge list and will return a list of those edges which could not be added
        because their nodes were missing.

        Parameters
        ----------
        edges: dict
            A dictionary of edges to themselves.

        Returns
        -------
        list
            A list of edges which could not be added.  This maybe because the
            nodes required to add the edge do not exist in the graph or because
            the edge is already present.
        '''
        missedEdges = []
        for edge in edges:
            try:
                self.addEdge(edge)
            except ValueError:
                missedEdges.append(edge)
        return missedEdges

    def getSubNetwork(self, parameterName, parameterValue, mode):
        '''
        Get subnetwork

        Deprecated function to return a subnetwork.  Removes the edges from the
        current network and returns a second network as described in
        .

        Parameters
        ----------
        parameterName: str
            The name of the parameter on an edge to use for subsetting the
            graph.  See the  function for available options.
        parameterValue: str or int
            The value of the parameter to be checked against.
        mode: str
            The method of comparison to use when checking the state of a
            parameter's value on an edge.

        Returns
        -------
        See .
        '''
        if((parameterName == 'Weight') and (mode == 'lt')):
            return self.deleteEdges(attr=parameterName, value=parameterValue)
        elif((parameterName == 'Source') and (mode == 'equal')):
            return self.deleteEdges(attr=parameterName, value=parameterValue)
        elif((parameterName == 'Mode') and (mode == 'equal')):
            return self.deleteEdges(attr=parameterName, value=parameterValue)
        elif((parameterName == 'Year') and (mode == 'gte')):
            return self.deleteEdges(attr=parameterName, value=parameterValue)
        else:
            raise NotImplementedError('This should be implemented!')

    def excludeEdgeType(self, type1, type2):
        '''
        Exclude an edge type

        A deprecated function used to return a subnetwork based on edge mode
        types.  Uses  to perform this behavior.

        Parameters
        ----------
        type1: str
            One of the two modes to check for on an edge.
        type2: str
            The other of the two modes to check for on an edge.

        Returns
        -------
        See .
        '''
        return self.deleteEdges(attr='Mode', value='{}-{}'.format(type1, type2))

    def subnetworkByNodes(self, parameterName, parameterValue, mode):
        '''
        Subnetwork network by nodes

        A deprecated function used to return a subnetwork based on nodes.

        Parameters
        ----------
        parameterName: str
            The name of the parameter on a node to use for subsetting the
            graph.  See the deleteNodes function for available options.
        parameterValue: str or int
            The value of the parameter to be checked against.
        mode: str
            The method of comparison to use when checking the state of a
            parameter's value on an edge.

        Returns
        -------
        See deleteNodes.
        '''
        if((parameterName == 'Mode') and (mode == 'Equal')):
            return self.deleteNodes(attr=parameterName, value=parameterValue)
        elif((parameterName == 'Node') and (mode == 'Equal')):
            return self.deleteNodes(attr=parameterName, value=parameterValue)
        elif((parameterName in ['Degree', 'In-Degree', 'Out-Degree']) and
             (mode == 'lte')):
            return self.deleteNodes(attr=parameterName, value=parameterValue)
        else:
            raise NotImplementedError('This should be implemented!')

    def idLookUp(self, entityName, keyWord=None):
        '''
        Look up the identifier for a node based on an alternate name which may
        be a canonical identifier or synonym, using a keyword if there is
        ambiguity.

        Parameters
        ----------
        entityName: str
            An entity name to search for in the network mapping resource.  If
            this name is a uid, canonical identifier, or synonym for a node
            then the uid of that node will be returned.
        keyWord: str or None (optional)
            An optional argument which allows the user to specify keywords to
            look for in the case that there are multiple uid's identified.
            The key word will be checked for in the canonical identifier and
            synonyms of the ambiguous uids.  Only uids which contain the
            keyWord will be kept.

        Returns:
        --------
        str
            A uid for a node in the network.  If no uid is found then None is
            returned.  If more than one uid is found, even after using the
            keyWord filter then a ValueError is raised after the ids are
            printed to standard out.

        '''
#         raise NotImplementedError('Unsupported function: idLookUp')
        if((self.canonicalIDs is None) or (self.synonyms is None)):
            self.uids = {}
            self.canonicalIDs = {}
            self.synonyms = {}
            fOpen = open(self.mappingFile, 'rb')
            fReader = reader(fOpen, delimiter='\t')
            header = None
            for line in fReader:
                if header is None:
                    header = line
                else:
                    uid = line[header.index('#IndexedID')]
                    canonical = line[header.index('CanonicalID')]
                    self.canonicalIDs[canonical] = uid
                    syns = line[header.index('Synonyms')]
                    self.uids[uid] = canonical + ';;' + syns
                    for syn in syns.split(';;'):
                        if(syn not in self.synonyms):
                            self.synonyms[syn] = []
                        self.synonyms[syn].append(uid)
        if(entityName in self.uids):
            return entityName
        elif(entityName in self.canonicalIDs):
            return self.canonicalIDs[entityName]
        elif(entityName in self.synonyms):
            ids = self.synonyms[entityName]
            if(len(ids) == 1):
                return ids[0]
            else:
                newIds = []
                for cid in ids:
                    if(keyWord in self.uids[cid]):
                        newIds.append(cid)
                if(len(newIds) == 1):
                    return newIds[0]
                # print ids
                raise ValueError('Too many/few ids for entity: {}'.format(
                    entityName))
        else:
            return None

    ##########################################################################
    # Currently unsupported functions
    ##########################################################################

    def setNodePosition(self):
        '''
        This method is not supported in the current GraphRepresentation5 since
        positions are computed on the fly in a deterministic fashion and not
        stored somewhere in the instance's state variables.
        '''
        raise NotImplementedError('Unsupported function: setNodePosition')

    def getMapping(self):
        '''
        This method is not supported in the current GraphRepresentation5 but
        may be added in the future.
        '''
        raise NotImplementedError('Unsupported function: getMapping')

    def getSynonyms(self):
        '''
        This method is not supported in the current GraphRepresentation5 but
        may be added in the future.
        '''
        raise NotImplementedError('Unsupported function: getSynonyms')

    def getCanonical(self):
        '''
        This method is not supported in the current GraphRepresentation5 but
        may be added in the future.
        '''
        raise NotImplementedError('Unsupported function: getCanonical')


if(__name__ == '__main__'):
    mappingFile = '../Mapping_Ccgdd16Sx91Extended.tsv'
    graph = GraphRepresentation5(mappingFile=mappingFile)
    print(str(graph))
