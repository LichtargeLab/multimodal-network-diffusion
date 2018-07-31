#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Sep 8, 2016

@author: daniel
'''
from operator import mul
from functools import reduce


class Edge(object):
    '''
    classdocs
    This class is intended to represent an edge for the GraphRepresentation
    class.
    '''

    def __init__(self, node1, node2, directed):
        '''
        Constructor

        Instantiates an instance of the edge class.

        Parameters
        ----------
        node1: Node
            First node which is connected by this edge.  If the edge is
            directed this should be the source node.
        node2: Node
            Second node which is connected by this edge.  If the edge is
            directed this should be the target node.
        directed: bool
            True if this edge is a directed edge from node1 to node2, otherwise
            False.

        Returns
        -------
        An instance of the edge class with the nodes instantiated.  All other
        data will be started with empty structures or the None value.  When
        this edge is created the degrees of the nodes will be incremented
        correctly, i.e. for an undirected edge both in and out degrees will
        be incremented for both nodes and for directed edges the out-degree
        of the source node and in-degree of the target node will be incremented.
        '''
        self.node1 = (node1.getUID(), node1.getMode())
        self.node2 = (node2.getUID(), node2.getMode())
        self.directed = directed
        self.edgeWeights = {}
        self.normalizedEdgeWeights = {}
        self.combinedEdgeWeight = None
        self.years = {}
        self.metaData = {}

    def __del__(self):
        '''
        Destructor

        Deletes the data stored by this class.
        '''
        self.node1 = None
        self.node2 = None
        del(self.directed)
        del(self.edgeWeights)
        del(self.normalizedEdgeWeights)
        del(self.combinedEdgeWeight)
        del(self.years)
        del(self.metaData)
        del(self)

    def __str__(self):
        '''
        To String

        Provides a meaningful string description of the instance of the node
        class object.

        Returns
        -------
        str
            String describing the edge represented by this instance of the
            edge class.
        '''
        objStr = '{} edge between {}-{}'.format(
            {True: 'Directed', False: 'Undirected'}[self.directed],
            self.node1[0], self.node2[0])
        return objStr

    def __cmp__(self, other):
        '''
        Comparator

        Compares the current instance of the edge class with another instance
        of the edge class.

        Parameters
        ----------
        other: Edge
            Another instance of the edge class to which the current instance
            should be compared.

        Returns
        -------
        int
            The function returns 0 if both edges are the same.  The method
            returns -1 if the current edge should come before the other edge
            provided for comparison.  The function returns 1 if the other edge
            instance should come first.
        '''
        if(self.directed < other.directed):
            return -1
        elif(self.directed > other.directed):
            return 1
        else:
            if(self.directed):
                sNodes = [self.node1, self.node2]
                oNodes = [other.node1, other.node2]
            else:
                sNodes = sorted([self.node1, self.node2])
                oNodes = sorted([other.node1, other.node2])
            if(sNodes[0] < oNodes[0]):
                return -1
            elif(sNodes[0] > oNodes[0]):
                return 1
            else:
                if(sNodes[1] < oNodes[1]):
                    return -1
                elif(sNodes[1] > oNodes[1]):
                    return 1
                else:
                    return 0

    def __eq__(self, other):
        '''
        Equality check.

        This method checks whether two instances of the edge class are equal.

        Parameters
        ----------
        other: Edge
            The other instance of the edge class to compare to.

        Returns
        -------
        bool
            This method returns True if the two edges are considered equivalent,
            and False if they are not.  Equivalence is tested by the cmp
            function which checks the directedness of the edges and their node
            composition.
        '''
        return (self.__cmp__(other) == 0)

    def __hash__(self):
        '''
        Hash function

        Produce a hash value for the edge for use in structures like
        dictionaries.  This relies on the hash of the unique identifiers of
        nodes 1 and 2.

        Returns
        -------
        int
            An integer that is supposed to uniquely represent the current
            instance of the edge class.
        '''
        return hash(self.node1) ^ hash(self.node2)

    def getNode1(self):
        '''
        Get Node 1

        Standard getter

        Returns
        -------
        Node
            Returns the Node stored at node1.  This method is not safe, it will
            return a reference to the node object, not the node itself.
        '''
        return self.node1

    def setNode1(self, node1):
        '''
        Set Node 1

        Standard setter

        Parameters
        ----------
        node1: Node
            Alters the Node stored at node1.
        '''
        self.node1 = node1

    def getNode2(self):
        '''
        Get Node 2

        Standard getter

        Returns
        -------
        Node
            Returns the Node stored at node2.  This method is not safe, it will
            return a reference to the node object, not the node itself.
        '''
        return self.node2

    def setNode2(self, node2):
        '''
        Set Node 2

        Standard setter

        Parameters
        -------
        node2: Node
            Alters the Node stored at node2.
        '''
        self.node2 = node2

    def getModes(self):
        '''
        Get edge modes.

        Retrieves the modes of each node in the edge to describe the modes of
        the edge.

        Returns
        -------
        tuple
            A tuple with two strings, one for the mode of each node in th edge.
        '''
        mode1 = self.node1[1]
        mode2 = self.node2[1]
        return (mode1, mode2)

    def getDirected(self):
        '''
        Get Directedness of the Edge

        Standard getter

        Returns
        -------
        bool
            Returns whether the edge is directed or not.  This method is not
            safe, it will return a reference to the node object, not the node
            itself.
        '''
        return self.directed

    def setDirected(self, directed):
        '''
        set Directedness of the Edge

        Standard setter

        Parameters
        ----------
        bool
            Alters whether the edge is directed or not.
        '''
        self.directed = directed

    def getEdgeWeights(self):
        '''
        Get edge weights

        Standard getter

        Returns
        -------
        dict
            Returns the dictionary mapping sources to edge weights/confidences.
            This method is not safe, it will return a reference to the
            dictionary, not the dictionary itself.
        '''
        return self.edgeWeights

    def setEdgeWeights(self, edgeWeights):
        '''
        Set edge weights

        Standard setter

        Parameters
        ----------
        edgeWeights: dict
            Alters the dictionary mapping sources to edge weights/confidences.
        '''
        self.edgeWeights = edgeWeights

    def getNormalizedEdgeWeights(self):
        '''
        Get normalized edge weights

        Standard getter

        Returns
        -------
        dict
            Returns the dictionary mapping sources to normalized edge
            weights/confidences.  This method is not safe, it will return a
            reference to the dictionary object, not the dictionary itself.
        '''
        return self.normalizedEdgeWeights

    def setNormalizedEdgeWeights(self, normalizedEdgeWeights):
        '''
        Set normalized edge weights

        Standard setter

        Parameters
        ----------
        normalizedEdgeWeights: dict
            Alters the dictionary mapping sources to normalized edge
            weights/confidences.
        '''
        self.normalizedEdgeWeights = normalizedEdgeWeights

    def getCombinedEdgeWeight(self):
        '''
        Get combined edge weight

        If the combined value of the edge weights has not been calculated yet,
        an attempt will be made to calculate the value.  If normalized edge
        weights have not been generated then this will fail and None will be
        returned.

        Returns
        -------
        float or None
            Returns the combined edge weight/confidence for this edge or None
            if it cannot be computed yet. This method is not safe, it will
            return a reference to the float, not the value itself.
        '''
        if(self.combinedEdgeWeight is None):
            self.combineEdgeWeights()
        return self.combinedEdgeWeight

    def setCombinedEdgeWeights(self, combinedEdgeWeight):
        '''
        Set combined edge weight

        Standard setter.

        Parameters
        ----------
        combinedEdgeWeight: float
            Alters the combined edge weight/confidence for this edge.
        '''
        if(self.combinedEdgeWeight is None):
            self.combineEdgeWeights()
        self.combinedEdgeWeight = combinedEdgeWeight

    def getYears(self):
        '''
        Get years

        Standard getter

        Returns
        -------
        dict
            Returns the dictionary mapping sources to years the edge was noted.
            This method is not safe, it will return a reference to the
            dictionary object, not the dictionary itself.
        '''
        return self.years

    def setYears(self, years):
        '''
        Set years

        Standard setter

        Parameters
        ----------
        years: dict
            Alters the dictionary mapping sources to years the edge was noted.
        '''
        self.years = years

    def getMetaData(self):
        '''
        Get meta data

        Standard getter

        Returns
        -------
        dict
            Returns the dictionary mapping sources to meta-data dictionaries.
            This method is not safe, it will return a reference to the
            dictionary object, not the dictionary itself.
        '''
        return self.metaData

    def setMetaData(self, metaData):
        '''
        Set meta data

        Standard setter

        Parameters
        ----------
        metaData:dict
            Alters the dictionary mapping sources to meta-data dictionaries.
        '''
        return self.metaData

    def getEarliestYear(self):
        '''
        Get Earliest Year

        This method searches through the years that this edge is seen in all
        sources, and returns the value of the earliest year.

        Returns
        -------
        int
            The first year in which this edge is observed in any source.
        '''
        years = self.years.values()
        minYear = min(years)
        return minYear

    def getLatestYear(self):
        '''
        Get Latest Year

        This method searches through the years that this edge is seen in all
        sources, and returns the value of the latest year.

        Returns
        -------
        int
            The last year in which this edge is observed in any source.
        '''
        years = self.years.values()
        maxYear = max(years)
        return maxYear

    def getNumSources(self):
        '''
        Get number of sources

        Determine the number of sources which support this edge.

        Returns
        -------
        int
            The number of sources which support the current edge.
        '''
        return len(self.edgeWeights.keys())

    def combineEdgeWeights(self):
        '''
        Calculate combined edge weight.

        This method calulates the combined edge weight of the edge given its
        normalized edge weight in all supporting sources.  Combination is
        performed using the formulat 1 - [(1-Weight1)*(1-Weight2)...(1-WeightN)]

        Returns
        -------
        float or None
            The value computed when combining normalized edge weights from
            supporting networks.
        '''
        try:
            toCombine = [(1 - self.normalizedEdgeWeights[x])
                         for x in self.normalizedEdgeWeights]
            combinedWeight = 1 - reduce(mul, toCombine, 1)
            self.combinedEdgeWeight = combinedWeight
        except:
            # toCombine = [(1 - x) for x in self.getWeights().values()]
            self.combinedEdgeWeight = None
        return self.combinedEdgeWeight

    def addSource(self, name, year, metaData, weight=0.0):
        '''
        Add source information.

        This method adds information about an edge from a specific source.
        Weight, year, and meta-data information are added for any source not
        yet reprsented in the edge. 

        Parameters
        ----------
        name: str
            The name of the source being added to the edge's information.
        year: int
            The year in which the edge was observed in that source.
        weight: float
            The weight or confidence of the edge in the source.
        metaData: dict
            The meta-data provided by the source about the edge.
        '''
        self.edgeWeights[name] = weight
        self.years[name] = year
        self.metaData[name] = metaData

    def updateSource(self, name, year, weight, metaData):
        '''
        Update source information.

        This method adds information about an edge from a specific source if it
        is not already in the edge information.  Otherwise, it updates the
        existing information. Year data is updated if the information is found
        earlier than previously recorded.  Weight/confidence information is
        updated if the weight is stronger than those previously recorded.
        Meta-data values provided are added to the existing dictionary such that
        any previously available data which different will be over-written.

        Parameters
        ----------
        name: str
            The name of the source being added to the edge's information.
        year: int
            The year in which the edge was observed in that source.
        weight: float
            The weight or confidence of the edge in the source.
        metaData: dict
            The meta-data provided by the source about the edge.
        '''
        if(name not in self.edgeWeights):
            self.addSource(name=name, year=year, weight=weight,
                           metaData=metaData)
        else:
            if(year < self.years[name]):
                self.years[name] = year
            if(weight > self.edgeWeights[name]):
                self.edgeWeights[name] = weight
            try:
                self.metaData[name].update(metaData)
            except:
                print self.metaData
                print str(self)
                exit()

    def inSource(self, name):
        '''
        Check for source

        This method checks whether this edge is supported by data from a
        specified source.

        Parameters
        ----------
        name: str
            The name of the source to check for in the edge information.

        Returns
        -------
        bool
            The function returns True if the source does have information
            about the current edge, otherwise false is returned.
        '''
        return (name in self.edgeWeights)

    def hasNode(self, node):
        '''
        Check for node

        This method checks whether either of the nodes connected by this edge
        is the specified node.

        Parameters
        ----------
        node: Node
            The node to look for in this edge.

        Returns
        -------
        bool
            The function returns True if either node1 or node2 is considered
            equal to the node provided.  Otherwise, False is returned.
        '''
        nTuple = (node.getUID(), node.getMode())
        if(nTuple == self.node1):
            return True
        elif(nTuple == self.node2):
            return True
        else:
            return False

    def isDirected(self):
        '''
        Checks the directedness of the edge.

        This function checks whether the present edge is directed or not.

        Returns
        -------
        bool
            If the edge is directed this returns True, else it returns False.
        '''
        if(self.directed):
            return True
        else:
            return False

    def hasModes(self, type1, type2):
        '''
        Check for edge mode

        This function checks the modes of the two nodes against the specified
        modes.

        Parameters
        ----------
        type1: str
            Mode for one of the two nodes.
        type2: str
            Mode for the other of the two nodes

        Returns
        -------
        bool
            This function returns True if both modes are represented in the two
            nodes and false otherwise.
        '''
        mode1 = self.node1[1]
        mode2 = self.node2[1]
        if((type1 == mode1) and (type2 == mode2)):
            return True
        elif((type1 == mode2) and (type2 == mode1)):
            return True
        else:
            return 0

    def getParameterValue(self, parameterName):
        '''
        Retrieve meta data values

        This method retrieves all meta data values for a specific attribute
        name.

        Parameters
        ----------
        parameterName: str
            The name of the attribute for which to look for values in the
            meta-data dictionary.

        Returns
        -------
        list
            A set of values found in the meta-data dictionary for the specified
            attribute name.
        '''
        values = []
        for s in self.metaData:
            if(parameterName in self.metaData[s]):
                paramVal = self.source[s]
                if(hasattr(paramVal, '__iter__')):
                    values += paramVal
                else:
                    values.append(paramVal)
        values = list(set(values))
        print('Len values: {}'.format(len(values)))
        if(len(values) == 1):
            return values[0]
        else:
            return values
