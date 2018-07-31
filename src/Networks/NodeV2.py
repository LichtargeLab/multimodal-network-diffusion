#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Sep 8, 2016

@author: daniel
'''


class Node(object):
    '''
    classdocs
    This class is meant to represent a single node in the GraphRepresentation
    class.
    '''

    def __init__(self, uID, canonical, synonyms, relSynonyms, mode, sources, metaData):
        '''
        Constructor

        Initializes the state of a node class object.

        Parameters
        ----------
        uID: str
            Unique identifier generally pulled from a mapping table.
        canonical: str
            The name by which this node is best known.
        synonyms: list
            Set of names by which this node is also known.
        relSynonyms: list
            Another set of names by which this node is also known.
        mode: str
            String or letter describing what type of entity this node is.
        sources: list
            Set of names of sources or mappings which include this node.
        metaData: dict
            Set of attributes and values this node has.

        Returns
        -------
        An instance of the node class.  This also includes a parameter tracking
        the edges connected to a node, which also allows for degree tracking.
        '''
        self.uID = uID
        self.canonical = canonical
        self.synonyms = synonyms
        self.relSynonyms = relSynonyms
        modeConversion = {'C': 'Chemical',
                          'D': 'Disease',
                          'G': 'Gene',
                          'P': 'Protein'}
        if(mode in modeConversion):
            self.mode = modeConversion[mode]
        else:
            self.mode = mode
        self.sources = sources
        self.metaData = metaData
        self.outEdges = []
        self.inEdges = []

    def __del__(self):
        '''
        Destructor

        Deletes each instance variable before destroying the node class object.
        '''
        del(self.uID)
        del(self.canonical)
        del(self.synonyms)
        del(self.mode)
        del(self.sources)
        del(self.metaData)
        del(self.outEdges)
        del(self.inEdges)
        del(self)

    def __str__(self):
        '''
        To String

        Provides a meaningful string description of the instance of the node
        class object.

        Returns
        -------
        str
            String containing the id of the node, its type, and its current
            degree.
        '''
        objStr = 'ID: {}\t'.format(self.uID)
        objStr += 'Type: {}\t'.format(self.mode)
        objStr += 'Degree: {}'.format(self.getDegree())
        return objStr

    def __cmp__(self, other):
        '''
        Comparator

        Compare two instances of the node class.

        Parameters
        ----------
        other: Node
            The other instance of the node class which you are comparing the
            current instance to.

        Returns
        -------
        int
            This method returns 0 if the two instances are exactly the same.
            The method returns -1 if the current instance should come before
            the other instance passed in.  It return 1 if the other instance
            passed in should come first.
        '''
        if(self.mode < other.mode):
            return -1
        elif(self.mode > other.mode):
            return 1
        else:
            if(self.uID == other.uID):
                return 0
            elif(self.uID < other.uID):
                return -1
            else:
                return 1

    def __eq__(self, other):
        '''
        Equality Check

        Determines if two node class instances are equivalent.

        Parameters
        ----------
        other: Node
            The other instance of the node class which you are comparing the
            current instance to.

        Returns
        -------
        bool
            This method returns True if two instances of the node class are
            equivalent as determined by the compare function.  This method
            returns False if the two instances of the node class are not
            equivalent.
        '''
        return (self.__cmp__(other) == 0)

    def __hash__(self):
        '''
        Hash function

        Produce a hash value for the node for use in structures like
        dictionaries.  This relies on the hash of the unique identifier of
        the node and its mode.

        Returns
        -------
        int
            An integer that is supposed to uniquely represent the current
            instance of the node class.
        '''
        return hash(self.uID) ^ hash(self.mode)

    def getUID(self):
        '''
        Get unique identifier

        Standard getter

        Returns
        -------
        str
            Returns the current uID of the instance of the node class.  This
            method is not safe, it returns an actual reference to the current
            state of the variable.
        '''
        return self.uID

    def setUID(self, uID):
        '''
        Set unique identifier

        Standard setter

        Parameters
        ----------
        uID: str
            Alters the current uID of the instance of the node class to that
            passed in by the user.
        '''
        self.uID = uID

    def getCanonical(self):
        '''
        Get canonical name

        Standard getter

        Returns
        -------
        str
            Returns the current canonical id of the instance of the node class.
            This method is not safe, it returns an actual reference to the
            current state of the variable.
        '''
        return self.canonical

    def setCanonical(self, canonical):
        '''
        Set canonical name

        Standard setter

        Parameters
        ----------
        canonical: str
            Alters the current canonical id of the instance of the node class
            to that passed in by the user.
        '''
        self.canonical = canonical

    def getSynonyms(self):
        '''
        Get synonyms

        Standard getter

        Returns
        -------
        list
            Returns the current synonyms of the instance of the node class.
            This method is not safe, it returns an actual reference to the
            current state of the variable.
        '''
        return self.synonyms

    def setSynonyms(self, synonyms):
        '''
        Set synonyms

        Standard setter

        Parameters
        ----------
        synonyms: list
            Alters the current synonyms of the instance of the node class to
            those passed in by the user.
        '''
        self.synonyms = synonyms

    def getRelSynonyms(self):
        '''
        Get relevant synonyms

        Standard getter

        Returns
        -------
        list
            Returns the current relevant synonyms of the instance of the node class.
            This method is not safe, it returns an actual reference to the
            current state of the variable.
        '''
        return self.relSynonyms

    def setRelSynonyms(self, relSynonyms):
        '''
        Set relevant synonyms

        Standard setter

        Parameters
        ----------
        relSynonyms: list
            Alters the current relevant synonyms of the instance of the node class to
            those passed in by the user.
        '''
        self.relSynonyms = relSynonyms

    def getMode(self):
        '''
        Get mode

        Standard getter

        Returns
        -------
        str
            Returns the current mode of the instance of the node class.  This
            method is not safe, it returns an actual reference to the current
            state of the variable.
        '''
        return self.mode

    def setMode(self, mode):
        '''
        Set mode

        Standard setter

        Parameters
        ----------
        mode: str
            Alters the current mode of the instance of the node class to that
            passed in by the user.
        '''
        self.mode = mode

    def getSources(self):
        '''
        Get sources

        Standard getter

        Returns
        -------
        list
            Returns the current sources of the instance of the node class.  This
            method is not safe, it returns an actual reference to the current
            state of the variable.
        '''
        return self.sources

    def setSources(self, sources):
        '''
        Set sources

        Standard setter

        Parameters
        ----------
        sources: list
            Alters the current sources of the instance of the node class to
            those passed in by the user.
        '''
        self.sources = sources

    def getMetaData(self):
        '''
        Get meta data

        Standard getter

        Returns
        -------
        dict
            Returns the current meta data of the instance of the node class.
            This method is not safe, it returns an actual reference to the
            current state of the variable.
        '''
        return self.metaData

    def setMetaData(self, metaData):
        '''
        Set meta data

        Standard setter

        Parameters
        ----------
        metaData: dict
            Alters the current meta data of the instance of the node class to
            the dictionary of attributes and values passed in by the user.
        '''
        self.metadata = metaData

    def getOutEdges(self):
        '''
        Get outgoing edges

        Standard getter

        Returns
        -------
        list
            Returns the current list of outgoing edges attached to this node.
            This method is not safe, it returns an actual reference to the
            current state of the variable.
        '''
        return self.outEdges

    def setOutEdges(self, edges):
        '''
        Set outgoing edges

        Standard setter

        Parameters
        ----------
        edges: list
            Alters the current list of outgoing edges of this instance of the
            node class to the list of Edge objects passed in by user.
        '''
        self.edges = edges

    def getInEdges(self):
        '''
        Get incoming edges

        Standard getter

        Returns
        -------
        list
            Returns the current list of incoming edges attached to this node.
            This method is not safe, it returns an actual reference to the
            current state of the variable.
        '''
        return self.inEdges

    def setInEdges(self, edges):
        '''
        Set incoming edges

        Standard setter

        Parameters
        ----------
        edges: list
            Alters the current incoming edge list of the instance of the node
            class to the list of Edge objects passed in by user.
        '''
        self.inEdges = edges

    def getOutDegree(self):
        '''
        Get out-degree

        Determines the out-degree of the nodes by looking at the number of edges
        which start from it.

        Returns
        -------
        int
            Returns the current out-degree of the instance of the node class.
        '''
        return len(self.outEdges)

    def getInDegree(self):
        '''
        Get in-degree

        Determines the in-degree of the nodes by looking at the number of edges
        which start from it.

        Returns
        -------
        int
            Returns the current in-degree of the instance of the node class.
        '''
        return len(self.inEdges)

    def getDegree(self):
        '''
        Get degree

        Determines the degree of the nodes by looking at the number of edges
        which connect to it.

        Returns
        -------
        int
            Returns the current degree of the instance of the node class.
        '''
        return (self.getInDegree() + self.getOutDegree() -
                len(set(self.outEdges).intersection(set(self.inEdges))))

    def getDegreeByMode(self):
        '''
        getDegreeByMode

        Returns a dictionary breaking down the in degree and out degree of this
        node by which other node types it is assocated with.

        Returns:
        --------
        dict:
            A nested dictionary where the first level is 'InDegree' or
            'OutDegree'. The second level are dictionaries where the key is the
            mode of the neighboring node and the value is the number of nodes
            of that type that this node shares an edge with.
        '''
        deg = {'InDegree': {}, 'OutDegree': {}}
        for direction in deg.keys():
            if direction == 'InDegree':
                edges = self.inEdges
            else:
                edges = self.outEdges
            for edge in edges:
                mode1 = edge.node1[1]
                mode2 = edge.node2[1]
                if(mode1 == self.mode):
                    currMode = mode2
                else:
                    currMode = mode1
                if(currMode not in deg[direction]):
                    deg[direction][currMode] = 0
                deg[direction][currMode] += 1
        return deg

    def getName(self):
        '''
        Get name

        Standard getter

        Returns
        -------
        str
            Returns the current uID of the instance of the node class.  This
            method is not safe, it returns an actual reference to the current
            state of the variable.  This method is deprecated from previous
            iterations of this class and actually performs getUID().
        '''
        return self.getUID()

    def resetEdges(self):
        '''
        Reset edges

        This method sets the edge list connected to this node to an empty list.
        '''
        self.inEdges = []
        self.outEdges = []

    def addEdge(self, edge):
        '''
        Add an edge to the node.

        Adds another edge to the list of edges connected to this node.

        Parameters
        ----------
        edge: EdgeV2
            An edge object to be added to this node.

        Returns
        -------
        bool
            True is returned if the edge is successfully added to this node.
            If the edge is already attached to this node addition is not
            possible and False is returned.
        '''
        if(edge.hasNode(self)):
            if((not(edge in self.outEdges)) and (not(edge in self.inEdges))):
                if(edge.getNode1() == edge.getNode2()):
                    self.outEdges.append(edge)
                    self.inEdges.append(edge)
                elif(edge.getNode1() == (self.uID, self.mode)):
                    self.outEdges.append(edge)
                else:
                    self.inEdges.append(edge)
                return True
            else:
                print(edge)
                print(self.outEdges)
                print(self.inEdges)
                print('Edge already connected to node.')
                return False
        else:
            raise ValueError('Edge to be added does not connect to this node!')

    def removeEdge(self, edge):
        '''
        Remove edge.

        Removes an edge from the list of edges connected to this node.

        Parameters
        ----------
        edge: EdgeV2
            An edge object to be added to this node.
        '''
        if((edge in self.outEdges) and (edge in self.inEdges)):
            self.outEdges.remove(edge)
            self.inEdges.remove(edge)
            return True
        if(edge in self.outEdges):
            self.outEdges.remove(edge)
            return True
        elif(edge in self.inEdges):
            self.inEdges.remove(edge)
            return True
        else:
            return False

    def hasEdge(self, edge):
        '''
        Edge check.

        This function checks whether a node is connected to an edge.

        Parameters
        ----------
        edge: EdgeV2
            The edge to check for in the current node.
        '''
        if(edge in self.outEdges):
            return True
        elif(edge in self.inEdges):
            return True
        else:
            return False

    def hasNeighbor(self, node):
        '''
        Check for a neighboring node.

        This method iterates over the edges connected to this node to determine
        if the specified node is its neighbor.

        Parameters
        ----------
        node: NodeV2
            The node to look for as a neighbor based on existing edges.

        Returns
        -------
        EdgeV2
            This method returns the edge which connects the two nodes if one
            exists, otherwise it returns None.
        '''
        for edge in self.outEdges:
            if(edge.hasNode(node)):
                return edge
        for edge in self.inEdges:
            if(edge.hasNode(node)):
                return edge
        return None

    def inSource(self, name):
        '''
        Check if this node comes from a specific source.

        Allows a user to determine if a node is generated based on a specific
        source or not.

        Parameters
        ----------
        name: str
            The name of the source to check for in this node.

        Returns
        -------
        bool
            True is returned if the source specified is found in the sources
            list of this node.  False is returned if the source specified is
            not in the list of sources for this node.
        '''
        return (name in self.sources)

    def getNumSource(self):
        '''
        Number of sources.

        Provides the number of supporting sources for a node

        Returns
        -------
        int
            The method returns the number of unique sources which support the
            node.
        '''
        return len(self.sources)

    def hasCanonical(self, name):
        '''
        Canonical ID check.

        This method checks if the nodes canonical id is a specific name.

        Parameters
        ----------
        name: str
            The string of the name you wish to compare to.

        Returns
        -------
        bool
            If the canonical id of the node is the same as the name provided,
            True is returned.  Otherwise False is returned.
        '''
        return (self.canonical == name)

    def hasSynonym(self, syn):
        '''
        Synonym check.

        Checks whether a specified synonym is in the list of synonyms for the
        current node.

        Parameters
        ----------
        syn: string
            The synonym to check for in the synonyms held by this class.

        Returns
        -------
        bool
            If the specified synonym is in the node class instance, True is
            returned.  Otherwise False is returned.
        '''
        return (syn in self.synonyms)

    def hasMode(self, mode):
        '''
        Check mode

        This method determines if the node has the mode specified or not.

        Parameters
        ----------
        mode: str
            The mode type to check for in this instance of the node class.
            If you prefer to use single letter identifiers you may use 'C' for
            Chemical, 'D' for Disease, 'G' for Gene, or 'P' for Protein.

        Returns
        -------
        bool
            If the mode specified is the mode of this instance of the node
            class True is returned.  Otherwise false is returned.
        '''
        modeConversion = {'C': 'Chemical',
                          'D': 'Disease',
                          'G': 'Gene',
                          'P': 'Protein'}
        if(mode in modeConversion):
            mode = modeConversion[mode]
        return (self.mode == mode)

    def getParameterValue(self, parameterName):
        '''
        Get the value of a specific meta data attribute.

        This method allows for the return of the specific value of an attribute
        stored in the node's meta data dictionary.

        Parameters
        ----------
        parameterName: str
            A string specifying the name of the attribute to retrieve the value
            for.

        Returns
        -------
        str or other
            The value associated with the attribute named by parameterName.
            This method is not safe and will return the reference to the value
            in the metaData dictionary.
        '''
        return self.metadata[parameterName]


if(__name__ == '__main__'):
    node1 = Node(uID='C.1', canonical='Test', mode='C', synonyms=None,
                 sources=None, metaData=None)
    node2 = Node(uID='C.1', canonical='Test', mode='Chemical', synonyms=None,
                 sources=None, metaData=None)
    node3 = Node(uID='G.1', canonical='Test', mode='G', synonyms=None,
                 sources=None, metaData=None)
    print('Pre-Source')
    print node1 == node2
    print node1 == node3
    test = [node1, node3]
    print(node1 in test)
    print(node3 in test)
    print(node2 in test)
