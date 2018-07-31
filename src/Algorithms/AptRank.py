#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on May 15, 2017

@author: Chih-Hsu Lin

Python wrapper for AptRank in Julia
'''
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix, csgraph, identity, rand
from scipy.stats import zscore
import numpy as np
import os
from AlgorithmParent import AlgorithmParent
import julia
from IPython import embed
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
PROJECT_DIR = os.getenv("PROJECT_DIR")

class AptRank(AlgorithmParent):
    '''
    Algorithm class of AptRank
    '''

    def __init__(self, network=None, validation=None, selectedQueryMode=None, selectedPredictionMode=None, method='AptRank', param=dict()):
        '''
        Constructor:
        Calls the parent constructor
        '''
        if validation is None:
            self.network = network
            self.validation = validation
        else:
            try:
                method = method + str(param['diff_type'])
            except:
                method = method + '1'
            for k in param:
                if k =='K' or k =='S' or k =='lowerBoundFactor':
                    method = method + '_' + str(k) + str(param[k])
            AlgorithmParent.__init__(self, method=method, network=network,
                                     validation=validation, selectedQueryMode=selectedQueryMode,
                                     selectedPredictionMode=selectedPredictionMode)
        self.j = julia.Julia()
        self.alpha = None
        # K            - the number of terms used in the APTRANK, default is 8
        self.K = 8
        # S            - the number of independent experiments during training,
        #                default is 5
        self.S = 5
        # ratio        - the split ratio between fiiting set and validation
        #                set, default is 0.8
        self.ratio = 0.5
        # diff_type    - choose what kind of diffusion matrix you want, 1 for G/rho,
        #               2 for G*D^{-1}, 3 for (D - G)/rho and 4 for (I - D^{-1/2}*G*D^{-1/2})/norm(L,1),
        #               where rho is the spectral radius and D is the out-degree of each
        #               vertex, defalut is 1
        self.diff_type = 1
        # sampling_type- 1 for randomly sampling using ratio, 2 for S-fold
        # cross validation, default is 1
        self.sampling_type = 1
        self.lowerBoundFactor = 2
        self.__dict__.update(param)
        self.lowerBound = 1.0/((self.K-1)**self.lowerBoundFactor)


    def callAptRank(self, trainIdx=None, predIdx=None):
        self.adjMatrixToJuliaSparse()
        del(self.network)
        path = PROJECT_DIR+'/src/Algorithms/aptrank/'
        self.j.eval("""include("{}/general_APTRANK_eff.jl")""".format(path))
        _ = self.j.eval("ratio = {}".format(self.ratio))
        _ = self.j.eval("diff_type = {}".format(self.diff_type))
        _ = self.j.eval("K = {}".format(self.K))
        _ = self.j.eval("S = {}".format(self.S))
        _ = self.j.eval("sampling_type = {}".format(self.sampling_type))
        _ = self.j.eval("lower_bound = {}".format(self.lowerBound))
        _ = self.j.eval("adjMatrix = spones(adjMatrix);")
        if trainIdx is None:
            _ = self.j.eval(
                """
                split = Int(round(size(adjMatrix,1)/2))
                train_rows = [1:split]
                push!(train_rows,split:size(adjMatrix,1))
                split = Int(round(size(adjMatrix,2)/2))
                train_cols = [1:split]
                push!(train_cols,split:size(adjMatrix,2))
                """)
        else:
            _ = self.j.eval(
                """
                train_rows = [{}:{}];
                train_cols = [{}:{}];
                """.format(trainIdx[0][0][0] + 1, trainIdx[0][0][1] + 1, trainIdx[0][1][0] + 1, trainIdx[0][1][1] + 1))
            for i in range(1, len(trainIdx)):
                _ = self.j.eval(
                    """
                    push!(train_rows,{}:{});
                    push!(train_cols,{}:{});
                    """.format(trainIdx[i][0][0] + 1, trainIdx[i][0][1] + 1, trainIdx[i][1][0] + 1, trainIdx[i][1][1] + 1))

        if predIdx is None:
            _ = self.j.eval(
                """
                predict_rows = 1:size(adjMatrix,1);
                predict_cols = 1:size(adjMatrix,2);
                """)
        else:
            _ = self.j.eval(
                """
                predict_rows = [];
                predict_cols = [];
                """)
            for i in range(len(predIdx)):
                _ = self.j.eval(
                    """
                    predict_rows = vcat(predict_rows,{}:{});
                    predict_cols = vcat(predict_cols,{}:{});
                    """.format(predIdx[i][0][0] + 1, predIdx[i][0][1] + 1, predIdx[i][1][0] + 1, predIdx[i][1][1] + 1))

        _ = self.j.eval(
            """
            ei,ej,v = findnz(adjMatrix);
            """)

        # self.j.eval(
        #     """
        #     println(adjMatrix.m)
        #     println(adjMatrix.n)
        #     println(train_cols)
        #     println(train_rows)
        #     println(predict_rows)
        #     println(predict_cols)
        #     """)
        self.resultmat, self.alpha, self.all_alpha = self.j.eval(
            """
            Xa,alpha,all_alpha = general_APTRANK_eff(ei,ej,v,adjMatrix.m,adjMatrix.n,train_rows,train_cols,predict_rows,predict_cols;diff_type=diff_type,K=K,S=S,sampling_type=sampling_type,lower_bound=lower_bound)
            """)
        # print "alpha = {}".format(self.alpha)
        # print "all_alpha = {}".format(self.all_alpha)
        try:
            self.saveAlpha()
            self.resultmat = csr_matrix(self.resultmat)
            if not hasattr(self, 'validation'):
                self.saveMatrixResult(self.saveFile)
        except:
            print('Did not save alphas and results.')

    def perform(self, trainIdx=None, predIdx=None):
        if hasattr(self, 'validation'):
            self.callAptRank(trainIdx, predIdx)
        else:
            os.chdir(self.resultDir)
            if(os.path.exists(self.saveFile + '.npz')):
                self.loadMatrixResult(self.saveFile)
            else:
                os.chdir(self.networkDir)
                self.callAptRank(trainIdx, predIdx)
        return

    def adjMatrixToJuliaSparse(self):
        _ = self.j.eval("using NPZ")
        try:
            nFileName = '{}_AdjacencyMatrixIJV'.format(self.network.name)
        except:
            nFileName = 'test_AdjacencyMatrixIJV'
        if not os.path.exists(nFileName + '.npz'):
            self.saveAdjMatrixIJV()
        _ = self.j.eval("""
            data = npzread("{}.npz");""".format(nFileName))
        _ = self.j.eval("""i = data["i"];
            i = convert(Array{Int64,1}, i);
            j = data["j"];
            j = convert(Array{Int64,1}, j);
            m = data["m"];
            n = data["n"];
            v = data["v"];
            adjMatrix = sparse(i,j,v,m,n);
            """)
        print "Adjacency matrix {} was loaded.".format(nFileName)
        print('Matrix size ', self.j.eval("size(adjMatrix)"))
        print('Matrix type ', self.j.eval("typeof(adjMatrix)"))
        print('Edge #', self.j.eval("size(v)"))
        return

    def getresult(self, problem='PredictionMatrix'):
        '''
        This returns the results in a specified format.

        problem=['PredictionMatrix', 'RelationshipList_Index', 'RelationshipList_Name']

        '''
        if problem == 'PredictionMatrix':
            return self.resultmat
        elif problem == 'ZScoreMatrix':
            try:
                return csr_matrix(zscore(self.resultmat, axis=1))
            except:
                return csr_matrix(zscore(self.resultmat.todense(), axis=1))
        elif problem == 'RelationshipList_Index':
            res = self.result
            if(res is None):
                res = {}
                resMat = self.resultmat
                indices = resMat.nonzero()
                for i in len(indices[0]):
                    row = indices[0][i]
                    col = indices[1][i]
                    res[(row, col)] = resMat[row, col]
            return res
        elif problem == 'RelationshipList_Name':
            currRes = self.getresult('RelationshipList_Index')
            res = {}
            positionNode = self.network.getPositionNode()
            for key in currRes:
                row = positionNode[key[0]]
                col = positionNode[key[1]]
                res[(row, col)] = currRes[key]
            return res
        else:
            raise ValueError('This return type is not supported')

    def saveAdjMatrixIJV(self):
        if type(self.network) == csr_matrix:
            adjMatrix = self.network
        else:
            adjMatrix = self.network.getAdjacencyMatrix()
        try:
            nFileName = '{}_AdjacencyMatrixIJV'.format(self.network.name)
        except:
            nFileName = 'test_AdjacencyMatrixIJV'

        # Get indices of nonzero values
        ind_i, ind_j = adjMatrix.nonzero()
        # Get values
        v = np.array(adjMatrix[ind_i, ind_j])[0]
        # Julia is 1-based indexing
        ind_i += 1
        ind_j += 1
        np.savez(nFileName, i=ind_i, j=ind_j, v=v,
                 m=adjMatrix.shape[0],
                 n=adjMatrix.shape[1])
        print 'adjMatrix was saved'
        print 'shape: {}'.format(adjMatrix.shape)
        print '{} edges'.format(len(v))
        return

    def saveAlpha(self):
        os.chdir(self.resultDir)
        fileName = self.saveFile + '.alpha'
        f = open(fileName,'w')
        f.write("{}\n{}\n".format(self.alpha,self.all_alpha))
        f.close()
        return

if(__name__ == '__main__'):
    adjMatrix = csr_matrix(np.matrix([[0, 1, 0, 0, 1],
                                      [1, 0, 1, 0, 0],
                                      [0, 1, 0, 1, 1],
                                      [0, 0, 1, 0, 1],
                                      [1, 0, 0, 1, 0]]))
    adjMatrix = rand(100, 100, format='csr', random_state=1)
    adjMatrix = adjMatrix + adjMatrix.T
    aptrank = AptRank(adjMatrix)
    aptrank.perform()
    Xa = aptrank.getresult()
    alpha = aptrank.alpha
    print(Xa)
    print(alpha)
