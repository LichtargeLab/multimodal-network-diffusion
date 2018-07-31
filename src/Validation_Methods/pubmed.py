#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Aug 8, 2017

@author: Chih-Hsu Lin
'''
from Bio import Entrez
from Bio import Medline
from scipy.stats import fisher_exact
from IPython import embed
from multiprocessing import cpu_count, Process, Queue
import numpy as np
import os
import cPickle as pickle

def sigAssociatedPubMed(term, pThreshold=0.001, pubmedTotal=27000000):
    # if (int(searchPubmed(term)) + searchGoogleScholar(term)) > 0:
    obs = [[0, 0], [0, 0]]
    obs[0][0] = int(searchPubmed(term))
    obs[0][1] = int(searchPubmed(term[0])) - obs[0][0]
    obs[1][0] = int(searchPubmed(term[1])) - obs[0][0]
    obs[1][1] = pubmedTotal - obs[0][0] - obs[0][1] - obs[1][0]
    try:
        oddsratio, pvalue = fisher_exact(obs, alternative='greater')
    except:
        print term, 'cannot compute fisher_exact'
        print obs
        pvalue = 1
    if pvalue < pThreshold:
        return True
    else:
        return False


def pAssociatedPubMed(term, pThreshold=0.001, pubmedTotal=27000000):
    # if (int(searchPubmed(term)) + searchGoogleScholar(term)) > 0:
    obs = [[0, 0], [0, 0]]
    obs[0][0] = int(searchPubmed(term))
    obs[0][1] = int(searchPubmed(term[0])) - obs[0][0]
    obs[1][0] = int(searchPubmed(term[1])) - obs[0][0]
    obs[1][1] = pubmedTotal - obs[0][0] - obs[0][1] - obs[1][0]
    try:
        oddsratio, pvalue = fisher_exact(obs, alternative='greater')
    except:
        print term, 'cannot compute fisher_exact'
        print obs
        pvalue = 1
    return pvalue, obs


def searchPubmed(term, max_count=1):
    if hasattr(term, '__iter__'):
        # term = ["\""+x+"\"" for x in term]
        term = [x.upper() for x in term]
        term = ["(MIR{} or MIR-{})".format(x.split('MIR')[1].replace("-", ""), x.split('MIR')
                                           [1].replace("-", "")) if 'MIR' in x else "\"" + x + "\"[Text Word]" for x in term]
        term = " AND ".join(term)
        # term = "\""+term+"\""
    elif 'MIR' in term.upper():
        term = term.upper()
        term = "(MIR{} or MIR-{})".format(term.split('MIR')
                                          [1].replace("-", ""), term.split('MIR')[1].replace("-", ""))
    else:
        term = term.upper()
        term = "\"" + term + "\"[Text Word]"
    Entrez.email = 'A.N.Other@example.com'
    n_search = 0
    while 1:
        n_search += 1
        if n_search >= 10:
            print term, 'Cannot be searched for {} times.'.format(n_search)
            return 0
        try:
            h = Entrez.esearch(db='pubmed', retmax=max_count, term=term)
            result = Entrez.read(h)
            # print term
            # print result['Count']
            # embed()
            try:
                if len(result['ErrorList']['PhraseNotFound']) > 0:
                    return 0
                else:
                    print term, result['ErrorList']
                    return (result['Count'])
            except:
                return (result['Count'])
        except:
            pass


def searchEntrezGene(entrezID):
    # entrezID = 100379158
    Entrez.email = 'A.N.Other@example.com'
    handle = Entrez.esummary(db="gene", id=entrezID)
    try:
        record = Entrez.read(handle)
        handle.close()
        name = record[u'DocumentSummarySet'][u'DocumentSummary'][0]['Name']
        return name
    except:
        print('No results found with Entrez ID {}'.format(entrezID))
        return

def multithreadSearch(terms, numThreads=10):
    # Create a queue
    q = Queue()
    if cpu_count() < numThreads:
        numThreads = cpu_count() - 1
    # Build a shared dictionary of adjMatrix by manager
    # print "Performing multithread search of {} cores".format(numThreads)
    numTerms = len(terms)
    for i in range(numThreads):
        argsInput = terms[
            int(i * numTerms / numThreads):int((i + 1) * numTerms / numThreads)], i
        p1 = Process(
            target=search_multiprocess, args=(argsInput, q))
        p1.start()
    results = list(np.zeros(numThreads))
    for i in range(numThreads):
        # set block=True to block until we get a result
        output = q.get(True)
        # Put the results in the original order
        results[output[1]] = output[0]
    results = [item for sublist in results for item in sublist]
    return results


def search_multiprocess(argsInput, q):
    subInputs, i = argsInput
    output = []
    n = 0
    total = len(subInputs)
    for term in subInputs:
        # n+=1
        # if n%1 ==0:
        #    print('Performing search on {}/{} nodes in process {}'.format(n,total,i))
        output.append(searchPubmed(term))
    q.put([output, i])


def multithreadAssociation(terms, numThreads=10, load_path=None, save_path=None, precomputed_dict={}):
    if load_path and len(precomputed_dict) == 0:
        precomputed_dict.update(loadPubMedResults(load_path))
    # Create a queue
    q = Queue()
    if cpu_count() < numThreads:
        numThreads = cpu_count() - 1    
    # print "Performing multithread search of {} cores".format(numThreads)
    other_terms = [t for t in terms if (t not in precomputed_dict) and (t[::-1] not in precomputed_dict)]
    print("Searching for {} terms. {} terms were precomputed.".format(
        len(other_terms), len(terms) - len(other_terms)))
    numTerms = len(other_terms)
    if numTerms < numThreads:
        numThreads = numTerms
    for i in range(numThreads):
        argsInput = other_terms[
            int(i * numTerms / numThreads):int((i + 1) * numTerms / numThreads)], i
        p1 = Process(
            target=pAssociation_multiprocess, args=(argsInput, q))
        p1.start()
    result_dict = {}
    for i in range(numThreads):
        # set block=True to block until we get a result
        output = q.get(True)
        result_dict.update(output)
    result_dict.update(precomputed_dict)
    try:
        results = [result_dict[t] if t in result_dict else result_dict[t[::-1]] for t in terms]
    except:
        embed()
    if save_path and result_dict != precomputed_dict:
        savePubMedResults(result_dict, save_path)
    return results, result_dict


def pAssociation_multiprocess(argsInput, q):
    subInputs, i = argsInput
    # output = []
    # n = 0
    # total = len(subInputs)
    result_dict = {}
    for term in subInputs:
        # n+=1
        # if n%1 ==0:
        #    print('Performing search on {}/{} nodes in process {}'.format(n,total,i))
        result_dict[term] = pAssociatedPubMed(term)
    q.put(result_dict)


def savePubMedResults(result_dict, output_path):
    '''
    Save PubMed search results
    '''
    pFile = open(output_path + 'PubMedDict.pkl', 'wb')
    pickle.dump(result_dict, pFile, protocol=pickle.HIGHEST_PROTOCOL)
    pFile.close()
    print('Saving precomputed saved PubMed search results of {} term pairs'.format(len(result_dict)))
    return


def loadPubMedResults(load_path):
    '''
    Load PubMed search results
    '''
    pFN = load_path + 'PubMedDict.pkl'
    try:
        pFile = open(pFN, 'rb')
        loadedData = pickle.load(pFile)
        pFile.close()
        print('Loading precomputed saved PubMed search results of {} term pairs'.format(len(loadedData)))
        return loadedData
    except:
        return {}

if __name__ == '__main__':
    terms = [['SLC22A1', 'Cisplatin'], ['SLC22A1', 'Nicotine'], ['Breast cancer', 'CAD'],[
        'SLC22A1', 'Cocaine'], ['TP53', 'PTEN'], ["Coronary Artery Disease", 'EDAR'], ['liver cancer','MIR-340'],['Coronary Artery Disease','GSTT1'],['Colon cancer', 'Tamoxifen']]
    result, result_dict = multithreadAssociation(terms, 2, load_path='./', save_path='./')
    print(result)
    terms = [('TP53','PIK3CA')]
    result, result_dict = multithreadAssociation(terms, 2, load_path='./', save_path='./', precomputed_dict = result_dict)
    print(result)
