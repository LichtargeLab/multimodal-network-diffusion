#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Jul 20, 2018

@author: Chih-Hsu Lin, Meng Liu

This script is for converting output predictions from prospective experiment to a html

"""
import sys
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
PROJECT_DIR = os.getenv("PROJECT_DIR")
sys.path.append(PROJECT_DIR)
import paths
from IPython import embed
import pandas as pd
from glob import glob as glob
from decimal import Decimal
import argparse
import numpy as np


def load_prediction(alg, mode_number, query_mode, pred_mode, file=""):
    '''
    Load prediction files of prospective experiment.
    Currently, only (`query_mode`,pred_mode`)==('Disease','Gene'), 
    ('Disease','Chemical'), ('Gene','Chemical') are available.

    Parameters
    ----------
    alg: string, {'RW', 'GID', 'AR'}
        Specify the used algorithm for prediction
    mode_number: int, {1, 3, 6}
        Specify the mode number of used networks for prediction
    query_mode: string, {'Disease', 'Gene'}
        Specify the mode of query
    pred_mode: string, {'Gene', 'Chemical'}
        Specify the mode of prediction
    file: string,
        Specify the path to the prediction file. The default file will be used if not provided.

    Returns
    -------
    Pandas.DataFrame
        The prediction results of specified setting.
        Each row is one predicted entity.
        ID, Synonyms, and Degree are the attributes of the predicted entities. 
        Each query entity has one column of prediction score, and 
        one column of whether it exists in network.
        Higher scores means the query entiy and predicted entity are more likely to be associated.
    '''
    if mode_number == 3:
        net = 'CTD2016CG_CTD2016CD_CTD2016DG'
    elif mode_number == 6:
        net = 'CTD2016CG_CTD2016CD_CTD2016DG_STRING10_900_MeSH2016DD_MeSH2016CC'
    elif query_mode.lower() == 'disease' and pred_mode.lower() == 'gene':
        net = 'CTD2016DG'
    elif query_mode.lower() == 'disease' and pred_mode.lower() == 'chemical':
        net = 'CTD2016CD'
    elif query_mode.lower() == 'gene' and pred_mode.lower() == 'chemical':
        net = 'CTD2016CG'
    if alg == 'RW':
        alg = 'RndWalk'
    elif alg == 'GID':
        alg = 'Diffusion'
    elif alg == 'AR':
        alg = 'AptRank'
    pred_files = glob(paths.VALIDATION_RESULT_DIR +
                      '/prospective/Diffusion2018predictions/{}?_{}_Degree_1*qMode*{}*pMode*{}*_pred.tsv'.format(alg, net, query_mode, pred_mode))
    # print(pred_files)
    if len(pred_files) < 1:
        print(load_prediction.__doc__)
        raise ValueError(
            'Make sure precomputed prediction files were downloaded and parameters were correct.')
    df = pd.read_csv(pred_files[0], sep="\t")
    return df, alg, net


def output_html(df, alg, mode, net, query, pred, outpath='.'):
    '''
    Convert predictions from prospective experiments to a html file

    Parameters
    ----------
    df: pandas.DataFrame
        The DataFrame of predictions from prospective experiments
    alg: string
        Specify the used algorithm for prediction
    mode: string, {'1', '3', '6'}
        Specify the mode number of used networks for prediction
    query: string, {'Disease', 'Gene'}
        Specify the mode of query
    pred: string, {'Gene', 'Chemical'}
        Specify the mode of prediction
    outpath: string
        Specify the path for outputing html file. The default is '.'.

    '''
    df['Synonyms'][df['Synonyms'].isnull()] = df['ID'][df['Synonyms'].isnull()]
    df.index = df['Synonyms']
    df = df.sort_values(df.columns[3], ascending=False)
    query_names = df.index
    query_names_string = "var query_names = [" + ",".join(["\"" + s.replace(
        ",", "").replace("score_", "") + "\"" for s in query_names]) + "];\n"
    pred_names = [list(df.columns)[i] for i in range(3, df.shape[1], 2)]
    pred_names_string = "var pred_names = [" + ",".join(["\"" + s.replace(
        ",", "").replace("score_", "") + "\"" for s in pred_names]) + "];\n"
    ranks = df[pred_names].rank(ascending=False).values
    ranks_string_list = []
    for row in ranks:
        temp = "[" + ",".join([str(i) for i in row]) + "]\n"
        ranks_string_list.append(temp)
    ranks_string = "var mat = [" + ",".join(ranks_string_list) + "];\n"
    color_pos = []
    for i in range(4, df.shape[1], 2):
        labels = df.iloc[:, i].values
        for j in range(len(labels)):
            if labels[j] > 0:
                color_pos.append([j + 1, i / 2 - 1])
    color_pos_string_list = []
    for row in color_pos:
        temp = "[" + ",".join([str(i) for i in row]) + "]\n"
        color_pos_string_list.append(temp)
    color_pos_string = "var color_pos = [" + \
        ",".join(color_pos_string_list) + "];\n"
    alg_string = "var algo = \"%s\";\n" % alg
    net_string = "var net = \"%s\";\n" % net
    query_string = "var query = \"%s\";\n" % query
    pred_string = "var pred = \"%s\";\n" % pred
    name = "_".join([alg, str(mode) + 'mode', query, pred]) + ".html"
    outfile = "/".join([outpath, name])
    rptr = open(PROJECT_DIR + "/src/Validation_Methods/template.html", "r")
    wptr = open(outfile, "w")
    lines = rptr.readlines()
    for line in lines:
        wptr.write(line)
        if "place names and values here" in line:
            wptr.writelines([query_names_string, pred_names_string, ranks_string,
                             color_pos_string, alg_string, net_string, query_string, pred_string])
    rptr.close()
    wptr.close()
    # print("html file has been written to "+outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("alg", choices=["RW", "GID", "AR"],
                        help="load results from which algorithm")
    parser.add_argument("mode", type=int, choices=[1, 3, 6],
                        help="the mode number of used networks for prediction")
    parser.add_argument("query", choices=['Disease', 'Gene'],
                        help="the mode of query")
    parser.add_argument("pred", choices=['Gene', 'Chemical'],
                        help="the mode of prediction")
    parser.add_argument("-i", metavar="infile", type=str, nargs=1, dest="infile",
                        help="use which file to load the prediction", default="")
    parser.add_argument("-o", metavar="outpath", type=str, nargs=1, dest="outpath",
                        help="use which directory to store the html file", default=".")
    args = vars(parser.parse_args())
    alg = args['alg']
    mode = args['mode']
    query = args['query']
    pred = args['pred']
    infile = args['infile']
    df, alg, net = load_prediction(alg, mode, query, pred, file=infile)
    output_html(df, alg, mode, net, query, pred)
