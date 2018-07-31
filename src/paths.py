#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Jul 20, 2018

@author: Chih-Hsu Lin

This script is for loading environment variables.
"""
import os
import sys
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
PROJECT_DIR = os.getenv("PROJECT_DIR")
NETWORK_DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'mapped_networks')
NETWORK_MAPPING_FILE = NETWORK_DATA_DIR + \
    '/Mapping_Ccgdd16Sx91Extended_101217.tsv'
NETWORK_TMP_DIR = os.path.join(PROJECT_DIR, 'data', 'interim', 'networks')
ALGORITHM_RESULT_TMP_DIR = os.path.join(
    PROJECT_DIR, 'data', 'interim', 'algorithms')
VALIDATION_RESULT_DIR = os.path.join(PROJECT_DIR, 'data', 'results')
sys.path.append(PROJECT_DIR + '/src/Networks')
sys.path.append(PROJECT_DIR + '/src/Algorithms')
sys.path.append(PROJECT_DIR + '/src/Validation_Methods')
