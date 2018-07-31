#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Jun 17, 2016

@author: dmkonecki
'''
import os,sys
import csv
from copy import deepcopy
from IPython import embed

# https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072
maxInt = sys.maxsize
decrement = True

while decrement:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.
    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True

class GeneralParser(object):
    '''
    This parser is intended to be a powerful easily modifiable tool for reading
    files with a fixed delimiter where only certain fields are of interest.
    '''

    def __init__(self, fileName, delimiter, numHeaderLines, columnNames,
                 skipChar):
        '''
        Constructor:
        Takes the:
        File path to the file of interest
        Delimiter separating columns in the file
        Column names of interest
        '''
        if('/' in fileName):
            self.__filePath = fileName
        else:
            self.__filePath = os.getcwd() + '/' + fileName
        self.__delimiter = delimiter
        self.__numHeaderLines = numHeaderLines
        self.__columnNames = columnNames
        self.__data = {x: [] for x in self.__columnNames}
        self.__count = 0
        self.__skipChar = skipChar

    def setFilePath(self, fileName):
        if('/' in fileName):
            self.__filePath = fileName
        else:
            self.__filePath = os.getcwd() + '/' + fileName

    def setDelimiter(self, delimiter):
        self.__delimiter = delimiter

    def setNumHeaderLines(self, num):
        self.__numHeaderLines = num

    def setColumnNames(self, columnNames):
        self.__columnNames = list(set(columnNames))
        self.__data = {x: [] for x in self.__columnNames}

    def addColumnName(self, columnName):
        self.__columnNames = list(set(self.__columnNames + [columnName]))
        self.__data[columnName] = []

    def removeColumnName(self, columnName):
        if(columnName in self.__columnNames):
            index = self.__columnNames.index(columnName)
            del(self.__columnNames[index])
            del(self.__data[columnName])

    def getFilePath(self):
        return deepcopy(self.__filePath)

    def getDelimiter(self):
        return deepcopy(self.__delimiter)

    def getNumHeaderLines(self):
        return deepcopy(self.__numHeaderLines)

    def getColumnNames(self):
        return deepcopy(self.__columnNames)

    def getData(self):
        return deepcopy(self.__data)

    def updateData(self, data):
        self.__data = data

    def getCount(self):
        return deepcopy(self.__count)

    def displayFilePath(self):
        print('The file for this parser is: {}'.format(self.__filePath))

    def displayDelimiter(self):
        print(
            'The delimiter specified for this file is: {}'.format(self.__delimiter))

    def displayNumHeaderLines(self):
        print('The parser is expecting {} header lines.'.format(
            self.__numHeaderLines))

    def displayColumnNames(self):
        columnStr = ', '.join(self.__columnNames)
        print('The columns specified for this parser are: ' + columnStr)

    def displayData(self):
        print('Data:')
        print(self.__data)
        print('There were {} data lines parsed'.format(
            len(self.__data.keys())))

    def displayCount(self):
        print('Lines parsed: {}'.format(self.__count))

    def save(self, filename):

        with open(filename, 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter=self.__delimiter)
            self.__columnNames[0]='#'+self.__columnNames[0]
            writer.writerow(self.__columnNames)

            self.__columnNames[0]=self.__columnNames[0][1:]
            lists=[self.__data[col] for col in self.__columnNames]
            for i in range(self.__count):
                writer.writerow([x[i] for x in lists])

    def __str__(self):
        objStr = 'File path: {}'.format(self.__filePath)
        objStr += '\nDelimiter: {}'.format(self.__delimiter)
        objStr += '\nNumber of Header Lines: {}'.format(self.__numHeaderLines)
        objStr += '\nColumns: {}'.format(', '.join(self.__columnNames))
        objStr += '\nData:'
        for key in self.__data:
            objStr += '\n{}: '.format(key)
            objStr += ', '.join(self.__data[key])
        objStr += '\nLines Parsed: {}'.format(self.__count)
        return objStr

    def __del__(self):
        del self

    def __cmp__(self, x):
        raise StandardError('Unsupported function!')

    def __repr__(self):
        return str(self).replace('\n', ';')

    def parse(self):
        fileHandle = open(self.__filePath, 'rb')
        fileReader = csv.reader(fileHandle, delimiter=self.__delimiter)
        count = 0
        if(self.__numHeaderLines == 0):
            indices = {x: int(x) for x in self.__columnNames}
        else:
            indices = {}
        for line in fileReader:
            if line[0][0] == self.__skipChar:
                line[0] = line[0][1:]
            if(count < self.__numHeaderLines):
                # print line
                for colName in self.__columnNames:
                    if colName in line:
                        if colName not in indices:
                            indices[colName] = line.index(colName)
                # print indices
            elif(line[0].startswith(self.__skipChar)):
                continue
            else:
                for column in self.__columnNames:
                    self.__data[column].append(line[indices[column]])
            count += 1
        count -= self.__numHeaderLines
        fileHandle.close()
        self.__count = count
        return self.getCount(), self.getData()
