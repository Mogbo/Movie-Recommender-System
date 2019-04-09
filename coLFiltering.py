# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:12:16 2019

@author: Iuliia_K
"""

import os
from pyspark import SparkContext
import pandas as pd

def readRatingsData(ratings_file, sc):
    """
    Takes in a ratings_csv file and returns and RDD of elements (userID, movieID, rating)
    :param ratings_file:
    :param sc:
    :return:
    """

    ratings_raw_data = sc.textFile(ratings_file)
    ratings_raw_data_header = ratings_raw_data.take(1)[0]

    ratings_data = ratings_raw_data.filter(lambda line: line != ratings_raw_data_header) \
        .map(lambda line: line.split(","))\
        .map(lambda tokens: (tokens[0], tokens[1], tokens[2]))

    return ratings_data



def readTitleData(title_file, sc):
    """
    Takes in a ratings_csv file and returns and RDD of elements (userID, movieID, rating)
    :param ratings_file:
    :param sc:
    :return:
    """

    title_raw_data = sc.textFile(ratings_file)
    title_raw_data_header = title_raw_data.take(1)[0]

    title_data = title_raw_data.filter(lambda line: line != title_raw_data_header) \
        .map(lambda line: line.split(","))\
        .map(lambda tokens: (tokens[5], tokens[8]))

    return rtitle_data


if __name__ == '__main__':
    sc = SparkContext('local[40]', 'User Based Filtering')

    # File Paths
    datasetsPath = os.path.join('.', 'the-movies-dataset')
    ratingsPath = os.path.join(datasetsPath, 'ratings.csv')


    ratingsRDD = readRatingsData(ratingsPath, sc).cache() # ratings RDD is (userID, movieID, rating)

    moviesPath = os.path.join(datasetsPath, 'movies_metadata.csv')


    moviesRDD = readTitleData(moviesPath, sc).cache() # ratings RDD is (userID, movieID, rating)
