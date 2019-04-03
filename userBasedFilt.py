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



if __name__ == '__main__':
    sc = SparkContext('local[*]', 'User Based Filtering')

    # File Paths
    datasetsPath = os.path.join('..', 'ml-latest-small')
    ratingsPath = os.path.join(datasetsPath, 'ratings.csv')


    ratingsRDD = readRatingsData(ratingsPath, sc).cache() # ratings RDD is (userID, movieID, rating)

    #print(ratingsRDD.take(5))
