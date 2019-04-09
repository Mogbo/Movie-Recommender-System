import os
from pyspark import SparkContext

def readFileWithHeader(fileName, sc):
    """
    Takes in a file and returns and RDD with each line as an element of the RDD. The header file is skipped
    :param fileName: string
    :param sc: SparkContext
    :return: RDD
    """

    raw_data = sc.textFile(fileName)
    header = raw_data.take(1)[0]

    actual_data = raw_data.filter(lambda line: line != header) \
        .map(lambda line: line.split(","))\
        .map(lambda tokens: tuple(tokens))

    return actual_data



if __name__ == '__main__':
    N = 40 # Number of partitions


    sc = SparkContext('local[*]', 'User Based Filtering')

    # File Paths
    dataFolder = os.path.join('..', 'the-movies-dataset')

    ratingsPath = os.path.join(dataFolder, 'ratings.csv')
    moviesPath = os.path.join(dataFolder, 'movies.csv')
    moviesDataPath = os.path.join(dataFolder, 'movies_metadata_clean.csv')

    ratingsRDD = readFileWithHeader(ratingsPath, sc)  \
        .map(lambda (userID, movieID, rating, timeStamp): (eval(userID), eval(movieID), float(rating))).cache() # ratings RDD is (userID, movieID, rating)
    moviesRDD = readFileWithHeader(moviesPath, sc) \
        .map(lambda tup: [eval(tup[0])]+list(tup[1:])).map(tuple) # moviesRDD is (movieId, title, genres)
    moviesDataRDD = readFileWithHeader(moviesDataPath, sc) \
        .map(lambda tup: [eval(tup[0])]+list(tup[1:])).map(tuple) # moviesDataRDD

    ratingsByMovie = ratingsRDD.map(lambda (uID, mID, rating): (mID, (uID, rating)))

    # Now we extract the movies that are present in the metadata file. We then use a join
    desiredMovieRDD = moviesDataRDD.map(lambda tup: (tup[0], 1)).join(ratingsByMovie, numPartitions=N) # format (mID, (1, (uID, rating))
    desiredMovieRDD = desiredMovieRDD.mapValues(lambda (plHol, ratingsTup): ratingsTup) # format (mID, (uID, rating))
    desiredMovieRDD = desiredMovieRDD.map(lambda (mID, (uID, rating)): (uID, mID, rating)) # format (uID, mID, rating)

    print(desiredMovieRDD).take(1)
    print(desiredMovieRDD.count())

    smallData = desiredMovieRDD.takeSample(False, 450000, seed=4)

    with open(os.path.join(dataFolder, 'small-data/train'), 'w') as f:
        for item in smallData[:400000]:
            f.write(str(item)+"\n" )

    with open(os.path.join(dataFolder, 'small-data/test'), 'w') as f:
        for item in smallData[400000:]:
            f.write(str(item)+"\n" )