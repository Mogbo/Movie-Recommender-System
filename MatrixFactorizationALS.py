import os
from pyspark import SparkContext
import pandas as pd
import sys,argparse
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'ALS Matrix Factorization.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--saveTo' ,help = 'File where results are to be stored', default='../results/defaultResultsFile.csv')

    args = parser.parse_args()

    N = 40 # Number of partitions


    sc = SparkContext('local[*]', 'User Based Filtering')

    # File Paths
    dataFolder = os.path.join('..', 'the-movies-dataset/small-data/')

    train = sc.textFile(dataFolder+'train').map(eval)
    val = sc.textFile(dataFolder+'test').map(eval)

    numIterationsTup = (3,5,7,10,13)
    rankTup = tuple(range(2,21))
    lamTup = (0.01, 0.03, 0.1, 0.3, 1., 3, 10, 30, 100, 300, 1000)

    trainMSEList = []
    valMSEList = []

    for numIterations in numIterationsTup:
        for rank in rankTup:
            for lam in lamTup:

                # Build the recommendation model using Alternating Least Squares
                blocks = -1
                model = ALS.train(train, rank, iterations=numIterations, lambda_=lam, nonnegative=False, blocks=blocks, seed=5)

                # Evaluate the model on training data
                trainTruncated= train.map(lambda p: (p[0], p[1]))
                trainPreds = model.predictAll(trainTruncated).map(lambda r: ((r[0], r[1]), r[2]))
                trainRatesAndPreds = train.map(lambda r: ((r[0], r[1]), r[2])).join(trainPreds)
                trainMSE = trainRatesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
                trainMSEList.append(trainMSE)

                # Evaluate the model on validation data
                valTruncated= val.map(lambda p: (p[0], p[1]))
                valPreds = model.predictAll(valTruncated).map(lambda r: ((r[0], r[1]), r[2]))
                valRatesAndPreds = val.map(lambda r: ((r[0], r[1]), r[2])).join(valPreds)
                valMSE = valRatesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
                valMSEList.append(valMSE)

                print("numIterations = %d. Rank = %d. lam = %.4f. Train MSE = %.4f. Val MSE = %.4f" %(numIterations, rank, lam, trainMSE, valMSE))

    # Analyze and Save
    df = pd.DataFrame({'numIterations': numIterationsTup, 'rank': rankTup, 'lambda_': lamTup, 'trainMSE':trainMSEList, 'valMSE':valMSEList})
    df.to_csv(args.saveTo)



    '''
    # Save and load model
    model.save(sc, "target/tmp/myCollaborativeFilter")
    sameModel = MatrixFactorizationModel.load(sc, "target/tmp/myCollaborativeFilter")
    '''
