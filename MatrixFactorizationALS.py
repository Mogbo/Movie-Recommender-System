import os
from pyspark import SparkContext
import pandas as pd
import sys,argparse
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from time import time

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'ALS Matrix Factorization.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--saveTo' ,help = 'File where results are to be stored', default='../results/defaultResultsFile.csv')
    parser.add_argument('--kf', help='which fold', default=0, type=int)

    args = parser.parse_args()

    N = 28 # Number of partitions


    sc = SparkContext('local[*]', 'User Based Filtering')

    # File Paths
    dataFolder = os.path.join('..', 'the-movies-dataset/folds/')

    fold0 = sc.textFile(dataFolder+'fold0').map(eval)
    fold1 = sc.textFile(dataFolder + 'fold1').map(eval)
    fold2 = sc.textFile(dataFolder + 'fold2').map(eval)
    fold3 = sc.textFile(dataFolder + 'fold3').map(eval)
    fold4 = sc.textFile(dataFolder + 'fold4').map(eval)

    folds = [fold0, fold1, fold2, fold3, fold4]

    train_folds = [folds[j] for j in range(len(folds)) if j is not args.kf]
    train = train_folds[0]

    for fold in train_folds[1:]:
        train = train.union(fold)
    train.cache()

    val = folds[args.kf].cache()

    numIterationsTup = (20,)
    rankTup = tuple(range(2,9))
    rankTup = (3,)
    lamTup = (3.0,)

    trainMSEList = []
    valMSEList = []

    for numIterations in numIterationsTup:
        for rank in rankTup:
            for lam in lamTup:

                start = time()
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

                print("%.4f" % (time()-start))
                print("numIterations = %d. Rank = %d. lam = %.4f. Train MSE = %.4f. Val MSE = %.4f" %(numIterations, rank, lam, trainMSE, valMSE))

    '''
    # Analyze and Save
    df = pd.DataFrame({'numIterations': numIterationsTup, 'rank': rankTup, 'lambda_': lamTup, 'trainMSE':trainMSEList, 'valMSE':valMSEList})
    df.to_csv(args.saveTo)
    '''



    '''
    # Save and load model
    model.save(sc, "target/tmp/myCollaborativeFilter")
    sameModel = MatrixFactorizationModel.load(sc, "target/tmp/myCollaborativeFilter")
    '''
