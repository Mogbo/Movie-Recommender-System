# -*- coding: utf-8 -*-
import numpy as np
from time import time
import sys,argparse
from pyspark import SparkContext
from operator import add
from pyspark.mllib.random import RandomRDDs


def swap((x,y)):
    """ Swap the elements of a pair tuple.
    """ 
    return (y,x)
   
def predict(u,v):
    """ Given a user profile uprof and an item profile vprof, 
        predict the rating given by the user to this item

        Inputs are:
           -u: user profile, in the form of a numpy array
           -v: item profile, in the form of a numpy array

        The return value is
           - the inner product <u,v>
    """
    return np.dot(u,v)
    

def pred_diff(r,u,v):
    """ Given a rating, a user profile u and an item profile v, 
        compute the difference between the prediction and actual rating

        Inputs are:
           -r: the rating a user gave to an item
           -u: user profile, in the form of a numpy array
           -v: item profile, in the form of a numpy array

        The return value is the difference
          - δ =  <u,v> - r 
    """
    delta = predict(u,v) - r
    return  delta   

def gradient_u(delta,u,v):
    """ Given a user profile u and an item profile v, and the difference 
        in rating predictions δ, compute the gradient

              ∇_u l(u,v)   = 2 (<u,v> - r ) v 

             of the square error loss:
         
              l(u,v) = (<u,v> - r)^2

        Inputs are:
           -δ: the difference  <u,v> - r 
           -u: user profile, in the form of a numpy array
           -v: item profile, in the form of a numpy array

        The return value is 
          - The gradient w.r.t. u
    """
    return 2.0*delta*v



def gradient_v(delta,u,v):
    """ Given a user profile u and an item profile v, and the difference 
        in rating predictions δ, compute the gradient

              ∇_v l(u,v)= 2 (<u,v> - r) u 

        of the square error loss:

              l(u,v) = (<u,v> - r)^2

        Inputs are:
           -δ: the difference  <u,v> - r 
           -u: user profile, in the form of a numpy array
           -v: item profile, in the form of a numpy array

        The return value is 
          - the gradient w.r.t. v  
    """
    return 2.0*delta*u

def readRatings(file,sparkContext):
    """ Read the ratings from a file and store them in an rdd 
        containing tuples of the form:
                (i,j,rij)

        where i,j are integers and rij is a floating number.

        Inputs are:
             -The name of a file that contains the ratings, in form:
                   i,j,rij 

              per line
             -A Spark context

        The return value is the constructed rdd
    """
    return sparkContext.textFile(file).map(lambda x: tuple(x.split(','))).map(lambda (i,j,rij):(int(i),int(j),float(rij)))


def generateUserProfiles(R,d,seed,sparkContext,N):
    """ Generate the user profiles from rdd R and store them in an RDD containing tuples of the form 
            (i,ui)
        where u is a random np.array of dimension d.

        The random uis are generated using normalVectorRDD(), a function in RandomRDDs.
        
        Inputs are:
             - R: an RDD that contains the ratings in (user, item, rating) form
             - d: the dimension of the user profiles
             - seed: a seed to be used for in generating the random vectors
             - sparkContext: a spark context
             - N: the number of partitions to be used during joins, etc.

        The return value is an RDD containing the user profiles
    """
    # exctract user ids
    U = R.map(lambda (i,j,rij):i).distinct(numPartitions = N)
    numUsers = U.count()
    randRDD = RandomRDDs.normalVectorRDD(sparkContext, numUsers, d,numPartitions=N, seed=seed)
    U = U.zipWithIndex().map(swap)
    randRDD = randRDD.zipWithIndex().map(swap)
    return U.join(randRDD,numPartitions = N).values()

def generateItemProfiles(R,d,seed,sparkContext,N):
    """ Generate the item profiles from rdd R and store them in an RDD containing tuples of the form 
            (j,vj)
        where v is a random np.array of dimension d.

        The random uis are generated using normalVectorRDD(), a function in RandomRDDs.
        
        Inputs are:
             - R: an RDD that contains the ratings in (user, item, rating) form
             - d: the dimension of the user profiles
             - seed: a seed to be used for in generating the random vectors
             - sparkContext: a spark context
             - N: the number of partitions to be used during joins, etc.

        The return value is an RDD containing the item profiles
    """
    V = R.map(lambda (i,j,rij): j).distinct(numPartitions = N)
    numItems = V.count()
    randRDD = RandomRDDs.normalVectorRDD(sparkContext, numItems, d, numPartitions= N, seed = seed)
    V = V.zipWithIndex().map(swap)
    randRDD = randRDD.zipWithIndex().map(swap)
    
    return V.join(randRDD,numPartitions = N).values()


    


def joinAndPredictAll(R,U,V,N):
    """ Receives as inputs the ratings R, the user profiles U, and the items V, and constructs a joined RDD.

        Inputs are:
         - R: an RDD containing tuples of the form (i,j,rij)
         - U: an RDD containing tuples of the form (i,ui)
         - V: an RDD containing tuples of the form (j,vj)
         - N: the number of partitions to be used during joins, etc.

        The output is a joined RDD containing tuples of the form:

        (i,j,δij,ui,vj)

        where 
          δij = <u,v>-rij
        is the prediction difference.
        
        
        
       """

    joinedRDD = R.map(lambda (i,j,r): (i,(j,r))).join(U,numPartitions = N)\
                .map(lambda x: (x[0], ) + x[1])\
                .map(lambda (i,(j,r),u): (j,(i,r,u))).join(V,numPartitions = N)\
                .map(lambda x: (x[0], ) + x[1])\
                .map(lambda (j,(i,r,u),v): (i,j,pred_diff(r,u,v),u,v))
    
    return joinedRDD
    

def SE(joinedRDD):
    """ Receives as input a joined RDD as well as a λ and a μ and computes the MSE:

        SE(R,U,V) = Σ_{i,j in data} (<ui,vj>-rij)^2 

        The input is
         -joinedRDD: an RDD with tuples of the form (i,j,δij,ui,vj), 
         where δij = <ui,vj> - rij is the prediction difference.
1.
        The output is the SE.
    """
    return joinedRDD.map(lambda (i,j,d,u,v): d**2 ).reduce(add)
 
def normSqRDD(profileRDD,param):
    """ Receives as input an RDD of profiles (e.g., U) 
        as well as a parameter (e.g., λ) and computes the square of norms:
        λ Σ_i ||ui||_2^2         

        The input is:
          -profileRDD: an RDD of the form (i,u), 
           where i is an index and u is a numpy array
          -param: a scalar λ>0

        The return value is:
        λ Σ_i ||ui||_2^2         
    """
    ui = profileRDD.map(lambda (i,u): np.linalg.norm(u)).reduce(add)
    return param*ui


def adaptU(joinedRDD,gamma,lam,N):
    """ Receives as input a joined RDD 
        as well as a gain γ, and regularization parameters λ and μ,  
        and constructs a new RDD of user profiles of the form 
 
        ui = ui - γ  ∇_ui RegSE(R,U,V)

        where 

        RegSE(R,U,V) = Σ_{i,j in R} (<ui,vj>-rij)^2 + λ Σ_i ||ui||_2^2 + μ Σ_j ||vj||_2^2                
                

        Inputs are
         -joinedRDD: an RDD with tuples of the form (i,j,δij,ui,vj), 
          where δij = <ui,vj> - rij
         -gamma: the gain γ
         -lam: the regularization parameter λ
         -N: the number of partitions to be used in reduceByKey operations

        The return value  is an RDD with tuples of the form (i,ui). 
        The returned rdd contains exactly N partitions.
    """
    
 
    profileRDD= joinedRDD.map(lambda (i,j,delta, u,v): (i,u))\
               .reduceByKey(lambda i,u: i)
               
    grad = joinedRDD.map(lambda (i,j,delta, u,v): (i, gradient_u(delta,u,v)+ 2.0*lam*u)).reduceByKey(add)
    Ui = profileRDD.join(grad, numPartitions = N).map(lambda (i, (u,x)): (i,(u - gamma*x)))
   
    return Ui






def adaptV(joinedRDD,gamma,mu,N):
    """ Receives as input a joined RDD 
        as well as a gain γ, and regularization parameters λ and μ,  and constructs a new RDD of user profiles of the form 
 
        ui = ui - γ  ∇_ui RegSE(R,U,V)

        where 

        RegSE(R,U,V) = Σ_{i,j in R} (<ui,vj>-rij)^2 + λ Σ_i ||ui||_2^2 + μ Σ_j ||vj||_2^2                
                

        Inputs are
         -joinedRDD: an RDD with tuples of the form (i,j,δij,ui,vj), where δij = <ui,vj> - rij
         -gamma: the gain γ
         -mu: the regularization parameter μ
         -N: the number of partitions to be used in reduceByKey operations

        The return value  is an RDD with tuples of the form (j,vj). The returned rdd contains exactly N partitions.
    """
    
    profileRDD= joinedRDD.map(lambda (i,j,delta, u,v): (j,v))\
                .reduceByKey(lambda j,v: j)
    grad = joinedRDD.map (lambda (i,j,delta, u,v): (j, gradient_v(delta,u,v)+ 2.0*mu*v))\
           .reduceByKey(add)
    Vj = profileRDD.join(grad, numPartitions = N).map(lambda (j, (v,x)): (j, (v - gamma*x)))
    return Vj

if __name__=="__main__":
        

    parser = argparse.ArgumentParser(description = 'Parallele Matrix Factorization.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data',help = 'Directory containing folds. The folds should be named fold0, fold1, ..., foldK.')
    parser.add_argument('folds',type = int,help = 'Number of folds')
    parser.add_argument('--gain',default=0.001,type=float,help ="Gain")
    parser.add_argument('--power',default=0.2,type=float,help ="Gain Exponent")
    parser.add_argument('--epsilon',default=1.e-99,type=float,help ="Desired objective accuracy")
    parser.add_argument('--lam',default=1.0,type=float,help ="Regularization parameter for user features")
    parser.add_argument('--mu',default=1.0,type=float,help ="Regularization parameter for item features")
    parser.add_argument('--d',default=10,type=int,help ="Number of latent features")
    parser.add_argument('--outputfile',help = 'Output file')
    parser.add_argument('--maxiter',default=20,type=int, help='Maximum number of iterations')
    parser.add_argument('--N',default=40,type=int, help='Parallelization Level')
    parser.add_argument('--seed',default=1234567,type=int, help='Seed used in random number generator')
    parser.add_argument('--output',default=None, help='If not None, cross validation is skipped, and U,V are trained over entire dataset and store it in files output_U and output_V')

    verbosity_group = parser.add_mutually_exclusive_group(required=False)
    verbosity_group.add_argument('--verbose', dest='verbose', action='store_true')
    verbosity_group.add_argument('--silent', dest='verbose', action='store_false')
    parser.set_defaults(verbose=False)

 
    args = parser.parse_args()

    sc = SparkContext(appName='Parallel MF')
    
    if not args.verbose :
        sc.setLogLevel("ERROR")        

       

    folds = {}

    if args.output is None:
        for k in range(args.folds):
            folds[k] = readRatings(args.data+"/fold"+str(k),sc)
    else:
        folds[0] = readRatings(args.data,sc)

      

 
    cross_val_rmses = []
    for k in folds:
        train_folds = [folds[j] for j in folds if j is not k ]

        if len(train_folds)>0:
            train = train_folds[0]
            for fold in  train_folds[1:]:
                train=train.union(fold)
            train.repartition(args.N).cache()
            test = folds[k].repartition(args.N).cache()
            Mtrain=train.count()
            Mtest=test.count()
              
            print("Initiating fold %d with %d train samples and %d test samples" % (k,Mtrain,Mtest) )
        else:
            train = folds[k].repartition(args.N).cache()
            test = train
            Mtrain=train.count()
            Mtest=test.count()
            print("Running single training over training set with %d train samples. Test RMSE computes RMSE on training set" % Mtrain )
            
      

        i = 0
        change = 1.e99
        obj = 1.e99
    

        #Generate user profiles
        U = generateUserProfiles(train,args.d,args.seed,sc,args.N).cache()
        V = generateItemProfiles(train,args.d,args.seed,sc,args.N).cache()

        print "Training set contains %d users and %d items" %(U.count(),V.count())
        
        start = time()
        gamma = args.gain

        while i<args.maxiter and change > args.epsilon:
            i += 1

            joinedRDD = joinAndPredictAll(train,U,V,args.N).cache()
        
            oldObjective = obj
            obj = SE(joinedRDD) + normSqRDD(U,args.lam) + normSqRDD(V,args.lam)         
            change = np.abs(obj-oldObjective) 

            testRMSE = np.sqrt(1.*SE(joinAndPredictAll(test,U,V,args.N))/Mtest)

            gamma = args.gain / i**args.power
            U.unpersist()
            V.unpersist()
            U = adaptU(joinedRDD,gamma,args.lam,args.N).cache()
            V = adaptV(joinedRDD,gamma,args.mu,args.N).cache()

            now = time()-start
            print "Iteration: %d\tTime: %f\tObjective: %f\tTestRMSE: %f" % (i,now,obj,testRMSE)

            joinedRDD.unpersist()
            
        cross_val_rmses.append(testRMSE)

        train.unpersist()
        test.unpersist()

    if args.output is None:
       print "%d-fold cross validation error is: %f " % (args.folds, np.mean(cross_val_rmses))
    else:
       print "Saving U and V RDDs"
       U.saveAsTextFile(args.output+'_U')
       V.saveAsTextFile(args.output+'_V')
