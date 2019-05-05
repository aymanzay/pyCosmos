from dag import *
from spotapi import *
from dfops import *
import spotipy
import warnings
import pickle
import numpy as np
import pandas as pd
from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql import Row
import pyspark.sql.types as pst
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from sklearn.decomposition import PCA
#from sklearn.cluster import KMeans, MeanShift
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import KNeighborsClassifier  

warnings.simplefilter(action='ignore', category=FutureWarning)

#conf = SparkConf().setAppName("Cosmos").setMaster("spark//master:7077")
conf = SparkConf().setAppName("Cosmos")
conf = (conf.setMaster('local[*]')
        .set('spark.executor.memory', '4G')
        .set('spark.driver.memory', '4G')
        .set('spark.driver.maxResultSize', '4G'))
sc = SparkContext(conf=conf)

sql_sc = SQLContext(sc)


# Evaluate clustering by computing Within Set Sum of Squared Errors
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))


if __name__ == '__main__':
    sc.setLogLevel("ERROR")

    #import library
    with open('./library/aymanzay_lib.pkl', 'rb') as f:
        library = pickle.load(f)
        print('loaded')
    
    #create df duplicates
    s_main = sql_sc.createDataFrame(library)
    s_main = s_main.drop('track_href', 'uri', 'analysis', 'analysis_url', 'id', 'type')

    lib_analysis = library['analysis']
    ids = library['id']

    #create analysis dataframe
    analysis_df = refactorAnalysisDF(lib_analysis, ids)

    #lib_features = library.drop(columns=['analysis'])
    s_features = s_main

    #lib_tech = library.as_matrix(columns=['tempo', 'key', 'loudness', 'valence', 'time_signature', 'liveness', 'energy', 'danceability'])
    s_tech = s_main.select('tempo', 'key', 'loudness', 'valence', 'time_signature', 'liveness', 'energy', 'danceability')

    #lib_song_info = library.as_matrix(columns=['artist', 'genre', 'speechiness', 'acousticness', 'instrumentalness'])
    s_info = s_main.select('speechiness', 'acousticness', 'instrumentalness')



    #get initial root vectors
    #init meanshift clustering model
    '''
    model = KMeans.train(d, k=12, maxIterations=10)
    # Shows the result.
    centers = model.clusterCenters()
    print("Cluster Centers: ")
    for center in centers:
        print(center)
    '''
    #clusters = KMeans.train(analysis_rdd, 2, maxIterations=10, initializationMode="random")
    #WSSSE = analysis_rdd.map(lambda point: error(point)).reduce(lambda x, y: x + y)
    #print("Within Set Sum of Squared Error = " + str(WSSSE))

    #labels = cluster.fit_predict(lib_analysis)
    #print(labels)
    
    #get list of centroids to extract root vectors

    #centroids = labels.cluster_centers_ # array, [n_clusters, n_features]
    #print(centroids)
    #get vertex distances in order to assign them to weights
    #distances = []
    #for i, (cx, cy) in enumerate(centroids):
    #    costs, roots = k_mean_distance(centroids, cx, cy, i, clusters)
    #    distances.append(mean_distance)
    
    #create graph
    g = Graph()

    #add list of root vertices to graph

    #loop through root vectors
    #perform knn
    
    #add to graph + connect knn results to input vector

    #perform Expectation-Maximization GMM

    #perform Mean-Shift Clustering
    #save root vectors

    #add to graph + connect


