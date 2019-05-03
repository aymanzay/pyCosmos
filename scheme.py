from dag import *
from spotapi import *
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
    
    #lib_a = library.filter(['analysis'])
    #s_analysis = s_main.select('analysis') #cannot use spark since dicts are cast as None
    lib_analysis = library.as_matrix(columns=['analysis'])
    lib_ids = library.as_matrix(columns=['id'])
    full_df = pd.DataFrame()
    songlist = []

    for i in range(len(lib_analysis)):
        mapping = lib_analysis[i][0]
        keylist = mapping.keys()
        song = []
        #iterating the dict by key
        for key in keylist:
            #checking if value is array of dicts
            if isinstance(mapping[key], (list,)):
                a_map = mapping[key]
                tempa = []
                for x in mapping[key]:
                    tempa.append(list(x.values()))
                #tup = (key, lib_ids[i],tempa)
                tup = tempa
                song.append(tup)
            else:
                a_map = list(mapping[key].values())
                #tup = (key, lib_ids[i], a_map)
                song.append(a_map)
        songlist.append(song)
    
    #print(songlist[0])
    #create analysis dataframe from array
    an_songdf = pd.DataFrame(songlist, columns=list(lib_analysis[0][0].keys()))
    s_a = sql_sc.createDataFrame(an_songdf)
    s_analysis = s_a.select('bars', 'beats', 'tatums', 'sections', 'segments')

    cols = s_analysis.columns
    va = VectorAssembler(inputCols=cols, outputCol="features")
    #vdf = va.transform(s_analysis)
    d = va.rdd.map(lambda row: Vectors.dense([x for x in row["features"]])).collect()
    
    def filterNones(rddlists):
        for i in range(len(rddlists)):
            for j in range(len(rddlists[i])):
                place = rddlists[i][j]
                for k in range(len(place)):
                    if rddlists[i][j][k] is None:
                        rddlists[i][j][k] = 0
        return rddlists
        
    #analysis_rdd = s_analysis.rdd.map(filterNones, s_analysis)

    #newrdd = analysis_rdd.foreach(filterNones)
    #print(analysis_rdd.take(1))

    #lib_features = library.drop(columns=['analysis'])
    #s_features = s_main.drop('analysis')

    #lib_tech = library.as_matrix(columns=['tempo', 'key', 'loudness', 'valence', 'time_signature', 'liveness', 'energy', 'danceability'])
    #s_tech = s_main.select('tempo', 'key', 'loudness', 'valence', 'time_signature', 'liveness', 'energy', 'danceability')

    #lib_song_info = library.as_matrix(columns=['artist', 'genre', 'speechiness', 'acousticness', 'instrumentalness'])
    #s_info = s_main.select('speechiness', 'acousticness', 'instrumentalness')

    #in order to properly process analysis data, all keys must be flattened and cast
    #into arrays while maintaining the order in the dict
    #print(s_analysis)

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


