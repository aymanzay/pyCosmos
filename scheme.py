print(__doc__)

from dag import *
from spotapi import *
from dfops import *
import spotipy

import time
import warnings
import pickle
import numpy as np
import pandas as pd

from spark_sklearn import Converter
from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql import Row
import pyspark.sql.types as pst
#from pyspark.ml.clustering import KMeans
#from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MeanShift
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets.samples_generator import make_blobs
from scipy.sparse import csr_matrix

import plotly.plotly as py
import plotly.graph_objs as go
import plotly
import matplotlib.pyplot as plt

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
    print("Extracting analysis data")
    start = time.time()
    m_analysis = refactorAnalysisDF(lib_analysis, ids, sql_sc)
    end = time.time()
    print('Extracted in', (end-start), 'seconds')
    #possible approach may be to process each column separately

    #lib_features = library.drop(columns=['analysis'])
    s_features = s_main
    print("Extracting feature data")
    assembler = VectorAssembler(inputCols=s_features.schema.names, outputCol="features")
    v_features = assembler.transform(s_features)

    #lib_tech = library.as_matrix(columns=['tempo', 'key', 'loudness', 'valence', 'time_signature', 'liveness', 'energy', 'danceability'])
    s_tech = s_main.select('tempo', 'key', 'loudness', 'valence', 'time_signature', 'liveness', 'energy', 'danceability')
    print("Extracting tech data")
    assembler = VectorAssembler(inputCols=s_tech.schema.names, outputCol="tech_features")
    v_tech = assembler.transform(s_tech)

    #lib_song_info = library.as_matrix(columns=['artist', 'genre', 'speechiness', 'acousticness', 'instrumentalness'])
    s_info = s_main.select('speechiness', 'acousticness', 'instrumentalness')
    print("Extracting song info data")
    assembler = VectorAssembler(inputCols=s_info.schema.names, outputCol="tech_features")
    v_info = assembler.transform(s_info)



    print('Converting ')
    start = time.time()
    converter = Converter(sc)
    features = converter.toPandas(v_features)
    m_features = features.values
    print(type(m_features[0]), m_features[0].shape)
    ml_features = normalize_matrix(m_features)
    end = time.time()
    print('Converted in', (end-start), 'seconds')

    print("Performing analysis clustering")
    #get initial root vectors
    #init meanshift clustering model

    start = time.time()
    clustering = MeanShift(n_jobs=-1)
    clustering.fit(ml_features)

    labels = clustering.labels_
    cluster_centers = clustering.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters = len(labels_unique)
    end = time.time()
    print('Clustered in', ((end - start)/60), 'minutes')

    print(clustering.cluster_centers_)


    #plot results


    #get list of centroids to extract root vectors

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


