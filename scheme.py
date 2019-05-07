print(__doc__)

from dag import *
from spotapi import *
from dfops import *
import spotipy

import os
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
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

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
    with open('./library/pkls/aymanzay_lib.pkl', 'rb') as f:
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
    ml_analysis = refactorAnalysisDF(lib_analysis, ids, sql_sc)
    end = time.time()
    print('Extracted in', (end-start), 'seconds')

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

    print('Converting all dataframes to dense matrices')
    start = time.time()
    converter = Converter(sc)
    features, tech, info = converter.toPandas(v_features), converter.toPandas(v_tech), converter.toPandas(v_info)
    m_features, m_tech, m_info = features.values, tech.values, info.values
    ml_features, ml_tech, ml_info = normalize_matrix(m_features), normalize_matrix(m_tech), normalize_matrix(m_info)
    end = time.time()
    print('Converted in', (end-start), 'seconds')

    song_matrices = [ml_analysis, ml_features, ml_tech, ml_info]
    matrix_labels = ['analysis', 'features', 'tech', 'info']
    print("Performing clustering on each matrix")

    performClustering = False
    for label in matrix_labels:
        if not (os.path.isfile('models/meanshift_' + label + '.model') and os.path.isfile('models/knn_' + label + '.model')):
            performClustering = True

    if performClustering:
        # getting initial root vectors
        start = time.time()
        clustering = MeanShift(n_jobs=-1) # init MeanShift clustering model

        for f_matrix, m_label in zip(song_matrices, matrix_labels):
            ##START ANALYSIS CODE BLOCK
            print(m_label)

            outputfile = 'models/meanshift_' + m_label + '.model'
            clustering.fit(f_matrix)
            joblib.dump(clustering, outputfile)

            cluster_centers = clustering.cluster_centers_
            n_clusters = len(cluster_centers)
            end = time.time()
            print(clustering.cluster_centers_)
            print('Clustered in', (((end-start)/60), 'minutes' if ((end - start) > 60) else (end-start),'seconds'), 'with ', n_clusters, 'clusters')

            root_indices = []
            #returns list of 'labels' corresponding to the song index in the lib array
            for center in cluster_centers:
                roots, _ = pairwise_distances_argmin_min([center], f_matrix)
                root_indices.append(roots)

            #given list of roots, perform knn on each one to get neighboring nodes
            for root in root_indices:
                neighbors = NearestNeighbors(n_neighbors=(50), algorithm='auto', n_jobs=-1)
                neighbors.fit(f_matrix)
                outputfile = 'models/knn_' + m_label + '.model'
                joblib.dump(neighbors, outputfile)
                distances, indices = neighbors.kneighbors(f_matrix[root])
                print('distances', distances)
                print('indices', indices)

    #get distances of each node that correspond to weight connections
    
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


