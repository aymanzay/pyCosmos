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

from sklearn.cluster import KMeans, MeanShift
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.externals import joblib

import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

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

    num_samples = len(library)

    #create df duplicates
    s_main = sql_sc.createDataFrame(library)
    s_main = s_main.drop('track_href', 'uri', 'analysis', 'analysis_url', 'id', 'type')

    lib_analysis = library['analysis']
    ids = library['id'] #important for graph construction

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

    root_vector_indices = []
    root_neighbors = []
    root_neighbor_weights = []

    a_start = time.time()
    for f_matrix, m_label in zip(song_matrices, matrix_labels):
        if not performClustering:
        # getting initial root vectors
            ##START ANALYSIS CODE BLOCK
            print(m_label)
            clustering = MeanShift(n_jobs=-1)  # init MeanShift clustering model
            outputfile = 'models/meanshift_' + m_label + '.model'

            # perform Mean-Shift Clustering
            clustering.fit(f_matrix)
            joblib.dump(clustering, outputfile)

            cluster_centers = clustering.cluster_centers_
            n_clusters = len(cluster_centers)
            end = time.time()
            print(clustering.cluster_centers_)
            print('Clustered in', ((end-start)/60), 'minutes' if ((end - start) > 60) else ((end-start),'seconds'), 'with ', n_clusters, 'clusters')

            # save root vectors
            root_indices = []
            #returns list of 'labels' corresponding to the song index in the lib array
            for center in cluster_centers:
                roots, _ = pairwise_distances_argmin_min([center], f_matrix)
                root_indices.append(roots)

            root_vector_indices.append(root_indices)
            #given list of roots, perform knn on each one to get neighboring nodes
            for root in root_indices:
                neighbors = NearestNeighbors(n_neighbors=((num_samples/5)/2), algorithm='auto', n_jobs=-1)
                neighbors.fit(f_matrix)
                outputfile = 'models/knn_' + m_label + '.model'
                joblib.dump(neighbors, outputfile)
                distances, indices = neighbors.kneighbors(f_matrix[root])
                #print('distances', distances)
                #print('indices', indices)
                root_neighbors.append(indices)
                root_neighbor_weights.append(distances)
        else:
            print('loading', m_label, 'model')
            outputfile = 'models/meanshift_' + m_label + '.model'
            clustering = joblib.load(outputfile)

            cluster_centers = clustering.cluster_centers_
            n_clusters = len(cluster_centers)

            # save root vectors
            root_indices = []
            # returns list of 'labels' corresponding to the song index in the lib array
            for center in cluster_centers:
                roots, _ = pairwise_distances_argmin_min([center], f_matrix)
                root_indices.append(roots)

            root_vector_indices.append(root_indices)
            # given list of roots, perform knn on each one to get neighboring nodes
            for root in root_indices:
                outputfile = 'models/knn_' + m_label + '.model'
                neighbors = joblib.load(outputfile)
                distances, indices = neighbors.kneighbors(f_matrix[root])
                #print('distances', distances)
                #print('indices', indices)
                root_neighbors.append(indices)
                root_neighbor_weights.append(distances)
    a_end = time.time()
    print('total node generation time:', (a_end-a_start), 'seconds')

    root_vector_indices = np.asarray(root_vector_indices)
    root_neighbors = np.asarray(root_neighbors)
    root_neighbor_weights = np.asarray(root_neighbor_weights)

    #create graph
    g = Graph()
    G = nx.MultiDiGraph()
    G.depth = {}

    #add list of root vertices to graph
    for r in range(root_vector_indices.shape[0]):
        for ri in range(len(root_vector_indices[r])):
            # loop through
            root_index_value = root_vector_indices[r][ri][0]
            root_id = ids[root_index_value]
            g.add_vertex(root_id)
            G.add_node(root_id)
            neighbor_arrays = np.asarray(root_neighbors[ri][0])
            neighbor_weights = np.asarray(root_neighbor_weights[ri][0])
            for n,w in zip(range(len(neighbor_arrays)), range(len(neighbor_weights))):
                neighbor_index = neighbor_arrays[n]
                conn_weight = neighbor_weights[w]
                neighbor_id = ids[neighbor_index]
                if (neighbor_id != root_id) and (conn_weight > 0):
                    # add to graph + connect
                    g.add_vertex(neighbor_id)
                    g.add_edge(root_id, neighbor_id, float(conn_weight))
                    G.add_edge(root_id, neighbor_id, weight=float(conn_weight))

    colors = range(G.number_of_edges())
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_color='#A0CBE2', edge_color=colors,
                     width=4, edge_cmap=plt.cm.Blues, with_labels=False)
    plt.savefig("graphs/2d_graph.png")
    plt.show()

    '''
    print('print graph')
    for v in g:
        for w in v.get_connections():
            vid = v.get_id()
            wid = w.get_id()
            print('( %s , %s, %3f)' % (vid, wid, v.get_weight(w)))

    for v in g:
        print('g.vert_dict[%s]=%s' % (v.get_id(), g.vert_dict[v.get_id()]))
    '''