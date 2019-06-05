print(__doc__)

from conf import *
from dag import *
from schema_functions import *
from spotapi import *
from dfops import *
import spotipy

import os
import sys
import time
import warnings
import pickle
import numpy as np
import pandas as pd

from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql import Row
import pyspark.sql.types as pst
#from pyspark.ml.clustering import KMeans
#from pyspark.ml.evaluation import ClusteringEvaluator

import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

warnings.simplefilter(action='ignore', category=FutureWarning)

conf = SparkConf().setAppName("Cosmos")
conf = (conf.setMaster('local[*]')
        .set('spark.executor.memory', '4G')
        .set('spark.driver.memory', '4G')
        .set('spark.driver.maxResultSize', '4G'))

sc = SparkContext(conf=conf)
sql_sc = SQLContext(sc)

saved_matrixData = False
savedModels = False

# TODO: perform nlp on songs with lyrics and perform word2vec to add another dimension of connection in the graph.

if __name__ == '__main__':
    sc.setLogLevel("ERROR")

    if (os.path.isfile(matrices_dir) and os.path.isfile(lib_ids_dir)):
        # Load library data, parse and produce matrices for graph construction
        print('Pre-saved matrices not found, loading library instead')
        # Import library
        library, num_samples = load_library(library_dir)
        ids = library['id']  # important for graph traversal

        # Create df duplicates
        s_main = sql_sc.createDataFrame(library)
        s_main = s_main.drop('track_href', 'uri', 'analysis', 'analysis_url', 'id', 'type')

        # Create spark vector dataframe representations for each dimension of features; note ml_analysis is a np array
        ml_analysis, v_features, v_tech, v_info, v_ids = spark_dfToVectors(s_main, library)

        print('Producing numpy matrices')
        # Create numpy ml-processable versions of the vector dfs
        ml_features, ml_tech, ml_info, ml_ids = vectors_to_matrices(sc, v_features, v_tech, v_info, v_ids)

        song_matrices = [ml_analysis, ml_features, ml_tech, ml_info]
        print('Saving matrices')
        # np.save(matrices_dir, np.asarray(song_matrices)) #save dataframe collection
        ids.to_pickle(lib_ids_dir)
        with open(matrices_dir, 'wb') as fp:
            pickle.dump(song_matrices, fp)
        print('Saved.')
    else:
        # Load pre-saved matrices
        print('Loading pre-saved matrices and ids')
        ids = pd.read_pickle(lib_ids_dir)
        with open(matrices_dir, 'rb') as fp:
            song_matrices = pickle.load(fp)

    savedModels = findmodels() # Decide if new models should be fit
    saved_matrixData = checkSavedMatrices()

    a_start = time.time()
    if savedModels: # Saved models not found
        print('Produce clustering + kNN models, fit, then produce graph data')
        root_vector_indices, root_neighbors, root_neighbor_weights = fit_transform_graphData(song_matrices, matrix_labels)
        print('Saving graph data')
        np.save(root_indices_dir, root_vector_indices)
        np.save(root_neighbors_dir, root_neighbors)
        np.save(neighbor_weights_dir, root_neighbor_weights)
    else:
        if saved_matrixData:
            print('Loading graph data in numpy matrix format')
            root_vector_indices = np.load(root_indices_dir, allow_pickle=True)
            root_neighbors = np.load(root_neighbors_dir, allow_pickle=True)
            root_neighbor_weights = np.load(neighbor_weights_dir, allow_pickle=True)
        else:
            root_vector_indices, root_neighbors, root_neighbor_weights = load_transform_graphData(song_matrices, matrix_labels)
    a_end = time.time()
    print('Nodes and edge weights generated in', ((a_end-a_start)/60),
          'minutes' if ((a_end - a_start) > 60) else ((a_end-a_start), 'seconds'))

    # If saved feature collection, models, and graph data are found are found, this will be the starting point
    root_vector_indices = np.asarray(root_vector_indices)
    root_neighbors = np.asarray(root_neighbors)
    root_neighbor_weights = np.asarray(root_neighbor_weights)

    # Function in dag.py: Creates graphs from produced root vector+neighbors indices+weights, and spotipy ids for each
    # vertex; where g is an instance of dag.py, and
    # G is an instance of networkx.MultiGraph()
    g, G, num_nodes = populate_graphs(root_vector_indices, root_neighbors, root_neighbor_weights, ids)

    # TODO: given graph, write algorithm to connect all nodes in the graph to all other nodes (if not already connected)

    colors = range(G.number_of_edges())
    print('Graph contains', G.number_of_nodes(), 'nodes')
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_color='#A0CBE2', edge_color=colors,
                     width=4, edge_cmap=plt.cm.Blues, with_labels=False)
    plt.savefig("graphs/2d_graph_"+str(num_nodes)+"nodes.png")
    plt.show()

    # Print adjacency-matrix
    #print_adj_matrix(g)