import os
import pickle
import time
import numpy as np
import pandas as pd

from dfops import *
from conf import *

from spark_sklearn import Converter
from pyspark.ml.feature import VectorAssembler
from sklearn.cluster import MeanShift
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.externals import joblib

# Returns user pre-pickled library as pandas dataframe, and number of samples in df
def load_library(lib_dir):
    with open(lib_dir, 'rb') as f:
        library = pickle.load(f)
        print('loaded')
        num_samples = len(library)

    return library, num_samples

def findmodels():
    temp_bool = True
    for label in matrix_labels:
        if not (os.path.isfile('models/meanshift_' + label + '.model') and os.path.isfile('models/knn_' + label + '.model')):
            temp_bool = False

    return temp_bool

# Returns ml_analysis containing numpy matrices depicting analysis data in library['analysis'] column
# Also returns 3 other spark vector representations of the selected feature dimensions to graph
def spark_dfToVectors(main_spark, lib):

    lib_analysis = lib['analysis']

    # create analysis dataframe
    print("Extracting analysis data")
    start = time.time()
    ml_analysis = refactorAnalysisNP(lib_analysis)
    end = time.time()
    print('Extracted in', (end - start), 'seconds')

    # create features dataframe
    s_features = main_spark
    print("Extracting feature data")
    assembler = VectorAssembler(inputCols=s_features.schema.names, outputCol="features")
    v_features = assembler.transform(s_features)

    # create technical dataframe
    s_tech = main_spark.select('tempo', 'key', 'loudness', 'valence', 'time_signature', 'liveness', 'energy', 'danceability')
    print("Extracting tech data")
    assembler = VectorAssembler(inputCols=s_tech.schema.names, outputCol="tech_features")
    v_tech = assembler.transform(s_tech)

    # create info dataframes
    s_info = main_spark.select('speechiness', 'acousticness', 'instrumentalness')
    print("Extracting song info data")
    assembler = VectorAssembler(inputCols=s_info.schema.names, outputCol="info_features")
    v_info = assembler.transform(s_info)

    # create song id dataframes
    s_ids = main_spark.select('name', 'artist', 'album')
    print('Extracting song id info')
    assembler = VectorAssembler(inputCols=s_ids.schema.names, outputCol="id_features")
    v_ids = assembler.transform(s_info)

    return ml_analysis, v_features, v_tech, v_info, v_ids

def separate_dfs(library, ids):

    lib_analysis = library['analysis']
    ids = pd.DataFrame(ids, columns=['id'])

    lib = library.drop(columns=['track_href', 'uri', 'analysis', 'analysis_url', 'id', 'type'])

    #analysis
    analysis_df = refactorAnalysisDF(lib_analysis)
    analysis_df = pd.concat([analysis_df, ids], sort=False)

    #main features
    main_df = pd.concat([lib, ids], sort=False)

    #technical features
    tech_df = library[['tempo', 'key', 'loudness', 'valence', 'time_signature', 'liveness', 'energy', 'danceability']]
    tech_df = pd.concat([tech_df, ids], sort=False)

    #song info
    song_df = lib[['speechiness', 'acousticness', 'instrumentalness']]
    song_df = pd.concat([song_df, ids], sort=False)

    #song ids
    songIds_df = lib[['name', 'artist', 'album']]
    songIds_df = pd.concat([songIds_df, ids], sort=False)

    return analysis_df, main_df, tech_df, song_df, songIds_df

# Converts all input vectorAssembler-transformed Spark DFs to sklearn processable numpy matrices
def vectors_to_matrices(sc, v_features, v_tech, v_info, v_ids):
    print('Converting all vector dataframes to dense matrices')
    start = time.time()
    converter = Converter(sc)
    features, tech, info, ids = converter.toPandas(v_features), converter.toPandas(v_tech), converter.toPandas(v_info), converter.toPandas(v_ids)
    m_features, m_tech, m_info, m_ids = features.values, tech.values, info.values, ids.values
    ml_features, ml_tech, ml_info, ml_ids = normalize_matrix(m_features), normalize_matrix(m_tech), normalize_matrix(m_info), normalize_matrix(m_ids)
    end = time.time()
    print('Converted in', (end - start), 'seconds')
    return ml_features, ml_tech, ml_info, ml_ids

def load_transform_graphData(song_matrices, matrix_labels):

    root_vector_indices = []
    root_neighbors = []
    root_neighbor_weights = []

    for f_matrix, m_label in zip(song_matrices, matrix_labels):
        print('loading', m_label, 'model')
        outputfile = 'models/meanshift_' + m_label + '.model'
        clustering = joblib.load(outputfile)
        cluster_centers = clustering.cluster_centers_
        n_clusters = len(cluster_centers)

        # Store root vectors
        root_indices = []
        # Returns list of 'labels' corresponding to the song index in the lib array
        for center in cluster_centers:
            roots, _ = pairwise_distances_argmin_min([center], f_matrix)
            root_indices.append(roots)

        root_vector_indices.append(root_indices)
        # Given list of roots, perform knn on each one to get neighboring nodes
        for root in root_indices:
            outputfile = 'models/knn_' + m_label + '.model'
            neighbors = joblib.load(outputfile)
            distances, indices = neighbors.kneighbors(f_matrix[root])
            root_neighbors.append(indices)
            root_neighbor_weights.append(distances)

    return np.asarray(root_vector_indices), np.asarray(root_neighbors), np.asarray(root_neighbor_weights)

def fit_transform_graphData(song_matrices, matrix_labels):

    root_vector_indices = []
    root_neighbors = []
    root_neighbor_weights = []

    for f_matrix, m_label in zip(song_matrices, matrix_labels):
        print(m_label)
        clustering = MeanShift(n_jobs=-1)  # init MeanShift clustering model
        outputfile = 'models/meanshift_' + m_label + '.model'

        start = time.time()
        # Perform Mean-Shift Clustering
        clustering.fit(f_matrix)
        joblib.dump(clustering, outputfile)  # Save MeanShift-clustering model

        cluster_centers = clustering.cluster_centers_
        n_clusters = len(cluster_centers)
        end = time.time()
        print('Clustered in', ((end - start) / 60), 'minutes' if ((end - start) > 60) else ((end - start), 'seconds'), 'with ', n_clusters, 'clusters')

        # Save root vectors
        root_indices = []
        # returns list of 'labels' corresponding to the song index in the lib array
        for center in cluster_centers:
            roots, _ = pairwise_distances_argmin_min([center], f_matrix)
            root_indices.append(roots)

        root_vector_indices.append(root_indices)
        num_samples = f_matrix.shape[0]
        num_neighbors = int((num_samples / 3))
        print('number of neighbors per root:', num_neighbors)

        # given list of roots, perform knn on each one to get neighboring nodes
        for root in root_indices:
            neighbors = NearestNeighbors(n_neighbors=num_neighbors, algorithm='auto', n_jobs=-1)
            neighbors.fit(f_matrix)

            outputfile = 'models/knn_' + m_label + '.model'
            joblib.dump(neighbors, outputfile)  # Save k-Nearest Neighbors model

            distances, indices = neighbors.kneighbors(f_matrix[root])
            root_neighbors.append(indices)
            root_neighbor_weights.append(distances)

    return np.asarray(root_vector_indices), np.asarray(root_neighbors), np.asarray(root_neighbor_weights)


def checkSavedMatrices():
    temp_bool = False

    if os.path.isfile(root_indices_dir) and os.path.isfile(root_neighbors_dir) and os.path.isfile(neighbor_weights_dir):
        temp_bool = True

    return temp_bool
