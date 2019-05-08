# pyCosmos
ML-based Recommendation system using pySpark and Directed Weighted Graphs

conf - contains required directories and variables to load and store required variables

spot-api - containing spotipy api to download full user’s library given
an input token, returned in a pandas df, saved to a pickle file.

dag - contains api for a directed weighted graph, including list transformation functions for graph compatibility.

schema_functions - contains the functions scheme requires to get data, fit/load models, etc.

scheme:
 - Uses pySpark to create Spark DataFrames for later clustering and graph construction.
 - Using spark-sklearn to convert Spark DataFrames -> Pandas -> Numpy arrays
 - Clustering is done using sklearn MeanShift, k-Nearest Neighbors.
 - Generate a weighted directed acyclic graph (DAG) from data recovered from clustering using networkx.
 - Plot the DAG representing user library schema; depicted in the three graphs below.

![Alt text](2d_graph.png?raw=true "Library spherical clustered MultiDiGraph representation")
![Alt text](graphs/2d_graphnet.png?raw=true "Library clustered DAG representation")
![Alt text](graphs/2d_graph_2221nodes.png?raw=true "Library clustered MultiGraph representation containing 2221 Vertices.")