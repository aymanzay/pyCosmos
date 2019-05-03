# pyCosmos
ML-based Recommendation system using pySpark and Directed Weighted Graphs

spot-api - containing spotipy api to download full user’s library given
an input token, returned in a pandas df, saved to a pickle file.

dag - contains api for a directed weighted graph, including list
transformation functions for graph compatibility.

scheme - uses pyspark to create RDDs for later clustering and graph
construction.
