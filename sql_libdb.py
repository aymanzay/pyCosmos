from schema_functions import *
from conf import *
import sqlite3
from sqlalchemy import create_engine
from functools import reduce
from pyspark import SparkConf, SparkContext, SQLContext
import numpy as np
import pandas as pd

conf = SparkConf().setAppName("Cosmos")
conf = (conf.setMaster('local[*]')
        .set('spark.executor.memory', '4G')
        .set('spark.driver.memory', '4G')
        .set('spark.driver.maxResultSize', '4G'))

sc = SparkContext(conf=conf)
sql_sc = SQLContext(sc)

connection = sqlite3.connect('library/dbs/aymanzay.db')

# cursor
crsr = connection.cursor()

library, num_samples = load_library(library_dir)
ids = library['id']

# Create df duplicates
# Create spark vector dataframe representations for each dimension of features; note ml_analysis is a np array
analysis_df, features_df, tech_df, info_df, songIds_df = separate_dfs(library, ids)

#full_df = [analysis_df, features_df, tech_df, info_df]
#df_merged = reduce(lambda left, right: pd.merge(left, right, on='id', how='outer'), full_df).fillna(0)
#print(full_df.columns)

engine = create_engine(sql_url, echo=False)

# TODO: implement time-series sql schema for analysis data (possibly might have to separate each separate column in analysis).

features_df.to_sql('features', con=engine)
tech_df.to_sql('tech', con=engine)
info_df.to_sql('info', con=engine)
songIds_df.to_sql('ids', con=engine)
#analysis_df.to_sql('analysis', con=engine)

def getSongDBs(engine, id):

    with engine.connect() as con:
        fs = con.execute('SELECT * from features where features.id ==', id)
        ts = con.execute('SELECT * from tech where tech.id ==', id)
        Is = con.execute('SELECT * from info where info.id ==', id)
        IDs = con.execute('SELECT * from ids where ids.id ==', id)

        return fs, ts, Is