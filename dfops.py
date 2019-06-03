import pandas as pd
import numpy as np
from pyspark.sql.functions import udf
from pyspark.mllib.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler

def normalize_matrix(full_arr):
    largest = 0
    for i in range(full_arr.shape[0]):
        for index in range(0, full_arr.shape[1] - 1):
            if isinstance(np.asarray(full_arr[i][index]), (list,)):
                length = np.asarray(full_arr[i][index]).shape[0]
            else:
                length = len(full_arr[i])
            if length > largest:
                largest = length

    ml_matrix = np.zeros((full_arr.shape[0], largest))
    for i in range(ml_matrix.shape[0]):
        temp_a = []
        for index in range(0, full_arr.shape[1] - 1):
            temp = np.asarray(full_arr[i][index])
            temp_a.append(temp)
        full_a = []
        ##OPTIMIZE
        for x in temp_a:
            x = np.asarray(x).flatten()
            for y in x:
                full_a.append(y)
        full_a = np.asarray(full_a)

        for zi, z in zip(range(len(full_a)), range(len(ml_matrix[i]))):
            ml_matrix[i][z] = full_a[zi]

    return ml_matrix


def refactorAnalysisNP(lib_analysis, sql_context=None):
    keylist = lib_analysis[0].keys()

    full_arr = []
    for songindex in range(lib_analysis.shape[0]): #iterate songs
        song = lib_analysis[songindex]
        factored = []
        for key in keylist: #iterate song keys
            metadata = song[key]
            if isinstance(metadata, (list,)):
                arrays = []
                for item in metadata: #iterate list of dicts
                    subvals = np.asarray(list(item.values()))
                    arrays.append(subvals)
                temp_d = arrays #now key -> array, instead of key to array[dicts]
                #flatten = [iteml for sublist in temp_d for iteml in sublist]
                factored.append(temp_d)
            else:
                items = np.asarray(list(metadata.values()))
                #temp_d = items
                #flatten = [iteml for sublist in temp_d for iteml in sublist]
                #factored.append(temp_d)
        full_arr.append(factored)

    # Spark data frame vectorization closure; not applicable for data structure or sklearn processing
    '''
    #analysis_df = pd.DataFrame(full_arr, columns=keylist)
    #analysis_df = analysis_df.drop(columns=['meta'])

    s_analysis = sql_context.createDataFrame(analysis_df)
    list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
    df_w_vectors = s_analysis.select(
        list_to_vector_udf(s_analysis["bars"]).alias("bars"),
        list_to_vector_udf(s_analysis["beats"]).alias("beats"),
        list_to_vector_udf(s_analysis["tatums"]).alias("tatums"),
        list_to_vector_udf(s_analysis["sections"]).alias("sections"),
        list_to_vector_udf(s_analysis["segments"]).alias("segments")
    )

    #assembler = VectorAssembler(inputCols=df_w_vectors.schema.names, outputCol="features")
    #v_analysis = assembler.transform(df_w_vectors)
    '''
    full_arr = np.asarray(full_arr)
    ml_analysis = normalize_matrix(full_arr)


    return np.asarray(ml_analysis)

def refactorAnalysisDF(lib_analysis):
    keylist = list(lib_analysis[0].keys())

    full_arr = []
    for songindex in range(lib_analysis.shape[0]):  # iterate songs
        song = lib_analysis[songindex]
        factored = []
        for key in keylist:  # iterate song keys
            metadata = song[key]
            if isinstance(metadata, (list,)):
                arrays = []
                for item in metadata:  # iterate list of dicts
                    subvals = np.asarray(list(item.values()))
                    arrays.append(subvals)
                temp_d = arrays  # now key -> array, instead of key to array[dicts]
                # flatten = [iteml for sublist in temp_d for iteml in sublist]
                factored.append(temp_d)
            else:
                items = np.asarray(list(metadata.values()))
                # temp_d = items
                # flatten = [iteml for sublist in temp_d for iteml in sublist]
                # factored.append(temp_d)
        full_arr.append(factored)

    keylist.remove('meta')
    keylist.remove('track')
    analysis_df = pd.DataFrame(full_arr, columns=keylist)

    return analysis_df