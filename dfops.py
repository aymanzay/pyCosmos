import pandas as pd
from pyspark.sql.functions import udf
from pyspark.mllib.linalg import Vectors, VectorUDT

def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res

def refactorAnalysisDF(lib_analysis, ids, sql_context):
    keylist = lib_analysis[0].keys()

    pd_ids = ids
    full_arr = []
    for songindex in range(lib_analysis.shape[0]): #iterate songs
        id = {'id': pd_ids[songindex]}
        song = lib_analysis[songindex]
        #song = Merge(song, id)
        factored = []
        for key in keylist: #iterate song keys
            metadata = song[key]
            if isinstance(metadata, (list,)):
                arrays = []
                for item in metadata: #iterate list of dicts
                    subvals = list(item.values())
                    arrays.append(subvals)
                temp_d = arrays #now key -> array, instead of key to array[dicts]
                flatten = [iteml for sublist in temp_d for iteml in sublist]
                factored.append(flatten)
            else:
                items = list(metadata.values())
                temp_d = items
                #flatten = [iteml for sublist in temp_d for iteml in sublist]
                factored.append(temp_d)
        full_arr.append(factored)

    analysis_df = pd.DataFrame(full_arr, columns=keylist)
    analysis_df = analysis_df.drop(columns=['meta'])

    s_analysis = sql_context.createDataFrame(analysis_df)
    list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
    df_w_vectors = s_analysis.select(
        list_to_vector_udf(s_analysis["bars"]).alias("bars"),
        list_to_vector_udf(s_analysis["beats"]).alias("beats"),
        list_to_vector_udf(s_analysis["tatums"]).alias("tatums"),
        list_to_vector_udf(s_analysis["sections"]).alias("sections"),
        list_to_vector_udf(s_analysis["segments"]).alias("segments")
    )

    return df_w_vectors
