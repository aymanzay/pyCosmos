import pandas as pd

def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res

def refactorAnalysisDF(lib_analysis, ids):
    keylist = lib_analysis[0].keys()

    pd_ids = ids
    full_arr = []
    for songindex in range(lib_analysis.shape[0]): #iterate songs
        id = {'id': pd_ids[songindex]}
        song = lib_analysis[songindex]
        song = Merge(song, id)
        factored = []
        for key in keylist: #iterate song keys
            metadata = song[key]
            if isinstance(metadata, (list,)):
                arrays = []
                for item in metadata: #iterate list of dicts
                    subvals = list(item.values())
                    arrays.append(subvals)
                temp_d = arrays #now key -> array, instead of key to array[dicts]
                factored.append(temp_d)
            else:
                items = list(metadata.values())
                temp_d = items
                factored.append(temp_d)
        full_arr.append(factored)

    analysis_df = pd.DataFrame(full_arr, columns=keylist)

    return analysis_df
