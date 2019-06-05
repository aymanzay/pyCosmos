import sys
import spotipy
import json
import pandas as pd
import time
import spotipy.oauth2 as oauth2
import spotipy.util as util
from multiprocessing import Process, Pool, TimeoutError

scope = 'user-library-read'

username = 'aymanzay'

token = util.prompt_for_user_token(username, scope)
sp = spotipy.Spotify(auth=token)

def scrape_userlib(token):
    offset = 0
    if token:
        start = time.time()
        try:
            first = sp.current_user_saved_tracks(offset=offset, limit=1)
            f_uri = first['items'][0]['track']['uri']

            fe_cols = sp.audio_features(tracks=f_uri)
            an_cols = sp.audio_analysis(f_uri)

            df = pd.DataFrame.from_dict(fe_cols)
            df['analysis'] = [an_cols]

            offset = offset + 1
            while True:
                results = sp.current_user_saved_tracks(offset=offset, limit=50) # function to multiprocess
                for item in results['items']:
                    track = item['track']
                    uri = track['uri']

                    name = track['name']
                    artist = track['artists'][0]['name']
                    album = track['album']
                    song_info = {'name': name, 'artist': artist, 'album': album}

                    songI_df = pd.DataFrame(song_info)
                    analysis = sp.audio_analysis(uri)
                    features = sp.audio_features(tracks=uri)
                    tmp_df = pd.DataFrame.from_dict(features)
                    tmp_df['analysis'] = [analysis]

                    frames = [songI_df, df, tmp_df]
                    df = pd.concat(frames, ignore_index=True, sort=False)
                    offset = offset + 1

                    #print(track['name'] + ' - ' + track['artists'][0]['name'] + ' - ' + track['uri'])
                print(offset, 'songs added to dataframe')
        except:
            print('saving dataframe of', offset, 'songs')
            df.to_pickle("./aymanzay_lib.pkl")
            print('end of lib')
            end = time.time()
            print(len(df))
            print('total library extraction time:',
                  str((end - start) / 60)+' minutes.' if (end - start) > 60 else str((end-start))+' seconds.')
            sys.exit()

    else:
        print("Can't get token for", username)


if __name__ == '__main__':
    scrape_userlib(token)