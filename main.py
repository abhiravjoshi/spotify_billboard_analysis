import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

############################## THESIS QUESTIONS ###########################
#
# THESIS: WHAT MAKES A SONG POPULAR AND AN ARTIST STAY POPULAR?
#
# DEFINE A HIT: A HIT IS SOMETHING ON THE BILLBOARD TOP 30 FOR AT LEAST
# 4 WEEKS.
# A) WHAT WERE THE ACOUSTIC FEATURES AND CHARACTERISTICS OF THEIR HITS?
#       What does that tell us about the pop music listening
#       population as a whole?
# B) WHAT NOTES/KEY IS MOST LIKELY TO BE POPULAR?
# C) DO COLLABORATIONS RESULT IN MORE POPULARITY?
# E) WHAT'S THE AVERAGE AMOUNT OF TIME THAT A SONG THAT WAS ONCE A HIT STAY ON
#    THE BILLBOARD?
# F) HOLIDAY SONGS? BIRTHDAY SONGS? REPEATABLE RELEVANCY? TO WHAT EXTENT
#    DOES YOUR POPULARITY AFFECT THAT TYPE OF SONG'S SUCCESS?
# G) ARE WE TALKING ABOUT AN ARTIST THAT HAS GONE VIRAL OR AN
#    ESTABLISHED ARTIST?
#
# NUMERICAL / REGRESSION / CONTINUOUS DATA
# CATEGORICAL / CLASSIFICATION / DISCRETE DATA


def main():
    # THIS IS THE TRANSFER SECTION (TO BE COMPLETED AND VISUALIZED ON JUPYTER

    # ACOUSTIC FEATURES + SONG CHART + SONG POP + songs_df
    # DATAFRAME OF INTEREST 1
    df1 = pd.merge(acoustic_features_df, song_chart_df,
                                        on='song_id')
    df1 = pd.merge(df1, song_pop_df, on='song_id')
    df1 = pd.merge(df1, songs_df, on='song_id')
    df1['week'] = pd.to_datetime(df1['week'], format='%Y-%m-%d')
    # df1 = pd.merge(df1, tracks_df, on='song_id')
    # songs duplicates if singles should be removed but preserve duplicates if
    # released with album
    # DEFINE THE SONG HITS
    df1_1 = df1.drop(df1[df1['peak_position'] > 30].index, inplace=False)
    df1_1['peak_position']
    df1_song_hits_only = df1_1.drop(df1_1[df1_1['weeks_on_chart'] < 4].index,
                                    inplace=False)
    df1_song_hits_only.reset_index(drop=True, inplace=True)

    df1_2010s = df1_song_hits_only.drop(
        df1_song_hits_only[df1_song_hits_only['year'] < 2011].index,
        inplace=False)
    df1_2010s.reset_index(drop=True, inplace=True)

    df1_2000s = df1_song_hits_only.drop(
        df1_song_hits_only[df1_song_hits_only['year'] < 2001].index,
        inplace=False)
    df1_2000s = df1_2000s.drop(df1_2000s[df1_2000s['year'] > 2010].index,
                               inplace=False)
    df1_2000s.reset_index(drop=True, inplace=True)

    df1_90s = df1_song_hits_only.drop(
        df1_song_hits_only[df1_song_hits_only['year'] < 1991].index,
        inplace=False)
    df1_90s = df1_90s.drop(df1_90s[df1_90s['year'] > 2000].index,
                           inplace=False)
    df1_90s.reset_index(drop=True, inplace=True)

    df1_80s = df1_song_hits_only.drop(
        df1_song_hits_only[df1_song_hits_only['year'] < 1981].index,
        inplace=False)
    df1_80s = df1_80s.drop(df1_80s[df1_80s['year'] > 1990].index,
                           inplace=False)
    df1_80s.reset_index(drop=True, inplace=True)

    df1_70s = df1_song_hits_only.drop(
        df1_song_hits_only[df1_song_hits_only['year'] < 1971].index,
        inplace=False)
    df1_70s = df1_70s.drop(df1_70s[df1_70s['year'] > 1980].index,
                           inplace=False)
    df1_70s.reset_index(drop=True, inplace=True)

    df1_60s = df1_song_hits_only.drop(
        df1_song_hits_only[df1_song_hits_only['year'] < 1961].index,
        inplace=False)
    df1_60s = df1_60s.drop(df1_60s[df1_60s['year'] > 1970].index,
                           inplace=False)
    df1_60s.reset_index(drop=True, inplace=True)

    # ALL ANALYSIS BASED OFF DF1

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Song Popularity")
    ax.set_ylabel("Acousticness")
    x = np.array(df1_2010s['song_popularity'])
    y = np.array(df1_2010s['acousticness'])
    ax.scatter(x, y, color='black', label='Acousticness')
    ax.legend(loc=0)

    ay = plt.twinx()
    ay.set_xlabel("Song Popularity")
    ay.set_ylabel("Positivity")
    y2 = np.array(df1_2010s['valence'])
    ay.scatter(x, y2, label='Positivity', color='red')
    ay.legend(loc=1)

    # obtain m (slope) and b(intercept) of linear regression line
    m, b = np.polyfit(x, y2, 1)

    # add linear regression line to scatterplot
    plt.plot(x, m * x + b)

    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.array(df1_2010s['valence'])
    y1 = np.array(df1_2010s['danceability'])
    ax.set_xlabel("Positivity")
    ax.set_ylabel("Danceable")
    ax.scatter(x, y1, color='black', label='Danceability')
    ax.legend(loc=2)

    ay = plt.twinx()
    y2 = np.array(df1_2010s['song_popularity'])
    ay.set_ylabel("Popularity")
    ay.scatter(x, y2, label='Popularity', color='red')
    ay.legend(loc=1)

    # obtain m (slope) and b(intercept) of linear regression line
    m, b = np.polyfit(x, y1, 1)

    # add linear regression line to scatterplot
    plt.plot(x, m * x + b)
    plt.show()

    dur = np.array(
        df1_2010s['duration_ms'] / 60000)  # 60k milliseconds in one minute
    danz = np.array(df1_2010s['danceability'])
    pos = np.array(df1_2010s['valence'])

    fig = plt.figure()
    counts, bins = np.histogram(dur)
    plt.hist(bins[:-1], bins, weights=counts, edgecolor='black', color='red')
    fig = plt.figure()
    counts, bins = np.histogram(danz)
    plt.hist(bins[:-1], bins, weights=counts, edgecolor='black', color='green')
    fig = plt.figure()
    counts, bins = np.histogram(pos)
    plt.hist(bins[:-1], bins, weights=counts, edgecolor='black', color='cyan')








    print(df1)

    # NEXT WE WANT TO TRACK ALBUM RELEASES + ARTIST POPULARITY
    # HOW MUCH DOES ARTIST'S POPULARITY CONTRIBUTE TO ALBUM'S POPULARITY
    # (AND VICE VERSA)?
    # DATAFRAME OF INTEREST 2
    df2 = pd.merge(albums_df, releases_df, on=["album_id", "artist_id"])
    df2 = pd.merge(df2, artists_df, on=["artist_id", "artist_name"])
    df2 = pd.merge(df2, album_pop_df, on="album_id")
    df2 = pd.merge(df2, album_chart_df, on="album_id")
    df2 = pd.merge(df2, tracks_df, on=['album_id'])
    song_id_col = df2.pop("track_id")
    df2.insert(0, "track_id", song_id_col)
    df2.drop(['release_date_precision_x', 'release_date_x'],
             axis='columns', inplace=True)
    df2.rename(columns={'release_date_y': 'release_date',
                        'release_date_precision_y': 'release_date_precision'},
               inplace=True)
    print(df2)

    # HOW MUCH DOES ARTIST'S POPULARITY CONTRIBUTE TO SONG'S POPULARITY
    # (AND VICE VERSA)?
    # DATAFRAME OF INTEREST 3
    df3 = pd.merge(artists_df, releases_df, on="artist_id")
    df3 = pd.merge(df3, artist_chart_df, on="artist_id")
    df3 = pd.merge(df3, artist_pop_df, on="artist_id")
    df3 = pd.merge(df3, songs_df, on=["artist_id","artist_name"])
    df3 = pd.merge(df3, tracks_df,
                   on=["album_id", "release_date", "release_date_precision"])
    df3['week'] = pd.to_datetime(df3['week'], format='%Y-%m-%d')
    #
    #
    #



# DEFINE HELPER FUNCTIONS

def drop_dups(df, subs):
    # df    REPRESENTS PANDAS DATAFRAME TO BE CHANGED
    # subs  ARG REPRESENTS SINGLE STRING OR LIST OF STRINGS FOR COLUMN NAMES
    #       WHERE DUPLICATES REMAIN
    #
    # GOAL OF THIS HELPER FUNCTION IS TO REMOVE DUPLICATES THAT MAY EXIST IN A
    # ROW IN DATAFRAME. FOR EXAMPLE, WE WON'T NEED TO SEE ONE SONG/ARTIST/ALBUM
    # MULTIPLE TIMES IN THE CHARTS SINCE WE'RE ALSO GIVEN THE AMOUNT OF WEEKS
    # THAT THE ARTIST HAS BEEN ON THE CHARTS.
    df.drop_duplicates(subset=subs, keep='first', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def extract_artist_ids(df):
    # df    REPRESENTS PANDAS DATAFRAME TO BE CHANGED
    #
    # GOAL OF THIS IS TO EXTRACT ARTIST_ID AND ARTIST_NAME FROM VAGUE 'ARTISTS'
    # COLUMN. COLUMN VALUES ARE IN A STRING FORMAT TO LOOK LIKE A DICTIONARY
    # BUT IT IS NOT BEING RECOGNIZED AS A DICTIONARY BY read_csv
    artist_ids = []
    artist_names = []
    for i in df['artists']:
        splitted = i.split("'")
        artist_ids.append(splitted[1])
        artist_names.append(splitted[3])
    # print(artist_ids)
    # print(artist_names)
    ids = pd.Series(artist_ids)
    names = pd.Series(artist_names)
    df['artist_id'] = ids
    df['artist_name'] = names
    df.drop('artists', axis='columns', inplace=True)
    return df


# START OF ETL SECTION
# EXTRACTING DATA FROM CSV SOURCES
# TRANSFER DATA INTO A STANDARD DATA MODEL (ON JUPYTER)
# LOAD IT INTO A DATABASE AT THE VERY END

# BELOW IS THE EXTRACT SECTION

# METADATA

albums_df = pd.read_csv("musicoset_metadata/albums.csv", sep='	')
albums_df = extract_artist_ids(albums_df)
albums_df.rename(columns={'name': 'album_name', 'image_url': 'album_image_url',
                          'popularity': 'album_popularity'}, inplace=True)

artists_df = pd.read_csv("musicoset_metadata/artists.csv", sep='	')
artists_df.rename(
    columns={'name': 'artist_name', 'image_url': 'artist_image_url',
             'popularity': 'artist_popularity'}, inplace=True)

releases_df = pd.read_csv("musicoset_metadata/releases.csv", sep='	')
songs_df = pd.read_csv("musicoset_metadata/songs.csv", sep='	')
songs_df = extract_artist_ids(songs_df)
songs_df.rename(
    columns={'name': 'song_name', 'popularity': 'song_popularity'}, inplace=True)

tracks_df = pd.read_csv("musicoset_metadata/tracks.csv", sep='	')
tracks_df.rename(columns={'song_id': 'track_id'}, inplace=True)

# POPULARITY
album_chart_df = pd.read_csv("musicoset_popularity/album_chart.csv",
                             sep='	')
album_chart_df = drop_dups(album_chart_df, ['album_id'])
album_pop_df = pd.read_csv("musicoset_popularity/album_pop.csv", sep='	')

artist_chart_df = pd.read_csv("musicoset_popularity/artist_chart.csv",
                              sep='	')
artist_chart_df = drop_dups(artist_chart_df, ['artist_id'])
artist_pop_df = pd.read_csv("musicoset_popularity/artist_pop.csv", sep='	')

song_chart_df = pd.read_csv("musicoset_popularity/song_chart.csv", sep='	')
song_chart_df = drop_dups(song_chart_df, ['song_id'])

song_pop_df = pd.read_csv("musicoset_popularity/song_pop.csv", sep='	')

# SONG FEATURES
acoustic_features_df = pd.read_csv(
    "musicoset_songfeatures/acoustic_features.csv", sep='	')
lyrics_df = pd.read_csv("musicoset_songfeatures/lyrics.csv", sep='	')

if __name__ == "__main__":
    main()
