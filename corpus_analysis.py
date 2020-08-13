import os, json, timeit, cProfile
from features import get_beatwise_chords
from alignment import get_alignment
from matplotlib import pyplot as plt
import seaborn as sns

corpus = '../../FAST/fifteen-songs-dataset2/'
audio = os.path.join(corpus, 'tuned_audio')
features = os.path.join(corpus, 'features')
with open(os.path.join(corpus, 'dataset.json')) as f:
    dataset = json.load(f)

def get_subdirs(path):
    return [f.path for f in os.scandir(path) if f.is_dir()]

songs = list(dataset.keys())

def get_feature_path(song, version):
    id = version.replace('.mp3','.wav').replace('.','_').replace('/','_')
    return os.path.join(features, id, id)

def get_sequences(song):
    print("loading")
    version = list(dataset[song].keys())[60]
    beatsFile = get_feature_path(song, version) + '_madbars.json'
    chordsFile = get_feature_path(song, version) + '_gochords.json'
    chords = get_beatwise_chords(beatsFile, chordsFile)
    #pr = cProfile.Profile()
    #pr.enable()
    alignment = get_alignment(chords, chords, 16, 4)
    #pr.disable()
    #pr.print_stats()
    print(timeit.timeit(lambda: get_alignment(chords, chords, 16, 4), number=1))
    sns.heatmap(alignment, xticklabels=False, yticklabels=False, cmap=sns.cm.rocket_r)
    plt.show()

get_sequences(songs[0])