# -*- coding: utf-8 -*-
""" Laplacian segmentation """

# Code source: Brian McFee, Florian Thalmann
# License: ISC

from collections import defaultdict
import numpy as np
import scipy
import json, sys

import sklearn.cluster

import librosa

import matplotlib.pyplot as plt
import librosa.display

from ..util import plot_matrix


BPO = 12 * 3
N_OCTAVES = 7
EVEC_SMOOTH = 9
REC_SMOOTH = 9
MAX_TYPES = 12
REC_WIDTH = 9


def make_beat_sync_features(y, sr):

    #print('\tseparating harmonics..')
    yh = librosa.effects.harmonic(y, margin=8)

    #print('\tcomputing CQT...')
    C = librosa.amplitude_to_db(librosa.cqt(y=yh, sr=sr,
                                            bins_per_octave=BPO,
                                            n_bins=N_OCTAVES * BPO),
                                ref=np.max)

    #print('\ttracking beats...')
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    Csync = librosa.util.sync(C, beats, aggregate=np.median)

    beat_times = librosa.frames_to_time(librosa.util.fix_frames(beats,
                                                                x_min=0,
                                                                x_max=C.shape[1]),
                                        sr=sr)
    #print(len(beats), len(beat_times))
    
    #print('\tcomputing MFCCs...')
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    Msync = librosa.util.sync(mfcc, beats)

    return Csync, Msync, beat_times


def embed_beats(A_rep, A_loc):

    #print('\tbuilding recurrence graph...')
    R = librosa.segment.recurrence_matrix(A_rep, width=REC_WIDTH,
                                          mode='affinity',
                                          metric='cosine',
                                          sym=True)
    #plot_matrix(R, 'est00.png')
    
    # Enhance diagonals with a median filter (Equation 2)
    df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
    Rf = df(R, size=(1, REC_SMOOTH))
    
    #plot_matrix(Rf, 'est000.png')

    #print('\tbuilding local graph...')
    path_distance = np.sum(np.diff(A_loc, axis=1)**2, axis=0)
    sigma = np.median(path_distance)
    path_sim = np.exp(-path_distance / sigma)

    R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)

    ##########################################################
    # And compute the balanced combination (Equations 6, 7, 9)
    #print('\tcomputing the Laplacian')

    deg_path = np.sum(R_path, axis=1)
    deg_rec = np.sum(Rf, axis=1)

    mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec)**2)

    A = mu * Rf + (1 - mu) * R_path
    #A = Rf

    #####################################################
    # Now let's compute the normalized Laplacian (Eq. 10)
    L = scipy.sparse.csgraph.laplacian(A, normed=True)

    # and its spectral decomposition
    evals, evecs = scipy.linalg.eigh(L)

    # We can clean this up further with a median filter.
    # This can help smooth over small discontinuities
    evecs = scipy.ndimage.median_filter(evecs, size=(EVEC_SMOOTH, 1))

    return evecs

def decompose(A):
    
    #####################################################
    # Now let's compute the normalized Laplacian (Eq. 10)
    L = scipy.sparse.csgraph.laplacian(A, normed=True)
    
    np.set_printoptions(threshold=sys.maxsize)
    plt.matshow(L)

    # and its spectral decomposition
    evals, evecs = scipy.linalg.eigh(L)
    
    # plt.clf()
    # plt.plot(evals)
    # plt.show(block=False)

    # We can clean this up further with a median filter.
    # This can help smooth over small discontinuities
    evecs = scipy.ndimage.median_filter(evecs, size=(EVEC_SMOOTH, 1))

    return evecs
    

def cluster(evecs, Cnorm, k):
    X = evecs[:, :k] / Cnorm[:, k-1:k]
    X = np.nan_to_num(X).astype('float64')
    #print(evecs)
    #print(Cnorm)

    KM = sklearn.cluster.KMeans(n_clusters=k, n_init=50, max_iter=500)

    seg_ids = KM.fit_predict(X)
    
    
    # plt.subplot(1, MAX_TYPES, k)
    # librosa.display.specshow(np.atleast_2d(seg_ids).T, cmap=colors)
    

    ###############################################################
    # Locate segment boundaries from the label sequence
    bound_beats = 1 + np.flatnonzero(seg_ids[:-1] != seg_ids[1:])

    # Count beats 0 as a boundary
    bound_beats = librosa.util.fix_frames(bound_beats, x_min=0)

    # Compute the segment label for each boundary
    bound_segs = list(seg_ids[bound_beats])

    ivals, labs = [], []
    for interval, label in zip(zip(bound_beats, bound_beats[1:]), bound_segs):
        ivals.append(interval)
        labs.append(str(label))

    return ivals, labs


def _reindex_labels(ref_int, ref_lab, est_int, est_lab):
    # for each estimated label
    #    find the reference label that is maximally overlaps with

    score_map = defaultdict(lambda: 0)

    for r_int, r_lab in zip(ref_int, ref_lab):
        for e_int, e_lab in zip(est_int, est_lab):
            score_map[(e_lab, r_lab)] += max(0, min(e_int[1], r_int[1]) -
                                             max(e_int[0], r_int[0]))

    r_taken = set()
    e_map = dict()

    hits = [(score_map[k], k) for k in score_map]
    hits = sorted(hits, reverse=True)

    while hits:
        cand_v, (e_lab, r_lab) = hits.pop(0)
        if r_lab in r_taken or e_lab in e_map:
            continue
        e_map[e_lab] = r_lab
        r_taken.add(r_lab)

    # Anything left over is unused
    unused = set(est_lab) - set(ref_lab)

    for e, u in zip(set(est_lab) - set(e_map.keys()), unused):
        e_map[e] = u

    return [e_map[e] for e in est_lab]


def reindex(hierarchy):
    new_hier = [hierarchy[0]]
    for i in range(1, len(hierarchy)):
        ints, labs = hierarchy[i]
        labs = _reindex_labels(new_hier[i-1][0], new_hier[i-1][1], ints, labs)
        new_hier.append((ints, labs))

    return new_hier

#convert into levels where each segment has granularity 1 (so it can be aligned with other hierarchies)
def to_levels(segmentations, length):
    print([(len(s[0]), len(s[1])) for s in segmentations])
    levels = []
    for s in segmentations:
        if len(s[0]) > 0:
            # if (s == segmentations[-1]):
            #     print(list(zip(*s)))
            levels.append(np.concatenate(
                [np.repeat(int(seg[1]), seg[0][1]-seg[0][0]) for seg in zip(*s)]))
    #not needed... levels.insert(0, np.repeat(0, len(levels[0])))
    #add padding due to method yielding incomplete levels
    #levels = [np.concatenate([l, np.repeat(-1, length-len(l))]) for l in levels]
    print([(len(s[0]), len(s[1])) for s in levels])
    return np.array(levels)

def add_beat_times(idseg, times):
    return [(np.array([[times[i], times[j]] for (i,j) in l[0]]), l[1]) for l in idseg]

def segment_file(filename):
    #print('Loading {}'.format(filename))
    y, sr = librosa.load(filename)

    #print('Extracting features...'.format(filename))
    Csync, Msync, beat_times = make_beat_sync_features(y=y, sr=sr)

    #print('Constructing embedding...'.format(filename))
    embedding = embed_beats(Csync, Msync)

    Cnorm = np.cumsum(embedding**2, axis=1)**0.5

    #print('Clustering...'.format(filename))
    segmentations = []
    for k in range(1, MAX_TYPES):
        #print('\tk={}'.format(k))
        segmentations.append(cluster(embedding, Cnorm, k))

    #print('done.')
    #return to_levels(reindex(segmentations), len(beat_times)), beat_times
    return tuple(zip(*add_beat_times(reindex(segmentations), beat_times)))

def laplacian_segmentation(matrix):#pass in an affinity matrix
    matrix = np.matrix(matrix, dtype=int)
    matrix = np.maximum(matrix, matrix.transpose())
    
    embedding = decompose(matrix)
    Cnorm = np.cumsum(embedding**2, axis=1)**0.5

    segmentations = []
    for k in range(1, MAX_TYPES):
        segmentations.append(cluster(embedding, Cnorm, k))

    return reindex(segmentations)

#-------------

from ..alignment.affinity import get_affinity_matrix

#same as above method for comparison
def get_smooth_affinity_matrix(filename):
    y, sr = librosa.load(filename)
    Csync, Msync, beat_times = make_beat_sync_features(y=y, sr=sr)
    R = librosa.segment.recurrence_matrix(Csync, width=REC_WIDTH,
        mode='affinity', metric='cosine', sym=True)
    df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
    #beat_times, tuple(zip(*add_beat_times(reindex(segmentations), beat_times)))
    return df(R, size=(1, REC_SMOOTH)), beat_times

def get_laplacian_struct_from_affinity(affinity):
    struct = laplacian_segmentation(affinity)
    return to_levels(struct, affinity.shape[0])#np.append(struct, [sequence], axis=0)

def get_laplacian_struct_from_affinity2(affinity, beat_times):
    struct = laplacian_segmentation(affinity)
    struct = tuple(zip(*add_beat_times(struct, beat_times)))
    return struct[0], [np.array(s, dtype=int).tolist() for s in struct[1]]#np.append(struct, [sequence], axis=0)

def get_laplacian_struct_from_audio(audio):
    return segment_file(audio)
    #return struct, beat_times#np.append(struct, [sequence], axis=0)
