import sys, json, operator, time, itertools
import numpy as np
from os import path
import argparse
from profile_hmm import ProfileHMM, FlankedProfileHMM

def load_json(path):
    with open(path) as ofile:
        return json.load(ofile)

def load_data(path):
    return load_json(path)["data"]#, loaded["labels"]

def train_model_from_data(data, verbose, match_match, delete_insert,
        dist_inertia, edge_inertia, max_iterations,
        model_length_func=np.median, model_type=ProfileHMM, flank_prob=0.999999):
    target_length = int(model_length_func([len(d) for d in data]))
    #take sequence closest to target length as init sequence
    init_sequence = sorted(data, key=lambda d: abs(len(d) - target_length))[0]
    training_data = data#np.array(np.delete(data, data.index(init_sequence), 0))
    num_features = max([d for r in data for d in r])+1
    if verbose:
        print('version count', len(data))
        print('model length', len(init_sequence))
        print('num features', num_features)
    model = model_type(len(init_sequence), num_features, init_sequence,
        match_match, delete_insert, flank_prob)
    if verbose:
        print('fitting model')
    before = time.time()
    history = model.fit(data, dist_inertia, edge_inertia, max_iterations)
    if verbose:
        print('total improvement', history.total_improvement[-1],
            'epochs', history.epochs[-1])
        print('took', round(time.time()-before), 'seconds')
    return model

def get_dist_max(dist):
    return max(dist.parameters[0], key=dist.parameters[0].get)

def print_viterbi_paths(data, model):
    for sequence in data[:20]:
        logp, path = model.viterbi(sequence)
        print(''.join( '-' if state.name[0] == 'I' #or state.name[0] == 'F' #insert or flank
            else str(chr(65+(get_dist_max(state.distribution)%26))) if state.name[0] == 'M'
            #else str(chr(65+(int(state.name[1:])%61))) if state.name[0] == 'M'
            else ''#state.name[0]
            for idx, state in path[1:-1]))

def get_results(data, model):
    viterbis = [model.viterbi(d) for d in data]
    logps = [v[0] for v in viterbis]
    paths = [v[1] for v in viterbis]
    no_del = [[state.name for idx, state in path[1:-1] if state.name[0] != 'D']
        for path in paths] #remove deletes
    msa = [[s if s[0] == 'M' else '' for s in path] for path in no_del] #hide inserts
    return msa, logps

def align_sequences(sequences, match_match=0.999, delete_insert=0.01,
        max_iterations=1, dist_inertia=0.8, edge_inertia=0.8, verbose=False,
        model_length_func=np.median, model_type=ProfileHMM, flank_prob=None):
    if verbose and isinstance(sequences[0][0], int):
        for sequence in map(list, sequences[:20]):
            print(''.join(str(chr(65+(s%26))) for s in sequence))
    model = train_model_from_data(sequences, verbose, match_match, delete_insert,
        dist_inertia, edge_inertia, max_iterations, model_length_func,
        model_type, flank_prob)
    print_viterbi_paths(sequences, model.model)
    return get_results(sequences, model.model, outfile)
