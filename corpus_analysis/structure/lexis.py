import os, subprocess, json
import numpy as np
from graph_tool.all import graph_draw
from matplotlib import pyplot as plt

lexis_path = './corpus_analysis/Lexis/'

def lexis(sequences):
    with open(lexis_path+'sequences', 'w') as tf:
        tf.write('\n'.join([' '.join([str(i) for i in t]) for t in sequences]))
    wd = os.getcwd()
    os.chdir(lexis_path)
    subprocess.call('python Lexis.py -t i -q sequences', shell=True)
    os.chdir(wd)
    with open(lexis_path+'output-dag') as f:
        dag = json.load(f)
    with open(lexis_path+'output-core') as f:
        core = json.load(f)
    return dag, core

def lexis_sections(sequences):
    #lexis code cannot have types < 1 (assigns 0 to sequences)
    min = np.min(np.concatenate(sequences))
    offset = 1-min if min < 1 else 0
    sequences = [s+offset for s in sequences]
    #run lexis
    dag, core = lexis(sequences)
    lex = {int(k):v for k,v in dag.items()}
    #convert types back
    seqs = [np.array(s)-offset for s in lex[0]]
    sections = {k-offset:np.array(v)-offset for k,v in lex.items() if k != 0}
    occs = np.bincount(np.concatenate(list(sections.values())+seqs)+1)[1:]#ignore -1
    occs = {k:np.repeat(0, occs[k]) if k < len(occs) else 0 for k in sections.keys()}#dummy occs for now
    #print('core', core)
    #print(len(core), len(sections.values()))
    return seqs, sections, occs, core