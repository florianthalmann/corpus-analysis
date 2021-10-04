import os, subprocess, json, uuid
import numpy as np
from graph_tool.all import graph_draw
from matplotlib import pyplot as plt

lexis_path = './corpus_analysis/Lexis/'

def lexis(sequences):
    #generate id and write sequences to file
    id = str(uuid.uuid1())
    with open(lexis_path+id, 'w') as tf:
        tf.write('\n'.join([' '.join([str(i) for i in t]) for t in sequences]))
    #run lexis
    wd = os.getcwd()
    os.chdir(lexis_path)
    subprocess.call('python Lexis.py -t i -r r -q '+id, shell=True)
    #load result
    os.chdir(wd)
    with open(lexis_path+id+'-dag') as f:
        dag = json.load(f)
    with open(lexis_path+id+'-core') as f:
        core = json.load(f)
    #delete files
    os.remove(lexis_path+id)
    os.remove(lexis_path+id+'-dag')
    os.remove(lexis_path+id+'-core')
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