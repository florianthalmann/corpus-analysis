import os, subprocess, json
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
    with open(lexis_path+'output') as f:
        dag = json.load(f)
    #networkx.draw(nxg)
    #graph_draw(g, output_size=(1000, 1000), output="results/lexis.png")
    return dag

def lexis_sections(sequences):
    lex = {int(k):v for k,v in lexis(sequences).items()}
    seqs = [np.array(s) for s in lex[0]]
    sections = {k:np.array(v) for k,v in lex.items() if k != 0}
    occs = np.bincount(np.concatenate(list(sections.values())+seqs)+1)[1:]#ignore -1
    occs = {k:np.repeat(0, occs[k]) if k < len(occs) else 0 for k in lex.keys()}#dummy occs for now
    return seqs, sections, occs