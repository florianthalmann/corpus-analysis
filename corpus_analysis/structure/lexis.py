import os, subprocess, pyintergraph, networkx
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
    nxg = networkx.read_gpickle(lexis_path+'output')
    networkx.draw(nxg)
    plt.show()
    print(nxg)
    g = pyintergraph.nx2gt(nxg)
    graph_draw(g, output_size=(1000, 1000), output="results/lexis.png")
    return g