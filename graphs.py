from itertools import product, groupby, chain
from collections import Counter
import numpy as np
from graph_tool.all import Graph, graph_draw, GraphView, edge_endpoint_property,\
    remove_parallel_edges
from graph_tool.topology import label_components
from graph_tool.spectral import adjacency
from graph_tool.inference.blockmodel import BlockState
from graph_tool.topology import transitive_closure, all_paths, max_cliques
from util import group_adjacent, plot, plot_matrix

def adjacency_matrix(graph):
    return adjacency(graph).toarray().T

def prune_isolated_vertices(g):
    not_isolated = g.new_vertex_property("bool")
    not_isolated.a = g.get_total_degrees(g.get_vertices()) > 0
    return GraphView(g, vfilt=not_isolated)

def graph_from_matrix(matrix):
    g = Graph(directed=False)
    g.add_vertex(len(matrix))
    g.add_edge_list(list(zip(*np.nonzero(matrix))))
    #graph_draw(g, output_size=(1000, 1000), output="results/structure.pdf")
    return g

def alignment_graph(lengths=[], pairings=[], alignments=[]):
    print('making graph')
    g = Graph(directed=False)
    seq_index = g.new_vertex_property("int")
    time = g.new_vertex_property("int")
    #add vertices
    g.add_vertex(sum(lengths))
    seq_index.a = np.concatenate([np.repeat(i,l) for i,l in enumerate(lengths)])
    time.a = np.concatenate([np.arange(l) for l in lengths])
    #add edges (alignments)
    for i,a in enumerate(alignments):
        if len(a) > 0:
            j, k = pairings[i]
            pairs = np.concatenate(a, axis=0)
            indicesJ = (np.arange(lengths[j]) + sum(lengths[:j]))[pairs.T[0]]
            indicesK = (np.arange(lengths[k]) + sum(lengths[:k]))[pairs.T[1]]
            g.add_edge_list(np.vstack([indicesJ, indicesK]).T)
    #g.add_edge_list([(b, a) for (a, b) in g.edges()])
    print('created alignment graph', g)
    #g = prune_isolated_vertices(g)
    #print('pruned alignment graph', g)
    #g = transitive_closure(g)
    #graph_draw(g, output_size=(1000, 1000), output="results/casey_jones_bars.pdf")
    return g, seq_index, time

#remove all alignments that are not reinforced by others from a simple a-graph
def clean_up(g, time, self_alignment):
    plot_matrix(np.triu(adjacency_matrix(g)), "results/clean0.png")
    graph_draw(g, output_size=(1000, 1000), output="results/clean_up0.pdf")
    segs = sorted(self_alignment, key=lambda a: len(a), reverse=True)
    reduced = Graph(directed=False)
    reduced.add_vertex(len(g.get_vertices()))
    votes = []
    for v in g.get_vertices():
        n = g.get_out_neighbors(v)
        if len(n) > 0:
            #split into connected segments
            combis = list(product(*group_adjacent(sorted(n))))
            #print(combis)
            for i,n in enumerate(combis):
                nn = [w for v in n for w in g.get_out_neighbors(v) if w > v]
                nn = np.unique(np.concatenate([n, np.array(nn, dtype=int)]))
                #print(v, n, nn)
                #make a subgraph for nn
                filt = g.new_vertex_property("bool")
                filt.a[nn] = True
                gg = GraphView(g, vfilt=filt)
                #check if elements of n in same clique
                cliques = list(max_cliques(gg))
                samecli = any(len(np.intersect1d(n, c)) == len(n) for c in cliques)
                #comps = label_components(gg)[0].fa[np.isin(nn, n).nonzero()[0]]
                #samecomp = np.all(comps == comps[0])
                if samecli:
                    votes.append(np.array(n) - v)
                    reduced.add_edge_list([(a,b) for i,a in enumerate(n) for b in n[i+1:]])
                #graph_draw(gg, vertex_text=g.vertex_index, output_size=(1000, 1000),
                #    output="results/clean_upp"+str(i)+".pdf")
    remove_parallel_edges(reduced)
    print(reduced)
    plot_matrix(np.triu(adjacency_matrix(reduced)), "results/clean1.png")
    graph_draw(reduced, output_size=(1000, 1000), output="results/clean_up1.pdf")
    votes = [tuple(v) for v in votes]
    print('counting')
    print(Counter(votes))
    # for s in segs[:1]:
    #     s = [[e[0],e[1]] for e in s]
    #     print(s)
    #     filt = g.new_edge_property("bool")
    #     filt.a = [[e[0],e[1]] not in s for e in g.get_edges()]
    #     gg = GraphView(g, efilt=filt)
    #     #print(gg)
    #     comps = label_components(gg)[0].a
    #     p = [comps[e[0]] == comps[e[1]] for e in s]
    #     print(p)
    #graph_draw(g, output_size=(1000, 1000), output="results/clean_up1.pdf")

def structure_graph(msa, alignment_graph, mask_threshold=.5):
    msa = [[int(m[1:]) if len(m) > 0 else -1 for m in a] for a in msa]
    matches = alignment_graph.new_vertex_property("int")
    matches.a = np.concatenate(msa)
    num_partitions = np.max(matches.a)+1
    # graph_draw(alignment_graph, output_size=(1000, 1000), vertex_fill_color=matches,
    #     output="results/box_of_rain_bars_c.pdf")
    #create connection matrix
    edge_ps = matches.a[alignment_graph.get_edges()] #partition memberships for edges
    edge_ps = edge_ps[np.where(np.all(edge_ps != -1, axis=1))] #filter out non-partitioned
    conn_matrix = np.zeros((num_partitions, num_partitions), dtype=int)
    np.add.at(conn_matrix, tuple(edge_ps.T), 1)
    np.add.at(conn_matrix, tuple(np.flip(edge_ps, axis=1).T), 1)
    conn_matrix = np.triu(conn_matrix, k=1)
    max_conn = np.max(conn_matrix)
    conn_matrix[np.where(conn_matrix < mask_threshold*max_conn)] = 0
    #create graph
    g = graph_from_matrix(conn_matrix)
    print('created structure graph', g)
    return g, conn_matrix, matches

def pattern_graph(sequences, pairings, alignments):
    #patterns = 
    return

def component_labels(g):
    labels, hist = label_components(g)
    return labels.a
