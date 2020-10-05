from itertools import product, groupby, chain
from collections import Counter
import numpy as np
import sortednp as snp
from graph_tool.all import Graph, graph_draw, GraphView, edge_endpoint_property,\
    remove_parallel_edges
from graph_tool.topology import label_components
from graph_tool.spectral import adjacency
from graph_tool.inference.blockmodel import BlockState
from graph_tool.topology import transitive_closure, all_paths, max_cliques
from graph_tool.util import find_edge
from util import group_adjacent, plot, plot_matrix

def adjacency_matrix(graph, weight=None):
    return adjacency(graph, weight).toarray().T

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
    alm_index = g.new_edge_property("int")
    seg_index = g.new_edge_property("int")
    for i,a in enumerate(alignments):
        if len(a) > 0:
            j, k = pairings[i]
            pairs = np.concatenate(a, axis=0)
            indicesJ = (np.arange(lengths[j]) + sum(lengths[:j]))[pairs.T[0]]
            indicesK = (np.arange(lengths[k]) + sum(lengths[:k]))[pairs.T[1]]
            seg_indices = np.concatenate([np.repeat(i, len(a))
                for i,a in enumerate(a)])
            g.add_edge_list(np.vstack([indicesJ, indicesK,
                np.repeat(i, len(pairs)), seg_indices]).T,
                eprops=[alm_index, seg_index])
    #g.add_edge_list([(b, a) for (a, b) in g.edges()])
    print('created alignment graph', g)
    #g = prune_isolated_vertices(g)
    #print('pruned alignment graph', g)
    #g = transitive_closure(g)
    #graph_draw(g, output_size=(1000, 1000), output="results/casey_jones_bars.pdf")
    return g, seq_index, time, alm_index, seg_index

#combine a sequence of potential segment combinations into a consensus
#(individual seg_combos have to be sorted)
#can easily be made into beam search to speed up
def integrate_segment_combos(seg_combos):
    sets = []
    ratings = []
    subset_buffer = {}
    for k,sc in enumerate(seg_combos):
        if len(sets) == 0:
            sets = [c for c in sc]
            ratings = np.array([len(c) for c in sc])
            for c in sc: subset_buffer[str(c)] = np.array([], dtype=int)
        else:
            for c in sc:
                key = str(c)
                lenc = len(c)
                exists = key in subset_buffer
                subset_locs = subset_buffer[key] if exists \
                    else np.array([], dtype=int)
                #calculate intersections where c not a known subset
                isects = [c if i in subset_locs and subset_locs[i]
                    else snp.intersect(c, s) for i,s in enumerate(sets)]
                ilengths = np.array([len(i) for i in isects])
                #update locations where c subset of existing sets
                subset_locs = np.concatenate((subset_locs,
                    ilengths[len(subset_locs):] == lenc))
                subset_buffer[key] = subset_locs
                #update ratings where c subset
                np.add.at(ratings, np.nonzero(subset_locs), lenc)
                #add set if not added before
                if not exists:
                    #get max rating at locations where c superset
                    setlengths = np.array([len(s) for s in sets])
                    superset_locs = np.nonzero(ilengths == setlengths)[0]
                    max_rating = lenc
                    if len(superset_locs) > 0:
                        max_rating += np.max(ratings[superset_locs])
                    #append c to sets with appropriate rating
                    sets.append(c)
                    ratings = np.append(ratings, [max_rating])
        #print(k, len(sc), len(sets), len(ratings), max(ratings) if len(ratings) > 0 else 0, sum(ratings) if len(ratings) > 0 else 0)
    print(len(sets))
    return sets, ratings

#remove all alignments that are not reinforced by others from a simple a-graph
def clean_up(g, time, seg_index):
    #plot_matrix(np.triu(adjacency_matrix(g)), "results/clean0.png")
    #graph_draw(g, output_size=(1000, 1000), output="results/clean_up0.pdf")
    out_edges = [g.get_out_edges(v, [g.edge_index]) for v in g.get_vertices()]
    edge_combos = []
    #for each vertex find best combinations of neighbors 
    for v in g.get_vertices():#[49:50]:
        n = sorted(g.get_out_neighbors(v))
        #split into connected segments
        vertex_combos = list(product(*group_adjacent(sorted(n))))
        edge_combos.append([])
        for i,c in enumerate(vertex_combos):
            #collect internal edges of subgraph
            vertices = [v]+list(c)
            edges = np.concatenate([out_edges[v] for v in vertices])
            shared = edges[np.where(np.isin(edges[:,1], vertices))]
            edge_combos[-1].append(sorted(shared[:,2]))
            # filt = g.new_vertex_property("bool")
            # filt.a[[v]+list(c)] = True
            # gg = GraphView(g, vfilt=filt)
            # #check if elements of n in same clique
            # #cliques = list(max_cliques(gg))
            # #samecli = any(len(np.intersect1d(n, c)) == len(n) for c in cliques)
            # #or more simply, rate by how many edges there are
            # edge_combos[-1].append([e[2] for e in gg.get_edges([g.edge_index])])
            
            #if samecli:
            #    votes.append(np.array(n) - v)
            #    reduced.add_edge_list([(a,b) for i,a in enumerate(n) for b in n[i+1:]])
            #graph_draw(gg, vertex_text=g.vertex_index, output_size=(1000, 1000),
            #    output="results/clean_upp"+str(i)+".pdf")
    
    #look for combos with highest number of edges and lowest number of segments
    seg_combos = []
    for ec in edge_combos:
        max_num_edges = max([len(c) for c in ec])
        largest_combos = [c for c in ec if len(c) == max_num_edges]
        #convert to segment combos
        segs = [np.unique(seg_index.a[lc]) for lc in largest_combos]
        min_num_segs = min([len(s) for s in segs])
        if min_num_segs > 0:
            seg_combos.append([s for s in segs if len(s) == min_num_segs])
        else:
            seg_combos.append([])
        #print(max_num_edges, min_num_segs)
    
    #iteratively select best segment combination for remaining nodes
    sets, ratings = integrate_segment_combos(seg_combos)
    best = list(sets[np.argmax(ratings)])
    print(len(sets), np.max(ratings), best)
    
    threshold = 0.1 #0.01 works well for first example....
    edges = g.get_edges([seg_index])
    while max(ratings) > threshold*g.num_edges():
        involved_edges = edges[np.where(np.isin(edges[:,2], best))]
        involved_vertices = np.unique(np.concatenate(involved_edges[:,:2]))
        remaining_vertices = np.setdiff1d(g.get_vertices(), involved_vertices)
        print(len(involved_vertices), len(remaining_vertices), len(seg_combos))
        remaining_combos = [seg_combos[v] for v in remaining_vertices]
        
        sets, ratings = integrate_segment_combos(remaining_combos)
        if max(ratings) > threshold*g.num_edges():
            best = best + list(sets[np.argmax(ratings)])
            print(len(sets), np.max(ratings), list(sets[np.argmax(ratings)]))
    
    #print(edges[:100])
    reduced = Graph(directed=False)
    reduced.add_vertex(len(g.get_vertices()))
    edges = g.get_edges([seg_index])
    edges = edges[np.where(np.isin(edges[:,2], best))]
    reduced.add_edge_list(edges)
    print(reduced)
    #plot_matrix(np.triu(adjacency_matrix(reduced)), "results/cleani2.png")
    #graph_draw(reduced, output_size=(1000, 1000), output="results/clean_up1.pdf")
    return reduced

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
