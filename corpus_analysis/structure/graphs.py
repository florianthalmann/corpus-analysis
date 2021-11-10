import tqdm
from itertools import product, chain, groupby, islice
from functools import reduce
from collections import Counter, OrderedDict
import numpy as np
import sortednp as snp
from graph_tool.all import Graph, graph_draw, GraphView, edge_endpoint_property,\
    remove_parallel_edges
from graph_tool.spectral import adjacency
from graph_tool.topology import transitive_closure, all_paths, max_cliques, label_components
from graph_tool.util import find_edge
from ..alignment.affinity import segments_to_matrix, matrix_to_segments
from ..util import group_adjacent, plot, plot_matrix

def adjacency_matrix(graph, weight=None):
    return adjacency(graph, weight).toarray().T

def prune_isolated_vertices(g):
    return GraphView(g, vfilt=g.get_total_degrees(g.get_vertices()) > 0)

def graph_from_matrix(matrix, directed=False):
    g = Graph(directed=directed)
    g.add_vertex(len(matrix))
    weights = g.new_ep("float")
    edges = np.nonzero(matrix)
    edges = np.append(edges, [matrix[edges]], axis=0)
    g.add_edge_list(np.transpose(edges), eprops=[weights])
    #graph_draw(g, output_size=(1000, 1000), output="results/structure.pdf")
    return g, weights

def alignment_graph(lengths=[], pairings=[], alignments=[]):
    #print('making graph')
    g = Graph(directed=False)
    seq_index = g.new_vertex_property("int")
    time = g.new_vertex_property("int")
    #add vertices
    g.add_vertex(sum(lengths))
    seq_index.a = np.concatenate([np.repeat(i,l) for i,l in enumerate(lengths)])
    time.a = np.concatenate([np.arange(l) for l in lengths])
    #add edges (alignments)
    alignment_index = g.new_edge_property("int")
    segment_index = g.new_edge_property("int")
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
                eprops=[alignment_index, segment_index])
    #g.add_edge_list([(b, a) for (a, b) in g.edges()])
    #print('created alignment graph', g)
    #g = prune_isolated_vertices(g)
    #print('pruned alignment graph', g)
    #g = transitive_closure(g)
    #graph_draw(g, output_size=(1000, 1000), output="results/casey_jones_bars.pdf")
    return g, seq_index, time, alignment_index, segment_index

#combine a sequence of potential segment combinations into a consensus
#(individual seg_combos have to be sorted)
#can easily be made into beam search to speed up
def integrate_segment_combos(seg_combos, incomp_segs):
    seg_combos = [s for s in seg_combos if len(s) > 0]
    sets = []
    ratings = []
    subset_buffer = {}
    positions = {}
    for k,sc in enumerate(seg_combos):
        if len(sets) == 0:
            sets = [c for c in sc]
            ratings = np.array([len(c) for c in sc])
            for i,c in enumerate(sc):
                key = str(list(c))
                positions[key] = i
                subset_buffer[key] = np.array([], dtype=int)
        else:
            for c in sc:
                key = str(list(c))
                lenc = len(c)
                exists = key in positions
                #keep a buffer of all the locations where c a subset
                subset_locs = subset_buffer[key] if exists \
                    else np.array([], dtype=int)
                #calculate intersections where c not a known subset
                isects = [c if len(subset_locs) > i and subset_locs[i] > 0
                    else snp.intersect(c, s) for i,s in enumerate(sets)]
                ilengths = np.array([len(i) for i in isects])
                #update locations where c subset of existing sets
                subset_locs = np.concatenate((subset_locs,
                    ilengths[len(subset_locs):] == lenc))
                subset_buffer[key] = subset_locs
                #update ratings where c subset
                np.add.at(ratings, np.nonzero(subset_locs), lenc)
                #add set if not added before and if not a subset of another
                if not exists and not np.any(subset_locs):
                    #get max rating at locations where c superset
                    setlengths = np.array([len(s) for s in sets])
                    superset_locs = np.nonzero(ilengths == setlengths)[0]
                    max_rating = lenc
                    if len(superset_locs) > 0:
                        max_rating += np.max(ratings[superset_locs])
                    #append c to sets with appropriate rating
                    sets.append(c)
                    subset_buffer[key] = np.array([], dtype=int)
                    ratings = np.append(ratings, [max_rating])
                    #append union if possible
                    for k,i in enumerate(isects):
                        #neither is a subset of the other
                        if len(i) < len(c) and len(i) < len(sets[k]):
                            union = np.unique(np.concatenate([c, sets[k]]))
                            ukey = str(list(union))
                            #union not added yet and no incompatible segments
                            if not ukey in positions and\
                                    not any(len(snp.intersect(j, union)) > 1 for j in incomp_segs):
                                positions[ukey] = len(sets)
                                sets.append(union)
                                subset_buffer[ukey] = np.array([], dtype=int)
                                #rating of union = sum - rating of intersection
                                rating = max_rating+ratings[k]
                                ikey = str(list(i))
                                if ikey in positions:
                                    rating -= ratings[positions[ikey]]
                                ratings = np.append(ratings, [rating])
        #print(k, len(sc), len(sets), len(ratings), max(ratings) if len(ratings) > 0 else 0, sum(ratings) if len(ratings) > 0 else 0)
    #[print(ratings[i], s) for i, s in enumerate(sets)]
    return sets, ratings

#returns a group index for each vertex. the segments of vertices with the same
#index are incompatible
def get_incompatible_segments(g, seg_index, out_edges):
    incomp_graph = Graph(directed=False)
    num_segs = np.max(seg_index.a)+1
    incomp_graph.add_vertex(num_segs)
    for v in g.get_vertices():
        for vs in group_adjacent(sorted(g.get_out_neighbors(v))):
            edges = out_edges[v][np.where(np.isin(out_edges[v][:,1], vs))][:,2]
            segments = list(np.unique(seg_index.a[edges]))
            [incomp_graph.add_edge(s,t)
                for i,s in enumerate(segments) for t in segments[i+1:]]
    return label_components(incomp_graph)[0].a

def get_edge_combos(g):
    out_edges = [g.get_out_edges(v, [g.edge_index]) for v in g.get_vertices()]
    edge_combos = []
    #for each vertex find best combinations of neighbors 
    for v in g.get_vertices():#[49:50]:
        n = sorted(g.get_out_neighbors(v))
        #split into connected segments
        vertex_combos = list(product(*group_adjacent(sorted(n))))
        edge_combos.append([])
        for c in vertex_combos:
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
    return edge_combos

def remove_subarrays(arrays, proper=True):
    arrays = sorted(arrays, key=lambda a: len(a))
    to_remove = []
    for i,a in enumerate(arrays):
        #bs = [b for b in arrays[i+1:] if len(b) < len(a)] if proper else arrays[i+1:]
        for b in arrays[i+1:]:
            if len(snp.intersect(a, b)) == len(a):
                to_remove.append(i)
                break
    print(len(arrays), len(to_remove))
    return [a for i,a in enumerate(arrays) if i not in to_remove]

def is_subarray(array, arrays):
    for a in arrays:
        if len(snp.intersect(array, a)) == len(a):
            return True
    return False

def sorted_unique(array):
    return snp.merge(array, np.array([]), duplicates=snp.DROP)

#no incompatible nodes
def validate_quasi_clique(quasi_clique, validated, incomp):
    if quasi_clique not in validated:
        incomps = np.array([incomp[u] for u in quasi_clique])
        validated[quasi_clique] = len(sorted_unique(incomps)) == len(incomps)

#TODO change to breadth first and see... probably faster!
def quasi_clique_expansions(quasi_cliques, cliques, incomp):
    max_expansions = set()
    temp_expansions = set([tuple(q) for q in quasi_cliques])
    expanded = set()
    valid = dict()
    while len(temp_expansions) > 0:
        current = temp_expansions.pop()
        expanded.add(current)
        new_expansions = set()
        #partition cliques into compatible and incompatible?
        #TODO NOT NECESSARY TO DO ALL!! ONLY CLIQUES THAT WERE NOT TRIED YET!
        #keep index path for each in temp ([qc[i], c[j], c[k], ....])
        #keep compatibility set for each in temp
        #----------
        #make pairwise comp graph for cs. possible combinations are max cliques.
        #for each qc calculate comp with each c and take subgraph
        unions = [tuple(snp.merge(np.array(current), c, duplicates=snp.DROP))
            for c in cliques]
        [validate_quasi_clique(u, valid, incomp) for u in unions]
        new_expansions = [u for u in unions if valid[u] and len(u) > len(current)] #== len(current)+1
        #append and remove all sub-quasi-cliques
        if len(new_expansions) == 0:
            max_expansions.add(current)
        new_expansions = [e for e in new_expansions if e not in expanded]
        temp_expansions.update(new_expansions)
        #temp_expansions = remove_subarrays(temp_expansions)#<---------------
    print(len(max_expansions), len(expanded), len(valid))
    max_expansions = [np.array(e) for e in max_expansions]
    #max_expansions = remove_subarrays(max_expansions)#<---------------
    #print(len(max_expansions))
    return max_expansions if len(max_expansions) > 1 else []

def valid_clique(c, incomp):
    incomps = np.array([incomp[u] for u in c])
    return len(np.unique(incomps)) == len(incomps)

def valid_combo(c1, c2, incomp):
    return valid_clique(snp.merge(c1, c2, duplicates=snp.DROP), incomp)

def quasi_clique_expansions2(quasi_cliques, cliques, incomp):
    array_cliques = [np.array(c) for c in cliques.keys()]
    c_comps = np.array([[valid_combo(array_cliques[i], array_cliques[j], incomp)
        if j > i else 0
        for j in range(len(array_cliques))] for i in range(len(array_cliques))])
    #print('ccomps', c_comps.shape)
    c_comp_graph = graph_from_matrix(c_comps)[0]
    #print('cgraph', c_comp_graph)
    max_expansions = dict()
    for q,r in quasi_cliques.items():
        q = np.array(q)
        qcomps = np.array([valid_combo(q, array_cliques[i], incomp)
            for i in range(len(array_cliques))])
        #print('qcomps', len(qcomps))
        if np.any(qcomps):
            view = GraphView(c_comp_graph, vfilt=qcomps > 0)
            #get max cliques of compatible patterns
            combos = list(max_cliques(view))
            #add compatible isolated vertices missing in cliques
            vs = view.get_vertices()
            isolated = vs[view.get_total_degrees(vs) == 0]
            combos += [[i] for i in isolated]
            #build combos and merge into quasi-cliques
            combos = [[array_cliques[i] for i in c] for c in combos]
            qcliques = [tuple(list(snp.kway_merge(*([q]+co), duplicates=snp.DROP))) 
                for co in combos]
            qratings = [r+sum([cliques[tuple(c)] for c in co]) for co in combos]
            max_expansions.update(zip(qcliques, qratings))
            #print('upd')
    return max_expansions#[np.array(m) for m in max_expansions] if len(max_expansions) > 1 else []

def chunks(dict, SIZE=10000):
    it = iter(dict)
    for i in range(0, len(dict), SIZE):
        yield {k:dict[k] for k in islice(it, SIZE)}

#iteratively build all possible combinations of cliques while respecting incompatible nodes
def maximal_quasi_cliques(cliques, incomp_vertices):
    max_size = max([len(c) for c in cliques])
    chunk_size = 25
    clique_chunks = [{c:r for c, r in cliques.items() if len(c) == s}
        for s in range(max_size, 0, -1)]
    clique_chunks = [x for c in clique_chunks for x in chunks(c, chunk_size)]
    quasi_cliques = dict()
    #go through groups of cliques by size, start with largest cliques
    for g in clique_chunks:
        if len(quasi_cliques) == 0:
            quasi_cliques = g
        #merge cliques depth-first in all combinations until no longer possible
        new_qcs = quasi_clique_expansions2(quasi_cliques, g, incomp_vertices)
        #append and remove all sub-quasi-cliques
        quasi_cliques.update(new_qcs)
        #keep best rated ones
        best = sorted(quasi_cliques.items(), key=lambda c: c[1], reverse=True)[:10]
        quasi_cliques = {c:r for c,r in best}
        #super = remove_subarrays([np.array(q) for q in quasi_cliques.keys()])
        #super = [tuple(s) for s in super]
        #quasi_cliques = {s:quasi_cliques[s] for s in super}
    return quasi_cliques

def get_edge_combos2(g):
    edge_combos = []
    #for each vertex find best combinations of neighbors 
    for v in g.get_vertices():#[150:155]:#[49:50]:
        n = sorted(g.get_out_neighbors(v))
        #split into connected segments
        #print(v, len(n))
        combos = []
        if len(n) > 0:
            vs = np.array([v]+list(n))
            adjacents = group_adjacent(vs)
            filt = g.new_vertex_property("bool")
            filt.a[vs] = True
            gg = GraphView(g, vfilt=filt)
            cliques = [np.sort(c) for c in list(max_cliques(gg))]
            qcs = maximal_quasi_cliques(cliques, adjacents)
            if len(qcs) > 0:
                ratings = []
                edges_ids = []
                for q in qcs:
                    filt = g.new_vertex_property("bool")
                    filt.a[q] = True
                    ggg = GraphView(g, vfilt=filt)
                    ratings.append(ggg.num_edges() / ggg.num_vertices())
                    edges_ids.append(ggg.get_edges([g.edge_index])[:,2])
                max_rating = max(ratings)
                best = [q for i,q in enumerate(qcs) if ratings[i] == max_rating]
                print(v, len(n), max_rating, len(best))
                combos = edges_ids
        edge_combos.append(combos)
    return edge_combos

def add_to_counting_dict(item, counting_dict):
    if item not in counting_dict:
        counting_dict[item] = 0
    counting_dict[item] += 1

def integrate_subsets(clique_dict):
    integrated = dict()
    cliques = sorted(clique_dict.items(), key=lambda i: len(i[0]), reverse=True)
    integrated[cliques[0][0]] = cliques[0][1]
    for s in cliques[1:]:
        subset = False
        #if s is subset somewhere, increase ratings of parents, else add to dict
        for t in integrated.keys():
            a1, a2 = np.array(s[0]), np.array(t)
            if len(a1) < len(a2) and len(snp.intersect(a1, a2)) == len(a1):
                integrated[t] += s[1]
                subset = True
        if not subset:
            integrated[s[0]] = s[1]
    return integrated

def get_segment_combos(g, seg_index):
    #print(seg_index.a)
    out_edges = [g.get_out_edges(v, [g.edge_index]) for v in g.get_vertices()]
    incomp_segs = get_incompatible_segments(g, seg_index, out_edges)
    #print('incomp', len(incomp_segs))
    segment_combos = []
    segment_cliques = dict()
    for v in g.get_vertices():
        n = sorted(g.get_out_neighbors(v))
        vs = np.array([v]+list(n))
        filt = g.new_vertex_property("bool")
        filt.a[vs] = True
        gg = GraphView(g, vfilt=filt)
        cliques = [list(np.sort(c)) for c in list(max_cliques(gg))]
        cliq_edges = [out_edges[v][np.where(np.isin(out_edges[v][:,1], c))][:,2]
            for c in cliques]
        cliq_segs = [sorted(seg_index.a[e]) for e in cliq_edges]
        [add_to_counting_dict(tuple(np.unique(s)), segment_cliques) for s in cliq_segs]
    #print('cliques', len(segment_cliques))
    #print(segment_cliques)
    segment_cliques = {c:n for c,n in segment_cliques.items()
        if valid_clique(c, incomp_segs)}
    #print('valid cliques', len(segment_cliques))
    #print(segment_cliques)
    segment_cliques = integrate_subsets(segment_cliques)
    #print('integrated', len(segment_cliques))
    #print(segment_cliques)
    return maximal_quasi_cliques(segment_cliques, incomp_segs)

#remove all alignments that are not reinforced by others from a simple a-graph
def clean_up(g, seg_index):
    #plot_matrix(np.triu(adjacency_matrix(g)), "results/clean0.png")
    #graph_draw(g, output_size=(1000, 1000), output="results/clean_up0.pdf")
    
    seg_combos = get_segment_combos(g, seg_index)
    best = sorted(seg_combos.items(), key=lambda c: c[1], reverse=True)#[:200]
    #print(best)
    best = best[0][0]
    #print(best)
    
    #print(edges[:100])
    reduced = Graph(directed=False)
    reduced.add_vertex(len(g.get_vertices()))
    edges = g.get_edges([seg_index])
    edges = edges[np.where(np.isin(edges[:,2], best))]
    reduced.add_edge_list(edges)
    #print(reduced)
    #plot_matrix(np.triu(adjacency_matrix(reduced)), "results/cleani2.png")
    #graph_draw(reduced, output_size=(1000, 1000), output="results/clean_up1.pdf")
    return reduced

def structure_graph(msa, alignment_graph, max_segments=None, mask_threshold=.5):
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
    segments = matrix_to_segments(conn_matrix)
    print(len(segments))
    avgcounts = [np.mean(conn_matrix[s.T[0], s.T[1]]) for s in segments]
    segments = sorted(zip(segments, avgcounts), key=lambda s: s[1], reverse=True)
    if max_segments: segments = [s[0] for s in segments[:max_segments]]
    conn_matrix = np.triu(segments_to_matrix(segments, conn_matrix.shape), k=1)
    #create graph
    g = graph_from_matrix(conn_matrix)[0]
    #print('created structure graph', g)
    return g, conn_matrix, matches

def component_labels(g):
    labels, hist = label_components(g)
    return labels.a

